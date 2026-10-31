//! Deep handlers for stream and event lifecycle: the calls the
//! happens-before partial order is reconstructed from (ADR-0004).
//!
//! Capture enforces no order. What it must do instead is leave every edge
//! findable: a `cuEventRecord` names the stream it closed over and the
//! event that carries it, and a `cuStreamWaitEvent` names the stream that
//! blocks and the same event. Those two records are the whole of a
//! cross-stream edge, so both halves record the handles by id and by name —
//! a role binding alone could not say which of two consumed handles was the
//! stream.

use std::ffi::{c_int, c_uint, c_void};

use gputrace_core::{ArgValue, Argument, HandleId};

use crate::handlers::golden::{self, Settles};
use crate::handlers::handles::{HandleKind, handle_id, stream_id};
use crate::recorder;
use crate::shim::{CUDA_ERROR_NOT_FOUND, CUresult, DriverFn, RecordArg};

const CUDA_SUCCESS: CUresult = 0;

type CUstream = *mut c_void;
type CUevent = *mut c_void;

/// What a record calls the stream it is ordered in. The offline
/// reconstruction finds a call's stream by this name, so every handler that
/// touches one uses it.
pub(crate) const STREAM_ARG: &str = "hStream";
/// What a record calls the event it records or waits on.
pub(crate) const EVENT_ARG: &str = "hEvent";

pub(crate) fn stream_arg(stream: CUstream) -> Argument {
    Argument::new(STREAM_ARG, ArgValue::Handle(stream_of(stream)))
}

pub(crate) fn stream_of(stream: CUstream) -> HandleId {
    stream_id(stream as usize as u64)
}

fn event_arg(event: CUevent) -> Argument {
    Argument::new(
        EVENT_ARG,
        match event_of(event) {
            Some(id) => ArgValue::Handle(id),
            None => ArgValue::Null,
        },
    )
}

fn event_of(event: CUevent) -> Option<HandleId> {
    handle_id(HandleKind::Event, event as usize as u64)
}

/// `cuStreamCreate` and its priority-taking twin, which differ only in the
/// parameters following `flags`.
macro_rules! stream_create_shims {
    ($($symbol:ident($flags_name:literal $(, $extra:ident: $ty:ty as $recorded:literal)*);)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(
            phstream: *mut CUstream,
            flags: c_uint,
            $($extra: $ty,)*
        ) -> CUresult {
            static REAL: DriverFn<
                unsafe extern "C" fn(*mut CUstream, c_uint $(, $ty)*) -> CUresult,
            > = DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let result = match REAL.get() {
                Some(real) => unsafe { real(phstream, flags $(, $extra)*) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            let created = if result == CUDA_SUCCESS && !phstream.is_null() {
                unsafe { *phstream }
            } else {
                std::ptr::null_mut()
            };
            let deep = call
                .deep(vec![
                    stream_arg(created),
                    Argument::new($flags_name, ArgValue::Unsigned(u64::from(flags))),
                    $(Argument::new($recorded, RecordArg::record(&$extra)),)*
                ])
                .produces(Some(stream_of(created)));
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// `cuStreamSynchronize`, which blocks the calling thread until the stream
/// drains — and so is the moment every read-back queued into that stream
/// finally means something (see [`crate::handlers::golden`]).
macro_rules! stream_sync_shims {
    ($($symbol:ident;)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(hstream: CUstream) -> CUresult {
            static REAL: DriverFn<unsafe extern "C" fn(CUstream) -> CUresult> =
                DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let stream = stream_of(hstream);
            let deep = call.deep(vec![stream_arg(hstream)]).consumes(Some(stream));
            let result = match REAL.get() {
                Some(real) => unsafe { real(hstream) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            let deep = if result == CUDA_SUCCESS {
                unsafe { golden::settle(deep, Settles::Stream(stream)) }
            } else {
                deep
            };
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// `cuEventSynchronize`, which blocks the calling thread until the event is
/// reached, and so settles the read-backs its recording closed over.
macro_rules! event_sync_shims {
    ($($symbol:ident;)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(hevent: CUevent) -> CUresult {
            static REAL: DriverFn<unsafe extern "C" fn(CUevent) -> CUresult> =
                DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let event = event_of(hevent);
            let deep = call.deep(vec![event_arg(hevent)]).consumes(event);
            let result = match REAL.get() {
                Some(real) => unsafe { real(hevent) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            let deep = match event.filter(|_| result == CUDA_SUCCESS) {
                Some(event) => unsafe { golden::settle(deep, Settles::Event(event)) },
                None => deep,
            };
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// `cuStreamDestroy`: one stream, ended.
macro_rules! stream_shims {
    ($($symbol:ident ends_the_stream: $destroys:literal;)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(hstream: CUstream) -> CUresult {
            static REAL: DriverFn<unsafe extern "C" fn(CUstream) -> CUresult> =
                DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let args = vec![stream_arg(hstream)];
            let id = Some(stream_of(hstream));
            let deep = if $destroys {
                call.deep(args).destroys(id)
            } else {
                call.deep(args).consumes(id)
            };
            let result = match REAL.get() {
                Some(real) => unsafe { real(hstream) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// `cuStreamWaitEvent`: one half of every cross-stream edge.
macro_rules! wait_event_shims {
    ($($symbol:ident($flags_name:literal);)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(
            hstream: CUstream,
            hevent: CUevent,
            flags: c_uint,
        ) -> CUresult {
            static REAL: DriverFn<
                unsafe extern "C" fn(CUstream, CUevent, c_uint) -> CUresult,
            > = DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let deep = call
                .deep(vec![
                    stream_arg(hstream),
                    event_arg(hevent),
                    Argument::new($flags_name, ArgValue::Unsigned(u64::from(flags))),
                ])
                .consumes(Some(stream_of(hstream)))
                .consumes(event_of(hevent));
            let result = match REAL.get() {
                Some(real) => unsafe { real(hstream, hevent, flags) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            call.finish_deep(result, deep);
            result
        }
    )*};
}

macro_rules! event_create_shims {
    ($($symbol:ident;)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(phevent: *mut CUevent, flags: c_uint) -> CUresult {
            static REAL: DriverFn<unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult> =
                DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let result = match REAL.get() {
                Some(real) => unsafe { real(phevent, flags) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            let created = if result == CUDA_SUCCESS && !phevent.is_null() {
                unsafe { *phevent }
            } else {
                std::ptr::null_mut()
            };
            let deep = call
                .deep(vec![
                    event_arg(created),
                    Argument::new("Flags", ArgValue::Unsigned(u64::from(flags))),
                ])
                .produces(event_of(created));
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// `cuEventRecord`: the other half of every cross-stream edge. The event
/// closes over everything issued into `hStream` so far, which is why the
/// record has to name the stream and not only the event.
macro_rules! event_record_shims {
    ($($symbol:ident($($flags:ident: $ty:ty),*);)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(
            hevent: CUevent,
            hstream: CUstream,
            $($flags: $ty,)*
        ) -> CUresult {
            static REAL: DriverFn<
                unsafe extern "C" fn(CUevent, CUstream $(, $ty)*) -> CUresult,
            > = DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let deep = call
                .deep(vec![
                    event_arg(hevent),
                    stream_arg(hstream),
                    $(Argument::new(stringify!($flags), RecordArg::record(&$flags)),)*
                ])
                .consumes(event_of(hevent))
                .consumes(Some(stream_of(hstream)));
            let result = match REAL.get() {
                Some(real) => unsafe { real(hevent, hstream $(, $flags)*) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            if result == CUDA_SUCCESS {
                golden::recorded(event_of(hevent), stream_of(hstream));
            }
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// `cuEventDestroy`: one event, ended. It names no stream, so it orders
/// nothing between streams.
macro_rules! event_shims {
    ($($symbol:ident ends_the_event: $destroys:literal;)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(hevent: CUevent) -> CUresult {
            static REAL: DriverFn<unsafe extern "C" fn(CUevent) -> CUresult> =
                DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let args = vec![event_arg(hevent)];
            let id = event_of(hevent);
            let deep = if $destroys {
                call.deep(args).destroys(id)
            } else {
                call.deep(args).consumes(id)
            };
            let result = match REAL.get() {
                Some(real) => unsafe { real(hevent) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            call.finish_deep(result, deep);
            result
        }
    )*};
}

stream_create_shims! {
    cuStreamCreate("Flags");
    cuStreamCreateWithPriority("flags", priority: c_int as "priority");
}

stream_sync_shims! {
    cuStreamSynchronize;
    cuStreamSynchronize_ptsz;
}

stream_shims! {
    cuStreamDestroy ends_the_stream: true;
    cuStreamDestroy_v2 ends_the_stream: true;
}

wait_event_shims! {
    cuStreamWaitEvent("Flags");
    cuStreamWaitEvent_ptsz("Flags");
}

event_create_shims! {
    cuEventCreate;
}

event_record_shims! {
    cuEventRecord();
    cuEventRecord_ptsz();
    cuEventRecordWithFlags(flags: c_uint);
    cuEventRecordWithFlags_ptsz(flags: c_uint);
}

event_sync_shims! {
    cuEventSynchronize;
}

event_shims! {
    cuEventDestroy ends_the_event: true;
    cuEventDestroy_v2 ends_the_event: true;
}
