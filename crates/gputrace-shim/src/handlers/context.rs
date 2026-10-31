//! Deep handlers for context lifecycle.
//!
//! A context is the unit a thread's work belongs to: which GPU it runs on,
//! and which handles it owns. These handlers bind each context to a
//! [`HandleId`](gputrace_core::HandleId) and keep the recorder's idea of
//! what device the calling thread is on in step with the calls that move it,
//! so a device-agnostic call in a multi-GPU process is still attributable.
//!
//! Every `cuCtxCreate` spelling is covered: a target built against any CUDA
//! version must record the same way, and CUDA 13's `cuda.h` remaps plain
//! `cuCtxCreate` onto `_v4`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_int, c_uint, c_void};
use std::sync::{LazyLock, RwLock};

use gputrace_core::{ArgValue, Argument, HandleId};

use crate::generated::cuda_types::CUcontext;
use crate::handlers::golden::{self, Settles};
use crate::handlers::handles::{HandleKind, handle_id, handle_value};
use crate::recorder;
use crate::shim::{CUDA_ERROR_NOT_FOUND, CUDA_SUCCESS, CUresult, DriverFn, RecordArg};

/// Which device each live context belongs to.
///
/// Only context creation and teardown write it and only a call that changes
/// the current context reads it — never the common record path, where a lock
/// would serialize the very concurrency capture exists to observe (ADR-0004).
static CONTEXT_DEVICES: LazyLock<RwLock<HashMap<u64, u32>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

thread_local! {
    /// What a `cuCtxPopCurrent` on this thread goes back to, mirroring the
    /// driver's own per-thread context stack.
    static PUSHED_DEVICES: RefCell<Vec<Option<u32>>> = const { RefCell::new(Vec::new()) };
}

fn remember(context: u64, device: c_int) {
    let Ok(device) = u32::try_from(device) else {
        return;
    };
    if let Ok(mut known) = CONTEXT_DEVICES.write() {
        known.insert(context, device);
    }
}

fn forget(context: u64) {
    if let Ok(mut known) = CONTEXT_DEVICES.write() {
        known.remove(&context);
    }
}

fn device_of(context: u64) -> Option<u32> {
    CONTEXT_DEVICES.read().ok()?.get(&context).copied()
}

/// A context handle is an opaque address with no meaning to a replay; its id
/// is the whole of what the record can say.
fn context_arg(context: u64) -> Argument {
    Argument::new("ctx", handle_value(context_id(context)))
}

fn context_id(context: u64) -> Option<HandleId> {
    handle_id(HandleKind::Context, context)
}

/// `cuCtxCreate`, whose spellings differ only in the parameters they take
/// before `flags` and `dev`.
macro_rules! create_shims {
    ($($symbol:ident($($extra:ident: $ty:ty as $recorded:literal),*);)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(
            pctx: *mut CUcontext,
            $($extra: $ty,)*
            flags: c_uint,
            dev: c_int,
        ) -> CUresult {
            static REAL: DriverFn<
                unsafe extern "C" fn(*mut CUcontext, $($ty,)* c_uint, c_int) -> CUresult,
            > = DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let result = match REAL.get() {
                Some(real) => unsafe { real(pctx, $($extra,)* flags, dev) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            let created = if result == CUDA_SUCCESS && !pctx.is_null() {
                (unsafe { *pctx }) as usize as u64
            } else {
                0
            };
            if created != 0 {
                remember(created, dev);
            }
            let deep = call
                .deep(vec![
                    context_arg(created),
                    $(Argument::new($recorded, RecordArg::record(&$extra)),)*
                    Argument::new("flags", ArgValue::Unsigned(u64::from(flags))),
                    Argument::new("dev", ArgValue::Signed(i64::from(dev))),
                ])
                .produces(context_id(created));
            call.finish_deep(result, deep);
            result
        }
    )*};
}

macro_rules! destroy_shims {
    ($($symbol:ident;)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(ctx: CUcontext) -> CUresult {
            static REAL: DriverFn<unsafe extern "C" fn(CUcontext) -> CUresult> =
                DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let context = ctx as usize as u64;
            let call = recorder::begin(stringify!($symbol));
            let deep = call
                .deep(vec![context_arg(context)])
                .destroys(context_id(context));
            let result = match REAL.get() {
                Some(real) => unsafe { real(ctx) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            if result == CUDA_SUCCESS {
                forget(context);
            }
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// `cuCtxSetCurrent` and `cuCtxPushCurrent`: both move the calling thread
/// onto the named context's device. The record is tagged with where the
/// thread lands, the way a `cuCtxCreate` record is.
macro_rules! enter_shims {
    ($($symbol:ident, remembers_the_previous: $stacks:literal;)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(ctx: CUcontext) -> CUresult {
            static REAL: DriverFn<unsafe extern "C" fn(CUcontext) -> CUresult> =
                DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let context = ctx as usize as u64;
            let call = recorder::begin(stringify!($symbol));
            let deep = call
                .deep(vec![context_arg(context)])
                .consumes(context_id(context));
            let result = match REAL.get() {
                Some(real) => unsafe { real(ctx) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            if result == CUDA_SUCCESS {
                if $stacks {
                    PUSHED_DEVICES
                        .with_borrow_mut(|stack| stack.push(recorder::bound_device()));
                }
                recorder::bind_device(device_of(context));
            }
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// `cuCtxPopCurrent`, which hands back the context it leaves. The record is
/// tagged with the device the thread was on, the way a teardown is; the
/// thread only moves once the call is recorded.
macro_rules! leave_shims {
    ($($symbol:ident;)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(pctx: *mut CUcontext) -> CUresult {
            static REAL: DriverFn<unsafe extern "C" fn(*mut CUcontext) -> CUresult> =
                DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let result = match REAL.get() {
                Some(real) => unsafe { real(pctx) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            let left = if result == CUDA_SUCCESS && !pctx.is_null() {
                (unsafe { *pctx }) as usize as u64
            } else {
                0
            };
            let deep = call
                .deep(vec![context_arg(left)])
                .consumes(context_id(left));
            call.finish_deep(result, deep);
            if result == CUDA_SUCCESS {
                let restored = PUSHED_DEVICES.with_borrow_mut(|stack| stack.pop());
                recorder::bind_device(restored.flatten());
            }
            result
        }
    )*};
}

/// `cuCtxGetCurrent` only reports; it moves nothing.
macro_rules! report_shims {
    ($($symbol:ident;)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(pctx: *mut CUcontext) -> CUresult {
            static REAL: DriverFn<unsafe extern "C" fn(*mut CUcontext) -> CUresult> =
                DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let result = match REAL.get() {
                Some(real) => unsafe { real(pctx) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            let current = if result == CUDA_SUCCESS && !pctx.is_null() {
                (unsafe { *pctx }) as usize as u64
            } else {
                0
            };
            let deep = call
                .deep(vec![context_arg(current)])
                .consumes(context_id(current));
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// `cuCtxSynchronize`, which blocks the calling thread until everything its
/// context has been asked to do is done — including every read-back this
/// thread staged.
macro_rules! context_sync_shims {
    ($($symbol:ident($($arg:ident: $ty:ty),*);)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol($($arg: $ty),*) -> CUresult {
            static REAL: DriverFn<unsafe extern "C" fn($($ty),*) -> CUresult> =
                DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let deep = call.deep(vec![$(
                Argument::new(stringify!($arg), RecordArg::record(&$arg))
            ),*]);
            let result = match REAL.get() {
                Some(real) => unsafe { real($($arg),*) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            let deep = if result == CUDA_SUCCESS {
                unsafe { golden::settle(deep, Settles::Thread) }
            } else {
                deep
            };
            call.finish_deep(result, deep);
            result
        }
    )*};
}

create_shims! {
    cuCtxCreate();
    cuCtxCreate_v2();
    cuCtxCreate_v3(params_array: *mut c_void as "paramsArray", num_params: c_int as "numParams");
    cuCtxCreate_v4(create_params: *mut c_void as "ctxCreateParams");
}

destroy_shims! {
    cuCtxDestroy;
    cuCtxDestroy_v2;
}

enter_shims! {
    cuCtxSetCurrent, remembers_the_previous: false;
    cuCtxPushCurrent, remembers_the_previous: true;
    cuCtxPushCurrent_v2, remembers_the_previous: true;
}

leave_shims! {
    cuCtxPopCurrent;
    cuCtxPopCurrent_v2;
}

report_shims! {
    cuCtxGetCurrent;
}

context_sync_shims! {
    cuCtxSynchronize();
    cuCtxSynchronize_v2(ctx: CUcontext);
}
