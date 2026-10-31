//! Deep handlers for the memory subset: allocation, release, and the
//! host-to-device copy.
//!
//! What a pass-through shim cannot say about these calls is what replay
//! needs most — which allocation a copy targeted, and what was copied. Each
//! spelling of a call therefore binds the device pointer to a
//! [`HandleId`](gputrace_core::HandleId) and, for a copy, stages the source
//! buffer for the blob store. Every `_vN`/`_ptds` spelling is covered: a
//! target built against any CUDA version must record the same way.

use std::ffi::{c_uint, c_void};

use gputrace_core::{ArgValue, Argument};

use crate::handlers::handles::{HandleKind, handle_id};
use crate::handlers::stream::{stream_arg, stream_of};
use crate::recorder;
use crate::shim::{CUDA_ERROR_NOT_FOUND, CUresult, DriverFn, RecordArg};

const CUDA_SUCCESS: CUresult = 0;

/// `cuMemAlloc`, which hands back a device pointer the trace must be able to
/// follow. The pointer is recorded as the call's product, not as the host
/// address it was written through.
macro_rules! alloc_shims {
    ($($symbol:ident($pointer:ty, $size:ty);)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(dptr: *mut $pointer, bytesize: $size) -> CUresult {
            static REAL: DriverFn<unsafe extern "C" fn(*mut $pointer, $size) -> CUresult> =
                DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let result = match REAL.get() {
                Some(real) => unsafe { real(dptr, bytesize) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            let allocated = if result == CUDA_SUCCESS && !dptr.is_null() {
                u64::from(unsafe { *dptr })
            } else {
                0
            };
            let deep = call
                .deep(vec![
                    Argument::new("dptr", ArgValue::DevicePtr(allocated)),
                    Argument::new("bytesize", RecordArg::record(&bytesize)),
                ])
                .produces(handle_id(HandleKind::DevicePtr, allocated));
            call.finish_deep(result, deep);
            result
        }
    )*};
}

macro_rules! free_shims {
    ($($symbol:ident($pointer:ty);)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(dptr: $pointer) -> CUresult {
            static REAL: DriverFn<unsafe extern "C" fn($pointer) -> CUresult> =
                DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let pointer = u64::from(dptr);
            let call = recorder::begin(stringify!($symbol));
            let deep = call
                .deep(vec![Argument::new("dptr", ArgValue::DevicePtr(pointer))])
                .destroys(handle_id(HandleKind::DevicePtr, pointer));
            let result = match REAL.get() {
                Some(real) => unsafe { real(dptr) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// `cuMemcpyHtoD`. The payload is staged before the driver runs: it is what
/// the caller sent, and the caller is free to overwrite the buffer the
/// instant the copy returns.
macro_rules! copy_to_device_shims {
    ($($symbol:ident($pointer:ty, $size:ty);)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(
            dst_device: $pointer,
            src_host: *const c_void,
            byte_count: $size,
        ) -> CUresult {
            static REAL: DriverFn<
                unsafe extern "C" fn($pointer, *const c_void, $size) -> CUresult,
            > = DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let destination = u64::from(dst_device);
            let call = recorder::begin(stringify!($symbol));
            let deep = unsafe {
                call.deep(vec![
                    Argument::new("dstDevice", ArgValue::DevicePtr(destination)),
                    Argument::new("srcHost", ArgValue::HostPtr(src_host as usize as u64)),
                    Argument::new("ByteCount", RecordArg::record(&byte_count)),
                ])
                .consumes(handle_id(HandleKind::DevicePtr, destination))
                .payload(HOST_TO_DEVICE, src_host.cast::<u8>(), byte_count as usize)
            };
            let result = match REAL.get() {
                Some(real) => unsafe { real(dst_device, src_host, byte_count) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// `cuMemcpyDtoH`, the only call in a trace that says what the GPU answered.
///
/// Its payload is taken *after* forwarding, the mirror image of a
/// host-to-device copy: until the driver has run, the destination still holds
/// whatever the caller last left there. A failed copy leaves it untouched
/// altogether, so there is no answer to keep.
macro_rules! copy_to_host_shims {
    ($($symbol:ident($pointer:ty, $size:ty);)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(
            dst_host: *mut c_void,
            src_device: $pointer,
            byte_count: $size,
        ) -> CUresult {
            static REAL: DriverFn<
                unsafe extern "C" fn(*mut c_void, $pointer, $size) -> CUresult,
            > = DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let source = u64::from(src_device);
            let call = recorder::begin(stringify!($symbol));
            let result = match REAL.get() {
                Some(real) => unsafe { real(dst_host, src_device, byte_count) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            let deep = call
                .deep(vec![
                    Argument::new("dstHost", ArgValue::HostPtr(dst_host as usize as u64)),
                    Argument::new("srcDevice", ArgValue::DevicePtr(source)),
                    Argument::new("ByteCount", RecordArg::record(&byte_count)),
                ])
                .consumes(handle_id(HandleKind::DevicePtr, source));
            let deep = if result == CUDA_SUCCESS {
                unsafe { deep.output(GOLDEN_OUTPUT, dst_host.cast::<u8>(), byte_count as usize) }
            } else {
                deep
            };
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// The stream-ordered allocator. It is the same allocation a replay has to
/// remap as the synchronous one, and it is also an operation in a stream's
/// order, so it records both: the pointer it produced and the stream it was
/// issued into.
macro_rules! alloc_async_shims {
    ($($symbol:ident;)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(
            dptr: *mut u64,
            bytesize: usize,
            h_stream: *mut c_void,
        ) -> CUresult {
            static REAL: DriverFn<
                unsafe extern "C" fn(*mut u64, usize, *mut c_void) -> CUresult,
            > = DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let result = match REAL.get() {
                Some(real) => unsafe { real(dptr, bytesize, h_stream) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            let allocated = if result == CUDA_SUCCESS && !dptr.is_null() {
                unsafe { *dptr }
            } else {
                0
            };
            let deep = call
                .deep(vec![
                    Argument::new("dptr", ArgValue::DevicePtr(allocated)),
                    Argument::new("bytesize", RecordArg::record(&bytesize)),
                    stream_arg(h_stream),
                ])
                .produces(handle_id(HandleKind::DevicePtr, allocated))
                .consumes(Some(stream_of(h_stream)));
            call.finish_deep(result, deep);
            result
        }
    )*};
}

macro_rules! free_async_shims {
    ($($symbol:ident;)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(dptr: u64, h_stream: *mut c_void) -> CUresult {
            static REAL: DriverFn<unsafe extern "C" fn(u64, *mut c_void) -> CUresult> =
                DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let deep = call
                .deep(vec![
                    Argument::new("dptr", ArgValue::DevicePtr(dptr)),
                    stream_arg(h_stream),
                ])
                .destroys(handle_id(HandleKind::DevicePtr, dptr))
                .consumes(Some(stream_of(h_stream)));
            let result = match REAL.get() {
                Some(real) => unsafe { real(dptr, h_stream) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// `cuMemcpyHtoDAsync`. The caller may overwrite its buffer as soon as the
/// call returns even though the copy has not run yet, so staging before
/// forwarding is not an optimisation here — it is the only correct moment.
macro_rules! copy_to_device_async_shims {
    ($($symbol:ident($pointer:ty, $size:ty);)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(
            dst_device: $pointer,
            src_host: *const c_void,
            byte_count: $size,
            h_stream: *mut c_void,
        ) -> CUresult {
            static REAL: DriverFn<
                unsafe extern "C" fn($pointer, *const c_void, $size, *mut c_void) -> CUresult,
            > = DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let destination = u64::from(dst_device);
            let call = recorder::begin(stringify!($symbol));
            let deep = unsafe {
                call.deep(vec![
                    Argument::new("dstDevice", ArgValue::DevicePtr(destination)),
                    Argument::new("srcHost", ArgValue::HostPtr(src_host as usize as u64)),
                    Argument::new("ByteCount", RecordArg::record(&byte_count)),
                    stream_arg(h_stream),
                ])
                .consumes(handle_id(HandleKind::DevicePtr, destination))
                .consumes(Some(stream_of(h_stream)))
                .payload(HOST_TO_DEVICE, src_host.cast::<u8>(), byte_count as usize)
            };
            let result = match REAL.get() {
                Some(real) => unsafe { real(dst_device, src_host, byte_count, h_stream) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// What a staged host-to-device buffer is called in a record's payloads.
pub(crate) const HOST_TO_DEVICE: &str = "host_to_device";
/// What a captured device-to-host result is called. The name a later
/// oracle's differential signal looks for (GLOSSARY: golden output).
pub(crate) const GOLDEN_OUTPUT: &str = "golden_output";

alloc_shims! {
    cuMemAlloc(c_uint, c_uint);
    cuMemAlloc_v2(u64, usize);
}

free_shims! {
    cuMemFree(c_uint);
    cuMemFree_v2(u64);
}

copy_to_device_shims! {
    cuMemcpyHtoD(c_uint, c_uint);
    cuMemcpyHtoD_v2(u64, usize);
    cuMemcpyHtoD_v2_ptds(u64, usize);
}

copy_to_host_shims! {
    cuMemcpyDtoH(c_uint, c_uint);
    cuMemcpyDtoH_v2(u64, usize);
    cuMemcpyDtoH_v2_ptds(u64, usize);
}

alloc_async_shims! {
    cuMemAllocAsync;
    cuMemAllocAsync_ptsz;
}

free_async_shims! {
    cuMemFreeAsync;
    cuMemFreeAsync_ptsz;
}

copy_to_device_async_shims! {
    cuMemcpyHtoDAsync(c_uint, c_uint);
    cuMemcpyHtoDAsync_v2(u64, usize);
    cuMemcpyHtoDAsync_v2_ptsz(u64, usize);
}
