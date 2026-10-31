//! Deep handlers for kernel launch.
//!
//! The launch configuration is in the signature and records as it stands.
//! The arguments are the hard part, and the two ways CUDA accepts them are
//! not equally recoverable:
//!
//! - `extra` is a null-terminated list of key/value pairs that *states* the
//!   argument buffer's address and its size. The whole buffer is recorded.
//! - `kernelParams` is a bare `void**`: an array of pointers to the argument
//!   values, with no element count and no element sizes anywhere in the ABI.
//!   Both live only in the module's metadata, which the driver holds and the
//!   interposer cannot see. Reading `kernelParams[0]` is already a guess, so
//!   capture records the array's address and stops there. Inventing a length
//!   would fault the target and store the wrong bytes when it did not.

use std::ffi::{c_uint, c_void};

use gputrace_core::{ArgValue, Argument, HandleId};

use crate::handlers::handles::{HandleKind, handle_id};
use crate::handlers::stream::{stream_arg, stream_of};
use crate::recorder::{self, DeepCall};
use crate::shim::{CUDA_ERROR_NOT_FOUND, CUresult, DriverFn};

type CUfunction = *mut c_void;
type CUstream = *mut c_void;

/// What a captured argument buffer is called in a record's payloads.
const KERNEL_PARAMS: &str = "kernel_params";

/// `CU_LAUNCH_PARAM_END`, `_BUFFER_POINTER` and `_BUFFER_SIZE`: CUDA uses
/// three unaligned low addresses as the keys of the `extra` list, so no real
/// pointer can be mistaken for one.
const PARAM_END: usize = 0x00;
const PARAM_BUFFER_POINTER: usize = 0x01;
const PARAM_BUFFER_SIZE: usize = 0x02;

fn function_id(function: CUfunction) -> Option<HandleId> {
    handle_id(HandleKind::Function, function as usize as u64)
}

/// The argument buffer `extra` describes, as an address and a length.
///
/// # Safety
/// `extra` must be null or a `CU_LAUNCH_PARAM_END`-terminated list.
unsafe fn packed_params(extra: *mut *mut c_void) -> Option<(*const u8, usize)> {
    if extra.is_null() {
        return None;
    }
    let (mut buffer, mut len) = (std::ptr::null::<u8>(), None);
    let mut at = extra;
    loop {
        let key = unsafe { *at } as usize;
        if key == PARAM_END {
            break;
        }
        let value = unsafe { *at.add(1) };
        match key {
            PARAM_BUFFER_POINTER => buffer = value.cast::<u8>(),
            // The size arrives behind a pointer, not by value.
            PARAM_BUFFER_SIZE if !value.is_null() => len = Some(unsafe { *value.cast::<usize>() }),
            _ => {}
        }
        at = unsafe { at.add(2) };
    }
    match (buffer.is_null(), len) {
        (false, Some(len)) => Some((buffer, len)),
        _ => None,
    }
}

/// # Safety
/// `extra` must be null or a `CU_LAUNCH_PARAM_END`-terminated list whose
/// buffer is readable for the length it declares.
unsafe fn with_params(deep: DeepCall, extra: *mut *mut c_void) -> DeepCall {
    match unsafe { packed_params(extra) } {
        Some((buffer, len)) => unsafe { deep.payload(KERNEL_PARAMS, buffer, len) },
        None => deep,
    }
}

macro_rules! launch_shims {
    ($($symbol:ident;)*) => {$(
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $symbol(
            f: CUfunction,
            grid_dim_x: c_uint,
            grid_dim_y: c_uint,
            grid_dim_z: c_uint,
            block_dim_x: c_uint,
            block_dim_y: c_uint,
            block_dim_z: c_uint,
            shared_mem_bytes: c_uint,
            h_stream: CUstream,
            kernel_params: *mut *mut c_void,
            extra: *mut *mut c_void,
        ) -> CUresult {
            static REAL: DriverFn<
                unsafe extern "C" fn(
                    CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
                    CUstream, *mut *mut c_void, *mut *mut c_void,
                ) -> CUresult,
            > = DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let deep = call
                .deep(vec![
                    Argument::new("f", match function_id(f) {
                        Some(id) => ArgValue::Handle(id),
                        None => ArgValue::Null,
                    }),
                    Argument::new("gridDimX", ArgValue::Unsigned(u64::from(grid_dim_x))),
                    Argument::new("gridDimY", ArgValue::Unsigned(u64::from(grid_dim_y))),
                    Argument::new("gridDimZ", ArgValue::Unsigned(u64::from(grid_dim_z))),
                    Argument::new("blockDimX", ArgValue::Unsigned(u64::from(block_dim_x))),
                    Argument::new("blockDimY", ArgValue::Unsigned(u64::from(block_dim_y))),
                    Argument::new("blockDimZ", ArgValue::Unsigned(u64::from(block_dim_z))),
                    Argument::new(
                        "sharedMemBytes",
                        ArgValue::Unsigned(u64::from(shared_mem_bytes)),
                    ),
                    stream_arg(h_stream),
                    Argument::new(
                        "kernelParams",
                        ArgValue::HostPtr(kernel_params as usize as u64),
                    ),
                    Argument::new("extra", ArgValue::HostPtr(extra as usize as u64)),
                ])
                .consumes(function_id(f))
                .consumes(Some(stream_of(h_stream)));
            let deep = unsafe { with_params(deep, extra) };
            let result = match REAL.get() {
                Some(real) => unsafe {
                    real(
                        f, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
                        block_dim_z, shared_mem_bytes, h_stream, kernel_params, extra,
                    )
                },
                None => CUDA_ERROR_NOT_FOUND,
            };
            call.finish_deep(result, deep);
            result
        }
    )*};
}

launch_shims! {
    cuLaunchKernel;
    cuLaunchKernel_ptsz;
}
