use std::ffi::{c_char, c_int, c_uint};

use crate::log::note;
use crate::out;
use crate::state::state;
use crate::types::*;

#[unsafe(no_mangle)]
pub extern "C" fn cuInit(_flags: c_uint) -> CUresult {
    note("cuInit");
    state().initialized = true;
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDriverGetVersion(driver_version: *mut c_int) -> CUresult {
    note("cuDriverGetVersion");
    if unsafe { out(driver_version, STUB_DRIVER_VERSION) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGetErrorName(error: CUresult, str_out: *mut *const c_char) -> CUresult {
    note("cuGetErrorName");
    let Some(name) = error_name(error) else {
        return CUDA_ERROR_INVALID_VALUE;
    };
    if unsafe { out(str_out, name.as_ptr().cast::<c_char>()) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGetErrorString(
    error: CUresult,
    str_out: *mut *const c_char,
) -> CUresult {
    note("cuGetErrorString");
    let Some(name) = error_name(error) else {
        return CUDA_ERROR_INVALID_VALUE;
    };
    if unsafe { out(str_out, name.as_ptr().cast::<c_char>()) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

fn error_name(error: CUresult) -> Option<&'static str> {
    Some(match error {
        CUDA_SUCCESS => "CUDA_SUCCESS\0",
        CUDA_ERROR_INVALID_VALUE => "CUDA_ERROR_INVALID_VALUE\0",
        CUDA_ERROR_OUT_OF_MEMORY => "CUDA_ERROR_OUT_OF_MEMORY\0",
        CUDA_ERROR_NOT_INITIALIZED => "CUDA_ERROR_NOT_INITIALIZED\0",
        CUDA_ERROR_INVALID_DEVICE => "CUDA_ERROR_INVALID_DEVICE\0",
        CUDA_ERROR_INVALID_CONTEXT => "CUDA_ERROR_INVALID_CONTEXT\0",
        CUDA_ERROR_INVALID_HANDLE => "CUDA_ERROR_INVALID_HANDLE\0",
        CUDA_ERROR_NOT_FOUND => "CUDA_ERROR_NOT_FOUND\0",
        _ => return None,
    })
}
