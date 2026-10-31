//! A stub `libcuda`: a deterministic, in-memory implementation of the subset
//! of the CUDA Driver API that gputrace's capture tests drive. Pointed at by
//! `GPUTRACE_LIBCUDA`, it stands in for `libcuda.so.1` so capture is testable
//! with no GPU present.
//!
//! Test support only — never a dependency of the shipped interposer or CLI.

mod log;
mod state;
mod types;

use std::ffi::{c_char, c_int, c_uint};

use log::note;
use state::state;
pub use types::*;

unsafe fn out<T>(ptr: *mut T, value: T) -> bool {
    if ptr.is_null() {
        return false;
    }
    unsafe { ptr.write(value) };
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn cuInit(_flags: c_uint) -> CUresult {
    note("cuInit");
    state().initialized = true;
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDriverGetVersion(driver_version: *mut c_int) -> CUresult {
    note("cuDriverGetVersion");
    if !unsafe { out(driver_version, STUB_DRIVER_VERSION) } {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDeviceGetCount(count: *mut c_int) -> CUresult {
    note("cuDeviceGetCount");
    if !state().initialized {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if !unsafe { out(count, STUB_DEVICE_COUNT) } {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
    note("cuDeviceGetName");
    if !state().initialized {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if dev != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if name.is_null() || len <= 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let bytes = STUB_DEVICE_NAME.as_bytes();
    let copied = bytes.len().min(len as usize - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), name, copied);
        name.add(copied).write(0);
    }
    CUDA_SUCCESS
}
