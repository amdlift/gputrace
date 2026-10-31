//! The ABI-level CUDA Driver API types the stub speaks, transcribed from
//! `cuda.h`. The stub deliberately depends on no CUDA headers or crates.

use std::ffi::{c_int, c_uint, c_void};
use std::sync::OnceLock;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUdeviceptrV1 = c_uint;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
pub const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;
pub const CUDA_ERROR_INVALID_CONTEXT: CUresult = 201;
pub const CUDA_ERROR_INVALID_HANDLE: CUresult = 400;
pub const CUDA_ERROR_NOT_FOUND: CUresult = 500;

pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

pub const CU_GET_PROC_ADDRESS_SUCCESS: c_int = 0;
pub const CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND: c_int = 1;
pub const CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT: c_int = 2;

pub const STUB_DRIVER_VERSION: c_int = 13040;
pub const STUB_DEVICE_NAME: &str = "gputrace-stub-device-0";
pub const STUB_DEVICE_COUNT: c_int = 1;
/// Names how many devices the stub presents, for a test that needs more than
/// one GPU in a process. Unset means [`STUB_DEVICE_COUNT`], so every fixture
/// that does not ask sees exactly the single-device driver it always saw.
pub const STUB_DEVICES_ENV: &str = "GPUTRACE_STUB_DEVICES";
pub const STUB_DEVICE_TOTAL_MEM: usize = 8 << 30;
pub const STUB_COMPUTE_CAPABILITY: (c_int, c_int) = (8, 6);

/// How many devices this process's stub presents, read once.
pub fn device_count() -> c_int {
    static COUNT: OnceLock<c_int> = OnceLock::new();
    *COUNT.get_or_init(|| {
        std::env::var(STUB_DEVICES_ENV)
            .ok()
            .and_then(|count| count.parse().ok())
            .filter(|count| *count >= 1)
            .unwrap_or(STUB_DEVICE_COUNT)
    })
}

pub fn device_exists(dev: CUdevice) -> bool {
    (0..device_count()).contains(&dev)
}

/// Device 0 keeps the name single-device fixtures already assert on.
pub fn device_name(dev: CUdevice) -> String {
    format!("gputrace-stub-device-{dev}")
}
