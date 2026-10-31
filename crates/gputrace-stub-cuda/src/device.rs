use std::ffi::{c_char, c_int, c_uint};

use crate::log::note;
use crate::out;
use crate::state::state;
use crate::types::*;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDeviceGetCount(count: *mut c_int) -> CUresult {
    note("cuDeviceGetCount");
    if !state().initialized {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if unsafe { out(count, device_count()) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult {
    note("cuDeviceGet");
    if !state().initialized {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if !device_exists(ordinal) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if unsafe { out(device, ordinal) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
    note("cuDeviceGetName");
    if !state().initialized {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if !device_exists(dev) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if name.is_null() || len <= 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let name_of_device = device_name(dev);
    let bytes = name_of_device.as_bytes();
    let copied = bytes.len().min(len as usize - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), name, copied);
        name.add(copied).write(0);
    }
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDeviceGetAttribute(
    pi: *mut c_int,
    attrib: c_int,
    dev: CUdevice,
) -> CUresult {
    note("cuDeviceGetAttribute");
    if !state().initialized {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if !device_exists(dev) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    let value = match attrib {
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK => 1024,
        CU_DEVICE_ATTRIBUTE_WARP_SIZE => 32,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT => 48,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => STUB_COMPUTE_CAPABILITY.0,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => STUB_COMPUTE_CAPABILITY.1,
        _ => return CUDA_ERROR_INVALID_VALUE,
    };
    if unsafe { out(pi, value) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDeviceComputeCapability(
    major: *mut c_int,
    minor: *mut c_int,
    dev: CUdevice,
) -> CUresult {
    note("cuDeviceComputeCapability");
    if !state().initialized {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if !device_exists(dev) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if unsafe { out(major, STUB_COMPUTE_CAPABILITY.0) && out(minor, STUB_COMPUTE_CAPABILITY.1) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult {
    note("cuDeviceTotalMem_v2");
    if !state().initialized {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if !device_exists(dev) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if unsafe { out(bytes, STUB_DEVICE_TOTAL_MEM) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

/// The pre-CUDA-3.2 spelling: its out-parameter is 32-bit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDeviceTotalMem(bytes: *mut c_uint, dev: CUdevice) -> CUresult {
    note("cuDeviceTotalMem");
    if !state().initialized {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if !device_exists(dev) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    let total = c_uint::try_from(STUB_DEVICE_TOTAL_MEM).unwrap_or(c_uint::MAX);
    if unsafe { out(bytes, total) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}
