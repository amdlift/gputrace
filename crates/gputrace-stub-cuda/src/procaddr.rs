//! The dispatch-table path. `cuda.h` hides a versioned spelling behind each
//! public name, so a caller asking for `cuMemAlloc` expects `cuMemAlloc_v2`
//! back; asking with the per-thread-default-stream flag expects the `_ptsz`
//! twin. Both arities the driver exports are provided: v1 (4 arguments,
//! CUDA 11.3) and v2 (5, CUDA 12+), because a v1 caller reaching a v2 body
//! would read an argument that was never passed.

use std::ffi::{CStr, c_char, c_int, c_void};

use crate::log::note;
use crate::out;
use crate::types::*;

const FIRST_VERSION_WITH_PROC_ADDRESS: c_int = 11030;
const PER_THREAD_DEFAULT_STREAM: u64 = 1 << 1;

macro_rules! entry_points {
    ($($module:ident :: $name:ident),* $(,)?) => {
        fn exact(symbol: &str) -> Option<*mut c_void> {
            match symbol {
                $(stringify!($name) => {
                    Some(crate::$module::$name as *const () as *mut c_void)
                })*
                _ => None,
            }
        }
    };
}

entry_points! {
    init::cuInit,
    init::cuDriverGetVersion,
    init::cuGetErrorName,
    init::cuGetErrorString,
    device::cuDeviceGet,
    device::cuDeviceGetCount,
    device::cuDeviceGetName,
    device::cuDeviceGetAttribute,
    device::cuDeviceComputeCapability,
    device::cuDeviceTotalMem,
    device::cuDeviceTotalMem_v2,
    context::cuCtxCreate,
    context::cuCtxCreate_v2,
    context::cuCtxCreate_v3,
    context::cuCtxCreate_v4,
    context::cuCtxDestroy,
    context::cuCtxDestroy_v2,
    context::cuCtxSetCurrent,
    context::cuCtxGetCurrent,
    context::cuCtxSynchronize,
    memory::cuMemAlloc,
    memory::cuMemAlloc_v2,
    memory::cuMemFree,
    memory::cuMemFree_v2,
    memory::cuMemcpyHtoD,
    memory::cuMemcpyHtoD_v2,
    memory::cuMemcpyHtoD_v2_ptds,
    memory::cuMemcpyDtoH,
    memory::cuMemcpyDtoH_v2,
    memory::cuMemcpyDtoH_v2_ptds,
    memory::cuMemcpyDtoD_v2,
    memory::cuMemsetD8_v2,
    module::cuModuleLoadData,
    module::cuModuleLoadDataEx,
    module::cuModuleGetFunction,
    module::cuModuleUnload,
    launch::cuLaunchKernel,
    launch::cuLaunchKernel_ptsz,
    stream::cuStreamCreate,
    stream::cuStreamCreateWithPriority,
    stream::cuStreamDestroy,
    stream::cuStreamDestroy_v2,
    stream::cuStreamSynchronize,
    stream::cuStreamSynchronize_ptsz,
    stream::cuStreamWaitEvent,
    stream::cuStreamWaitEvent_ptsz,
    event::cuEventCreate,
    event::cuEventRecord,
    event::cuEventRecord_ptsz,
    event::cuEventSynchronize,
    event::cuEventDestroy,
    event::cuEventDestroy_v2,
    event::cuEventElapsedTime,
    event::cuEventElapsedTime_v2,
    procaddr::cuGetProcAddress,
    procaddr::cuGetProcAddress_v2,
}

fn newest_spelling(symbol: &str) -> Option<String> {
    ["_v4", "_v3", "_v2"]
        .iter()
        .map(|suffix| format!("{symbol}{suffix}"))
        .find(|candidate| exact(candidate).is_some())
        .or_else(|| exact(symbol).map(|_| symbol.to_owned()))
}

fn resolve(symbol: &str, cuda_version: c_int, flags: u64) -> Result<*mut c_void, c_int> {
    if cuda_version < FIRST_VERSION_WITH_PROC_ADDRESS {
        return Err(CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT);
    }
    let spelling = newest_spelling(symbol).ok_or(CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND)?;
    if flags & PER_THREAD_DEFAULT_STREAM != 0 {
        for suffix in ["_ptsz", "_ptds"] {
            if let Some(address) = exact(&format!("{spelling}{suffix}")) {
                return Ok(address);
            }
        }
    }
    exact(&spelling).ok_or(CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND)
}

fn get_proc_address(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cuda_version: c_int,
    flags: u64,
    symbol_status: *mut c_int,
) -> CUresult {
    if symbol.is_null() || pfn.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let Ok(symbol) = unsafe { CStr::from_ptr(symbol) }.to_str() else {
        return CUDA_ERROR_INVALID_VALUE;
    };

    match resolve(symbol, cuda_version, flags) {
        Ok(address) => {
            unsafe {
                out(pfn, address);
                out(symbol_status, CU_GET_PROC_ADDRESS_SUCCESS);
            }
            CUDA_SUCCESS
        }
        Err(status) => {
            unsafe {
                out(pfn, std::ptr::null_mut());
                out(symbol_status, status);
            }
            CUDA_ERROR_NOT_FOUND
        }
    }
}

/// The CUDA 11.3 arity, kept because binaries from that era still call it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGetProcAddress(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cuda_version: c_int,
    flags: u64,
) -> CUresult {
    note("cuGetProcAddress");
    get_proc_address(symbol, pfn, cuda_version, flags, std::ptr::null_mut())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cuda_version: c_int,
    flags: u64,
    symbol_status: *mut c_int,
) -> CUresult {
    note("cuGetProcAddress_v2");
    get_proc_address(symbol, pfn, cuda_version, flags, symbol_status)
}
