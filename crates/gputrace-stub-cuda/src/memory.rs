use std::ffi::{c_uint, c_void};

use crate::log::note;
use crate::out;
use crate::state::{current_context, state};
use crate::stream::live_stream;
use crate::types::*;

fn alloc(dptr: *mut CUdeviceptr, bytes: usize) -> CUresult {
    if current_context() == 0 {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if bytes == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let base = state().allocate(bytes);
    if unsafe { out(dptr, base) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

fn free(dptr: CUdeviceptr) -> CUresult {
    if state().allocations.remove(&dptr).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

fn copy_host_to_device(dst: CUdeviceptr, src: *const c_void, len: usize) -> CUresult {
    if src.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let mut state = state();
    let Some(span) = state.span(dst, len) else {
        return CUDA_ERROR_INVALID_VALUE;
    };
    unsafe { std::ptr::copy_nonoverlapping(src.cast::<u8>(), span.as_mut_ptr(), len) };
    CUDA_SUCCESS
}

fn copy_device_to_host(dst: *mut c_void, src: CUdeviceptr, len: usize) -> CUresult {
    if dst.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let mut state = state();
    let Some(span) = state.span(src, len) else {
        return CUDA_ERROR_INVALID_VALUE;
    };
    unsafe { std::ptr::copy_nonoverlapping(span.as_ptr(), dst.cast::<u8>(), len) };
    CUDA_SUCCESS
}

fn copy_device_to_device(dst: CUdeviceptr, src: CUdeviceptr, len: usize) -> CUresult {
    let mut state = state();
    let Some(span) = state.span(src, len) else {
        return CUDA_ERROR_INVALID_VALUE;
    };
    let staged = span.to_vec();
    let Some(span) = state.span(dst, len) else {
        return CUDA_ERROR_INVALID_VALUE;
    };
    span.copy_from_slice(&staged);
    CUDA_SUCCESS
}

fn memset(dst: CUdeviceptr, byte: u8, len: usize) -> CUresult {
    let mut state = state();
    let Some(span) = state.span(dst, len) else {
        return CUDA_ERROR_INVALID_VALUE;
    };
    span.fill(byte);
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    note("cuMemAlloc_v2");
    alloc(dptr, bytesize)
}

/// The pre-CUDA-3.2 spelling: device pointers and sizes were 32-bit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemAlloc(dptr: *mut CUdeviceptrV1, bytesize: c_uint) -> CUresult {
    note("cuMemAlloc");
    let mut wide = 0;
    let result = alloc(&mut wide, bytesize as usize);
    if result != CUDA_SUCCESS {
        return result;
    }
    let Ok(narrow) = CUdeviceptrV1::try_from(wide) else {
        return CUDA_ERROR_OUT_OF_MEMORY;
    };
    if unsafe { out(dptr, narrow) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult {
    note("cuMemFree_v2");
    free(dptr)
}

#[unsafe(no_mangle)]
pub extern "C" fn cuMemFree(dptr: CUdeviceptrV1) -> CUresult {
    note("cuMemFree");
    free(CUdeviceptr::from(dptr))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemcpyHtoD_v2(
    dst_device: CUdeviceptr,
    src_host: *const c_void,
    byte_count: usize,
) -> CUresult {
    note("cuMemcpyHtoD_v2");
    copy_host_to_device(dst_device, src_host, byte_count)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemcpyHtoD_v2_ptds(
    dst_device: CUdeviceptr,
    src_host: *const c_void,
    byte_count: usize,
) -> CUresult {
    note("cuMemcpyHtoD_v2_ptds");
    copy_host_to_device(dst_device, src_host, byte_count)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemcpyHtoD(
    dst_device: CUdeviceptrV1,
    src_host: *const c_void,
    byte_count: c_uint,
) -> CUresult {
    note("cuMemcpyHtoD");
    copy_host_to_device(CUdeviceptr::from(dst_device), src_host, byte_count as usize)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemcpyDtoH_v2(
    dst_host: *mut c_void,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult {
    note("cuMemcpyDtoH_v2");
    copy_device_to_host(dst_host, src_device, byte_count)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemcpyDtoH_v2_ptds(
    dst_host: *mut c_void,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult {
    note("cuMemcpyDtoH_v2_ptds");
    copy_device_to_host(dst_host, src_device, byte_count)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemcpyDtoH(
    dst_host: *mut c_void,
    src_device: CUdeviceptrV1,
    byte_count: c_uint,
) -> CUresult {
    note("cuMemcpyDtoH");
    copy_device_to_host(dst_host, CUdeviceptr::from(src_device), byte_count as usize)
}

#[unsafe(no_mangle)]
pub extern "C" fn cuMemcpyDtoD_v2(
    dst_device: CUdeviceptr,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult {
    note("cuMemcpyDtoD_v2");
    copy_device_to_device(dst_device, src_device, byte_count)
}

#[unsafe(no_mangle)]
pub extern "C" fn cuMemsetD8_v2(dst_device: CUdeviceptr, uc: u8, n: usize) -> CUresult {
    note("cuMemsetD8_v2");
    memset(dst_device, uc, n)
}

/// The stream-ordered allocator. The stub completes every stream operation
/// the moment it is asked, so ordering only shows up in what it refuses:
/// a call naming a stream that was never created.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemAllocAsync(
    dptr: *mut CUdeviceptr,
    bytesize: usize,
    h_stream: CUstream,
) -> CUresult {
    note("cuMemAllocAsync");
    if !live_stream(h_stream) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    alloc(dptr, bytesize)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemAllocAsync_ptsz(
    dptr: *mut CUdeviceptr,
    bytesize: usize,
    h_stream: CUstream,
) -> CUresult {
    note("cuMemAllocAsync_ptsz");
    if !live_stream(h_stream) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    alloc(dptr, bytesize)
}

#[unsafe(no_mangle)]
pub extern "C" fn cuMemFreeAsync(dptr: CUdeviceptr, h_stream: CUstream) -> CUresult {
    note("cuMemFreeAsync");
    if !live_stream(h_stream) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    free(dptr)
}

#[unsafe(no_mangle)]
pub extern "C" fn cuMemFreeAsync_ptsz(dptr: CUdeviceptr, h_stream: CUstream) -> CUresult {
    note("cuMemFreeAsync_ptsz");
    if !live_stream(h_stream) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    free(dptr)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemcpyHtoDAsync_v2(
    dst_device: CUdeviceptr,
    src_host: *const c_void,
    byte_count: usize,
    h_stream: CUstream,
) -> CUresult {
    note("cuMemcpyHtoDAsync_v2");
    if !live_stream(h_stream) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    copy_host_to_device(dst_device, src_host, byte_count)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemcpyHtoDAsync_v2_ptsz(
    dst_device: CUdeviceptr,
    src_host: *const c_void,
    byte_count: usize,
    h_stream: CUstream,
) -> CUresult {
    note("cuMemcpyHtoDAsync_v2_ptsz");
    if !live_stream(h_stream) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    copy_host_to_device(dst_device, src_host, byte_count)
}

/// The pre-CUDA-3.2 spelling: device pointers and sizes were 32-bit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemcpyHtoDAsync(
    dst_device: CUdeviceptrV1,
    src_host: *const c_void,
    byte_count: c_uint,
    h_stream: CUstream,
) -> CUresult {
    note("cuMemcpyHtoDAsync");
    if !live_stream(h_stream) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    copy_host_to_device(CUdeviceptr::from(dst_device), src_host, byte_count as usize)
}
