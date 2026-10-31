use std::ffi::{c_uint, c_void};

use crate::log::note;
use crate::state::state;
use crate::types::*;

#[allow(clippy::too_many_arguments)]
fn launch(f: CUfunction) -> CUresult {
    if state().functions.contains_key(&(f as usize)) {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_HANDLE
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cuLaunchKernel(
    f: CUfunction,
    _grid_dim_x: c_uint,
    _grid_dim_y: c_uint,
    _grid_dim_z: c_uint,
    _block_dim_x: c_uint,
    _block_dim_y: c_uint,
    _block_dim_z: c_uint,
    _shared_mem_bytes: c_uint,
    _h_stream: CUstream,
    _kernel_params: *mut *mut c_void,
    _extra: *mut *mut c_void,
) -> CUresult {
    note("cuLaunchKernel");
    launch(f)
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cuLaunchKernel_ptsz(
    f: CUfunction,
    _grid_dim_x: c_uint,
    _grid_dim_y: c_uint,
    _grid_dim_z: c_uint,
    _block_dim_x: c_uint,
    _block_dim_y: c_uint,
    _block_dim_z: c_uint,
    _shared_mem_bytes: c_uint,
    _h_stream: CUstream,
    _kernel_params: *mut *mut c_void,
    _extra: *mut *mut c_void,
) -> CUresult {
    note("cuLaunchKernel_ptsz");
    launch(f)
}
