use std::ffi::{c_int, c_uint, c_void};

use crate::log::note;
use crate::out;
use crate::state::{as_handle, current_context, pop_stack, push_stack, set_current_context, state};
use crate::types::*;

fn create(device: CUdevice, pctx: *mut CUcontext) -> CUresult {
    let mut state = state();
    if !state.initialized {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if !device_exists(device) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    let handle = state.fresh_handle();
    state.contexts.insert(handle, device);
    drop(state);

    if !unsafe { out(pctx, as_handle(handle)) } {
        return CUDA_ERROR_INVALID_VALUE;
    }
    set_current_context(handle);
    CUDA_SUCCESS
}

/// The pre-CUDA-3.2 spelling, identical in shape to `_v2`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxCreate(
    pctx: *mut CUcontext,
    _flags: c_uint,
    dev: CUdevice,
) -> CUresult {
    note("cuCtxCreate");
    create(dev, pctx)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxCreate_v2(
    pctx: *mut CUcontext,
    _flags: c_uint,
    dev: CUdevice,
) -> CUresult {
    note("cuCtxCreate_v2");
    create(dev, pctx)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxCreate_v3(
    pctx: *mut CUcontext,
    _params_array: *mut c_void,
    _num_params: c_int,
    _flags: c_uint,
    dev: CUdevice,
) -> CUresult {
    note("cuCtxCreate_v3");
    create(dev, pctx)
}

/// What a target built against CUDA 12+ headers actually calls: `cuda.h`
/// remaps plain `cuCtxCreate` onto this spelling.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxCreate_v4(
    pctx: *mut CUcontext,
    _ctx_create_params: *mut c_void,
    _flags: c_uint,
    dev: CUdevice,
) -> CUresult {
    note("cuCtxCreate_v4");
    create(dev, pctx)
}

fn destroy(ctx: CUcontext) -> CUresult {
    let handle = ctx as usize;
    if state().contexts.remove(&handle).is_none() {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if current_context() == handle {
        set_current_context(0);
    }
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxDestroy(ctx: CUcontext) -> CUresult {
    note("cuCtxDestroy");
    destroy(ctx)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult {
    note("cuCtxDestroy_v2");
    destroy(ctx)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult {
    note("cuCtxSetCurrent");
    let handle = ctx as usize;
    if handle != 0 && !state().contexts.contains_key(&handle) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    set_current_context(handle);
    CUDA_SUCCESS
}

fn push_current(ctx: CUcontext) -> CUresult {
    let handle = ctx as usize;
    if !state().contexts.contains_key(&handle) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    push_stack(current_context());
    set_current_context(handle);
    CUDA_SUCCESS
}

unsafe fn pop_current(pctx: *mut CUcontext) -> CUresult {
    let handle = current_context();
    if handle == 0 {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if !unsafe { out(pctx, as_handle(handle)) } {
        return CUDA_ERROR_INVALID_VALUE;
    }
    set_current_context(pop_stack());
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxPushCurrent(ctx: CUcontext) -> CUresult {
    note("cuCtxPushCurrent");
    push_current(ctx)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult {
    note("cuCtxPushCurrent_v2");
    push_current(ctx)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxPopCurrent(pctx: *mut CUcontext) -> CUresult {
    note("cuCtxPopCurrent");
    unsafe { pop_current(pctx) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult {
    note("cuCtxPopCurrent_v2");
    unsafe { pop_current(pctx) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult {
    note("cuCtxGetCurrent");
    if !state().initialized {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if unsafe { out(pctx, as_handle(current_context())) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cuCtxSynchronize() -> CUresult {
    note("cuCtxSynchronize");
    if current_context() == 0 {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult {
    note("cuCtxGetDevice");
    let handle = current_context();
    let Some(&dev) = state().contexts.get(&handle) else {
        return CUDA_ERROR_INVALID_CONTEXT;
    };
    if unsafe { out(device, dev) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}
