use std::ffi::{CStr, c_char, c_uint, c_void};

use crate::log::note;
use crate::out;
use crate::state::{Module, as_handle, current_context, state};
use crate::types::*;

fn load(module: *mut CUmodule, image: *const c_void) -> CUresult {
    if current_context() == 0 {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if image.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // A real driver accepts PTX text or a cubin; the stub only ever sees the
    // nul-terminated PTX its fixtures hand it.
    let bytes = unsafe { CStr::from_ptr(image.cast::<c_char>()) }
        .to_bytes()
        .to_vec();

    let mut state = state();
    let handle = state.fresh_handle();
    state.modules.insert(
        handle,
        Module {
            image: bytes,
            ..Module::default()
        },
    );
    drop(state);

    if unsafe { out(module, as_handle(handle)) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult {
    note("cuModuleLoadData");
    load(module, image)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleLoadDataEx(
    module: *mut CUmodule,
    image: *const c_void,
    _num_options: c_uint,
    _options: *mut c_void,
    _option_values: *mut *mut c_void,
) -> CUresult {
    note("cuModuleLoadDataEx");
    load(module, image)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleGetFunction(
    hfunc: *mut CUfunction,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult {
    note("cuModuleGetFunction");
    if name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let Ok(name) = unsafe { CStr::from_ptr(name) }.to_str() else {
        return CUDA_ERROR_INVALID_VALUE;
    };

    let module_handle = hmod as usize;
    let mut state = state();
    let Some(module) = state.modules.get(&module_handle) else {
        return CUDA_ERROR_INVALID_HANDLE;
    };
    if let Some(&existing) = module.functions.get(name) {
        drop(state);
        return if unsafe { out(hfunc, as_handle(existing)) } {
            CUDA_SUCCESS
        } else {
            CUDA_ERROR_INVALID_VALUE
        };
    }
    if !module.defines(name) {
        return CUDA_ERROR_NOT_FOUND;
    }

    let handle = state.fresh_handle();
    state
        .modules
        .get_mut(&module_handle)
        .expect("module still live")
        .functions
        .insert(name.to_owned(), handle);
    state.functions.insert(handle, module_handle);
    drop(state);

    if unsafe { out(hfunc, as_handle(handle)) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cuModuleUnload(hmod: CUmodule) -> CUresult {
    note("cuModuleUnload");
    let module_handle = hmod as usize;
    let mut state = state();
    let Some(module) = state.modules.remove(&module_handle) else {
        return CUDA_ERROR_INVALID_HANDLE;
    };
    for handle in module.functions.into_values() {
        state.functions.remove(&handle);
    }
    CUDA_SUCCESS
}
