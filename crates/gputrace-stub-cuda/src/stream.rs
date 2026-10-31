use std::ffi::c_uint;

use crate::log::note;
use crate::out;
use crate::state::{as_handle, current_context, state};
use crate::types::*;

/// A null handle is CUDA's legacy default stream: always valid, never created.
pub(crate) fn live_stream(stream: CUstream) -> bool {
    let handle = stream as usize;
    handle == 0 || state().streams.contains(&handle)
}

fn create(phstream: *mut CUstream) -> CUresult {
    if current_context() == 0 {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    let mut state = state();
    let handle = state.fresh_handle();
    state.streams.insert(handle);
    drop(state);

    if unsafe { out(phstream, as_handle(handle)) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamCreate(phstream: *mut CUstream, _flags: c_uint) -> CUresult {
    note("cuStreamCreate");
    create(phstream)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamCreateWithPriority(
    phstream: *mut CUstream,
    _flags: c_uint,
    _priority: i32,
) -> CUresult {
    note("cuStreamCreateWithPriority");
    create(phstream)
}

fn destroy(hstream: CUstream) -> CUresult {
    if state().streams.remove(&(hstream as usize)) {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_HANDLE
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cuStreamDestroy(hstream: CUstream) -> CUresult {
    note("cuStreamDestroy");
    destroy(hstream)
}

#[unsafe(no_mangle)]
pub extern "C" fn cuStreamDestroy_v2(hstream: CUstream) -> CUresult {
    note("cuStreamDestroy_v2");
    destroy(hstream)
}

fn synchronize(hstream: CUstream) -> CUresult {
    if live_stream(hstream) {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_HANDLE
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cuStreamSynchronize(hstream: CUstream) -> CUresult {
    note("cuStreamSynchronize");
    synchronize(hstream)
}

#[unsafe(no_mangle)]
pub extern "C" fn cuStreamSynchronize_ptsz(hstream: CUstream) -> CUresult {
    note("cuStreamSynchronize_ptsz");
    synchronize(hstream)
}

fn wait_event(hstream: CUstream, hevent: CUevent) -> CUresult {
    if !live_stream(hstream) || !state().events.contains_key(&(hevent as usize)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn cuStreamWaitEvent(
    hstream: CUstream,
    hevent: CUevent,
    _flags: c_uint,
) -> CUresult {
    note("cuStreamWaitEvent");
    wait_event(hstream, hevent)
}

#[unsafe(no_mangle)]
pub extern "C" fn cuStreamWaitEvent_ptsz(
    hstream: CUstream,
    hevent: CUevent,
    _flags: c_uint,
) -> CUresult {
    note("cuStreamWaitEvent_ptsz");
    wait_event(hstream, hevent)
}
