use std::ffi::c_uint;

use crate::log::note;
use crate::out;
use crate::state::{as_handle, current_context, state};
use crate::stream::live_stream;
use crate::types::*;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuEventCreate(phevent: *mut CUevent, _flags: c_uint) -> CUresult {
    note("cuEventCreate");
    if current_context() == 0 {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    let mut state = state();
    let handle = state.fresh_handle();
    state.events.insert(handle, None);
    drop(state);

    if unsafe { out(phevent, as_handle(handle)) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

fn record(hevent: CUevent, hstream: CUstream) -> CUresult {
    if !live_stream(hstream) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    let mut state = state();
    if !state.events.contains_key(&(hevent as usize)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    let stamp = state.next_record();
    state.events.insert(hevent as usize, Some(stamp));
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn cuEventRecord(hevent: CUevent, hstream: CUstream) -> CUresult {
    note("cuEventRecord");
    record(hevent, hstream)
}

#[unsafe(no_mangle)]
pub extern "C" fn cuEventRecord_ptsz(hevent: CUevent, hstream: CUstream) -> CUresult {
    note("cuEventRecord_ptsz");
    record(hevent, hstream)
}

#[unsafe(no_mangle)]
pub extern "C" fn cuEventSynchronize(hevent: CUevent) -> CUresult {
    note("cuEventSynchronize");
    if state().events.contains_key(&(hevent as usize)) {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_HANDLE
    }
}

fn destroy(hevent: CUevent) -> CUresult {
    if state().events.remove(&(hevent as usize)).is_some() {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_HANDLE
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cuEventDestroy(hevent: CUevent) -> CUresult {
    note("cuEventDestroy");
    destroy(hevent)
}

#[unsafe(no_mangle)]
pub extern "C" fn cuEventDestroy_v2(hevent: CUevent) -> CUresult {
    note("cuEventDestroy_v2");
    destroy(hevent)
}

fn elapsed_time(milliseconds: *mut f32, hstart: CUevent, hend: CUevent) -> CUresult {
    let state = state();
    let (Some(Some(start)), Some(Some(end))) = (
        state.events.get(&(hstart as usize)),
        state.events.get(&(hend as usize)),
    ) else {
        return CUDA_ERROR_INVALID_HANDLE;
    };
    let span = end.abs_diff(*start) as f32;
    drop(state);

    if unsafe { out(milliseconds, span) } {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuEventElapsedTime(
    milliseconds: *mut f32,
    hstart: CUevent,
    hend: CUevent,
) -> CUresult {
    note("cuEventElapsedTime");
    elapsed_time(milliseconds, hstart, hend)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuEventElapsedTime_v2(
    milliseconds: *mut f32,
    hstart: CUevent,
    hend: CUevent,
) -> CUresult {
    note("cuEventElapsedTime_v2");
    elapsed_time(milliseconds, hstart, hend)
}
