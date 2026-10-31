//! Context lifecycle, shimmed only deeply enough to tell capture which device
//! a thread's calls target.
//!
//! Ticket #8 replaces these with deep handlers that bind handle identity.
//! When it does, delete the entry here rather than adding a second one: two
//! `#[no_mangle]` exports of one symbol do not link.

use std::ffi::{c_int, c_uint, c_void};

use crate::shim::shim;

shim! {
    fn cuCtxCreate_v4(
        pctx: *mut *mut c_void,
        ctx_create_params: *mut c_void,
        flags: c_uint,
        dev: c_int,
    ) -> CUresult;
    fn cuCtxDestroy_v2(ctx: *mut c_void) -> CUresult;
}
