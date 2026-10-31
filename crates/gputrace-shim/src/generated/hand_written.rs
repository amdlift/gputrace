//! Which `cu*` symbols are shimmed by hand, and therefore not generated.
//!
//! A hand-written shim and a generated one are both `#[unsafe(no_mangle)]`
//! exports of the same symbol, and two definitions of one symbol do not
//! link. The hand-written one wins: its symbol is listed below, which is the
//! whole of the exclusion. Adding a deep handler is one line here in the same
//! commit — the vendored output is not regenerated for it.

/// Expand one generated entry point into a pass-through shim, or into
/// nothing when the symbol is hand-written.
macro_rules! pass_through {
    // ── Hand-written: `handlers/init.rs` ──
    (fn cuInit $($rest:tt)*) => {};
    (fn cuDriverGetVersion $($rest:tt)*) => {};
    // ── Hand-written: `handlers/context.rs` ──
    (fn cuCtxCreate $($rest:tt)*) => {};
    (fn cuCtxCreate_v2 $($rest:tt)*) => {};
    (fn cuCtxCreate_v3 $($rest:tt)*) => {};
    (fn cuCtxCreate_v4 $($rest:tt)*) => {};
    (fn cuCtxDestroy $($rest:tt)*) => {};
    (fn cuCtxDestroy_v2 $($rest:tt)*) => {};
    (fn cuCtxGetCurrent $($rest:tt)*) => {};
    (fn cuCtxPopCurrent $($rest:tt)*) => {};
    (fn cuCtxPopCurrent_v2 $($rest:tt)*) => {};
    (fn cuCtxPushCurrent $($rest:tt)*) => {};
    (fn cuCtxPushCurrent_v2 $($rest:tt)*) => {};
    (fn cuCtxSetCurrent $($rest:tt)*) => {};
    // ── Hand-written: `handlers/memory.rs` ──
    (fn cuMemAlloc $($rest:tt)*) => {};
    (fn cuMemAlloc_v2 $($rest:tt)*) => {};
    (fn cuMemFree $($rest:tt)*) => {};
    (fn cuMemFree_v2 $($rest:tt)*) => {};
    (fn cuMemcpyHtoD $($rest:tt)*) => {};
    (fn cuMemcpyHtoD_v2 $($rest:tt)*) => {};
    (fn cuMemcpyHtoD_v2_ptds $($rest:tt)*) => {};
    // ── Hand-written: `dispatch.rs` ──
    (fn cuGetProcAddress $($rest:tt)*) => {};
    (fn cuGetProcAddress_v2 $($rest:tt)*) => {};

    (fn $symbol:ident ( $($arg:ident : $ty:ty),* $(,)? ) -> CUresult;) => {
        $crate::shim::shim! {
            fn $symbol($($arg: $ty),*) -> CUresult;
        }
    };
}

pub(crate) use pass_through;
