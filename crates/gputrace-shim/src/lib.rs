//! The gputrace interposer: an `LD_PRELOAD` cdylib that shadows the CUDA
//! Driver API. It exports the `cu*` symbols and hooks `cuGetProcAddress`,
//! records each call into the active trace, then forwards to the real
//! `libcuda.so.1`.
//!
//! Full-surface trampolines are generated from the driver headers at dev
//! time and vendored (see ADR-0005); hand-written deep handlers cover the
//! curated subset. This file is the scaffold entry point only.

use std::sync::OnceLock;

use libloading::Library;

static REAL: OnceLock<Library> = OnceLock::new();

/// Lazily `dlopen` the real CUDA driver so trampolines can resolve the
/// genuine `cu*` entry points at runtime. The shim links no CUDA itself,
/// so the shipped artifact runs on any Linux with the NVIDIA driver.
pub fn real_library() -> &'static Library {
    REAL.get_or_init(|| unsafe {
        Library::new("libcuda.so.1").expect("gputrace: cannot dlopen libcuda.so.1")
    })
}
