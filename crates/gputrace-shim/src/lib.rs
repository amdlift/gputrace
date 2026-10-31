//! The gputrace interposer: an `LD_PRELOAD` cdylib that shadows the CUDA
//! Driver API. It exports the `cu*` symbols and hooks `cuGetProcAddress`,
//! records each call into the active trace, then forwards to the real
//! `libcuda.so.1`.
//!
//! Full-surface trampolines are generated from the driver headers at dev
//! time and vendored (see ADR-0005); hand-written deep handlers cover the
//! curated subset. Both are written against the one pattern in [`shim`].

mod handlers;
pub mod real;
pub mod recorder;
pub mod shim;

pub use real::{
    DEFAULT_LIBCUDA, LIBCUDA_OVERRIDE_ENV, real_library, real_library_path, real_symbol,
};
pub use recorder::{BUNDLE_DIR_ENV, FIDELITY_ENV};
