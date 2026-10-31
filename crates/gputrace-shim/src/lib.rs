//! The gputrace interposer: an `LD_PRELOAD` cdylib that shadows the CUDA
//! Driver API. It exports the `cu*` symbols and hooks `cuGetProcAddress`,
//! records each call into the active trace, then forwards to the real
//! `libcuda.so.1`.
//!
//! Full-surface trampolines are generated from the driver headers at dev
//! time and vendored (see ADR-0005); hand-written deep handlers cover the
//! curated subset. This file is the scaffold entry point only.

pub mod real;

pub use real::{DEFAULT_LIBCUDA, LIBCUDA_OVERRIDE_ENV, real_library, real_library_path};
