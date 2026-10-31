//! Test support for gputrace capture: runs a target program under the
//! interposer with a stub `libcuda` behind it, and hands back what the run
//! produced — the stub's call log and the trace bundle — for assertion.
//!
//! Never a dependency of the shipped interposer or CLI.

mod artifacts;
mod driver;
mod harness;

pub use artifacts::{interposer, stub_libcuda, target_bin};
pub use driver::driver_symbol;
pub use harness::{BUNDLE_ENV, CaptureRun, Harness, LIBCUDA_ENV, STUB_LOG_ENV};

#[derive(Debug, thiserror::Error)]
pub enum TestkitError {
    #[error("building test artifacts failed: {0}")]
    Build(String),
    #[error("running the target failed: {0}")]
    Run(#[from] std::io::Error),
    #[error(transparent)]
    Bundle(#[from] gputrace_core::BundleError),
}
