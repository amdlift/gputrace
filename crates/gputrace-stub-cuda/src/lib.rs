//! A stub `libcuda`: a deterministic, in-memory implementation of the subset
//! of the CUDA Driver API that gputrace's capture tests drive. Pointed at by
//! `GPUTRACE_LIBCUDA`, it stands in for `libcuda.so.1` so capture is testable
//! with no GPU present.
//!
//! Entry points are exported under the spellings a real driver exports,
//! including the `_vN` and `_ptsz`/`_ptds` variants, because interposition is
//! purely by name: `libcuda.so.1` carries no ELF symbol versioning.
//!
//! Test support only — never a dependency of the shipped interposer or CLI.

mod context;
mod device;
mod event;
mod init;
mod launch;
mod log;
mod memory;
mod module;
mod state;
mod stream;
mod types;

pub use types::*;

pub(crate) unsafe fn out<T>(ptr: *mut T, value: T) -> bool {
    if ptr.is_null() {
        return false;
    }
    unsafe { ptr.write(value) };
    true
}
