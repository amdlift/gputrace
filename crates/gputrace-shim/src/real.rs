//! Resolution of the real CUDA driver the interposer forwards to.
//!
//! `GPUTRACE_LIBCUDA` overrides the backing library. That override is the
//! single seam capture tests drive: pointed at a stub driver, the whole
//! capture path runs with no GPU present.

use std::env;
use std::ffi::OsString;
use std::sync::OnceLock;

use libloading::Library;

pub const LIBCUDA_OVERRIDE_ENV: &str = "GPUTRACE_LIBCUDA";
pub const DEFAULT_LIBCUDA: &str = "libcuda.so.1";

static REAL: OnceLock<Library> = OnceLock::new();

/// The library the interposer forwards to: the `GPUTRACE_LIBCUDA` override
/// when it names one, otherwise the driver's stock soname.
pub fn real_library_path() -> OsString {
    match env::var_os(LIBCUDA_OVERRIDE_ENV) {
        Some(path) if !path.is_empty() => path,
        _ => OsString::from(DEFAULT_LIBCUDA),
    }
}

/// Lazily `dlopen` the real CUDA driver so shims can resolve the genuine
/// `cu*` entry points at runtime. The interposer links no CUDA itself, so
/// the shipped artifact runs on any Linux with the NVIDIA driver.
///
/// `dlopen` of a concrete object plus `dlsym` on its handle searches that
/// object, not the global scope, so this does not resolve back into the
/// interposer's own exported `cu*` symbols.
pub fn real_library() -> &'static Library {
    REAL.get_or_init(|| {
        let path = real_library_path();
        unsafe { Library::new(&path) }
            .unwrap_or_else(|e| panic!("gputrace: cannot dlopen {}: {e}", path.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_override(value: Option<&str>) -> (MutexGuard<'static, ()>, OsString) {
        let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            match value {
                Some(v) => std::env::set_var(LIBCUDA_OVERRIDE_ENV, v),
                None => std::env::remove_var(LIBCUDA_OVERRIDE_ENV),
            }
        }
        let resolved = real_library_path();
        (guard, resolved)
    }

    #[test]
    fn resolves_the_stock_driver_when_unset() {
        let (_guard, resolved) = with_override(None);
        assert_eq!(resolved, OsString::from("libcuda.so.1"));
    }

    #[test]
    fn resolves_the_override_when_set() {
        let (_guard, resolved) = with_override(Some("/tmp/libstub.so"));
        assert_eq!(resolved, OsString::from("/tmp/libstub.so"));
    }

    #[test]
    fn ignores_an_empty_override() {
        let (_guard, resolved) = with_override(Some(""));
        assert_eq!(resolved, OsString::from("libcuda.so.1"));
    }
}
