//! Resolution of the real CUDA driver the interposer forwards to.
//!
//! `GPUTRACE_LIBCUDA` overrides the backing library. That override is the
//! single seam capture tests drive: pointed at a stub driver, the whole
//! capture path runs with no GPU present.

use std::env;
use std::ffi::OsString;
use std::sync::OnceLock;

use libloading::{Library, Symbol};

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

/// Resolve one entry point from the backing driver. `name` must be
/// nul-terminated. `None` means the backing library does not export it, which
/// a generated trampoline reports as `CUDA_ERROR_NOT_FOUND` rather than
/// treating as fatal — 21 of `cuda.h`'s declarations are absent from a shipped
/// driver.
///
/// # Safety
///
/// `T` must match the entry point's real ABI.
pub unsafe fn real_symbol<T>(name: &[u8]) -> Option<Symbol<'static, T>> {
    unsafe { real_library().get(name) }.ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{OsString, c_char, c_int, c_uint};
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
    fn resolves_entry_points_from_the_library_the_override_names() {
        let stub = gputrace_testkit::stub_libcuda().expect("stub libcuda");
        let (_guard, _) = with_override(Some(stub.to_str().expect("utf-8 path")));

        let cu_init = unsafe { real_symbol::<unsafe extern "C" fn(c_uint) -> c_int>(b"cuInit\0") }
            .expect("cuInit");
        assert_eq!(unsafe { cu_init(0) }, 0);

        let cu_device_get_name = unsafe {
            real_symbol::<unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int>(
                b"cuDeviceGetName\0",
            )
        }
        .expect("cuDeviceGetName");
        let mut name = [0u8; 64];
        assert_eq!(
            unsafe { cu_device_get_name(name.as_mut_ptr().cast(), 64, 0) },
            0
        );
        let end = name.iter().position(|&b| b == 0).expect("terminator");
        assert_eq!(
            std::str::from_utf8(&name[..end]).expect("utf-8 name"),
            "gputrace-stub-device-0"
        );
    }

    #[test]
    fn ignores_an_empty_override() {
        let (_guard, resolved) = with_override(Some(""));
        assert_eq!(resolved, OsString::from("libcuda.so.1"));
    }
}
