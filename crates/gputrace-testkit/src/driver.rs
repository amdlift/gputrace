//! How a fixture target reaches the CUDA Driver API.
//!
//! Two ways, because they bind differently. [`driver_symbol`] looks a symbol
//! up in the process's global scope, the same scope a PLT-bound `cu*` call
//! resolves against, so an `LD_PRELOAD`ed interposer shadows it exactly as it
//! shadows a real CUDA program's calls. [`DlopenedDriver`] instead opens one
//! concrete library and searches that object alone — the lookup never consults
//! the global scope, so `LD_PRELOAD` has nothing to shadow. Resolving either
//! way keeps fixtures free of any link-time CUDA library.

use std::ffi::{CString, OsStr, c_char, c_int, c_void};
use std::os::unix::ffi::OsStrExt;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;
const RTLD_LOCAL: c_int = 0;

/// Resolve a driver entry point from the global scope.
pub fn driver_symbol(name: &str) -> Option<*mut c_void> {
    let symbol = CString::new(name).expect("symbol name");
    // RTLD_DEFAULT is a null handle on glibc and musl alike.
    let address = unsafe { dlsym(std::ptr::null_mut(), symbol.as_ptr()) };
    (!address.is_null()).then_some(address)
}

/// A driver library the target opened for itself, as a program that
/// `dlopen`s `libcuda.so.1` rather than linking it does.
pub struct DlopenedDriver {
    handle: *mut c_void,
}

impl DlopenedDriver {
    /// `dlopen` the library at `path` into the target's own scope.
    pub fn open(path: impl AsRef<OsStr>) -> Option<Self> {
        let path = CString::new(path.as_ref().as_bytes()).expect("library path");
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        (!handle.is_null()).then_some(Self { handle })
    }

    /// Resolve `name` within this library, bypassing the global scope.
    pub fn symbol(&self, name: &str) -> Option<*mut c_void> {
        let symbol = CString::new(name).expect("symbol name");
        let address = unsafe { dlsym(self.handle, symbol.as_ptr()) };
        (!address.is_null()).then_some(address)
    }
}

/// Declare a struct of driver entry points resolved from the global scope.
#[macro_export]
macro_rules! cuda_api {
    (
        $vis:vis struct $name:ident {
            $( fn $symbol:ident ( $($arg:ty),* $(,)? ) -> $ret:ty; )*
        }
    ) => {
        #[allow(non_snake_case)]
        $vis struct $name {
            $( $vis $symbol: unsafe extern "C" fn($($arg),*) -> $ret, )*
        }

        impl $name {
            /// Resolve every declared entry point, panicking on any that the
            /// loaded libraries do not export.
            $vis fn load() -> Self {
                Self {
                    $( $symbol: unsafe {
                        ::std::mem::transmute::<
                            *mut ::std::ffi::c_void,
                            unsafe extern "C" fn($($arg),*) -> $ret,
                        >(
                            $crate::driver_symbol(stringify!($symbol)).unwrap_or_else(|| {
                                panic!("no {} in the loaded driver", stringify!($symbol))
                            })
                        )
                    }, )*
                }
            }
        }
    };
}
