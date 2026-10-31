//! How a fixture target reaches the CUDA Driver API.
//!
//! Symbols are looked up in the process's global scope, the same scope a
//! PLT-bound `cu*` call resolves against, so an `LD_PRELOAD`ed interposer
//! shadows them exactly as it shadows a real CUDA program's calls. Resolving
//! this way keeps fixtures free of any link-time CUDA library.

use std::ffi::{CString, c_char, c_void};

unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

/// Resolve a driver entry point from the global scope.
pub fn driver_symbol(name: &str) -> Option<*mut c_void> {
    let symbol = CString::new(name).expect("symbol name");
    // RTLD_DEFAULT is a null handle on glibc and musl alike.
    let address = unsafe { dlsym(std::ptr::null_mut(), symbol.as_ptr()) };
    (!address.is_null()).then_some(address)
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
