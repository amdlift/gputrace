//! How one shim is written.
//!
//! A shim stands in for exactly one real `cu*` entry point: it opens a
//! [`CallScope`](crate::recorder::CallScope), records the arguments the caller
//! passed, forwards to the real driver, and closes the scope with the result.
//! That whole body is mechanical, so [`shim!`] generates it from the
//! signature alone — which is what lets the full driver surface be generated
//! and a deep handler be dropped in by hand for the same symbol.

use std::ffi::c_int;
use std::sync::OnceLock;

use gputrace_core::ArgValue;

use crate::real::real_symbol;

/// Every `cu*` entry point returns a `CUresult`.
pub type CUresult = c_int;

/// `CUDA_ERROR_NOT_FOUND`. What a shim answers when the backing driver does
/// not export the entry point it stands in for — 21 of `cuda.h`'s
/// declarations are absent from a shipped driver, so this is normal.
pub const CUDA_ERROR_NOT_FOUND: CUresult = 500;

/// The real entry point behind one shim, resolved once and cached.
pub struct DriverFn<T: 'static> {
    symbol: &'static [u8],
    resolved: OnceLock<Option<T>>,
}

impl<T: Copy + 'static> DriverFn<T> {
    /// `symbol` must be nul-terminated.
    pub const fn new(symbol: &'static [u8]) -> Self {
        Self {
            symbol,
            resolved: OnceLock::new(),
        }
    }

    /// The backing driver's function pointer, or `None` when it exports no
    /// such symbol.
    pub fn get(&self) -> Option<T> {
        *self
            .resolved
            .get_or_init(|| unsafe { real_symbol::<T>(self.symbol) }.map(|symbol| *symbol))
    }
}

/// How one argument of a shimmed function is recorded. Implemented for every
/// scalar and pointer type a `cu*` signature can use, so a shim body needs no
/// per-type knowledge.
pub trait RecordArg {
    fn record(&self) -> ArgValue;
}

macro_rules! record_as {
    ($variant:ident as $as:ty => $($ty:ty),*) => {$(
        impl RecordArg for $ty {
            fn record(&self) -> ArgValue {
                ArgValue::$variant(*self as $as)
            }
        }
    )*};
}

record_as!(Unsigned as u64 => u8, u16, u32, u64, usize);
record_as!(Signed as i64 => i8, i16, i32, i64, isize);
record_as!(Float as f64 => f32, f64);

impl RecordArg for bool {
    fn record(&self) -> ArgValue {
        ArgValue::Bool(*self)
    }
}

impl<T> RecordArg for *const T {
    fn record(&self) -> ArgValue {
        ArgValue::HostPtr(*self as usize as u64)
    }
}

impl<T> RecordArg for *mut T {
    fn record(&self) -> ArgValue {
        ArgValue::HostPtr(*self as usize as u64)
    }
}

/// Define pass-through shims from their signatures.
///
/// ```ignore
/// shim! {
///     fn cuInit(flags: c_uint) -> CUresult;
///     fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
/// }
/// ```
///
/// Each entry exports a `#[no_mangle]` symbol of exactly that ABI, records a
/// metadata-level `CallRecord` naming every argument, and forwards to the
/// real driver. Argument types only need to implement [`RecordArg`].
macro_rules! shim {
    ($(
        fn $symbol:ident ( $($arg:ident : $ty:ty),* $(,)? ) -> CUresult;
    )*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol($($arg: $ty),*) -> $crate::shim::CUresult {
            static REAL: $crate::shim::DriverFn<
                unsafe extern "C" fn($($ty),*) -> $crate::shim::CUresult,
            > = $crate::shim::DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = $crate::recorder::begin(stringify!($symbol));
            let args = vec![$(
                ::gputrace_core::Argument::new(
                    stringify!($arg),
                    $crate::shim::RecordArg::record(&$arg),
                )
            ),*];
            let result = match REAL.get() {
                Some(real) => unsafe { real($($arg),*) },
                None => $crate::shim::CUDA_ERROR_NOT_FOUND,
            };
            call.finish(result, args);
            result
        }
    )*};
}

pub(crate) use shim;
