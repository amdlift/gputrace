//! The dispatch table (ADR-0003).
//!
//! `cuGetProcAddress` hands the caller a raw function pointer, so a program —
//! or the CUDA Runtime, which resolves every entry point this way — never
//! binds the interposer's exported `cu*` symbols and would escape capture
//! entirely. The hook answers with the interposer's own trampoline for every
//! entry point it shims, and with whatever the real driver answered for the
//! rest.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;
use std::sync::OnceLock;

use gputrace_core::{ArgValue, Argument};

use crate::recorder;
use crate::shim::{CUDA_ERROR_NOT_FOUND, CUresult, DriverFn};

const CUDA_SUCCESS: CUresult = 0;
const CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND: c_int = 1;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cuda_version: c_int,
    flags: u64,
    symbol_status: *mut c_int,
) -> CUresult {
    static REAL: DriverFn<
        unsafe extern "C" fn(*const c_char, *mut *mut c_void, c_int, u64, *mut c_int) -> CUresult,
    > = DriverFn::new(b"cuGetProcAddress_v2\0");

    let call = recorder::begin("cuGetProcAddress_v2");
    let args = lookup_args(symbol, cuda_version, flags);
    let result = match REAL.get() {
        Some(real) => unsafe { real(symbol, pfn, cuda_version, flags, symbol_status) },
        None => unsafe { no_such_driver_entry_point(pfn, symbol_status) },
    };
    if result == CUDA_SUCCESS {
        unsafe { redirect(pfn) };
    }
    call.finish(result, args);
    result
}

fn lookup_args(symbol: *const c_char, cuda_version: c_int, flags: u64) -> Vec<Argument> {
    let name = match symbol_name(symbol) {
        Some(name) => ArgValue::Str(name.to_string_lossy().into_owned()),
        None => ArgValue::Null,
    };
    vec![
        Argument::new("symbol", name),
        Argument::new("cudaVersion", ArgValue::Signed(cuda_version.into())),
        Argument::new("flags", ArgValue::Unsigned(flags)),
    ]
}

/// What the hook answers when the backing driver has no `cuGetProcAddress`
/// of its own. The out-params are the caller's only channel, so they are
/// written even on the failure path.
unsafe fn no_such_driver_entry_point(pfn: *mut *mut c_void, symbol_status: *mut c_int) -> CUresult {
    unsafe {
        if !pfn.is_null() {
            *pfn = ptr::null_mut();
        }
        if !symbol_status.is_null() {
            *symbol_status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        }
    }
    CUDA_ERROR_NOT_FOUND
}

/// Swap the driver's answer for the interposer's trampoline, when there is
/// one for the entry point the driver just resolved.
unsafe fn redirect(pfn: *mut *mut c_void) {
    if pfn.is_null() {
        return;
    }
    let resolved = unsafe { *pfn };
    if resolved.is_null() {
        return;
    }
    if let Some(name) = resolved_entry_point_name(resolved)
        && let Some(trampoline) = trampoline(name)
    {
        unsafe { *pfn = trampoline };
    }
}

/// The name the driver's answer is exported under. Asking the dynamic linker
/// rather than re-deriving the `_vN` / `_ptsz` spelling the driver chose
/// keeps the hook free of any knowledge of that mangling.
fn resolved_entry_point_name(address: *mut c_void) -> Option<&'static CStr> {
    let info = describe(address)?;
    if info.symbol_address != address || info.symbol_name.is_null() {
        return None;
    }
    Some(unsafe { CStr::from_ptr(info.symbol_name) })
}

/// The interposer's own trampoline for `name`, or `None` when it shims no
/// such entry point.
///
/// The registry of trampolines is the interposer's own dynamic symbol table:
/// every shim is a `#[no_mangle]` export, so hand-written and generated
/// shims alike are answerable here with no list to keep in step.
fn trampoline(name: &CStr) -> Option<*mut c_void> {
    // RTLD_DEFAULT is a null handle on glibc: it searches the global scope in
    // load order, which the `LD_PRELOAD`ed interposer heads.
    let address = unsafe { dlsym(ptr::null_mut(), name.as_ptr()) };
    if address.is_null() {
        return None;
    }
    let base = describe(address)?.base;
    (base == interposer_base()?).then_some(address)
}

fn interposer_base() -> Option<*mut c_void> {
    static BASE: OnceLock<Option<usize>> = OnceLock::new();
    let base = *BASE.get_or_init(|| {
        describe(interposer_base as *const () as *mut c_void).map(|info| info.base as usize)
    });
    base.map(|base| base as *mut c_void)
}

fn symbol_name(symbol: *const c_char) -> Option<&'static CStr> {
    (!symbol.is_null()).then(|| unsafe { CStr::from_ptr(symbol) })
}

#[repr(C)]
struct DlInfo {
    file_name: *const c_char,
    base: *mut c_void,
    symbol_name: *const c_char,
    symbol_address: *mut c_void,
}

unsafe extern "C" {
    fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

fn describe(address: *mut c_void) -> Option<DlInfo> {
    let mut info = DlInfo {
        file_name: ptr::null(),
        base: ptr::null_mut(),
        symbol_name: ptr::null(),
        symbol_address: ptr::null_mut(),
    };
    (unsafe { dladdr(address, &mut info) } != 0).then_some(info)
}
