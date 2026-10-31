//! Deep handlers for module loading and kernel lookup.
//!
//! A launch is unreplayable without the code it launches, and the code
//! arrives as one pointer with no length: the ABI says `const void *image`
//! and nothing else. The driver infers the format from the leading bytes
//! and reads exactly as far as that format declares, so capture does the
//! same — see [`image_len`]. Guessing a larger size would fault the target
//! on the page after the image.

use std::ffi::{CStr, c_char, c_uint, c_void};

use gputrace_core::{ArgValue, Argument, HandleId};

use crate::handlers::handles::{HandleKind, handle_id};
use crate::recorder;
use crate::shim::{CUDA_ERROR_NOT_FOUND, CUresult, DriverFn, RecordArg};

const CUDA_SUCCESS: CUresult = 0;

type CUmodule = *mut c_void;
type CUfunction = *mut c_void;

/// What a captured module image is called in a record's payloads.
const MODULE_IMAGE: &str = "module_image";

/// The ELF wrapper NVIDIA packs a fat binary in, little-endian.
const FATBIN_MAGIC: u32 = 0xba55_ed50;

fn module_arg(name: &'static str, module: CUmodule) -> Argument {
    Argument::new(
        name,
        match module_id(module) {
            Some(id) => ArgValue::Handle(id),
            None => ArgValue::Null,
        },
    )
}

fn module_id(module: CUmodule) -> Option<HandleId> {
    handle_id(HandleKind::Module, module as usize as u64)
}

fn function_id(function: CUfunction) -> Option<HandleId> {
    handle_id(HandleKind::Function, function as usize as u64)
}

/// How many bytes of `image` belong to the module, or `None` when the
/// leading bytes match no format capture can measure.
///
/// # Safety
/// `image` must point at a readable module image.
unsafe fn image_len(image: *const u8) -> Option<usize> {
    if unsafe { *image } == 0 {
        return Some(0);
    }
    let header = |offset: usize, width: usize| -> u64 {
        let mut bytes = [0u8; 8];
        bytes[..width]
            .copy_from_slice(unsafe { std::slice::from_raw_parts(image.add(offset), width) });
        u64::from_le_bytes(bytes)
    };

    if header(0, 4) as u32 == FATBIN_MAGIC {
        // The wrapper states its own header size and the size of the
        // payload that follows it.
        return usize::try_from(header(4 + 2, 2) + header(8, 8)).ok();
    }
    if unsafe { std::slice::from_raw_parts(image, 4) } == b"\x7fELF" {
        // A cubin is an ELF whose section table is its last structure.
        let sections = header(0x28, 8) + header(0x3c, 2) * header(0x3a, 2);
        let segments = header(0x20, 8) + header(0x38, 2) * header(0x36, 2);
        return usize::try_from(sections.max(segments).max(header(0x34, 2))).ok();
    }
    // Anything else is PTX: nul-terminated source text, which is what the
    // driver itself falls back to reading.
    Some(
        unsafe { CStr::from_ptr(image.cast::<c_char>()) }
            .to_bytes()
            .len(),
    )
}

/// `cuModuleLoadData` and the spellings that differ only in what follows
/// the image pointer.
macro_rules! load_shims {
    ($($symbol:ident($($extra:ident: $ty:ty as $recorded:literal),*);)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(
            module: *mut CUmodule,
            image: *const c_void,
            $($extra: $ty,)*
        ) -> CUresult {
            static REAL: DriverFn<
                unsafe extern "C" fn(*mut CUmodule, *const c_void $(, $ty)*) -> CUresult,
            > = DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let result = match REAL.get() {
                Some(real) => unsafe { real(module, image $(, $extra)*) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            let loaded = if result == CUDA_SUCCESS && !module.is_null() {
                unsafe { *module }
            } else {
                std::ptr::null_mut()
            };
            let deep = call
                .deep(vec![
                    module_arg("module", loaded),
                    Argument::new("image", ArgValue::HostPtr(image as usize as u64)),
                    $(Argument::new($recorded, RecordArg::record(&$extra)),)*
                ])
                .produces(module_id(loaded));
            // Staging after the driver has run, unlike a copy: the module
            // handle is an out-parameter, so the record cannot be opened
            // before the call completes. The image is `const` and the caller
            // cannot free it until this shim returns, so it is still here.
            let len = if image.is_null() {
                None
            } else {
                unsafe { image_len(image.cast::<u8>()) }
            };
            let deep = match len {
                Some(len) => unsafe { deep.payload(MODULE_IMAGE, image.cast::<u8>(), len) },
                None => deep,
            };
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// `cuModuleGetFunction`. The name is the whole of what identifies a kernel
/// inside its module, so it is recorded as text and not as the pointer it
/// arrived as.
macro_rules! get_function_shims {
    ($($symbol:ident;)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(
            hfunc: *mut CUfunction,
            hmod: CUmodule,
            name: *const c_char,
        ) -> CUresult {
            static REAL: DriverFn<
                unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
            > = DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let result = match REAL.get() {
                Some(real) => unsafe { real(hfunc, hmod, name) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            let found = if result == CUDA_SUCCESS && !hfunc.is_null() {
                unsafe { *hfunc }
            } else {
                std::ptr::null_mut()
            };
            let deep = call
                .deep(vec![
                    Argument::new("hfunc", match function_id(found) {
                        Some(id) => ArgValue::Handle(id),
                        None => ArgValue::Null,
                    }),
                    module_arg("hmod", hmod),
                    Argument::new("name", match unsafe { kernel_name(name) } {
                        Some(name) => ArgValue::Str(name),
                        None => ArgValue::Null,
                    }),
                ])
                .produces(function_id(found))
                .consumes(module_id(hmod));
            call.finish_deep(result, deep);
            result
        }
    )*};
}

macro_rules! unload_shims {
    ($($symbol:ident;)*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(hmod: CUmodule) -> CUresult {
            static REAL: DriverFn<unsafe extern "C" fn(CUmodule) -> CUresult> =
                DriverFn::new(concat!(stringify!($symbol), "\0").as_bytes());

            let call = recorder::begin(stringify!($symbol));
            let deep = call
                .deep(vec![module_arg("hmod", hmod)])
                .destroys(module_id(hmod));
            let result = match REAL.get() {
                Some(real) => unsafe { real(hmod) },
                None => CUDA_ERROR_NOT_FOUND,
            };
            call.finish_deep(result, deep);
            result
        }
    )*};
}

/// # Safety
/// `name` must be null or a nul-terminated string.
unsafe fn kernel_name(name: *const c_char) -> Option<String> {
    (!name.is_null()).then(|| {
        unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned()
    })
}

load_shims! {
    cuModuleLoadData();
    cuModuleLoadFatBinary();
    cuModuleLoadDataEx(
        num_options: c_uint as "numOptions",
        options: *mut c_void as "options",
        option_values: *mut *mut c_void as "optionValues"
    );
}

get_function_shims! {
    cuModuleGetFunction;
}

unload_shims! {
    cuModuleUnload;
}

#[cfg(test)]
mod tests {
    use super::image_len;

    fn len_of(image: &[u8]) -> Option<usize> {
        unsafe { image_len(image.as_ptr()) }
    }

    #[test]
    fn ptx_source_runs_to_its_nul_terminator() {
        assert_eq!(len_of(b".version 8.0\n.target sm_86\n\0trailing"), Some(27));
    }

    /// A fat binary states its own size: a 16-byte wrapper header followed
    /// by `fatSize` bytes of payload.
    #[test]
    fn a_fat_binary_is_as_long_as_its_wrapper_header_says() {
        let mut image = Vec::new();
        image.extend_from_slice(&0xba55_ed50u32.to_le_bytes());
        image.extend_from_slice(&1u16.to_le_bytes());
        image.extend_from_slice(&16u16.to_le_bytes());
        image.extend_from_slice(&1024u64.to_le_bytes());
        assert_eq!(len_of(&image), Some(16 + 1024));
    }

    /// A cubin is an ELF, and its section table is the last thing in it.
    #[test]
    fn a_cubin_runs_to_the_end_of_its_section_table() {
        let mut image = vec![0u8; 0x40];
        image[..4].copy_from_slice(b"\x7fELF");
        image[0x20..0x28].copy_from_slice(&0x100u64.to_le_bytes());
        image[0x28..0x30].copy_from_slice(&0x800u64.to_le_bytes());
        image[0x34..0x36].copy_from_slice(&0x40u16.to_le_bytes());
        image[0x36..0x38].copy_from_slice(&0x38u16.to_le_bytes());
        image[0x38..0x3a].copy_from_slice(&2u16.to_le_bytes());
        image[0x3a..0x3c].copy_from_slice(&0x40u16.to_le_bytes());
        image[0x3c..0x3e].copy_from_slice(&4u16.to_le_bytes());
        assert_eq!(len_of(&image), Some(0x800 + 4 * 0x40));
    }
}
