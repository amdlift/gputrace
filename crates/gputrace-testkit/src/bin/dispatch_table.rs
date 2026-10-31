//! Resolves driver entry points through the dispatch table instead of
//! linking them, the way the CUDA Runtime does (ADR-0003). Reports one
//! `key=value` line per observation.

use std::ffi::{c_char, c_int, c_uint, c_void};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuGetProcAddress(*const c_char, *mut *mut c_void, c_int, u64) -> c_int;
        fn cuGetProcAddress_v2(*const c_char, *mut *mut c_void, c_int, u64, *mut c_int) -> c_int;
    }
}

const CUDA_VERSION: c_int = 13040;

fn main() {
    let cuda = Driver::load();
    unsafe {
        assert_eq!((cuda.cuInit)(0), 0);

        let mut pfn = std::ptr::null_mut();
        let mut status = -1;
        let lookup = (cuda.cuGetProcAddress_v2)(
            c"cuDriverGetVersion".as_ptr(),
            &mut pfn,
            CUDA_VERSION,
            0,
            &mut status,
        );
        println!("lookup={lookup}");
        println!("status={status}");

        let driver_get_version: unsafe extern "C" fn(*mut c_int) -> c_int =
            std::mem::transmute(pfn);
        let mut version = -1;
        println!("call={}", driver_get_version(&mut version));
        println!("version={version}");

        let mut v1_pfn = std::ptr::null_mut();
        let v1 =
            (cuda.cuGetProcAddress)(c"cuDriverGetVersion".as_ptr(), &mut v1_pfn, CUDA_VERSION, 0);
        println!("v1={v1}");
        println!("v1_agrees={}", v1_pfn == pfn);
    }
}
