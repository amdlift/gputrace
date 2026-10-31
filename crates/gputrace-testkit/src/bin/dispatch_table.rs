//! Resolves driver entry points through the dispatch table instead of
//! linking them, the way the CUDA Runtime does (ADR-0003). Reports one
//! `key=value` line per observation rather than asserting, so a run against
//! a backing library that answers nothing still finishes.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};

use gputrace_testkit::{cuda_api, driver_symbol};

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuGetProcAddress(*const c_char, *mut *mut c_void, c_int, u64) -> c_int;
        fn cuGetProcAddress_v2(*const c_char, *mut *mut c_void, c_int, u64, *mut c_int) -> c_int;
    }
}

const CUDA_VERSION: c_int = 13040;

struct Lookup {
    result: c_int,
    status: c_int,
    pfn: *mut c_void,
}

impl Driver {
    fn lookup(&self, symbol: &CStr, cuda_version: c_int) -> Lookup {
        let mut found = Lookup {
            result: -1,
            status: -1,
            pfn: std::ptr::null_mut(),
        };
        found.result = unsafe {
            (self.cuGetProcAddress_v2)(
                symbol.as_ptr(),
                &mut found.pfn,
                cuda_version,
                0,
                &mut found.status,
            )
        };
        found
    }
}

fn main() {
    let cuda = Driver::load();
    unsafe {
        println!("init={}", (cuda.cuInit)(0));

        let shimmed = cuda.lookup(c"cuDriverGetVersion", CUDA_VERSION);
        println!("lookup={}", shimmed.result);
        println!("status={}", shimmed.status);
        println!("lookup_null={}", shimmed.pfn.is_null());
        println!(
            "lookup_is_interposer={}",
            Some(shimmed.pfn) == driver_symbol("cuDriverGetVersion")
        );
        if !shimmed.pfn.is_null() {
            let driver_get_version: unsafe extern "C" fn(*mut c_int) -> c_int =
                std::mem::transmute(shimmed.pfn);
            let mut version = -1;
            println!("call={}", driver_get_version(&mut version));
            println!("version={version}");
        }

        let mut v1_pfn = std::ptr::null_mut();
        let v1 =
            (cuda.cuGetProcAddress)(c"cuDriverGetVersion".as_ptr(), &mut v1_pfn, CUDA_VERSION, 0);
        println!("v1={v1}");
        println!("v1_agrees={}", v1_pfn == shimmed.pfn);

        let unshimmed = cuda.lookup(c"cuDeviceGetCount", CUDA_VERSION);
        println!(
            "unshimmed_agrees={}",
            Some(unshimmed.pfn) == driver_symbol("cuDeviceGetCount")
        );
        if !unshimmed.pfn.is_null() {
            let device_get_count: unsafe extern "C" fn(*mut c_int) -> c_int =
                std::mem::transmute(unshimmed.pfn);
            let mut count = -1;
            println!("count_call={}", device_get_count(&mut count));
            println!("devices={count}");
        }

        let missing = cuda.lookup(c"cuNotAnEntryPoint", CUDA_VERSION);
        println!("missing={}", missing.result);
        println!("missing_status={}", missing.status);
        println!("missing_null={}", missing.pfn.is_null());

        let too_old = cuda.lookup(c"cuDriverGetVersion", 11000);
        println!("too_old={}", too_old.result);
        println!("too_old_status={}", too_old.status);

        // The base spelling the driver answers with its `_v2` symbol: the
        // dispatch table must keep handing back the interposer, or the CUDA
        // Runtime would resolve past it on the next lookup.
        let dispatch = cuda.lookup(c"cuGetProcAddress", CUDA_VERSION);
        println!(
            "dispatch_is_interposer={}",
            Some(dispatch.pfn) == driver_symbol("cuGetProcAddress_v2")
        );
    }
}
