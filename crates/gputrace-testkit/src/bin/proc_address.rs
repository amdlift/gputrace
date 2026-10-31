use std::ffi::{c_char, c_int, c_uint, c_void};

use gputrace_testkit::{cuda_api, driver_symbol};

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuGetProcAddress_v2(*const c_char, *mut *mut c_void, c_int, u64, *mut c_int) -> c_int;
        fn cuGetProcAddress(*const c_char, *mut *mut c_void, c_int, u64) -> c_int;
    }
}

const CUDA_VERSION: c_int = 13040;
const PER_THREAD_DEFAULT_STREAM: u64 = 1 << 1;

fn main() {
    let cuda = Driver::load();
    unsafe {
        assert_eq!((cuda.cuInit)(0), 0);

        let mut count_pfn = std::ptr::null_mut();
        let mut status = -1;
        let result = (cuda.cuGetProcAddress_v2)(
            c"cuDeviceGetCount".as_ptr(),
            &mut count_pfn,
            CUDA_VERSION,
            0,
            &mut status,
        );
        println!("lookup={result} status={status}");

        let device_get_count: unsafe extern "C" fn(*mut c_int) -> c_int =
            std::mem::transmute(count_pfn);
        let mut count = 0;
        assert_eq!(device_get_count(&mut count), 0);
        println!("devices={count}");

        let mut alloc_pfn = std::ptr::null_mut();
        (cuda.cuGetProcAddress_v2)(
            c"cuMemAlloc".as_ptr(),
            &mut alloc_pfn,
            CUDA_VERSION,
            0,
            &mut status,
        );
        println!(
            "mem_alloc_is_v2={}",
            alloc_pfn == driver_symbol("cuMemAlloc_v2").expect("cuMemAlloc_v2")
        );

        let mut sync_pfn = std::ptr::null_mut();
        (cuda.cuGetProcAddress_v2)(
            c"cuStreamSynchronize".as_ptr(),
            &mut sync_pfn,
            CUDA_VERSION,
            PER_THREAD_DEFAULT_STREAM,
            &mut status,
        );
        println!(
            "per_thread_default_stream={}",
            sync_pfn
                == driver_symbol("cuStreamSynchronize_ptsz").expect("cuStreamSynchronize_ptsz")
        );

        let mut missing_pfn = std::ptr::null_mut();
        let result = (cuda.cuGetProcAddress_v2)(
            c"cuNotAnEntryPoint".as_ptr(),
            &mut missing_pfn,
            CUDA_VERSION,
            0,
            &mut status,
        );
        println!(
            "missing={result} status={status} null={}",
            missing_pfn.is_null()
        );

        let mut old_pfn = std::ptr::null_mut();
        let result = (cuda.cuGetProcAddress_v2)(
            c"cuDeviceGetCount".as_ptr(),
            &mut old_pfn,
            11000,
            0,
            &mut status,
        );
        println!("too_old={result} status={status}");

        let mut v1_pfn = std::ptr::null_mut();
        let result =
            (cuda.cuGetProcAddress)(c"cuDeviceGetCount".as_ptr(), &mut v1_pfn, CUDA_VERSION, 0);
        println!("v1={result} agrees={}", v1_pfn == count_pfn);
    }
}
