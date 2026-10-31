use std::ffi::{c_int, c_uint, c_void};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDeviceGet(*mut c_int, c_int) -> c_int;
        fn cuCtxCreate_v4(*mut *mut c_void, *mut c_void, c_uint, c_int) -> c_int;
        fn cuMemAlloc_v2(*mut u64, usize) -> c_int;
        fn cuMemcpyHtoD_v2(u64, *const c_void, usize) -> c_int;
        fn cuMemFree_v2(u64) -> c_int;
        fn cuCtxDestroy_v2(*mut c_void) -> c_int;
    }
}

fn main() {
    let cuda = Driver::load();
    let first: [u8; 8] = [0x11; 8];
    let second: [u8; 8] = [0x22; 8];

    unsafe {
        assert_eq!((cuda.cuInit)(0), 0);

        let mut device = -1;
        assert_eq!((cuda.cuDeviceGet)(&mut device, 0), 0);

        let mut context = std::ptr::null_mut();
        assert_eq!(
            (cuda.cuCtxCreate_v4)(&mut context, std::ptr::null_mut(), 0, device),
            0
        );

        for payload in [&first, &second] {
            let mut dptr = 0u64;
            assert_eq!((cuda.cuMemAlloc_v2)(&mut dptr, payload.len()), 0);
            println!("device_ptr={dptr:#x}");
            assert_eq!(
                (cuda.cuMemcpyHtoD_v2)(dptr, payload.as_ptr().cast(), payload.len()),
                0
            );
            assert_eq!((cuda.cuMemFree_v2)(dptr), 0);
        }

        assert_eq!((cuda.cuCtxDestroy_v2)(context), 0);
    }
}
