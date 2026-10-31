//! The same work through the pre-CUDA-3.2 spellings, plus a per-thread-
//! default-stream copy: a target built against an older header must record
//! exactly as a current one does.

use std::ffi::{c_int, c_uint, c_void};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDeviceGet(*mut c_int, c_int) -> c_int;
        fn cuCtxCreate_v2(*mut *mut c_void, c_uint, c_int) -> c_int;
        fn cuMemAlloc(*mut c_uint, c_uint) -> c_int;
        fn cuMemcpyHtoD(c_uint, *const c_void, c_uint) -> c_int;
        fn cuMemcpyHtoD_v2_ptds(u64, *const c_void, usize) -> c_int;
        fn cuMemFree(c_uint) -> c_int;
        fn cuCtxDestroy(*mut c_void) -> c_int;
    }
}

pub const PAYLOAD: [u8; 16] = [0x5a; 16];

fn main() {
    let cuda = Driver::load();
    unsafe {
        assert_eq!((cuda.cuInit)(0), 0);
        let mut device = -1;
        assert_eq!((cuda.cuDeviceGet)(&mut device, 0), 0);
        let mut context = std::ptr::null_mut();
        assert_eq!((cuda.cuCtxCreate_v2)(&mut context, 0, device), 0);

        let mut dptr = 0u32;
        assert_eq!((cuda.cuMemAlloc)(&mut dptr, PAYLOAD.len() as c_uint), 0);
        println!("device_ptr={dptr:#x}");

        assert_eq!(
            (cuda.cuMemcpyHtoD)(dptr, PAYLOAD.as_ptr().cast(), PAYLOAD.len() as c_uint),
            0
        );
        assert_eq!(
            (cuda.cuMemcpyHtoD_v2_ptds)(u64::from(dptr), PAYLOAD.as_ptr().cast(), PAYLOAD.len()),
            0
        );

        assert_eq!((cuda.cuMemFree)(dptr), 0);
        assert_eq!((cuda.cuCtxDestroy)(context), 0);
    }
}
