//! Context lifecycle on one thread across two GPUs: create, push, pop and
//! set, with a device-agnostic call after each so the trace shows which GPU
//! the thread was on at the time.

use std::ffi::{c_int, c_uint, c_void};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDeviceGet(*mut c_int, c_int) -> c_int;
        fn cuCtxCreate_v4(*mut *mut c_void, *mut c_void, c_uint, c_int) -> c_int;
        fn cuCtxPushCurrent_v2(*mut c_void) -> c_int;
        fn cuCtxPopCurrent_v2(*mut *mut c_void) -> c_int;
        fn cuCtxSetCurrent(*mut c_void) -> c_int;
        fn cuDriverGetVersion(*mut c_int) -> c_int;
        fn cuCtxDestroy_v2(*mut c_void) -> c_int;
    }
}

fn main() {
    let cuda = Driver::load();
    unsafe {
        let mut version = 0;
        let mut anywhere = |cuda: &Driver| {
            assert_eq!((cuda.cuDriverGetVersion)(&mut version), 0);
        };

        assert_eq!((cuda.cuInit)(0), 0);
        let mut context = [std::ptr::null_mut(); 2];
        for (ordinal, context) in context.iter_mut().enumerate() {
            let mut device = -1;
            assert_eq!((cuda.cuDeviceGet)(&mut device, ordinal as c_int), 0);
            assert_eq!(
                (cuda.cuCtxCreate_v4)(context, std::ptr::null_mut(), 0, device),
                0
            );
        }
        anywhere(&cuda);

        assert_eq!((cuda.cuCtxPushCurrent_v2)(context[0]), 0);
        anywhere(&cuda);

        let mut popped = std::ptr::null_mut();
        assert_eq!((cuda.cuCtxPopCurrent_v2)(&mut popped), 0);
        assert_eq!(popped, context[0]);
        anywhere(&cuda);

        assert_eq!((cuda.cuCtxSetCurrent)(context[0]), 0);
        anywhere(&cuda);

        assert_eq!((cuda.cuCtxDestroy_v2)(context[0]), 0);
        assert_eq!((cuda.cuCtxSetCurrent)(context[1]), 0);
        assert_eq!((cuda.cuCtxDestroy_v2)(context[1]), 0);
    }
    println!("version={}", 13040);
}
