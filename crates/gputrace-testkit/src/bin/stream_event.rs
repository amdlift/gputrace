use std::ffi::{c_int, c_uint, c_void};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDeviceGet(*mut c_int, c_int) -> c_int;
        fn cuCtxCreate_v4(*mut *mut c_void, *mut c_void, c_uint, c_int) -> c_int;
        fn cuStreamCreate(*mut *mut c_void, c_uint) -> c_int;
        fn cuEventCreate(*mut *mut c_void, c_uint) -> c_int;
        fn cuEventRecord(*mut c_void, *mut c_void) -> c_int;
        fn cuStreamWaitEvent(*mut c_void, *mut c_void, c_uint) -> c_int;
        fn cuStreamSynchronize(*mut c_void) -> c_int;
        fn cuEventSynchronize(*mut c_void) -> c_int;
        fn cuEventElapsedTime_v2(*mut f32, *mut c_void, *mut c_void) -> c_int;
        fn cuEventDestroy_v2(*mut c_void) -> c_int;
        fn cuStreamDestroy_v2(*mut c_void) -> c_int;
    }
}

fn main() {
    let cuda = Driver::load();
    unsafe {
        assert_eq!((cuda.cuInit)(0), 0);
        let mut device = -1;
        assert_eq!((cuda.cuDeviceGet)(&mut device, 0), 0);
        let mut context = std::ptr::null_mut();
        assert_eq!(
            (cuda.cuCtxCreate_v4)(&mut context, std::ptr::null_mut(), 0, device),
            0
        );

        let mut producer = std::ptr::null_mut();
        let mut consumer = std::ptr::null_mut();
        assert_eq!((cuda.cuStreamCreate)(&mut producer, 0), 0);
        assert_eq!((cuda.cuStreamCreate)(&mut consumer, 0), 0);
        println!("producer={:#x}", producer as usize);
        println!("consumer={:#x}", consumer as usize);

        let mut start = std::ptr::null_mut();
        let mut finish = std::ptr::null_mut();
        assert_eq!((cuda.cuEventCreate)(&mut start, 0), 0);
        assert_eq!((cuda.cuEventCreate)(&mut finish, 0), 0);
        println!("start={:#x}", start as usize);
        println!("finish={:#x}", finish as usize);

        assert_eq!((cuda.cuEventRecord)(start, producer), 0);
        assert_eq!((cuda.cuStreamWaitEvent)(consumer, start, 0), 0);
        assert_eq!((cuda.cuEventRecord)(finish, consumer), 0);
        assert_eq!((cuda.cuStreamSynchronize)(producer), 0);
        assert_eq!((cuda.cuStreamSynchronize)(consumer), 0);
        assert_eq!((cuda.cuEventSynchronize)(finish), 0);

        let mut elapsed = -1.0f32;
        assert_eq!((cuda.cuEventElapsedTime_v2)(&mut elapsed, start, finish), 0);
        println!("elapsed={elapsed}");

        assert_eq!((cuda.cuEventDestroy_v2)(finish), 0);
        assert_eq!((cuda.cuStreamDestroy_v2)(consumer), 0);
        println!(
            "record_on_destroyed={}",
            (cuda.cuEventRecord)(start, consumer)
        );
    }
}
