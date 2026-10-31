//! One host thread per GPU, each working through a context of its own. Which
//! device each thread's calls targeted is only recoverable from the trace if
//! capture tags records with it: `cuDriverGetVersion` names no device at all.

use std::ffi::{c_int, c_uint, c_void};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDeviceGetCount(*mut c_int) -> c_int;
        fn cuCtxCreate_v4(*mut *mut c_void, *mut c_void, c_uint, c_int) -> c_int;
        fn cuCtxGetDevice(*mut c_int) -> c_int;
        fn cuDriverGetVersion(*mut c_int) -> c_int;
        fn cuCtxDestroy_v2(*mut c_void) -> c_int;
    }
}

unsafe extern "C" {
    fn gettid() -> c_int;
}

const CALLS_PER_THREAD: usize = 300;

fn main() {
    let cuda = Driver::load();
    let devices = unsafe {
        assert_eq!((cuda.cuInit)(0), 0);
        let mut devices = 0;
        assert_eq!((cuda.cuDeviceGetCount)(&mut devices), 0);
        devices
    };
    println!("devices={devices}");

    let reports: Vec<String> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..devices)
            .map(|device| {
                let cuda = &cuda;
                scope.spawn(move || unsafe {
                    let mut context = std::ptr::null_mut();
                    assert_eq!(
                        (cuda.cuCtxCreate_v4)(&mut context, std::ptr::null_mut(), 0, device),
                        0
                    );
                    let mut current = -1;
                    assert_eq!((cuda.cuCtxGetDevice)(&mut current), 0);
                    assert_eq!(current, device);

                    for _ in 0..CALLS_PER_THREAD {
                        let mut version = 0;
                        assert_eq!((cuda.cuDriverGetVersion)(&mut version), 0);
                    }

                    assert_eq!((cuda.cuCtxDestroy_v2)(context), 0);
                    format!("tid={} device={device}", gettid())
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("worker"))
            .collect()
    });
    for report in reports {
        println!("{report}");
    }
}
