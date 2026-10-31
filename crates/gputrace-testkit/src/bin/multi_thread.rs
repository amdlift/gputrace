use std::ffi::{c_int, c_uint, c_void};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDeviceGet(*mut c_int, c_int) -> c_int;
        fn cuCtxCreate_v4(*mut *mut c_void, *mut c_void, c_uint, c_int) -> c_int;
        fn cuMemAlloc_v2(*mut u64, usize) -> c_int;
        fn cuMemcpyHtoD_v2(u64, *const c_void, usize) -> c_int;
        fn cuMemcpyDtoH_v2(*mut c_void, u64, usize) -> c_int;
        fn cuMemFree_v2(u64) -> c_int;
        fn cuCtxDestroy_v2(*mut c_void) -> c_int;
    }
}

const THREADS: u8 = 4;

fn main() {
    let cuda = Driver::load();
    let device = unsafe {
        assert_eq!((cuda.cuInit)(0), 0);
        let mut device = -1;
        assert_eq!((cuda.cuDeviceGet)(&mut device, 0), 0);
        device
    };

    let round_trips = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..THREADS)
            .map(|id| {
                let cuda = &cuda;
                scope.spawn(move || unsafe {
                    let mut context = std::ptr::null_mut();
                    assert_eq!(
                        (cuda.cuCtxCreate_v4)(&mut context, std::ptr::null_mut(), 0, device),
                        0
                    );

                    let sent = [id; 32];
                    let mut received = [0u8; 32];
                    let mut dptr = 0u64;
                    assert_eq!((cuda.cuMemAlloc_v2)(&mut dptr, sent.len()), 0);
                    assert_eq!(
                        (cuda.cuMemcpyHtoD_v2)(dptr, sent.as_ptr().cast(), sent.len()),
                        0
                    );
                    assert_eq!(
                        (cuda.cuMemcpyDtoH_v2)(received.as_mut_ptr().cast(), dptr, received.len()),
                        0
                    );
                    assert_eq!((cuda.cuMemFree_v2)(dptr), 0);
                    assert_eq!((cuda.cuCtxDestroy_v2)(context), 0);
                    received == sent
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("worker"))
            .filter(|round_tripped| *round_tripped)
            .count()
    });
    println!("round_trips={round_trips}");
}
