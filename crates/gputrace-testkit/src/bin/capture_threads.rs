//! Several host threads each make the same shimmed call many times over, so
//! capture has to attribute them per thread and hand several buffers to the
//! writer per thread.

use std::ffi::{c_int, c_uint};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDriverGetVersion(*mut c_int) -> c_int;
    }
}

const THREADS: usize = 4;
const CALLS_PER_THREAD: usize = 600;

fn main() {
    let cuda = Driver::load();
    unsafe { assert_eq!((cuda.cuInit)(0), 0) };

    let agreed = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..THREADS)
            .map(|_| {
                let cuda = &cuda;
                scope.spawn(move || {
                    (0..CALLS_PER_THREAD).all(|_| {
                        let mut version = 0;
                        unsafe { (cuda.cuDriverGetVersion)(&mut version) == 0 && version == 13040 }
                    })
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("worker"))
            .filter(|agreed| *agreed)
            .count()
    });
    println!("agreed={agreed}");
}
