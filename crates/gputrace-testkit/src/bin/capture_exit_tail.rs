//! Worker threads that outlive `main`. Each fills more than one buffer and
//! leaves a partial one behind, then parks forever: nothing joins them, so
//! their tails reach the bundle only if capture flushes threads that are
//! still alive when the process exits.

use std::ffi::{c_int, c_uint};
use std::sync::mpsc;

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDriverGetVersion(*mut c_int) -> c_int;
    }
}

const THREADS: usize = 4;
const CALLS_PER_THREAD: usize = 300;

fn main() {
    let cuda: &'static Driver = Box::leak(Box::new(Driver::load()));
    unsafe { assert_eq!((cuda.cuInit)(0), 0) };

    let (report, reported) = mpsc::channel();
    for _ in 0..THREADS {
        let report = report.clone();
        std::thread::spawn(move || {
            for _ in 0..CALLS_PER_THREAD {
                let mut version = 0;
                assert_eq!(unsafe { (cuda.cuDriverGetVersion)(&mut version) }, 0);
                assert_eq!(version, 13040);
            }
            report.send(()).expect("report");
            loop {
                std::thread::park();
            }
        });
    }
    drop(report);

    for _ in 0..THREADS {
        reported.recv().expect("worker");
    }
    println!("parked={THREADS}");
}
