//! The process exits while its worker threads are still recording. Nothing
//! joins them and nothing stops them, so capture has to collect their buffers
//! out from under threads that are actively appending to them.
//!
//! With `abrupt` the target leaves through `_exit`, which runs no `atexit`
//! handler: capture gets no chance to finish the bundle, and must still not
//! change what the target's exit looks like.

use std::ffi::{c_int, c_uint};
use std::sync::mpsc;

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDriverGetVersion(*mut c_int) -> c_int;
    }
}

unsafe extern "C" {
    fn _exit(status: c_int) -> !;
}

const THREADS: usize = 4;
const CALLS_BEFORE_REPORTING: usize = 512;
const ABRUPT_STATUS: c_int = 7;

fn main() {
    let cuda: &'static Driver = Box::leak(Box::new(Driver::load()));
    unsafe { assert_eq!((cuda.cuInit)(0), 0) };

    let (report, reported) = mpsc::channel();
    for _ in 0..THREADS {
        let report = report.clone();
        std::thread::spawn(move || {
            let mut made = 0usize;
            loop {
                let mut version = 0;
                if unsafe { (cuda.cuDriverGetVersion)(&mut version) } != 0 || version != 13040 {
                    return;
                }
                made += 1;
                if made == CALLS_BEFORE_REPORTING {
                    report.send(()).expect("report");
                }
            }
        });
    }
    drop(report);

    for _ in 0..THREADS {
        reported.recv().expect("worker");
    }
    println!("recording={THREADS}");

    if std::env::args().nth(1).as_deref() == Some("abrupt") {
        unsafe { _exit(ABRUPT_STATUS) };
    }
}
