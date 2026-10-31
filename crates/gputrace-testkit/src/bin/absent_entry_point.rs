//! Reports what a shimmed call answers rather than asserting on it, so a run
//! against a backing library that exports no `cu*` at all still finishes.

use std::ffi::{c_int, c_uint};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDriverGetVersion(*mut c_int) -> c_int;
    }
}

fn main() {
    let cuda = Driver::load();
    let mut version = -1;
    unsafe {
        println!("init={}", (cuda.cuInit)(0));
        println!("version={}", (cuda.cuDriverGetVersion)(&mut version));
    }
    println!("read={version}");
}
