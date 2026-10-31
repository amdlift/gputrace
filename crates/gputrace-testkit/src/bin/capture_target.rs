//! A target for the `gputrace capture` wrapper: it makes shimmed driver
//! calls, writes to both streams, and exits with the code it is given.

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
    let version = unsafe {
        assert_eq!((cuda.cuInit)(0), 0);
        let mut version = 0;
        assert_eq!((cuda.cuDriverGetVersion)(&mut version), 0);
        version
    };
    println!("out:driver={version}");
    eprintln!("err:ready");

    let code = std::env::args()
        .nth(1)
        .map_or(0, |code| code.parse().expect("exit code"));
    std::process::exit(code);
}
