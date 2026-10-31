use std::ffi::{c_char, c_int, c_uint};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDriverGetVersion(*mut c_int) -> c_int;
        fn cuDeviceGetCount(*mut c_int) -> c_int;
        fn cuDeviceGetName(*mut c_char, c_int, c_int) -> c_int;
    }
}

fn main() {
    let cuda = Driver::load();
    unsafe {
        assert_eq!((cuda.cuInit)(0), 0);

        let mut version = 0;
        assert_eq!((cuda.cuDriverGetVersion)(&mut version), 0);
        println!("driver={version}");

        let mut count = 0;
        assert_eq!((cuda.cuDeviceGetCount)(&mut count), 0);
        println!("devices={count}");

        let mut name = [0u8; 64];
        assert_eq!((cuda.cuDeviceGetName)(name.as_mut_ptr().cast(), 64, 0), 0);
        let end = name.iter().position(|&b| b == 0).unwrap();
        println!("name={}", std::str::from_utf8(&name[..end]).unwrap());
    }
}
