//! Reaches the driver the way a program that `dlopen`s `libcuda.so.1` and
//! `dlsym`s its handle does, instead of letting the loader bind its `cu*`
//! (#13). Makes one call each way so a run shows both: the global-scope call
//! the interposer shadows, and the private-handle call it cannot see.
//!
//! Reports one `key=value` line per observation rather than asserting, so a
//! run against a backing library that answers nothing still finishes.

use std::ffi::{c_int, c_uint, c_void};

use gputrace_testkit::{DlopenedDriver, LIBCUDA_ENV, cuda_api, driver_symbol};

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
    }
}

fn main() {
    let cuda = Driver::load();
    println!("init={}", unsafe { (cuda.cuInit)(0) });

    let libcuda = std::env::var_os(LIBCUDA_ENV).unwrap_or_else(|| "libcuda.so.1".into());
    let Some(private) = DlopenedDriver::open(&libcuda) else {
        println!("dlopen=false");
        return;
    };
    println!("dlopen=true");

    let Some(pfn) = private.symbol("cuDeviceGetCount") else {
        println!("dlsym=false");
        return;
    };
    println!("dlsym=true");
    println!(
        "dlsym_is_interposer={}",
        Some(pfn) == driver_symbol("cuDeviceGetCount")
    );

    let device_get_count: unsafe extern "C" fn(*mut c_int) -> c_int =
        unsafe { std::mem::transmute::<*mut c_void, _>(pfn) };
    let mut count = -1;
    println!("call={}", unsafe { device_get_count(&mut count) });
    println!("devices={count}");
}
