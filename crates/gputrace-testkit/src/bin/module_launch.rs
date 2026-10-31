use std::ffi::{c_char, c_int, c_uint, c_void};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDeviceGet(*mut c_int, c_int) -> c_int;
        fn cuCtxCreate_v4(*mut *mut c_void, *mut c_void, c_uint, c_int) -> c_int;
        fn cuModuleLoadData(*mut *mut c_void, *const c_void) -> c_int;
        fn cuModuleGetFunction(*mut *mut c_void, *mut c_void, *const c_char) -> c_int;
        fn cuLaunchKernel(
            *mut c_void, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
            *mut c_void, *mut *mut c_void, *mut *mut c_void,
        ) -> c_int;
        fn cuModuleUnload(*mut c_void) -> c_int;
    }
}

const PTX: &[u8] = b".version 8.0\n.target sm_86\n.visible .entry kernel_a() { ret; }\n.visible .entry kernel_b() { ret; }\n\0";

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

        let mut module = std::ptr::null_mut();
        assert_eq!((cuda.cuModuleLoadData)(&mut module, PTX.as_ptr().cast()), 0);
        println!("module={:#x}", module as usize);

        let mut first = std::ptr::null_mut();
        assert_eq!(
            (cuda.cuModuleGetFunction)(&mut first, module, c"kernel_a".as_ptr()),
            0
        );
        let mut again = std::ptr::null_mut();
        assert_eq!(
            (cuda.cuModuleGetFunction)(&mut again, module, c"kernel_a".as_ptr()),
            0
        );
        let mut other = std::ptr::null_mut();
        assert_eq!(
            (cuda.cuModuleGetFunction)(&mut other, module, c"kernel_b".as_ptr()),
            0
        );
        println!("kernel_a={:#x}", first as usize);
        println!("kernel_a_again={:#x}", again as usize);
        println!("kernel_b={:#x}", other as usize);

        let mut absent = std::ptr::null_mut();
        println!(
            "absent={}",
            (cuda.cuModuleGetFunction)(&mut absent, module, c"kernel_missing".as_ptr())
        );

        println!(
            "launch={}",
            (cuda.cuLaunchKernel)(
                first,
                4,
                1,
                1,
                64,
                1,
                1,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        );

        assert_eq!((cuda.cuModuleUnload)(module), 0);
        println!(
            "launch_after_unload={}",
            (cuda.cuLaunchKernel)(
                first,
                4,
                1,
                1,
                64,
                1,
                1,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        );
    }
}
