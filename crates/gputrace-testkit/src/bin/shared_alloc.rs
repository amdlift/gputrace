//! One thread allocates, several use what it allocated: the case where a
//! device pointer's identity has to agree across host threads. Every worker
//! also sends one identical payload and one of its own, so a bundle can be
//! checked for payload dedup.

use std::ffi::{c_int, c_uint, c_void};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDeviceGet(*mut c_int, c_int) -> c_int;
        fn cuCtxCreate_v4(*mut *mut c_void, *mut c_void, c_uint, c_int) -> c_int;
        fn cuCtxSetCurrent(*mut c_void) -> c_int;
        fn cuMemAlloc_v2(*mut u64, usize) -> c_int;
        fn cuMemcpyHtoD_v2(u64, *const c_void, usize) -> c_int;
        fn cuMemFree_v2(u64) -> c_int;
        fn cuCtxDestroy_v2(*mut c_void) -> c_int;
    }
}

const THREADS: u8 = 4;
pub const SHARED_PAYLOAD: [u8; 16] = [0xa5; 16];

fn main() {
    let cuda = Driver::load();

    let (context, shared, private) = unsafe {
        assert_eq!((cuda.cuInit)(0), 0);
        let mut device = -1;
        assert_eq!((cuda.cuDeviceGet)(&mut device, 0), 0);
        let mut context = std::ptr::null_mut();
        assert_eq!(
            (cuda.cuCtxCreate_v4)(&mut context, std::ptr::null_mut(), 0, device),
            0
        );
        let mut shared = 0u64;
        let mut private = 0u64;
        assert_eq!((cuda.cuMemAlloc_v2)(&mut shared, SHARED_PAYLOAD.len()), 0);
        assert_eq!((cuda.cuMemAlloc_v2)(&mut private, SHARED_PAYLOAD.len()), 0);
        (context as usize, shared, private)
    };
    println!("shared={shared:#x}");
    println!("private={private:#x}");

    std::thread::scope(|scope| {
        for id in 0..THREADS {
            let cuda = &cuda;
            scope.spawn(move || unsafe {
                assert_eq!((cuda.cuCtxSetCurrent)(context as *mut c_void), 0);
                assert_eq!(
                    (cuda.cuMemcpyHtoD_v2)(
                        shared,
                        SHARED_PAYLOAD.as_ptr().cast(),
                        SHARED_PAYLOAD.len()
                    ),
                    0
                );
                let own = [id; SHARED_PAYLOAD.len()];
                assert_eq!(
                    (cuda.cuMemcpyHtoD_v2)(private, own.as_ptr().cast(), own.len()),
                    0
                );
            });
        }
    });

    unsafe {
        assert_eq!((cuda.cuMemFree_v2)(shared), 0);
        assert_eq!((cuda.cuMemFree_v2)(private), 0);
        assert_eq!((cuda.cuCtxDestroy_v2)(context as *mut c_void), 0);
    }
    println!("threads={THREADS}");
}
