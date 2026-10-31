//! Asynchronous read-backs settled by each of the three blocking calls that
//! can make one valid. Pass `stream`, `event` or `context` to choose which.

use std::ffi::{c_int, c_uint, c_void};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDeviceGet(*mut c_int, c_int) -> c_int;
        fn cuCtxCreate_v4(*mut *mut c_void, *mut c_void, c_uint, c_int) -> c_int;
        fn cuStreamCreate(*mut *mut c_void, c_uint) -> c_int;
        fn cuStreamSynchronize(*mut c_void) -> c_int;
        fn cuStreamDestroy_v2(*mut c_void) -> c_int;
        fn cuEventCreate(*mut *mut c_void, c_uint) -> c_int;
        fn cuEventRecord(*mut c_void, *mut c_void) -> c_int;
        fn cuEventSynchronize(*mut c_void) -> c_int;
        fn cuEventDestroy_v2(*mut c_void) -> c_int;
        fn cuCtxSynchronize() -> c_int;
        fn cuMemAlloc_v2(*mut u64, usize) -> c_int;
        fn cuMemcpyHtoD_v2(u64, *const c_void, usize) -> c_int;
        fn cuMemcpyDtoHAsync_v2(*mut c_void, u64, usize, *mut c_void) -> c_int;
        fn cuMemFree_v2(u64) -> c_int;
        fn cuCtxDestroy_v2(*mut c_void) -> c_int;
    }
}

fn main() {
    let settle = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "stream".to_owned());
    let cuda = Driver::load();
    let sent: [u8; 8] = [0xc0, 0xff, 0xee, 0x00, 0x11, 0x22, 0x33, 0x44];
    let mut received = [0u8; 8];

    unsafe {
        assert_eq!((cuda.cuInit)(0), 0);

        let mut device = -1;
        assert_eq!((cuda.cuDeviceGet)(&mut device, 0), 0);

        let mut context = std::ptr::null_mut();
        assert_eq!(
            (cuda.cuCtxCreate_v4)(&mut context, std::ptr::null_mut(), 0, device),
            0
        );

        let mut stream = std::ptr::null_mut();
        assert_eq!((cuda.cuStreamCreate)(&mut stream, 0), 0);

        let mut dptr = 0u64;
        assert_eq!((cuda.cuMemAlloc_v2)(&mut dptr, sent.len()), 0);
        assert_eq!(
            (cuda.cuMemcpyHtoD_v2)(dptr, sent.as_ptr().cast(), sent.len()),
            0
        );

        assert_eq!(
            (cuda.cuMemcpyDtoHAsync_v2)(received.as_mut_ptr().cast(), dptr, received.len(), stream),
            0
        );

        match settle.as_str() {
            "event" => {
                let mut event = std::ptr::null_mut();
                assert_eq!((cuda.cuEventCreate)(&mut event, 0), 0);
                assert_eq!((cuda.cuEventRecord)(event, stream), 0);
                assert_eq!((cuda.cuEventSynchronize)(event), 0);
                assert_eq!((cuda.cuEventDestroy_v2)(event), 0);
            }
            "context" => assert_eq!((cuda.cuCtxSynchronize)(), 0),
            _ => assert_eq!((cuda.cuStreamSynchronize)(stream), 0),
        }

        println!(
            "received={}",
            received
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        );

        assert_eq!((cuda.cuMemFree_v2)(dptr), 0);
        assert_eq!((cuda.cuStreamDestroy_v2)(stream), 0);
        assert_eq!((cuda.cuCtxDestroy_v2)(context), 0);
    }
}
