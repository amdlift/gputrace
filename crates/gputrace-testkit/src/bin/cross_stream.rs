//! Two streams joined by one event: the producer copies, records the event,
//! then copies again; the consumer waits on that event and copies. Only the
//! producer's *first* copy is ordered before the consumer's.

use std::ffi::{c_int, c_uint, c_void};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDeviceGet(*mut c_int, c_int) -> c_int;
        fn cuCtxCreate_v4(*mut *mut c_void, *mut c_void, c_uint, c_int) -> c_int;
        fn cuStreamCreate(*mut *mut c_void, c_uint) -> c_int;
        fn cuEventCreate(*mut *mut c_void, c_uint) -> c_int;
        fn cuMemAllocAsync(*mut u64, usize, *mut c_void) -> c_int;
        fn cuMemcpyHtoDAsync_v2(u64, *const c_void, usize, *mut c_void) -> c_int;
        fn cuEventRecord(*mut c_void, *mut c_void) -> c_int;
        fn cuStreamWaitEvent(*mut c_void, *mut c_void, c_uint) -> c_int;
        fn cuMemFreeAsync(u64, *mut c_void) -> c_int;
        fn cuStreamSynchronize(*mut c_void) -> c_int;
        fn cuEventDestroy_v2(*mut c_void) -> c_int;
        fn cuStreamDestroy_v2(*mut c_void) -> c_int;
    }
}

const BYTES: usize = 16;
/// The fill byte each copy sends, so the trace says which copy is which.
pub const BEFORE: u8 = 0x11;
pub const AFTER: u8 = 0x22;
pub const WAITER: u8 = 0x33;

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

        let mut producer = std::ptr::null_mut();
        let mut consumer = std::ptr::null_mut();
        assert_eq!((cuda.cuStreamCreate)(&mut producer, 0), 0);
        assert_eq!((cuda.cuStreamCreate)(&mut consumer, 0), 0);
        let mut done = std::ptr::null_mut();
        assert_eq!((cuda.cuEventCreate)(&mut done, 0), 0);

        let mut buffer = 0u64;
        assert_eq!((cuda.cuMemAllocAsync)(&mut buffer, BYTES, producer), 0);
        println!("buffer={buffer:#x}");

        let copy = |fill: u8, stream| {
            let sent = [fill; BYTES];
            assert_eq!(
                (cuda.cuMemcpyHtoDAsync_v2)(buffer, sent.as_ptr().cast(), BYTES, stream),
                0
            );
        };
        copy(BEFORE, producer);
        assert_eq!((cuda.cuEventRecord)(done, producer), 0);
        copy(AFTER, producer);
        assert_eq!((cuda.cuStreamWaitEvent)(consumer, done, 0), 0);
        copy(WAITER, consumer);

        assert_eq!((cuda.cuMemFreeAsync)(buffer, consumer), 0);
        assert_eq!((cuda.cuStreamSynchronize)(producer), 0);
        assert_eq!((cuda.cuStreamSynchronize)(consumer), 0);
        assert_eq!((cuda.cuEventDestroy_v2)(done), 0);
        assert_eq!((cuda.cuStreamDestroy_v2)(producer), 0);
        assert_eq!((cuda.cuStreamDestroy_v2)(consumer), 0);
    }
}
