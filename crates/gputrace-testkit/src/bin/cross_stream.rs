//! Two streams joined by one event: the producer copies, records the event,
//! then copies again; the consumer waits on that event and copies. Only the
//! producer's *first* copy is ordered before the consumer's.
//!
//! Run with `ptsz` to drive the per-thread-default-stream spellings of the
//! same calls, which is what a target built with `--default-stream
//! per-thread` binds.

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
        fn cuMemAllocAsync_ptsz(*mut u64, usize, *mut c_void) -> c_int;
        fn cuMemcpyHtoDAsync_v2(u64, *const c_void, usize, *mut c_void) -> c_int;
        fn cuMemcpyHtoDAsync_v2_ptsz(u64, *const c_void, usize, *mut c_void) -> c_int;
        fn cuEventRecord(*mut c_void, *mut c_void) -> c_int;
        fn cuEventRecord_ptsz(*mut c_void, *mut c_void) -> c_int;
        fn cuStreamWaitEvent(*mut c_void, *mut c_void, c_uint) -> c_int;
        fn cuStreamWaitEvent_ptsz(*mut c_void, *mut c_void, c_uint) -> c_int;
        fn cuMemFreeAsync(u64, *mut c_void) -> c_int;
        fn cuMemFreeAsync_ptsz(u64, *mut c_void) -> c_int;
        fn cuStreamSynchronize(*mut c_void) -> c_int;
        fn cuStreamSynchronize_ptsz(*mut c_void) -> c_int;
        fn cuEventDestroy(*mut c_void) -> c_int;
        fn cuEventDestroy_v2(*mut c_void) -> c_int;
        fn cuStreamDestroy(*mut c_void) -> c_int;
        fn cuStreamDestroy_v2(*mut c_void) -> c_int;
    }
}

const BYTES: usize = 16;
/// The fill byte each copy sends, so the trace says which copy is which.
pub const BEFORE: u8 = 0x11;
pub const AFTER: u8 = 0x22;
pub const WAITER: u8 = 0x33;

/// The spelling of each call this run drives.
struct Spelling {
    alloc_async: unsafe extern "C" fn(*mut u64, usize, *mut c_void) -> c_int,
    copy_async: unsafe extern "C" fn(u64, *const c_void, usize, *mut c_void) -> c_int,
    event_record: unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int,
    wait_event: unsafe extern "C" fn(*mut c_void, *mut c_void, c_uint) -> c_int,
    free_async: unsafe extern "C" fn(u64, *mut c_void) -> c_int,
    synchronize: unsafe extern "C" fn(*mut c_void) -> c_int,
    destroy_event: unsafe extern "C" fn(*mut c_void) -> c_int,
    destroy_stream: unsafe extern "C" fn(*mut c_void) -> c_int,
}

impl Spelling {
    fn of(cuda: &Driver, per_thread: bool) -> Self {
        if per_thread {
            Self {
                alloc_async: cuda.cuMemAllocAsync_ptsz,
                copy_async: cuda.cuMemcpyHtoDAsync_v2_ptsz,
                event_record: cuda.cuEventRecord_ptsz,
                wait_event: cuda.cuStreamWaitEvent_ptsz,
                free_async: cuda.cuMemFreeAsync_ptsz,
                synchronize: cuda.cuStreamSynchronize_ptsz,
                destroy_event: cuda.cuEventDestroy,
                destroy_stream: cuda.cuStreamDestroy,
            }
        } else {
            Self {
                alloc_async: cuda.cuMemAllocAsync,
                copy_async: cuda.cuMemcpyHtoDAsync_v2,
                event_record: cuda.cuEventRecord,
                wait_event: cuda.cuStreamWaitEvent,
                free_async: cuda.cuMemFreeAsync,
                synchronize: cuda.cuStreamSynchronize,
                destroy_event: cuda.cuEventDestroy_v2,
                destroy_stream: cuda.cuStreamDestroy_v2,
            }
        }
    }
}

fn main() {
    let cuda = Driver::load();
    let spelling = Spelling::of(&cuda, std::env::args().any(|arg| arg == "ptsz"));
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
        assert_eq!((spelling.alloc_async)(&mut buffer, BYTES, producer), 0);
        println!("buffer={buffer:#x}");

        let copy = |fill: u8, stream| {
            let sent = [fill; BYTES];
            assert_eq!(
                (spelling.copy_async)(buffer, sent.as_ptr().cast(), BYTES, stream),
                0
            );
        };
        copy(BEFORE, producer);
        assert_eq!((spelling.event_record)(done, producer), 0);
        copy(AFTER, producer);
        assert_eq!((spelling.wait_event)(consumer, done, 0), 0);
        copy(WAITER, consumer);

        assert_eq!((spelling.free_async)(buffer, consumer), 0);
        assert_eq!((spelling.synchronize)(producer), 0);
        assert_eq!((spelling.synchronize)(consumer), 0);
        assert_eq!((spelling.destroy_event)(done), 0);
        assert_eq!((spelling.destroy_stream)(producer), 0);
        assert_eq!((spelling.destroy_stream)(consumer), 0);
    }
}
