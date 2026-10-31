//! The two ways a target can hand `cuLaunchKernel` its arguments: the
//! self-describing `extra` packing, which states the buffer's size, and the
//! bare `kernelParams` array of pointers, which states nothing.

use std::ffi::{c_char, c_int, c_uint, c_void};

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDeviceGet(*mut c_int, c_int) -> c_int;
        fn cuCtxCreate_v4(*mut *mut c_void, *mut c_void, c_uint, c_int) -> c_int;
        fn cuModuleLoadData(*mut *mut c_void, *const c_void) -> c_int;
        fn cuModuleGetFunction(*mut *mut c_void, *mut c_void, *const c_char) -> c_int;
        fn cuStreamCreate(*mut *mut c_void, c_uint) -> c_int;
        fn cuLaunchKernel(
            *mut c_void, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
            *mut c_void, *mut *mut c_void, *mut *mut c_void,
        ) -> c_int;
    }
}

const PTX: &[u8] = b".version 8.0\n.target sm_86\n.visible .entry scale() { ret; }\n\0";

/// The packed argument buffer, byte for byte as a kernel would receive it.
const PACKED: [u8; 12] = [7, 0, 0, 0, 0, 0, 0xc0, 0x3f, 0xef, 0xbe, 0xad, 0xde];

/// CUDA's `extra` keys are three low addresses, never dereferenced.
const CU_LAUNCH_PARAM_END: *mut c_void = std::ptr::null_mut();
const CU_LAUNCH_PARAM_BUFFER_POINTER: *mut c_void = std::ptr::without_provenance_mut(1);
const CU_LAUNCH_PARAM_BUFFER_SIZE: *mut c_void = std::ptr::without_provenance_mut(2);

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
        let mut scale = std::ptr::null_mut();
        assert_eq!(
            (cuda.cuModuleGetFunction)(&mut scale, module, c"scale".as_ptr()),
            0
        );
        let mut stream = std::ptr::null_mut();
        assert_eq!((cuda.cuStreamCreate)(&mut stream, 0), 0);

        let mut size = PACKED.len();
        let mut extra = [
            CU_LAUNCH_PARAM_BUFFER_POINTER,
            PACKED.as_ptr() as *mut c_void,
            CU_LAUNCH_PARAM_BUFFER_SIZE,
            std::ptr::addr_of_mut!(size).cast(),
            CU_LAUNCH_PARAM_END,
        ];
        assert_eq!(
            (cuda.cuLaunchKernel)(
                scale,
                4,
                2,
                1,
                64,
                8,
                1,
                256,
                stream,
                std::ptr::null_mut(),
                extra.as_mut_ptr(),
            ),
            0
        );

        let mut count = 7u32;
        let mut factor = 1.5f32;
        let mut params = [
            std::ptr::addr_of_mut!(count).cast::<c_void>(),
            std::ptr::addr_of_mut!(factor).cast::<c_void>(),
        ];
        assert_eq!(
            (cuda.cuLaunchKernel)(
                scale,
                1,
                1,
                1,
                32,
                1,
                1,
                0,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            ),
            0
        );
        println!("kernel_params={:#x}", params.as_ptr() as usize);
    }
}
