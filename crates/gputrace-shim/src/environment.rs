//! What the capture ran against, asked of the driver directly.
//!
//! Every question here goes to the backing library through [`crate::real`],
//! never through this interposer's own exported `cu*`. A shimmed query would
//! append a record the target never made, and a trace whose own probes cannot
//! be told apart from the target's calls is not a recording of the target.
//!
//! An unanswerable question is answered `None`. A manifest that invented a
//! driver version would be worse than one that admits it does not know, since
//! warning on an incompatible environment is the whole point of recording one.

use std::ffi::c_int;

use gputrace_core::ComputeCapability;

use crate::real::real_symbol;

const CUDA_SUCCESS: c_int = 0;

/// `CUdevice_attribute`, transcribed from `cuda.h`.
const COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const COMPUTE_CAPABILITY_MINOR: c_int = 76;

/// The backing driver's version, in CUDA's `1000 * major + 10 * minor` form.
pub fn driver_version() -> Option<u32> {
    let get_version = *unsafe {
        real_symbol::<unsafe extern "C" fn(*mut c_int) -> c_int>(b"cuDriverGetVersion\0")
    }?;
    let mut version = 0;
    if unsafe { get_version(&mut version) } != CUDA_SUCCESS {
        return None;
    }
    u32::try_from(version).ok()
}

/// The architecture generation of the GPU the captured process named by the
/// `CUdevice` `device`.
pub fn compute_capability(device: u32) -> Option<ComputeCapability> {
    let device = c_int::try_from(device).ok()?;
    Some(ComputeCapability {
        major: attribute(COMPUTE_CAPABILITY_MAJOR, device)?,
        minor: attribute(COMPUTE_CAPABILITY_MINOR, device)?,
    })
}

fn attribute(attribute: c_int, device: c_int) -> Option<u32> {
    let get_attribute = *unsafe {
        real_symbol::<unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int>(
            b"cuDeviceGetAttribute\0",
        )
    }?;
    let mut value = 0;
    if unsafe { get_attribute(&mut value, attribute, device) } != CUDA_SUCCESS {
        return None;
    }
    u32::try_from(value).ok()
}
