use std::ffi::{c_int, c_uint};

use crate::shim::shim;

shim! {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDriverGetVersion(driver_version: *mut c_int) -> CUresult;
}
