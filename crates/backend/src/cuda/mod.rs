pub mod ops;

use anyhow::Result;
use cudarc::driver::{result, sys};

pub use ops::{Gpu, GpuMat};

pub fn device_name(ordinal: usize) -> Result<String> {
    let dev = result::device::get(ordinal as i32)?;
    let mut buf = [0u8; 256];
    unsafe {
        let r = sys::cuDeviceGetName(buf.as_mut_ptr() as *mut _, buf.len() as i32, dev);
        if r != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuDeviceGetName failed: {r:?}");
        }
    }
    Ok(std::ffi::CStr::from_bytes_until_nul(&buf)?.to_str()?.to_string())
}

pub fn available() -> bool {
    cudarc::driver::CudaContext::new(0).is_ok()
}
