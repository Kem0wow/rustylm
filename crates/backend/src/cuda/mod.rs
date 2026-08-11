use cudarc::driver::{CudaContext, result, sys};

// ====================================================
// GPU/CUDA Model Info
// =====================================================

pub fn device_name(ordinal: usize) -> anyhow::Result<String> {
    let dev = result::device::get(ordinal as i32)?;
    let mut name_buf = [0u8; 256];
    unsafe {
        let res = sys::cuDeviceGetName(name_buf.as_mut_ptr() as *mut std::ffi::c_char, name_buf.len() as i32, dev);
        if res != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuDeviceGetName failed: {:?}", res);
        }
    }
    let name_str = std::ffi::CStr::from_bytes_until_nul(&name_buf)?.to_str()?.to_string();
    Ok(name_str)
}

pub fn init() -> anyhow::Result<()> {
    let _ctx = CudaContext::new(0)?;
    println!("CUDA initialized");
    Ok(())
}