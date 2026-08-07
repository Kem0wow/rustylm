use candle_core::{Device, Result};

pub fn select_device() -> Result<Device> {
    if Device::cuda_is_available() {
        new_cuda(0)
        log_info!("Using CUDA")
    } else {
        Ok(Device::Cpu)
        log_info!("Using CPU")
    }
}