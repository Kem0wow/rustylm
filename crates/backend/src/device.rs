use std::fmt;
use crate::cpu;

#[derive(Debug, Clone)]
pub enum Device {
    Cpu {
        name: String,
    },

    #[cfg(feature = "cuda")]
    Cuda {
        index: usize,
        name: String,
    },
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu { name } => write!(f, "Using CPU - {}", name),
            #[cfg(feature = "cuda")]
            Device::Cuda { name, .. } => write!(f, "Using CUDA - {}", name),
        }
    }
}

pub fn auto() -> Device {
    #[cfg(feature = "cuda")]
    {
        if cudarc::driver::CudaContext::new(0).is_ok() {
            if let Ok(name) = crate::cuda::device_name(0) {
                return Device::Cuda { index: 0, name };
            }
        }
    }

    Device::Cpu {
        name: cpu::name(),
    }
}