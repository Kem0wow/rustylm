use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Auto,
    Cpu,
    Cuda,
}

impl Device {
    pub fn resolve(self) -> Self {
        match self {
            Self::Auto | Self::Cuda if cuda_available() => Self::Cuda,
            _ => Self::Cpu,
        }
    }
    pub fn is_cuda(self) -> bool { self == Self::Cuda }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Cpu => write!(f, "CPU ({})", crate::cpu::name()),
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                if let Ok(name) = crate::cuda::device_name(0) {
                    return write!(f, "CUDA ({name}) + CPU ({})", crate::cpu::name());
                }
                write!(f, "CUDA + CPU ({})", crate::cpu::name())
            }
        }
    }
}

pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")] { crate::cuda::available() }
    #[cfg(not(feature = "cuda"))] { false }
}
