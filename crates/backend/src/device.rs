use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Auto,
    Cpu,
    Cuda,
}

impl Device {
    /// Turn `Auto` into whatever this machine actually has.
    pub fn resolve(self) -> Self {
        match self {
            Self::Auto if cuda_available() => Self::Cuda,
            Self::Auto => Self::Cpu,
            other => other,
        }
    }

    pub fn is_cuda(self) -> bool {
        self == Self::Cuda
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Cpu => write!(f, "{}", crate::cpu::name()),
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                if let Ok(name) = crate::cuda::device_name(0) {
                    return write!(f, "{name} + {}", crate::cpu::name());
                }
                write!(f, "cuda")
            }
        }
    }
}

pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}
