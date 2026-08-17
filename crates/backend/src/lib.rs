pub mod cpu;
pub mod device;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use device::Device;
