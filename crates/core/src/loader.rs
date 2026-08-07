use std::fs::File;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use anyhow::Result;
use std::path::Path;

pub struct ModelLoader;

impl ModelLoader {
    pub fn inspect_safe<P: AsRef<Path>>(path: P) -> Result<()> {
        let file = File::open(path)?;
        
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        let tensors = SafeTensors::deserialize(&mmap)?;
        
        log_info!("Model loaded succesfully. Inspecting first 5 layers...");
        
        for (name, view) in tensors.tensors().take(5) {
            log_info!(
                "Layer: {:<40} | Type: {:?} | Shape: {:?}", 
                name, 
                view.dtype(), 
                view.shape()
            );
        }

        let total_weight_count: usize = tensors.tensors()
            .map(|(_, v)| v.shape().iter().product::<usize>())
            .sum();

        println!("---------------------------------------");
        log_info!("Total parameter count (Approximate): {} M", total_weight_count / 1_000_000);
        
        Ok(())
    }
}