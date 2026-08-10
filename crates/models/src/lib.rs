use anyhow::Result;
use safetensors::SafeTensors;
use std::fs;

pub fn inspect_model(path: &str) -> Result<()> {
    let data = fs::read(path)?;

    let tensors = SafeTensors::deserialize(&data)?;

    println!("Model: {}", path);

    for (name, tensor) in tensors.tensors() {
        println!(
            "{} | shape={:?} | dtype={:?}",
            name,
            tensor.shape(),
            tensor.dtype()
        );
    }

    Ok(())
}