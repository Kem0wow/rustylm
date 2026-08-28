use crate::quant::{f16_to_f32, QTensor};
use memmap2::MmapOptions;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

pub struct Shard { mmap: memmap2::Mmap }

impl Shard {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, BoxError> {
        let mmap = unsafe { MmapOptions::new().map(&File::open(path)?)? };
        SafeTensors::deserialize(&mmap)?;
        Ok(Self { mmap })
    }
    pub fn tensors(&self) -> SafeTensors<'_> { SafeTensors::deserialize(&self.mmap).expect("invalid header") }
}

pub struct Weights {
    shards: Vec<Shard>,
    index: HashMap<String, usize>,
}

impl Weights {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, BoxError> {
        let mut paths: Vec<_> = std::fs::read_dir(dir.as_ref())?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() { return Err(format!("no .safetensors in {}", dir.as_ref().display()).into()); }

        let (mut shards, mut index) = (Vec::new(), HashMap::new());
        for path in paths {
            let shard = Shard::open(&path)?;
            let id = shards.len();
            for name in shard.tensors().names() { index.insert(name.to_string(), id); }
            shards.push(shard);
        }
        Ok(Self { shards, index })
    }

    pub fn has(&self, name: &str) -> bool { self.index.contains_key(name) }
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, BoxError> { self.with(name, |v| to_f32(&v)) }
    pub fn quant(&self, name: &str) -> Result<QTensor, BoxError> { self.with(name, |v| QTensor::from_view(&v)) }

    fn with<T>(&self, name: &str, f: impl FnOnce(TensorView<'_>) -> T) -> Result<T, BoxError> {
        let id = *self.index.get(name).ok_or_else(|| format!("tensor not found: {name}"))?;
        Ok(f(self.shards[id].tensors().tensor(name)?))
    }
}

pub fn to_f32(view: &TensorView<'_>) -> Vec<f32> {
    let d = view.data();
    match view.dtype() {
        Dtype::F32 => d.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        Dtype::BF16 => d.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16)).collect(),
        Dtype::F16 => d.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))).collect(),
        other => panic!("unsupported dtype: {other:?}"),
    }
}
