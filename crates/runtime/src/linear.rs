use rustylm_backend::cpu::ops as cpu;
use rustylm_core::QTensor;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use rustylm_backend::cuda::{Gpu, GpuMat};

/// A quantized projection. Processes its workload directly on the GPU if offloaded.
pub struct Linear {
    w: Arc<QTensor>,
    bias: Option<Vec<f32>>,
    #[cfg(feature = "cuda")]
    gpu: Option<GpuMat>,
}

impl Linear {
    pub fn new(w: Arc<QTensor>, bias: Option<Vec<f32>>) -> Self {
        Self {
            w,
            bias,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn rows(&self) -> usize {
        self.w.rows
    }

    /// Move as many rows as `budget` bytes allow onto the GPU; returns bytes taken.
    #[cfg(feature = "cuda")]
    pub fn offload(&mut self, gpu: &Arc<Gpu>, budget: usize) -> usize {
        let rows = (budget / self.w.row_bytes()).min(self.w.rows);
        if rows < 64 {
            return 0;
        }
        match GpuMat::upload(gpu, &self.w, rows) {
            Ok(mat) => {
                self.gpu = Some(mat);
                rows * self.w.row_bytes()
            }
            Err(_) => 0,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; self.w.rows];
        self.matvec(x, &mut out);
        if let Some(b) = &self.bias {
            cpu::add_bias(&mut out, b);
        }
        out
    }

    #[cfg(not(feature = "cuda"))]
    fn matvec(&self, x: &[f32], out: &mut [f32]) {
        cpu::matvec(&self.w, x, 0, out);
    }

    #[cfg(feature = "cuda")]
    fn matvec(&self, x: &[f32], out: &mut [f32]) {
        let Some(gpu) = &self.gpu else {
            return cpu::matvec(&self.w, x, 0, out);
        };

        // Adım 1: GPU'da olan kısmı GPU'ya kitle (split iptal)
        let gpu_rows = gpu.rows();
        let (head, tail) = out.split_at_mut(gpu_rows);

        if gpu_rows > 0 {
            if gpu.matvec(x, head).is_err() {
                cpu::matvec(&self.w, x, 0, head); // Fallback
            }
        }

        // VRAM'e sığmayan kalan satırlar (eğer varsa) CPU'da hesaplanır
        if !tail.is_empty() {
            cpu::matvec(&self.w, x, gpu_rows, tail);
        }
    }
}
