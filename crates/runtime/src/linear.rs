use rustylm_backend::cpu::ops as cpu;
use rustylm_core::QTensor;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use rustylm_backend::cuda::{Gpu, GpuMat};
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU32, Ordering};
#[cfg(feature = "cuda")]
use std::time::Instant;

/// A quantized projection. When part of it lives in VRAM, every call splits
/// the output rows between GPU and CPU and re-tunes the split from the
/// throughput each side actually delivered.
pub struct Linear {
    w: Arc<QTensor>,
    bias: Option<Vec<f32>>,
    #[cfg(feature = "cuda")]
    frac: AtomicU32,
    #[cfg(feature = "cuda")]
    cap: f32,
    #[cfg(feature = "cuda")]
    gpu: Option<GpuMat>,
}

impl Linear {
    pub fn new(w: Arc<QTensor>, bias: Option<Vec<f32>>) -> Self {
        Self {
            w,
            bias,
            #[cfg(feature = "cuda")]
            frac: AtomicU32::new(0f32.to_bits()),
            #[cfg(feature = "cuda")]
            cap: 0.0,
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
                self.cap = (rows as f32 / self.w.rows as f32).min(0.98);
                self.frac.store((self.cap * 0.9).to_bits(), Ordering::Relaxed);
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

        let rows = out.len();
        let split = ((rows as f32 * self.load()) as usize).clamp(1, rows - 1);
        let (head, tail) = out.split_at_mut(split);

        let (gt, ct) = rayon::join(
            || {
                let t = Instant::now();
                if gpu.matvec(x, head).is_err() {
                    cpu::matvec(&self.w, x, 0, head);
                }
                t.elapsed().as_secs_f32()
            },
            || {
                let t = Instant::now();
                cpu::matvec(&self.w, x, split, tail);
                t.elapsed().as_secs_f32()
            },
        );

        self.tune(split as f32 / gt.max(1e-9), (rows - split) as f32 / ct.max(1e-9));
    }

    #[cfg(feature = "cuda")]
    fn load(&self) -> f32 {
        f32::from_bits(self.frac.load(Ordering::Relaxed))
    }

    /// Nudge the split toward the ratio of measured rows-per-second.
    #[cfg(feature = "cuda")]
    fn tune(&self, gpu_rate: f32, cpu_rate: f32) {
        let target = gpu_rate / (gpu_rate + cpu_rate);
        let next = (0.9 * self.load() + 0.1 * target).clamp(0.05, self.cap);
        self.frac.store(next.to_bits(), Ordering::Relaxed);
    }
}
