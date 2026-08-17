use anyhow::Result;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use rustylm_core::quant::QTensor;
use std::sync::{Arc, Mutex};

// ─── Kernel: one warp per output row of a Q8 matrix ───────────────────────

const KERNEL: &str = r#"
extern "C" __global__ void q8_matvec(
    const signed char* __restrict__ q, const float* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ y,
    const int rows, const int cols, const int blocks)
{
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    const int lane = threadIdx.x;
    const signed char* qr = q + (long long)row * cols;
    const float* sr = s + (long long)row * blocks;

    float acc = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const int i = b * 32 + lane;
        if (i < cols) acc += (float)qr[i] * x[i] * sr[b];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);

    if (lane == 0) y[row] = acc;
}
"#;

const ROWS_PER_BLOCK: u32 = 8;

// ─── Device handle ────────────────────────────────────────────────────────

pub struct Gpu {
    stream: Arc<CudaStream>,
    func: CudaFunction,
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Arc<Self>> {
        let ctx = CudaContext::new(ordinal)?;
        let module = ctx.load_module(cudarc::nvrtc::compile_ptx(KERNEL)?)?;
        Ok(Arc::new(Self {
            func: module.load_function("q8_matvec")?,
            stream: ctx.default_stream(),
        }))
    }

    pub fn free_bytes() -> usize {
        cudarc::driver::result::mem_get_info().map(|(free, _)| free).unwrap_or(0)
    }
}

// ─── GPU-resident slice of a Q8 matrix ────────────────────────────────────

struct Io {
    x: CudaSlice<f32>,
    y: CudaSlice<f32>,
}

pub struct GpuMat {
    gpu: Arc<Gpu>,
    q: CudaSlice<i8>,
    s: CudaSlice<f32>,
    io: Mutex<Io>,
    rows: usize,
    cols: usize,
    blocks: usize,
}

impl GpuMat {
    /// Upload rows `[0, rows)` of `w` to VRAM.
    pub fn upload(gpu: &Arc<Gpu>, w: &QTensor, rows: usize) -> Result<Self> {
        let stream = &gpu.stream;
        let q = stream.memcpy_stod(&w.q[..rows * w.cols])?;
        let s = stream.memcpy_stod(&w.s[..rows * w.blocks])?;
        let io = Io {
            x: stream.alloc_zeros::<f32>(w.cols)?,
            y: stream.alloc_zeros::<f32>(rows)?,
        };
        Ok(Self {
            gpu: gpu.clone(),
            q,
            s,
            io: Mutex::new(io),
            rows,
            cols: w.cols,
            blocks: w.blocks,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn matvec(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let mut io = self.io.lock().unwrap();
        let Io { x: dx, y: dy } = &mut *io;
        let stream = &self.gpu.stream;
        stream.memcpy_htod(x, dx)?;

        let cfg = LaunchConfig {
            grid_dim: ((out.len() as u32).div_ceil(ROWS_PER_BLOCK), 1, 1),
            block_dim: (32, ROWS_PER_BLOCK, 1),
            shared_mem_bytes: 0,
        };
        let (rows, cols, blocks) = (out.len() as i32, self.cols as i32, self.blocks as i32);
        unsafe {
            stream
                .launch_builder(&self.gpu.func)
                .arg(&self.q)
                .arg(&self.s)
                .arg(&*dx)
                .arg(&mut *dy)
                .arg(&rows)
                .arg(&cols)
                .arg(&blocks)
                .launch(cfg)?;
        }

        stream.memcpy_dtoh(&dy.slice(0..out.len()), out)?;
        stream.synchronize()?;
        Ok(())
    }
}
