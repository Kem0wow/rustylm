use rayon::prelude::*;
use safetensors::tensor::TensorView;
use safetensors::Dtype;

pub const BLOCK: usize = 32;

pub struct QTensor {
    pub q: Vec<i8>,
    pub s: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub blocks: usize,
}

impl QTensor {
    pub fn from_view(view: &TensorView<'_>) -> Self {
        let (rows, cols) = shape_2d(view.shape());
        let (data, dtype) = (view.data(), view.dtype());
        let blocks = cols.div_ceil(BLOCK);
        let mut q = vec![0i8; rows * cols];
        let mut s = vec![0f32; rows * blocks];

        let elem_size = match dtype {
            Dtype::F32 => 4,
            Dtype::BF16 | Dtype::F16 => 2,
            other => panic!("unsupported dtype: {other:?}"),
        };

        q.par_chunks_mut(cols).zip(s.par_chunks_mut(blocks)).enumerate().for_each(|(r, (qr, sr))| {
            let row_bytes = &data[r * cols * elem_size..(r + 1) * cols * elem_size];
            let mut buf = [0f32; BLOCK];
            for b in 0..blocks {
                let (lo, hi) = (b * BLOCK, (b * BLOCK + BLOCK).min(cols));
                let chunk = &row_bytes[lo * elem_size..hi * elem_size];
                let mut amax = 0f32;
                match dtype {
                    Dtype::F32 => for (i, c) in chunk.chunks_exact(4).enumerate() {
                        let v = f32::from_le_bytes(c.try_into().unwrap());
                        buf[i] = v; amax = amax.max(v.abs());
                    },
                    Dtype::BF16 => for (i, c) in chunk.chunks_exact(2).enumerate() {
                        let v = f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16);
                        buf[i] = v; amax = amax.max(v.abs());
                    },
                    Dtype::F16 => for (i, c) in chunk.chunks_exact(2).enumerate() {
                        let v = f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()));
                        buf[i] = v; amax = amax.max(v.abs());
                    },
                    _ => unreachable!(),
                }
                let scale = amax / 127.0;
                sr[b] = scale;
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                for i in 0..hi - lo {
                    qr[lo + i] = (buf[i] * inv).round().clamp(-127.0, 127.0) as i8;
                }
            }
        });

        Self { q, s, rows, cols, blocks }
    }

    pub fn from_f32(data: &[f32], rows: usize, cols: usize) -> Self {
        let blocks = cols.div_ceil(BLOCK);
        let mut q = vec![0i8; rows * cols];
        let mut s = vec![0f32; rows * blocks];

        q.par_chunks_mut(cols).zip(s.par_chunks_mut(blocks)).enumerate().for_each(|(r, (qr, sr))| {
            let row = &data[r * cols..(r + 1) * cols];
            for b in 0..blocks {
                let (lo, hi) = (b * BLOCK, (b * BLOCK + BLOCK).min(cols));
                let amax = row[lo..hi].iter().fold(0f32, |m, &x| m.max(x.abs()));
                let scale = amax / 127.0;
                sr[b] = scale;
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                for i in lo..hi {
                    qr[i] = (row[i] * inv).round().clamp(-127.0, 127.0) as i8;
                }
            }
        });

        Self { q, s, rows, cols, blocks }
    }

    pub fn row(&self, r: usize, out: &mut [f32]) {
        let (qr, sr) = (&self.q[r * self.cols..(r + 1) * self.cols], &self.s[r * self.blocks..(r + 1) * self.blocks]);
        for (b, &scale) in sr.iter().enumerate() {
            let (lo, hi) = (b * BLOCK, (b * BLOCK + BLOCK).min(self.cols));
            for i in lo..hi { out[i] = qr[i] as f32 * scale; }
        }
    }

    pub fn row_bytes(&self) -> usize { self.cols + self.blocks * 4 }
}

fn shape_2d(shape: &[usize]) -> (usize, usize) {
    let cols = *shape.last().unwrap_or(&1);
    let rows = shape.iter().take(shape.len().saturating_sub(1)).product::<usize>().max(1);
    (rows, cols)
}

#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let (sign, exp, mant) = (((bits >> 15) & 1) as u32, ((bits >> 10) & 0x1f) as u32, (bits & 0x3ff) as u32);
    match exp {
        0 if mant == 0 => f32::from_bits(sign << 31),
        0 => {
            let shift = mant.leading_zeros() - 21;
            f32::from_bits((sign << 31) | ((127 - 15 - shift) << 23) | ((mant << (shift + 13)) & 0x7f_ffff))
        }
        0x1f => f32::from_bits((sign << 31) | (0xff << 23) | (mant << 13)),
        _ => f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_round_trips_within_one_step() {
        let src: Vec<f32> = (0..96).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let t = QTensor::from_f32(&src, 3, 32);
        let mut out = vec![0f32; 32];
        for r in 0..3 {
            t.row(r, &mut out);
            let row = &src[r * 32..(r + 1) * 32];
            let step = row.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
            assert!(out.iter().zip(row).all(|(a, b)| (a - b).abs() <= step));
        }
    }
}
