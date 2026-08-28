use rayon::prelude::*;
use rustylm_core::quant::{QTensor, BLOCK};

pub fn matvec(w: &QTensor, x: &[f32], start: usize, out: &mut [f32]) {
    let (cols, blocks) = (w.cols, w.blocks);
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let r = start + i;
        let qr = &w.q[r * cols..(r + 1) * cols];
        let sr = &w.s[r * blocks..(r + 1) * blocks];
        let mut row_sum = 0.0f32;
        for (b, &scale) in sr.iter().enumerate() {
            let lo = b * BLOCK;
            let hi = (lo + BLOCK).min(cols);
            let mut blk_sum = 0.0f32;
            for j in lo..hi { blk_sum += (qr[j] as f32) * x[j]; }
            row_sum += blk_sum * scale;
        }
        *o = row_sum;
    });
}

pub fn add_bias(out: &mut [f32], bias: &[f32]) {
    for (o, b) in out.iter_mut().zip(bias) { *o += b; }
}

pub fn add_into(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) { *a += b; }
}

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    for (i, (&v, &w)) in x.iter().zip(weight).enumerate() { out[i] = w * v * inv; }
}

pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up) { *g = u * (*g / (1.0 + (-*g).exp())); }
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter_mut().map(|v| { *v = (*v - max).exp(); *v }).sum();
    let inv = 1.0 / sum;
    x.iter_mut().for_each(|v| *v *= inv);
}

pub fn rope_table(head_dim: usize, pos: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    (0..head_dim / 2).map(|i| {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        let (s, c) = (pos as f32 * freq).sin_cos();
        (c, s)
    }).unzip()
}

pub fn rope(x: &mut [f32], heads: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    for h in 0..heads {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        for i in 0..half {
            let (a, b) = (head[i], head[i + half]);
            head[i] = a * cos[i] - b * sin[i];
            head[i + half] = a * sin[i] + b * cos[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_matches_dense_reference() {
        let (rows, cols) = (5, 96);
        let w: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.11).cos()).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.23).sin()).collect();
        let q = QTensor::from_f32(&w, rows, cols);
        let mut got = vec![0f32; rows - 1];
        matvec(&q, &x, 1, &mut got);
        for (r, &value) in got.iter().enumerate() {
            let want: f32 = w[(r + 1) * cols..(r + 2) * cols].iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!((value - want).abs() < 0.05);
        }
    }
}
