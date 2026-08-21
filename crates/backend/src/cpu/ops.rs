use rayon::prelude::*;
use rustylm_core::quant::{QTensor, BLOCK};
use std::cell::Cell;
use std::time::{SystemTime, UNIX_EPOCH};

// ─── Quantized matrix-vector: out[i] = W[start + i] · x ───────────────────

pub fn matvec(w: &QTensor, x: &[f32], start: usize, out: &mut [f32]) {
    let (cols, blocks) = (w.cols, w.blocks);
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let r = start + i;
        let q = &w.q[r * cols..(r + 1) * cols];
        let s = &w.s[r * blocks..(r + 1) * blocks];
        let mut lanes = [0f32; LANES];
        for (b, &scale) in s.iter().enumerate() {
            let lo = b * BLOCK;
            let hi = (lo + BLOCK).min(cols);
            dot(&q[lo..hi], &x[lo..hi], scale, &mut lanes);
        }
        *o = lanes.iter().sum();
    });
}

const LANES: usize = 8;

/// Independent accumulators keep the float adds reorderable, so this
/// compiles to a straight SIMD loop.
#[inline(always)]
fn dot(q: &[i8], x: &[f32], scale: f32, lanes: &mut [f32; LANES]) {
    let mut acc = [0f32; LANES];
    let chunks = q.chunks_exact(LANES);
    let rest = chunks.remainder();
    for (qc, xc) in chunks.zip(x.chunks_exact(LANES)) {
        for j in 0..LANES {
            acc[j] += qc[j] as f32 * xc[j];
        }
    }
    for (k, &qv) in rest.iter().enumerate() {
        acc[0] += qv as f32 * x[q.len() - rest.len() + k];
    }
    for j in 0..LANES {
        lanes[j] += acc[j] * scale;
    }
}

pub fn add_bias(out: &mut [f32], bias: &[f32]) {
    for (o, b) in out.iter_mut().zip(bias) {
        *o += b;
    }
}

pub fn add_into(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

// ─── Normalization / activation ───────────────────────────────────────────

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().zip(weight).map(|(&v, &w)| w * v * inv).collect()
}

pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = u * (*g / (1.0 + (-*g).exp()));
    }
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter_mut().map(|v| { *v = (*v - max).exp(); *v }).sum();
    let inv = 1.0 / sum;
    x.iter_mut().for_each(|v| *v *= inv);
}

// ─── Rotary embeddings (NeoX half-split, as used by Qwen/Llama/Gemma) ─────

pub fn rope_table(head_dim: usize, pos: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    (0..head_dim / 2)
        .map(|i| {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let (s, c) = (pos as f32 * freq).sin_cos();
            (c, s)
        })
        .unzip()
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

// ─── Sampling ─────────────────────────────────────────────────────────────

const TOP_K: usize = 64;

pub fn repeat_penalty(logits: &mut [f32], recent: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &t in recent {
        if let Some(l) = logits.get_mut(t as usize) {
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
    }
}

pub fn sample(logits: &[f32], temperature: f32, top_p: f32) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }

    // Keep only the TOP_K largest logits; most values fail the first compare.
    let mut top: Vec<(f32, u32)> = Vec::with_capacity(TOP_K + 1);
    let mut floor = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if top.len() == TOP_K && l <= floor {
            continue;
        }
        let at = top.partition_point(|&(p, _)| p > l);
        top.insert(at, (l, i as u32));
        top.truncate(TOP_K);
        floor = top.last().unwrap().0;
    }

    let max = top[0].0;
    let mut sum = 0f32;
    for (p, _) in top.iter_mut() {
        *p = ((*p - max) / temperature).exp();
        sum += *p;
    }

    // Nucleus: smallest prefix whose mass reaches top_p.
    let mut mass = 0f32;
    let mut keep = 0;
    for (p, _) in top.iter() {
        mass += *p / sum;
        keep += 1;
        if mass >= top_p {
            break;
        }
    }

    let target = rng() * top[..keep].iter().map(|(p, _)| *p).sum::<f32>();
    let mut acc = 0f32;
    for &(p, id) in &top[..keep] {
        acc += p;
        if acc >= target {
            return id;
        }
    }
    top[0].1
}

pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn rng() -> f32 {
    thread_local! {
        static STATE: Cell<u64> = Cell::new({
            let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(42);
            (t as u64).wrapping_add(1).wrapping_mul(6364136223846793005)
        });
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    })
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
            assert!((value - want).abs() < 0.05, "row {r}: {value} vs {want}");
        }
    }

    #[test]
    fn greedy_sampling_picks_the_largest_logit() {
        let logits = [0.1, 9.0, -3.0, 2.0];
        assert_eq!(sample(&logits, 0.0, 1.0), 1);
        assert_eq!(sample(&logits, 0.01, 1.0), 1);
    }
}
