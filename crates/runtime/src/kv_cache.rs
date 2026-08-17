/// Flat per-layer K/V store: `[pos * dim + channel]`.
pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    dim: usize,
    len: usize,
}

impl KvCache {
    pub fn new(layers: usize, dim: usize, max_seq: usize) -> Self {
        let make = || (0..layers).map(|_| Vec::with_capacity(dim * max_seq)).collect();
        Self { k: make(), v: make(), dim, len: 0 }
    }

    pub fn push(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        self.k[layer].extend_from_slice(k);
        self.v[layer].extend_from_slice(v);
        if layer + 1 == self.k.len() {
            self.len += 1;
        }
    }

    pub fn k(&self, layer: usize) -> &[f32] {
        &self.k[layer]
    }

    pub fn v(&self, layer: usize) -> &[f32] {
        &self.v[layer]
    }

    pub fn seq_len(&self, layer: usize) -> usize {
        self.k[layer].len() / self.dim
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn clear(&mut self) {
        for (k, v) in self.k.iter_mut().zip(&mut self.v) {
            k.clear();
            v.clear();
        }
        self.len = 0;
    }
}
