pub struct KvCache {
    k_data: Vec<f32>,
    v_data: Vec<f32>,
    dim: usize,
    max_seq: usize,
    current_seq: usize,
}

impl KvCache {
    pub fn new(layers: usize, dim: usize, max_seq: usize) -> Self {
        let capacity = layers * max_seq * dim;
        Self { k_data: vec![0f32; capacity], v_data: vec![0f32; capacity], dim, max_seq, current_seq: 0 }
    }

    pub fn push(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        let offset = (layer * self.max_seq + self.current_seq) * self.dim;
        self.k_data[offset..offset + self.dim].copy_from_slice(k);
        self.v_data[offset..offset + self.dim].copy_from_slice(v);
    }

    pub fn increment_seq(&mut self) { self.current_seq += 1; }
    pub fn reset(&mut self) { self.current_seq = 0; }

    pub fn k(&self, layer: usize) -> &[f32] {
        let start = layer * self.max_seq * self.dim;
        let count = (self.current_seq + 1).min(self.max_seq);
        &self.k_data[start..start + count * self.dim]
    }

    pub fn v(&self, layer: usize) -> &[f32] {
        let start = layer * self.max_seq * self.dim;
        let count = (self.current_seq + 1).min(self.max_seq);
        &self.v_data[start..start + count * self.dim]
    }

    pub fn seq_len(&self, _layer: usize) -> usize { (self.current_seq + 1).min(self.max_seq) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache() {
        let mut kv = KvCache::new(2, 4, 10);
        kv.push(0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(kv.seq_len(0), 1);
        assert_eq!(kv.k(0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(kv.v(0), &[5.0, 6.0, 7.0, 8.0]);
        kv.increment_seq();
        kv.reset();
    }
}