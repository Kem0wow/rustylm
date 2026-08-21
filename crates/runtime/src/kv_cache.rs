/// Flat, zero-allocation KV Cache for LLM inference.
///
/// Memory Layout: Continuous 1D buffer per K/V tensor structured as `[layers][max_seq][dim]`.
/// All memory is pre-allocated upfront to guarantee zero heap allocations during model execution.
pub struct KvCache {
    k_data: Vec<f32>,
    v_data: Vec<f32>,
    layers: usize,
    dim: usize,
    max_seq: usize,
    current_seq: usize,
}

// =============================================================================
// SECTION 1: Initialization & Allocation
// =============================================================================

impl KvCache {
    pub fn new(layers: usize, dim: usize, max_seq: usize) -> Self {
        let capacity = layers * max_seq * dim;
        Self {
            k_data: vec![0.0f32; capacity],
            v_data: vec![0.0f32; capacity],
            layers,
            dim,
            max_seq,
            current_seq: 0,
        }
    }
}

// =============================================================================
// SECTION 2: Cache Mutators (Write & State Updates)
// =============================================================================

impl KvCache {
    /// Appends Key and Value vectors for a layer at the current sequence position.
    pub fn push(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        debug_assert!(layer < self.layers, "Layer index out of bounds");
        debug_assert!(self.current_seq < self.max_seq, "KV Cache sequence overflow");
        debug_assert_eq!(k.len(), self.dim, "Key dimension mismatch");
        debug_assert_eq!(v.len(), self.dim, "Value dimension mismatch");

        let offset = self.calculate_offset(layer, self.current_seq);
        let end = offset + self.dim;

        self.k_data[offset..end].copy_from_slice(k);
        self.v_data[offset..end].copy_from_slice(v);
    }

    /// Advances the global sequence step counter after processing a token across all layers.
    pub fn increment_seq(&mut self) {
        debug_assert!(self.current_seq < self.max_seq, "Sequence index exceeds max bound");
        self.current_seq += 1;
    }

    /// Resets sequence position back to zero without freeing allocated memory.
    pub fn reset(&mut self) {
        self.current_seq = 0;
    }
}

// =============================================================================
// SECTION 3: Cache Readers (Accessors & Slices)
// =============================================================================

impl KvCache {
    /// Active Key slice for a layer including current sequence position.
    pub fn k(&self, layer: usize) -> &[f32] {
        let start = self.calculate_offset(layer, 0);
        let count = (self.current_seq + 1).min(self.max_seq);
        let end = start + (count * self.dim);
        &self.k_data[start..end]
    }

    /// Active Value slice for a layer including current sequence position.
    pub fn v(&self, layer: usize) -> &[f32] {
        let start = self.calculate_offset(layer, 0);
        let count = (self.current_seq + 1).min(self.max_seq);
        let end = start + (count * self.dim);
        &self.v_data[start..end]
    }

    /// Alias for `k(layer)` slice access.
    pub fn k_slice(&self, layer: usize) -> &[f32] {
        self.k(layer)
    }

    /// Alias for `v(layer)` slice access.
    pub fn v_slice(&self, layer: usize) -> &[f32] {
        self.v(layer)
    }

    /// Returns the sequence length processed for a layer (including current token position).
    pub fn seq_len(&self, _layer: usize) -> usize {
        (self.current_seq + 1).min(self.max_seq)
    }

    pub fn total_seq_len(&self) -> usize {
        self.current_seq
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn layers(&self) -> usize {
        self.layers
    }
}

// =============================================================================
// SECTION 4: Layout & Indexing Math
// =============================================================================

impl KvCache {
    #[inline(always)]
    fn calculate_offset(&self, layer: usize, seq: usize) -> usize {
        (layer * self.max_seq + seq) * self.dim
    }
}

// =============================================================================
// SECTION 5: Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_push_and_read() {
        let mut kv = KvCache::new(2, 4, 10);
        let k0 = vec![1.0, 2.0, 3.0, 4.0];
        let v0 = vec![5.0, 6.0, 7.0, 8.0];
        kv.push(0, &k0, &v0);

        assert_eq!(kv.seq_len(0), 1);
        assert_eq!(kv.k(0), &k0[..]);
        assert_eq!(kv.v(0), &v0[..]);

        kv.increment_seq();
        assert_eq!(kv.total_seq_len(), 1);
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut kv = KvCache::new(1, 2, 5);
        kv.push(0, &[1.0, 2.0], &[3.0, 4.0]);
        kv.increment_seq();
        assert_eq!(kv.total_seq_len(), 1);

        kv.reset();
        assert_eq!(kv.total_seq_len(), 0);
    }
}