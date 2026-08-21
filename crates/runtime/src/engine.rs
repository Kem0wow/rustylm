use anyhow::{anyhow, Result};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rustylm_backend::cpu::ops as cpu;
use rustylm_backend::Device;
use rustylm_core::{Architecture, ModelConfig, QTensor, Tokenizer, Weights};

use crate::kv_cache::KvCache;
use crate::linear::Linear;

/// VRAM left free for activations and the CUDA context itself.
#[cfg(feature = "cuda")]
const VRAM_RESERVE: usize = 320 << 20;

// ─── Public API ───────────────────────────────────────────────────────────

pub struct Params {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub system: String,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            system: "You are a helpful assistant.".into(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Stats {
    pub prompt_tokens: usize,
    pub generated: usize,
    pub prefill_secs: f32,
    pub decode_secs: f32,
}

impl Stats {
    pub fn tokens_per_sec(&self) -> f32 {
        self.generated as f32 / self.decode_secs.max(1e-6)
    }
}

// ─── Model ────────────────────────────────────────────────────────────────

struct Cfg {
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    rope_theta: f32,
    max_seq: usize,
    scale_embed: bool,
}

impl Cfg {
    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
    fn groups(&self) -> usize {
        self.heads / self.kv_heads
    }
}

struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}

pub struct Engine {
    cfg: Cfg,
    arch: Architecture,
    device: Device,
    embed: Arc<QTensor>,
    layers: Vec<Layer>,
    out_norm: Vec<f32>,
    head: Linear,
    tok: Tokenizer,
    eos: Vec<u32>,
    vram: usize,
}

impl Engine {
    pub fn load(dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let dir = dir.as_ref();
        let device = device.resolve();
        let err = |e: Box<dyn std::error::Error + Send + Sync>| anyhow!("{e}");

        let raw = ModelConfig::load_config(dir.join("config.json")).map_err(err)?;
        let arch = Architecture::detect(&raw);
        let tok = Tokenizer::load(dir.join("tokenizer.json")).map_err(err)?;
        let w = Weights::open(dir).map_err(err)?;

        let gemma = matches!(
            arch,
            Architecture::Gemma | Architecture::Gemma2 | Architecture::Gemma3 | Architecture::Gemma4
        );
        let cfg = Cfg {
            hidden: raw.hidden_size,
            heads: raw.num_attention_heads,
            kv_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim(),
            eps: raw.rms_norm_eps,
            rope_theta: raw.rope_theta,
            max_seq: raw.max_position_embeddings.min(8192),
            scale_embed: gemma,
        };

        // Norm weights stay f32; Gemma stores them as (w - 1).
        let norm = |name: &str| -> Result<Vec<f32>> {
            let mut v = w.f32(name).map_err(err)?;
            if gemma {
                v.iter_mut().for_each(|x| *x += 1.0);
            }
            Ok(v)
        };
        let quant = |name: &str| -> Result<Arc<QTensor>> { Ok(Arc::new(w.quant(name).map_err(err)?)) };

        let embed = quant("model.embed_tokens.weight")?;
        let head_w = if w.has("lm_head.weight") { quant("lm_head.weight")? } else { embed.clone() };

        let mut layers = Vec::with_capacity(raw.num_hidden_layers);
        for i in 0..raw.num_hidden_layers {
            let n = |s: &str| format!("model.layers.{i}.{s}");
            let bias = |s: String| w.f32(&s).ok();
            layers.push(Layer {
                attn_norm: norm(&n("input_layernorm.weight"))?,
                ffn_norm: norm(&n("post_attention_layernorm.weight"))?,
                q: Linear::new(quant(&n("self_attn.q_proj.weight"))?, bias(n("self_attn.q_proj.bias"))),
                k: Linear::new(quant(&n("self_attn.k_proj.weight"))?, bias(n("self_attn.k_proj.bias"))),
                v: Linear::new(quant(&n("self_attn.v_proj.weight"))?, bias(n("self_attn.v_proj.bias"))),
                o: Linear::new(quant(&n("self_attn.o_proj.weight"))?, None),
                gate: Linear::new(quant(&n("mlp.gate_proj.weight"))?, None),
                up: Linear::new(quant(&n("mlp.up_proj.weight"))?, None),
                down: Linear::new(quant(&n("mlp.down_proj.weight"))?, None),
            });
        }

        let mut engine = Self {
            cfg,
            arch: arch.clone(),
            device,
            embed,
            layers,
            out_norm: norm("model.norm.weight")?,
            head: Linear::new(head_w, None),
            eos: eos_ids(&raw, &tok),
            tok,
            vram: 0,
        };

        if device.is_cuda() {
            engine.vram = engine.offload()?;
        }
        Ok(engine)
    }

    /// Fill VRAM with the heaviest matrices first; the rest stays on CPU.
    #[cfg(feature = "cuda")]
    fn offload(&mut self) -> Result<usize> {
        use rustylm_backend::cuda::Gpu;
        let gpu = Gpu::new(0)?;
        let mut budget = std::env::var("RUSTYLM_VRAM_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map_or_else(|| Gpu::free_bytes().saturating_sub(VRAM_RESERVE), |mb| mb << 20);
        let start = budget;

        let take = |l: &mut Linear, budget: &mut usize| {
            *budget = budget.saturating_sub(l.offload(&gpu, *budget));
        };
        take(&mut self.head, &mut budget);
        for layer in &mut self.layers {
            for l in [
                &mut layer.down,
                &mut layer.gate,
                &mut layer.up,
                &mut layer.o,
                &mut layer.q,
                &mut layer.k,
                &mut layer.v,
            ] {
                take(l, &mut budget);
            }
        }
        Ok(start - budget)
    }

    #[cfg(not(feature = "cuda"))]
    fn offload(&mut self) -> Result<usize> {
        Err(anyhow!("built without the `cuda` feature"))
    }

    /// Bytes of model weights currently resident in VRAM.
    pub fn vram_bytes(&self) -> usize {
        self.vram
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn architecture(&self) -> &Architecture {
        &self.arch
    }

    // ─── Forward pass (one token) ─────────────────────────────────────────

    fn forward(&self, token: u32, pos: usize, kv: &mut KvCache) -> Vec<f32> {
        let c = &self.cfg;
        let mut x = vec![0f32; c.hidden];
        self.embed.row(token as usize, &mut x);
        if c.scale_embed {
            let s = (c.hidden as f32).sqrt();
            x.iter_mut().for_each(|v| *v *= s);
        }

        let (cos, sin) = cpu::rope_table(c.head_dim, pos, c.rope_theta);

        for (i, layer) in self.layers.iter().enumerate() {
            let xn = cpu::rms_norm(&x, &layer.attn_norm, c.eps);
            let mut q = layer.q.forward(&xn);
            let mut k = layer.k.forward(&xn);
            let v = layer.v.forward(&xn);

            cpu::rope(&mut q, c.heads, c.head_dim, &cos, &sin);
            cpu::rope(&mut k, c.kv_heads, c.head_dim, &cos, &sin);
            kv.push(i, &k, &v);

            let attn = self.attend(&q, kv, i);
            cpu::add_into(&mut x, &layer.o.forward(&attn));

            let xn = cpu::rms_norm(&x, &layer.ffn_norm, c.eps);
            let mut gate = layer.gate.forward(&xn);
            cpu::swiglu(&mut gate, &layer.up.forward(&xn));
            cpu::add_into(&mut x, &layer.down.forward(&gate));
        }

        kv.increment_seq();

        let xn = cpu::rms_norm(&x, &self.out_norm, c.eps);
        self.head.forward(&xn)
    }

    /// Grouped-query attention over the cache, one rayon task per head.
    fn attend(&self, q: &[f32], kv: &KvCache, layer: usize) -> Vec<f32> {
        let c = &self.cfg;
        let (hd, kv_dim, groups) = (c.head_dim, c.kv_dim(), c.groups());
        let seq = kv.seq_len(layer);
        let scale = 1.0 / (hd as f32).sqrt();
        let (keys, vals) = (kv.k(layer), kv.v(layer));

        let mut out = vec![0f32; c.heads * hd];
        out.par_chunks_mut(hd).enumerate().for_each(|(h, o)| {
            let base = (h / groups) * hd;
            let qh = &q[h * hd..(h + 1) * hd];

            let mut scores: Vec<f32> = (0..seq)
                .map(|t| {
                    let ks = t * kv_dim + base;
                    keys[ks..ks + hd].iter().zip(qh).map(|(&a, &b)| a * b).sum::<f32>() * scale
                })
                .collect();
            cpu::softmax(&mut scores);

            for (t, &s) in scores.iter().enumerate() {
                let vs = t * kv_dim + base;
                for (acc, &val) in o.iter_mut().zip(&vals[vs..vs + hd]) {
                    *acc += s * val;
                }
            }
        });
        out
    }

    // ─── Generation ───────────────────────────────────────────────────────

    pub fn prompt(&self, user: &str, system: &str) -> String {
        template(&self.arch, user, system)
    }

    pub fn ask(&self, question: &str) -> Result<String> {
        let mut text = String::new();
        self.generate(question, &Params::default(), |chunk| text.push_str(chunk))?;
        Ok(text)
    }

    pub fn generate<F: FnMut(&str)>(&self, question: &str, p: &Params, mut on_text: F) -> Result<Stats> {
        let err = |e: Box<dyn std::error::Error + Send + Sync>| anyhow!("{e}");
        let ids = self
            .tok
            .encode(&self.prompt(question, &p.system), false)
            .map_err(err)?;

        let mut kv = KvCache::new(self.layers.len(), self.cfg.kv_dim(), self.cfg.max_seq);
        let mut stats = Stats { prompt_tokens: ids.len(), ..Default::default() };

        let t0 = Instant::now();
        let mut logits = Vec::new();
        for (pos, &token) in ids.iter().enumerate() {
            logits = self.forward(token, pos, &mut kv);
        }
        stats.prefill_secs = t0.elapsed().as_secs_f32();

        let t1 = Instant::now();
        let mut out: Vec<u32> = Vec::new();
        let mut shown = 0usize;
        for step in 0..p.max_tokens {
            let window = out.len().saturating_sub(64);
            cpu::repeat_penalty(&mut logits, &out[window..], p.repeat_penalty);

            let next = cpu::sample(&logits, p.temperature, p.top_p);
            if self.eos.contains(&next) || ids.len() + step + 1 >= self.cfg.max_seq {
                break;
            }
            out.push(next);

            // Decode the whole tail so multi-token characters render correctly.
            let text = self.tok.decode(&out, true).map_err(err)?;
            if let Some(chunk) = text.get(shown..).filter(|c| !c.is_empty()) {
                on_text(chunk);
                shown = text.len();
            }

            logits = self.forward(next, ids.len() + step, &mut kv);
        }
        stats.generated = out.len();
        stats.decode_secs = t1.elapsed().as_secs_f32();
        Ok(stats)
    }
}

// ─── Chat templates ───────────────────────────────────────────────────────

fn template(arch: &Architecture, user: &str, system: &str) -> String {
    match arch {
        Architecture::Gemma | Architecture::Gemma2 | Architecture::Gemma3 | Architecture::Gemma4 => {
            format!("<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n")
        }
        Architecture::Llama => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        _ => format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{user}<|im_end|>\n\
             <|im_start|>assistant\n"
        ),
    }
}

fn eos_ids(cfg: &ModelConfig, tok: &Tokenizer) -> Vec<u32> {
    let mut ids = Vec::new();
    if let Some(eos) = &cfg.eos_token_id {
        match eos {
            rustylm_core::config::TokenId::Single(id) => ids.push(*id),
            rustylm_core::config::TokenId::Multiple(v) => ids.extend(v),
        }
    }
    for name in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>", "<eos>"] {
        if let Some(id) = tok.token_to_id(name) {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}
