use anyhow::{anyhow, Result};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rustylm_backend::cpu::{ops as cpu, sampling};
use rustylm_backend::Device;
use rustylm_core::{Architecture, ModelConfig, QTensor, Tokenizer, Weights};

use crate::kv_cache::KvCache;
use crate::linear::Linear;
use crate::template::{eos_ids, template};

#[cfg(feature = "cuda")]
const VRAM_RESERVE: usize = 320 << 20;

pub struct Params {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub system: String,
}

impl Default for Params {
    fn default() -> Self {
        Self { max_tokens: 512, temperature: 0.7, top_p: 0.9, repeat_penalty: 1.1, system: "You are a helpful assistant.".into() }
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
    fn kv_dim(&self) -> usize { self.kv_heads * self.head_dim }
    fn groups(&self) -> usize { self.heads / self.kv_heads }
}

struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q: Linear, k: Linear, v: Linear, o: Linear,
    gate: Linear, up: Linear, down: Linear,
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

struct Scratch {
    x: Vec<f32>,
    xn: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    temp: Vec<f32>,
    logits: Vec<f32>,
}

impl Scratch {
    fn new(cfg: &Cfg, head_dim_total: usize, mlp_dim: usize, vocab_size: usize) -> Self {
        Self {
            x: vec![0f32; cfg.hidden],
            xn: vec![0f32; cfg.hidden],
            q: vec![0f32; cfg.heads * cfg.head_dim],
            k: vec![0f32; cfg.kv_heads * cfg.head_dim],
            v: vec![0f32; cfg.kv_heads * cfg.head_dim],
            attn: vec![0f32; head_dim_total],
            gate: vec![0f32; mlp_dim],
            up: vec![0f32; mlp_dim],
            temp: vec![0f32; cfg.hidden],
            logits: vec![0f32; vocab_size],
        }
    }
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

        let gemma = matches!(arch, Architecture::Gemma | Architecture::Gemma2 | Architecture::Gemma3 | Architecture::Gemma4);
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

        let norm = |name: &str| -> Result<Vec<f32>> {
            let mut v = w.f32(name).map_err(err)?;
            if gemma { v.iter_mut().for_each(|x| *x += 1.0); }
            Ok(v)
        };
        let quant = |name: &str| -> Result<Arc<QTensor>> { Ok(Arc::new(w.quant(name).map_err(err)?)) };

        let embed = quant("model.embed_tokens.weight")?;
        let head_w = if w.has("lm_head.weight") { quant("lm_head.weight")? } else { embed.clone() };

        let mut layers = Vec::with_capacity(raw.num_hidden_layers);
        for i in 0..raw.num_hidden_layers {
            let n = |s: &str| format!("model.layers.{i}.{s}");
            let proj = |s: &str, has_bias: bool| -> Result<Linear> {
                let bias = if has_bias { w.f32(&n(&format!("{s}.bias"))).ok() } else { None };
                Ok(Linear::new(quant(&n(&format!("{s}.weight")))?, bias))
            };
            layers.push(Layer {
                attn_norm: norm(&n("input_layernorm.weight"))?,
                ffn_norm: norm(&n("post_attention_layernorm.weight"))?,
                q: proj("self_attn.q_proj", true)?,
                k: proj("self_attn.k_proj", true)?,
                v: proj("self_attn.v_proj", true)?,
                o: proj("self_attn.o_proj", false)?,
                gate: proj("mlp.gate_proj", false)?,
                up: proj("mlp.up_proj", false)?,
                down: proj("mlp.down_proj", false)?,
            });
        }

        let mut engine = Self {
            cfg,
            arch,
            device,
            embed,
            layers,
            out_norm: norm("model.norm.weight")?,
            head: Linear::new(head_w, None),
            eos: eos_ids(&raw, &tok),
            tok,
            vram: 0,
        };

        if device.is_cuda() { engine.vram = engine.offload()?; }
        Ok(engine)
    }

    #[cfg(feature = "cuda")]
    fn offload(&mut self) -> Result<usize> {
        use rustylm_backend::cuda::Gpu;
        let gpu = Gpu::new(0)?;
        let mut budget = std::env::var("RUSTYLM_VRAM_CAP")
            .or_else(|_| std::env::var("RUSTYLM_VRAM_MB"))
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map_or_else(|| Gpu::free_bytes().saturating_sub(VRAM_RESERVE), |mb| mb << 20);
        let start = budget;

        let mut take = |l: &mut Linear| budget = budget.saturating_sub(l.offload(&gpu, budget));
        take(&mut self.head);
        for layer in &mut self.layers {
            for l in [&mut layer.down, &mut layer.gate, &mut layer.up, &mut layer.o, &mut layer.q, &mut layer.k, &mut layer.v] {
                take(l);
            }
        }
        Ok(start - budget)
    }

    #[cfg(not(feature = "cuda"))]
    fn offload(&mut self) -> Result<usize> { Err(anyhow!("built without the `cuda` feature")) }

    pub fn vram_bytes(&self) -> usize { self.vram }
    pub fn device(&self) -> Device { self.device }
    pub fn architecture(&self) -> &Architecture { &self.arch }

    // ─── Forward Pass ─────────────────────────────────────────────────────

    fn forward(&self, token: u32, pos: usize, kv: &mut KvCache, s: &mut Scratch) {
        let c = &self.cfg;
        self.embed.row(token as usize, &mut s.x);
        if c.scale_embed {
            let scale = (c.hidden as f32).sqrt();
            s.x.iter_mut().for_each(|v| *v *= scale);
        }

        let (cos, sin) = cpu::rope_table(c.head_dim, pos, c.rope_theta);

        for (i, layer) in self.layers.iter().enumerate() {
            cpu::rms_norm(&s.x, &layer.attn_norm, c.eps, &mut s.xn);
            layer.q.forward_into(&s.xn, &mut s.q);
            layer.k.forward_into(&s.xn, &mut s.k);
            layer.v.forward_into(&s.xn, &mut s.v);

            cpu::rope(&mut s.q, c.heads, c.head_dim, &cos, &sin);
            cpu::rope(&mut s.k, c.kv_heads, c.head_dim, &cos, &sin);
            kv.push(i, &s.k, &s.v);

            self.attend(&s.q, kv, i, &mut s.attn);
            layer.o.forward_into(&s.attn, &mut s.temp);
            cpu::add_into(&mut s.x, &s.temp);

            cpu::rms_norm(&s.x, &layer.ffn_norm, c.eps, &mut s.xn);
            layer.gate.forward_into(&s.xn, &mut s.gate);
            layer.up.forward_into(&s.xn, &mut s.up);
            cpu::swiglu(&mut s.gate, &s.up);
            layer.down.forward_into(&s.gate, &mut s.temp);
            cpu::add_into(&mut s.x, &s.temp);
        }

        kv.increment_seq();
        cpu::rms_norm(&s.x, &self.out_norm, c.eps, &mut s.xn);
        self.head.forward_into(&s.xn, &mut s.logits);
    }

    fn attend(&self, q: &[f32], kv: &KvCache, layer: usize, out: &mut [f32]) {
        let c = &self.cfg;
        let (hd, kv_dim, groups) = (c.head_dim, c.kv_dim(), c.groups());
        let seq = kv.seq_len(layer);
        let scale = 1.0 / (hd as f32).sqrt();
        let (keys, vals) = (kv.k(layer), kv.v(layer));

        out.par_chunks_mut(hd).enumerate().for_each(|(h, o)| {
            let base = (h / groups) * hd;
            let qh = &q[h * hd..(h + 1) * hd];

            let mut scores: Vec<f32> = (0..seq).map(|t| {
                let ks = t * kv_dim + base;
                keys[ks..ks + hd].iter().zip(qh).map(|(&a, &b)| a * b).sum::<f32>() * scale
            }).collect();
            cpu::softmax(&mut scores);

            o.fill(0.0);
            for (t, &s) in scores.iter().enumerate() {
                let vs = t * kv_dim + base;
                for (acc, &val) in o.iter_mut().zip(&vals[vs..vs + hd]) {
                    *acc += s * val;
                }
            }
        });
    }

    // ─── Generation API ───────────────────────────────────────────────────

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
        let ids = self.tok.encode(&self.prompt(question, &p.system), false).map_err(err)?;

        let mut kv = KvCache::new(self.layers.len(), self.cfg.kv_dim(), self.cfg.max_seq);
        let mut stats = Stats { prompt_tokens: ids.len(), ..Default::default() };

        let mlp_dim = self.layers.first().map(|l| l.gate.rows()).unwrap_or(self.cfg.hidden * 4);
        let mut scratch = Scratch::new(&self.cfg, self.cfg.heads * self.cfg.head_dim, mlp_dim, self.head.rows());

        let t0 = Instant::now();
        for (pos, &token) in ids.iter().enumerate() {
            self.forward(token, pos, &mut kv, &mut scratch);
        }
        stats.prefill_secs = t0.elapsed().as_secs_f32();

        let t1 = Instant::now();
        let mut out: Vec<u32> = Vec::new();
        let mut shown = 0usize;
        for step in 0..p.max_tokens {
            let window = out.len().saturating_sub(64);
            sampling::repeat_penalty(&mut scratch.logits, &out[window..], p.repeat_penalty);

            let next = sampling::sample(&scratch.logits, p.temperature, p.top_p);
            if self.eos.contains(&next) || ids.len() + step + 1 >= self.cfg.max_seq {
                break;
            }
            out.push(next);

            let text = self.tok.decode(&out, true).map_err(err)?;
            if let Some(chunk) = text.get(shown..).filter(|c| !c.is_empty()) {
                on_text(chunk);
                shown = text.len();
            }

            self.forward(next, ids.len() + step, &mut kv, &mut scratch);
        }
        stats.generated = out.len();
        stats.decode_secs = t1.elapsed().as_secs_f32();
        Ok(stats)
    }
}
