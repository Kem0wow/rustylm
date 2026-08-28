use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum TokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default, alias = "model_name")]
    pub model_type: String,
    #[serde(default)]
    pub text_config: Option<Box<ModelConfig>>,

    #[serde(default)]
    pub hidden_size: usize,
    #[serde(default, alias = "num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default, alias = "attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub vocab_size: usize,

    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,

    #[serde(default)]
    pub bos_token_id: Option<TokenId>,
    #[serde(default)]
    pub eos_token_id: Option<TokenId>,
}

fn default_eps() -> f32 { 1e-6 }
fn default_theta() -> f32 { 10000.0 }
fn default_max_pos() -> usize { 2048 }

impl ModelConfig {
    pub fn load_config(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut cfg: ModelConfig = serde_json::from_reader(BufReader::new(File::open(path)?))?;
        if let Some(t) = &cfg.text_config {
            if cfg.hidden_size == 0 { cfg.hidden_size = t.hidden_size; }
            if cfg.num_hidden_layers == 0 { cfg.num_hidden_layers = t.num_hidden_layers; }
            if cfg.num_attention_heads == 0 { cfg.num_attention_heads = t.num_attention_heads; }
            if cfg.num_key_value_heads == 0 { cfg.num_key_value_heads = t.num_key_value_heads; }
            if cfg.vocab_size == 0 { cfg.vocab_size = t.vocab_size; }
            if cfg.head_dim.is_none() { cfg.head_dim = t.head_dim; }
            if cfg.eos_token_id.is_none() { cfg.eos_token_id = t.eos_token_id.clone(); }
            if cfg.max_position_embeddings == 2048 && t.max_position_embeddings != 2048 {
                cfg.max_position_embeddings = t.max_position_embeddings;
            }
        }
        if cfg.num_key_value_heads == 0 {
            cfg.num_key_value_heads = cfg.num_attention_heads;
        }
        Ok(cfg)
    }

    pub fn head_dim(&self) -> usize {
        match self.head_dim {
            Some(dim) if dim > 0 => dim,
            _ if self.num_attention_heads > 0 => self.hidden_size / self.num_attention_heads,
            _ => 0,
        }
    }
}