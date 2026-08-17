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

impl TokenId {
    pub fn first(&self) -> u32 {
        match self {
            Self::Single(id) => *id,
            Self::Multiple(vec) => vec.first().copied().unwrap_or(0),
        }
    }

    pub fn is_match(&self, token_id: u32) -> bool {
        match self {
            Self::Single(id) => *id == token_id,
            Self::Multiple(vec) => vec.contains(&token_id),
        }
    }
}

fn deserialize_null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + serde::Deserialize<'de>,
{
    Ok(Option::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub architectures: Vec<String>,
    #[serde(default, alias = "model_name")]
    pub model_type: String,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub transformers_version: Option<String>,
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
    pub intermediate_size: usize,
    #[serde(default)]
    pub vocab_size: usize,

    #[serde(default = "default_hidden_act", alias = "hidden_activation")]
    pub hidden_act: String,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    #[serde(default)]
    pub bos_token_id: Option<TokenId>,
    #[serde(default)]
    pub eos_token_id: Option<TokenId>,
    #[serde(default)]
    pub pad_token_id: Option<TokenId>,

    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default)]
    pub use_cache: Option<bool>,

    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub max_window_layers: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: Option<bool>,

    #[serde(default)]
    pub initializer_range: Option<f32>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

fn default_hidden_act() -> String {
    "silu".to_string()
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_max_position_embeddings() -> usize {
    2048
}

impl ModelConfig {
    pub fn load_config(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = File::open(path)?;
        let mut config: ModelConfig = serde_json::from_reader(BufReader::new(file))?;

        if let Some(text_cfg) = &config.text_config {
            if config.hidden_size == 0 { config.hidden_size = text_cfg.hidden_size; }
            if config.num_hidden_layers == 0 { config.num_hidden_layers = text_cfg.num_hidden_layers; }
            if config.num_attention_heads == 0 { config.num_attention_heads = text_cfg.num_attention_heads; }
            if config.num_key_value_heads == 0 { config.num_key_value_heads = text_cfg.num_key_value_heads; }
            if config.intermediate_size == 0 { config.intermediate_size = text_cfg.intermediate_size; }
            if config.vocab_size == 0 { config.vocab_size = text_cfg.vocab_size; }
            if config.head_dim.is_none() { config.head_dim = text_cfg.head_dim; }
            if config.bos_token_id.is_none() { config.bos_token_id = text_cfg.bos_token_id.clone(); }
            if config.eos_token_id.is_none() { config.eos_token_id = text_cfg.eos_token_id.clone(); }
            if config.pad_token_id.is_none() { config.pad_token_id = text_cfg.pad_token_id.clone(); }
            if !text_cfg.hidden_act.is_empty() { config.hidden_act = text_cfg.hidden_act.clone(); }
            if config.max_position_embeddings == 2048 && text_cfg.max_position_embeddings != 2048 {
                config.max_position_embeddings = text_cfg.max_position_embeddings;
            }
        }

        if config.num_key_value_heads == 0 {
            config.num_key_value_heads = config.num_attention_heads;
        }

        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        match self.head_dim {
            Some(dim) if dim > 0 => dim,
            _ if self.num_attention_heads > 0 => self.hidden_size / self.num_attention_heads,
            _ => 0,
        }
    }

    pub fn num_key_value_groups(&self) -> usize {
        if self.num_key_value_heads == 0 { 1 } else { self.num_attention_heads / self.num_key_value_heads }
    }
}