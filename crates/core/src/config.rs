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
            TokenId::Single(id) => *id,
            TokenId::Multiple(vec) => vec.first().copied().unwrap_or(0),
        }
    }

    pub fn is_match(&self, token_id: u32) -> bool {
        match self {
            TokenId::Single(id) => *id == token_id,
            TokenId::Multiple(vec) => vec.contains(&token_id),
        }
    }
}

fn deserialize_null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + serde::Deserialize<'de>,
{
    let opt = Option::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // --- Model Identification & Metadata ---
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub architectures: Vec<String>,

    #[serde(default, alias = "model_name")]
    pub model_type: String,

    #[serde(default)]
    pub torch_dtype: Option<String>,

    #[serde(default)]
    pub transformers_version: Option<String>,

    // --- Nested Text Config (for Gemma4 / Multimodal / Unified Assistant Models) ---
    #[serde(default)]
    pub text_config: Option<Box<ModelConfig>>,

    // --- Core Architecture Parameters ---
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

    // --- Activation & Normalization ---
    #[serde(default = "default_hidden_act", alias = "hidden_activation")]
    pub hidden_act: String,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    // --- Positional Embeddings (RoPE) ---
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    // --- Special Token IDs ---
    #[serde(default)]
    pub bos_token_id: Option<TokenId>,

    #[serde(default)]
    pub eos_token_id: Option<TokenId>,

    #[serde(default)]
    pub pad_token_id: Option<TokenId>,

    // --- Weights & Attention Behavior ---
    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub attention_dropout: f32,

    #[serde(default)]
    pub use_cache: Option<bool>,

    // --- Sliding Window / Local Attention ---
    #[serde(default)]
    pub sliding_window: Option<usize>,

    #[serde(default)]
    pub max_window_layers: Option<usize>,

    #[serde(default)]
    pub use_sliding_window: Option<bool>,

    // --- Training / Initialization metadata ---
    #[serde(default)]
    pub initializer_range: Option<f32>,

    // --- Quantization Metadata ---
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
    /// Loads a ModelConfig from a JSON file path.
    pub fn load_config(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut config: ModelConfig = serde_json::from_reader(reader)?;

        // Fallback for multimodal/unified models (e.g. Gemma 4) where architecture is in text_config
        if let Some(text_cfg) = &config.text_config {
            if config.hidden_size == 0 {
                config.hidden_size = text_cfg.hidden_size;
            }
            if config.num_hidden_layers == 0 {
                config.num_hidden_layers = text_cfg.num_hidden_layers;
            }
            if config.num_attention_heads == 0 {
                config.num_attention_heads = text_cfg.num_attention_heads;
            }
            if config.num_key_value_heads == 0 {
                config.num_key_value_heads = text_cfg.num_key_value_heads;
            }
            if config.intermediate_size == 0 {
                config.intermediate_size = text_cfg.intermediate_size;
            }
            if config.vocab_size == 0 {
                config.vocab_size = text_cfg.vocab_size;
            }
            if config.head_dim.is_none() {
                config.head_dim = text_cfg.head_dim;
            }
            if config.bos_token_id.is_none() {
                config.bos_token_id = text_cfg.bos_token_id.clone();
            }
            if config.eos_token_id.is_none() {
                config.eos_token_id = text_cfg.eos_token_id.clone();
            }
            if config.pad_token_id.is_none() {
                config.pad_token_id = text_cfg.pad_token_id.clone();
            }
            if !text_cfg.hidden_act.is_empty() {
                config.hidden_act = text_cfg.hidden_act.clone();
            }
            if config.max_position_embeddings == 2048 && text_cfg.max_position_embeddings != 2048 {
                config.max_position_embeddings = text_cfg.max_position_embeddings;
            }
        }

        // Fallback for standard MHA models where num_key_value_heads isn't explicitly set
        if config.num_key_value_heads == 0 {
            config.num_key_value_heads = config.num_attention_heads;
        }

        Ok(config)
    }

    /// Computes dimension per attention head: explicit head_dim if set, or hidden_size / num_attention_heads.
    pub fn head_dim(&self) -> usize {
        if let Some(dim) = self.head_dim {
            if dim > 0 {
                return dim;
            }
        }
        if self.num_attention_heads == 0 {
            0
        } else {
            self.hidden_size / self.num_attention_heads
        }
    }

    /// Computes GQA ratio (heads per key-value head): num_attention_heads / num_key_value_heads.
    pub fn num_key_value_groups(&self) -> usize {
        if self.num_key_value_heads == 0 {
            1
        } else {
            self.num_attention_heads / self.num_key_value_heads
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_load_qwen2_config() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let config_path = manifest_dir.join("../../models/qwen2.5-0.5b-instruct/config.json");
        if config_path.exists() {
            let config = ModelConfig::load_config(&config_path).expect("Failed to load qwen2 config");
            assert_eq!(config.model_type, "qwen2");
            assert_eq!(config.hidden_size, 896);
            assert_eq!(config.num_hidden_layers, 24);
            assert_eq!(config.num_attention_heads, 14);
            assert_eq!(config.num_key_value_heads, 2);
            assert_eq!(config.vocab_size, 151936);
            assert_eq!(config.head_dim(), 64);
            assert_eq!(config.num_key_value_groups(), 7);
        }
    }

    #[test]
    fn test_load_llama3_config() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let config_path = manifest_dir.join("../../models/llama3.2-1b-instruct/config.json");
        if config_path.exists() {
            let config = ModelConfig::load_config(&config_path).expect("Failed to load llama3.2 config");
            assert_eq!(config.model_type, "llama");
            assert_eq!(config.hidden_size, 2048);
            assert_eq!(config.num_hidden_layers, 16);
            assert_eq!(config.num_attention_heads, 32);
            assert_eq!(config.num_key_value_heads, 8);
            assert_eq!(config.vocab_size, 128256);
            assert_eq!(config.head_dim(), 64);
            assert_eq!(config.num_key_value_groups(), 4);
            assert!(matches!(config.eos_token_id, Some(TokenId::Multiple(_))));
        }
    }

    #[test]
    fn test_load_gemma4_config() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let config_path = manifest_dir.join("../../models/gemma-4-12b-it-assistant/config.json");
        if config_path.exists() {
            let config = ModelConfig::load_config(&config_path).expect("Failed to load gemma4 config");
            assert!(config.model_type.contains("gemma"));
            assert_eq!(config.hidden_size, 1024);
            assert_eq!(config.num_hidden_layers, 4);
            assert_eq!(config.num_attention_heads, 16);
            assert_eq!(config.num_key_value_heads, 8);
            assert_eq!(config.head_dim(), 256);
            assert_eq!(config.vocab_size, 262144);
        }
    }
}