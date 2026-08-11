use crate::config::ModelConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Architecture {
    Qwen2,
    Llama,
    Gemma,
    Gemma2,
    Gemma3,
    Gemma4,
    DeepSeek,
    Kimi,
    Glm,
    Unknown(String),
}

impl Architecture {
    /// Detects the model architecture from ModelConfig.
    ///
    /// `model_type` is preferred because it is the primary
    /// architecture identifier in Hugging Face configs.
    /// `architectures` is used as a fallback.
    pub fn detect(config: &ModelConfig) -> Self {
        // First try model_type.
        if !config.model_type.is_empty() {
            if let Some(architecture) = Self::from_model_type(&config.model_type) {
                return architecture;
            }
        }

        // Fallback to the architectures array.
        for architecture_name in &config.architectures {
            if let Some(architecture) = Self::from_architecture_name(architecture_name) {
                return architecture;
            }
        }

        // Nothing recognized.
        if !config.model_type.is_empty() {
            Self::Unknown(config.model_type.clone())
        } else if let Some(name) = config.architectures.first() {
            Self::Unknown(name.clone())
        } else {
            Self::Unknown("unknown".to_string())
        }
    }

    /// Maps Hugging Face `model_type` values to Rust architectures.
    fn from_model_type(model_type: &str) -> Option<Self> {
        let model_type = model_type.to_lowercase();

        match model_type.as_str() {
            // Qwen
            "qwen" | "qwen2" | "qwen2_5" | "qwen2.5" => {
                Some(Self::Qwen2)
            }

            // Llama
            "llama" | "llama2" | "llama3" | "llama3_2" | "llama3.2" | "llama4" => {
                Some(Self::Llama)
            }

            // Gemma
            "gemma" => Some(Self::Gemma),
            "gemma2" | "gemma2_2b" | "gemma2_9b" => Some(Self::Gemma2),
            "gemma3" => Some(Self::Gemma3),
            "gemma4" | "gemma4_unified_assistant" | "gemma4_unified_text" => Some(Self::Gemma4),

            // DeepSeek
            "deepseek" | "deepseek_v2" | "deepseek_v3" => {
                Some(Self::DeepSeek)
            }

            // Kimi
            "kimi" | "kimi_k2" => Some(Self::Kimi),

            // GLM / ChatGLM
            "glm" | "chatglm" | "glm4" | "glm4_moe" => {
                Some(Self::Glm)
            }

            _ => None,
        }
    }

    /// Maps Hugging Face `architectures` values to Rust architectures.
    ///
    /// Example:
    /// `Qwen2ForCausalLM` -> `Architecture::Qwen2`
    fn from_architecture_name(name: &str) -> Option<Self> {
        let name = name.to_lowercase();

        if name.starts_with("qwen") {
            return Some(Self::Qwen2);
        }

        if name.starts_with("llama") {
            return Some(Self::Llama);
        }

        if name.starts_with("gemma4") {
            return Some(Self::Gemma4);
        }

        if name.starts_with("gemma3") {
            return Some(Self::Gemma3);
        }

        if name.starts_with("gemma2") {
            return Some(Self::Gemma2);
        }

        if name.starts_with("gemma") {
            return Some(Self::Gemma);
        }

        if name.starts_with("deepseek") {
            return Some(Self::DeepSeek);
        }

        if name.starts_with("kimi") {
            return Some(Self::Kimi);
        }

        if name.starts_with("glm") || name.starts_with("chatglm") {
            return Some(Self::Glm);
        }

        None
    }

    /// Returns a human-readable architecture name.
    pub fn name(&self) -> &str {
        match self {
            Self::Qwen2 => "Qwen2",
            Self::Llama => "Llama",
            Self::Gemma => "Gemma",
            Self::Gemma2 => "Gemma2",
            Self::Gemma3 => "Gemma3",
            Self::Gemma4 => "Gemma4",
            Self::DeepSeek => "DeepSeek",
            Self::Kimi => "Kimi",
            Self::Glm => "GLM",
            Self::Unknown(name) => name,
        }
    }

    /// Returns whether this architecture is currently supported
    /// by the RustyLM model implementations.
    pub fn is_supported(&self) -> bool {
        matches!(
            self,
            Self::Qwen2
                | Self::Llama
                | Self::Gemma
                | Self::Gemma2
                | Self::Gemma3
                | Self::Gemma4
                | Self::DeepSeek
                | Self::Kimi
                | Self::Glm
        )
    }
}


impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    fn test_config(model_type: &str) -> ModelConfig {
        ModelConfig {
            model_type: model_type.to_string(),
            architectures: Vec::new(),

            text_config: None,

            hidden_size: 0,
            intermediate_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            head_dim: None,
            vocab_size: 0,

            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,

            rope_theta: 10000.0,
            max_position_embeddings: 2048,

            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,

            tie_word_embeddings: false,

            torch_dtype: None,
            transformers_version: None,
            rope_scaling: None,
            attention_dropout: 0.0,
            use_cache: None,
            sliding_window: None,
            max_window_layers: None,
            use_sliding_window: None,
            initializer_range: None,
            quantization_config: None,
        }
    }

    #[test]
    fn detects_qwen2() {
        let config = test_config("qwen2");

        assert_eq!(
            Architecture::detect(&config),
            Architecture::Qwen2
        );
    }

    #[test]
    fn detects_llama() {
        let config = test_config("llama");

        assert_eq!(
            Architecture::detect(&config),
            Architecture::Llama
        );
    }

    #[test]
    fn detects_gemma() {
        let config = test_config("gemma3");

        assert_eq!(
            Architecture::detect(&config),
            Architecture::Gemma3
        );
    }

    #[test]
    fn detects_unknown_architecture() {
        let config = test_config("some_future_model");

        assert_eq!(
            Architecture::detect(&config),
            Architecture::Unknown("some_future_model".to_string())
        );
    }

    #[test]
    fn detects_from_architectures_fallback() {
        let mut config = test_config("");

        config.architectures = vec![
            "Qwen2ForCausalLM".to_string()
        ];

        assert_eq!(
            Architecture::detect(&config),
            Architecture::Qwen2
        );
    }
}