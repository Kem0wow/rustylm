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
    pub fn detect(config: &ModelConfig) -> Self {
        let tag = config.model_type.to_ascii_lowercase();
        let arch = config.architectures.first().map(|s| s.to_ascii_lowercase()).unwrap_or_default();
        let s = if tag.is_empty() { &arch } else { &tag };

        if s.contains("qwen") { Self::Qwen2 }
        else if s.contains("llama") { Self::Llama }
        else if s.contains("gemma4") { Self::Gemma4 }
        else if s.contains("gemma3") { Self::Gemma3 }
        else if s.contains("gemma2") { Self::Gemma2 }
        else if s.contains("gemma") { Self::Gemma }
        else if s.contains("deepseek") { Self::DeepSeek }
        else if s.contains("kimi") { Self::Kimi }
        else if s.contains("glm") { Self::Glm }
        else if !tag.is_empty() { Self::Unknown(config.model_type.clone()) }
        else if !arch.is_empty() { Self::Unknown(config.architectures[0].clone()) }
        else { Self::Unknown("unknown".into()) }
    }

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
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(model_type: &str) -> ModelConfig {
        ModelConfig { model_type: model_type.to_string(), ..Default::default() }
    }

    #[test]
    fn detects_architectures() {
        assert_eq!(Architecture::detect(&cfg("qwen2")), Architecture::Qwen2);
        assert_eq!(Architecture::detect(&cfg("llama")), Architecture::Llama);
        assert_eq!(Architecture::detect(&cfg("gemma3")), Architecture::Gemma3);
        assert_eq!(Architecture::detect(&cfg("future")), Architecture::Unknown("future".into()));
        assert_eq!(
            Architecture::detect(&ModelConfig { architectures: vec!["Qwen2ForCausalLM".into()], ..Default::default() }),
            Architecture::Qwen2
        );
    }
}