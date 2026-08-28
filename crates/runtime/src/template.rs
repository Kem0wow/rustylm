use rustylm_core::{Architecture, ModelConfig, Tokenizer};

pub fn template(arch: &Architecture, user: &str, system: &str) -> String {
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

pub fn eos_ids(cfg: &ModelConfig, tok: &Tokenizer) -> Vec<u32> {
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
