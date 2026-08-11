use rustylm_backend::device;
use rustylm_core::architecture::Architecture;
use rustylm_core::config::ModelConfig;
use std::path::PathBuf;

fn main() {
    println!("RustyLM Init");

    let device = device::auto();
    println!("Device: {}", device);

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model_dirs = [
        "models/qwen2.5-0.5b-instruct",
        "models/llama3.2-1b-instruct",
        "models/gemma-4-12b-it-assistant",
    ];

    for model_dir in model_dirs {
        let config_path = manifest_dir.join(model_dir).join("config.json");
        println!("\n--------------------------------------------------");
        println!("Checking model: {}", model_dir);

        if config_path.exists() {
            match ModelConfig::load_config(&config_path) {
                Ok(config) => {
                    let architecture = Architecture::detect(&config);

                    println!("[ModelConfig Loaded Successfully]");
                    println!("  Model type: {}", config.model_type);
                    println!("  Detected Architecture: {} (Supported: {})", architecture, architecture.is_supported());

                    println!("  Hidden size: {}", config.hidden_size);
                    println!("  Intermediate size: {}", config.intermediate_size);
                    println!("  Hidden layers: {}", config.num_hidden_layers);

                    println!(
                        "  Attention: {} Q heads / {} KV heads",
                        config.num_attention_heads,
                        config.num_key_value_heads
                    );

                    println!("  Head dimension: {}", config.head_dim());
                    println!("  KV groups (GQA): {}", config.num_key_value_groups());
                    println!("  Vocabulary size: {}", config.vocab_size);

                    println!("  Activation: {}", config.hidden_act);
                    println!("  RMSNorm epsilon: {}", config.rms_norm_eps);
                    println!("  RoPE theta: {}", config.rope_theta);
                    println!(
                        "  Max position embeddings: {}",
                        config.max_position_embeddings
                    );
                    if let Some(eos) = &config.eos_token_id {
                        println!("  EOS token ID(s): {:?}", eos);
                    }
                }
                Err(e) => {
                    eprintln!("Failed to load model config from {:?}: {}", config_path, e);
                }
            }
        } else {
            println!("Config file not found at {:?}", config_path);
        }
    }
}