use rustylm_backend::device;
use rustylm_models::inspect_model;

fn main() {
    println!("RustyLM Init");

    let device = device::auto();
    println!("{}", device);

    inspect_model("models/qwen-0.5b/model.safetensors")
        .expect("Failed to read model");
}