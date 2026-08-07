use rustylm_core::device;
use rustylm_core::loader::ModelLoader;

fn main() -> anyhow::Result<()> {
    // 1. Cihazı seç
    let dev = device::select_device()?;
    println!("Aktif Donanım: {:?}", dev);

    // 2. Modeli incele
    // NOT: Buraya gerçek bir dosya yolu yazmalısın!
    let path = "models/qwen/model.safetensors"; 

    if std::path::Path::new(path).exists() {
        ModelLoader::inspect_safe(path)?;
    } else {
        println!("Dosya bulunamadı: {}. Lütfen geçerli bir safetensors dosyası koy.", path);
    }

    Ok(())
}