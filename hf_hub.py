from huggingface_hub import snapshot_download

# snapshot_download klasörü otomatik yönetmek için daha iyidir
snapshot_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct", 
    local_dir="./models/qwen-3b",
    # Sadece bize lazım olan dosyaları çekelim (VRAM tasarrufu)
    allow_patterns=["*.safetensors", "*.json", "*.model"]
)

print("Model ./models/qwen-3b klasörüne başarıyla indirildi.")