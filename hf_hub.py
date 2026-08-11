from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/gemma-4-12B-it-assistant",
    local_dir="./models/gemma-4-12b-it-assistant",
    allow_patterns=[
        "*.safetensors",
        "*.json",
        "*.model",
    ],
)

print("Model başarıyla indirildi.")