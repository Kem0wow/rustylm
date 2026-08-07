import ollama
import time
import yaml
import subprocess
import psutil
from pathlib import Path

def get_gpu_vram():
    """Sistemdeki aktif VRAM kullanımını MB cinsinden döner."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"])
        return int(out.decode().split('\n')[0])
    except Exception:
        return 0

def get_ram_usage():
    """Sistemdeki aktif RAM kullanımını MB cinsinden döner."""
    try:
        return int(psutil.virtual_memory().used / (1024 * 1024))
    except Exception:
        return 0

def benchmark_ollama(model, prompt):
    print(f"--- {model} with Ollama ---")
    
    vram_before = get_gpu_vram()
    ram_before = get_ram_usage()
    start = time.perf_counter()
    
    res = ollama.generate(model=model, prompt=prompt)
    
    dur = time.perf_counter() - start
    vram_after = get_gpu_vram()
    ram_after = get_ram_usage()
    
    device = "GPU" if vram_after > vram_before + 50 else "CPU"
    tokens = res.get('eval_count', 0)
    tps = tokens / dur if dur > 0 else 0

    path = Path("bench/compare/results.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(path.read_text()) if path.exists() else {}

    model_entry = data.setdefault("model", {}).setdefault(model, {})
    if "rustylm" not in model_entry:
        model_entry["rustylm"] = None

    model_entry["ollama"] = {
        "duration": round(dur, 3),
        "tps": round(tps, 2),
        "peak_vram": f"{max(vram_before, vram_after)} MB",
        "avg_vram": f"{(vram_before + vram_after) // 2} MB",
        "device": device,
        "peak_ram": f"{max(ram_before, ram_after)} MB"
    }

    path.write_text(yaml.dump(data, sort_keys=False))
    print(f"Bitti: {device} | {tps:.2f} tps")

if __name__ == "__main__":
    benchmark_ollama("qwen2.5:3b", "Importance of quantization.")
