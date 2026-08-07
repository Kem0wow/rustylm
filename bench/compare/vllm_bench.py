from vllm import LLM, SamplingParams
import time, yaml, subprocess, psutil
from pathlib import Path

MODEL_MAP = {
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
}

def get_gpu_vram():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"])
        return int(out.decode().split('\n')[0])
    except: return 0

def get_ram_usage():
    return int(psutil.virtual_memory().used / (1024 * 1024))

def benchmark_vllm(model, prompt):
    print(f"--- {model} with vLLM ---")
    hf_model_id = MODEL_MAP.get(model, model)
    path = Path("bench/compare/results.yaml")
    
    vram_before = get_gpu_vram()
    ram_before = get_ram_usage()
    
    try:
        start = time.perf_counter()
        # GTX 1650 için çok agresif ayarlar
        llm = LLM(
            model=hf_model_id,
            gpu_memory_utilization=0.30, # VRAM'in sadece %30'unu KV Cache için ayır
            max_model_len=512,           # Context'i iyice kıstık
            enforce_eager=True,          # CUDA Graph'ları kapat (VRAM kazandırır)
            dtype="float16"              # GTX 1650 bfloat16 desteklemez
        )
        
        sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
        outputs = llm.generate([prompt], sampling_params)
        
        dur = time.perf_counter() - start
        vram_after = get_gpu_vram()
        ram_after = get_ram_usage()
        
        tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        tps = tokens / dur if dur > 0 else 0
        status = "Success"

    except Exception as e:
        print(f"vLLM Hata Aldı: {e}")
        dur, tps, vram_after, ram_after = 0, 0, vram_before, ram_before
        status = f"Failed (OOM): {str(e)[:50]}"

    # YAML Yazma
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    model_entry = data.setdefault("model", {}).setdefault(model, {})
    
    model_entry["vllm"] = {
        "status": status,
        "duration": round(dur, 3),
        "tps": round(tps, 2),
        "peak_vram": f"{vram_after} MB",
        "device": "GPU" if vram_after > vram_before + 50 else "CPU",
        "peak_ram": f"{ram_after} MB"
    }

    path.write_text(yaml.dump(data, sort_keys=False))
    print(f"Sonuç: {status}")

if __name__ == "__main__":
    # 3B yerine 0.5B denersen GTX 1650'de vLLM'in çalıştığını görebilirsin
    benchmark_vllm("qwen2.5:3b", "Importance of quantization.")