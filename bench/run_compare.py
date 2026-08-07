import sys
import yaml
from pathlib import Path

# Add compare directory to sys.path for simple imports
compare_dir = Path(__file__).parent / "compare"
if str(compare_dir) not in sys.path:
    sys.path.insert(0, str(compare_dir))

def run_compare(model="qwen2.5:3b", prompt="Importance of quantization."):
    print(f"==========================================")
    print(f"Running Benchmarks for Model: {model}")
    print(f"==========================================")

    # 1. Run Ollama Benchmark
    try:
        from ollama_bench import benchmark_ollama
        benchmark_ollama(model, prompt)
    except Exception as e:
        print(f"[-] Ollama benchmark skipped/failed: {e}")

    # 2. Run vLLM Benchmark
    try:
        from vllm_bench import benchmark_vllm
        benchmark_vllm(model, prompt)
    except Exception as e:
        print(f"[-] vLLM benchmark skipped/failed: {e}")

    # 3. Print Summary Results
    results_path = compare_dir / "results.yaml"
    if results_path.exists():
        print("\n==========================================")
        print("Updated Results (results.yaml):")
        print("==========================================")
        print(results_path.read_text())

if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b"
    prompt_arg = sys.argv[2] if len(sys.argv) > 2 else "Importance of quantization."
    run_compare(model_arg, prompt_arg)
