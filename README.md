# RustyLM

Small, fast local LLM inference in Rust. Loads a Hugging Face model directory
(`config.json` + `tokenizer.json` + `*.safetensors`) and answers questions on
CPU, on CUDA, or on both at once.

## Usage

```bash
cargo build --release                  # CPU only
cargo build --release --features cuda  # CPU + CUDA

./target/release/rustylm -m models/qwen2.5-1.5b-instruct        # chat REPL
./target/release/rustylm -p "Explain gravity briefly."          # answer once
```

| flag                   | meaning                                             |
| ---------------------- | --------------------------------------------------- |
| `-m, --model DIR`    | model directory (or`MODEL_DIR`)                   |
| `-p, --prompt TEXT`  | answer once and exit                                |
| `-t, --temp F`       | sampling temperature,`0` for greedy (default 0.7) |
| `-n, --max-tokens N` | generation limit (default 512)                      |
| `--cuda` / `--cpu` | force a backend (default: CUDA if present)          |
| `RUSTYLM_VRAM_MB`    | cap how much VRAM the weights may take              |

## Library API

```rust
use rustylm_runtime::{Device, Engine, Params};

let engine = Engine::load("models/qwen2.5-1.5b-instruct", Device::Auto)?;
println!("{}", engine.ask("What is the capital of France?")?);

// or stream, with sampling control
let stats = engine.generate("Write a haiku.", &Params::default(), |chunk| print!("{chunk}"))?;
println!("{:.1} tok/s", stats.tokens_per_sec());
```

The surface is deliberately three calls wide — `load`, `ask`, `generate` — so a
Python binding can be a thin wrapper over the same names.

## How it stays small and fast

- **Q8 weights.** Every tensor is quantized at load time to int8 with one f32
  scale per 32 values, so a 1.5B model needs ~1.6 GB instead of ~6 GB. Matrix
  work is memory-bound, so this is also the main speedup.
- **One CUDA kernel.** A single NVRTC-compiled warp-per-row `q8_matvec`; no
  cuBLAS, no `.cu` build step. Weights stay resident in VRAM.
- **VRAM budget, not all-or-nothing.** Projections are uploaded heaviest-first
  until the budget runs out; whatever does not fit simply runs on the CPU. A
  model larger than the card still runs.
- **Self-balancing hybrid.** Each projection splits its output rows between GPU
  and CPU and re-tunes the split every call from the rows-per-second each side
  actually delivered, so neither device waits on the other.
- **SIMD CPU path.** Independent lane accumulators keep the float adds
  reorderable, which is what lets the loop vectorize (2.5x over the naive sum).

Measured on a GTX 1650 (4 GB) + i5-10300H, Qwen2.5-1.5B-Instruct:

| backend                     | tokens/s |
| --------------------------- | -------- |
| CPU only                    | 10.4     |
| CUDA + CPU                  | ~39      |
| CUDA + CPU, 800 MB VRAM cap | 15.5     |

## Supported models

Qwen2/2.5, Llama, and Gemma-family checkpoints in f32/f16/bf16 safetensors.
The architecture and chat template are picked from `config.json`.

## Note on the build

`.cargo/config.toml` sets `-C target-cpu=native` for the SIMD path. Remove it
before building binaries meant to run on another machine.

---

## Future Plans

- [ ] Add more quantization types
- [ ] Expand model support
- [ ] Add a benchmark feature on the CLI to post benchmark results to our site: [kem0wow.github.io/rustylm](https://kem0wow.github.io/rustylm/benchmarks.html)
