# RustyLM

Small, fast local LLM inference in Rust. Loads a Hugging Face model directory
(`config.json` + `tokenizer.json` + `*.safetensors`) and answers questions on
CPU, on CUDA, or on both at once.

## Usage

```bash
cargo build --release                  # CPU only
cargo build --release --features cuda  # CPU + CUDA

./target/release/rustylm                                            # show available commands
./target/release/rustylm run models/qwen2.5-1.5b-instruct           # chat REPL
./target/release/rustylm run models/qwen2.5-1.5b-instruct "Explain gravity briefly."  # answer once
./target/release/rustylm list                                       # list available models
```

### Commands

| command                  | meaning                                         |
| ------------------------ | ----------------------------------------------- |
| `run <model> [prompt]` | Run a model (interactive REPL or single prompt) |
| `list`, `ls`         | List available local models                     |
| `help`                 | Help about any command                          |

### Flags (for `run`)

| flag                   | meaning                                             |
| ---------------------- | --------------------------------------------------- |
| `-p, --prompt TEXT`  | answer once and exit                                |
| `-t, --temp F`       | sampling temperature,`0` for greedy (default 0.7) |
| `-n, --max-tokens N` | generation limit (default 512)                      |
| `-d, --device DEV`   | device backend (`auto`, `cuda`, `cpu`)        |
| `--cuda` / `--cpu` | force a backend (default: auto)                     |
| `RUSTYLM_VRAM_CAP`   | cap how much VRAM the weights may take (in MiB)     |

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

---

Measured on a GTX 1650 (4 GB) + i5-10300H, Qwen2.5-1.5B-Instruct:

| backend                     | tokens/s |
| --------------------------- | -------- |
| CPU only                    | ~8       |
| CUDA + CPU                  | ~39      |
| CUDA + CPU, 800 MB VRAM cap | ~12      |

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
