use std::io::{self, Write};
use std::path::PathBuf;
use rustylm::{Device, Engine, Params};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("run") => cmd_run(args),
        Some("list" | "ls") => cmd_list(),
        Some("-v" | "--version" | "version") => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some("help") | Some("-h") | Some("--help") | None => help(),
        Some(other) => anyhow::bail!("unknown command '{other}'. Run 'rustylm --help' for usage."),
    }
}

fn cmd_run(mut args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let (mut model, mut prompt, mut device, mut params) = (None, None, default_device(), Params::default());

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return help_run(),
            "-m" | "--model" => model = args.next(),
            "-p" | "--prompt" => prompt = args.next(),
            "-t" | "--temp" => params.temperature = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.7),
            "-n" | "--max-tokens" => params.max_tokens = args.next().and_then(|v| v.parse().ok()).unwrap_or(512),
            "-d" | "--device" => if let Some(d) = args.next() {
                device = match d.to_lowercase().as_str() {
                    "cuda" => Device::Cuda,
                    "cpu" => Device::Cpu,
                    "auto" => Device::Auto,
                    other => anyhow::bail!("unknown device: '{other}', expected auto|cuda|cpu"),
                };
            },
            "--cuda" => device = Device::Cuda,
            "--cpu" => device = Device::Cpu,
            other if other.starts_with('-') => anyhow::bail!("unknown flag: {other}"),
            _ if model.is_none() => model = Some(arg),
            _ => prompt = Some(prompt.map_or(arg.clone(), |p| format!("{p} {arg}"))),
        }
    }

    let Some(model_name) = model else {
        anyhow::bail!("missing model name. Usage: rustylm run <model> [prompt]\nRun 'rustylm list' to see available models.");
    };

    let dir = resolve_model_dir(&model_name);
    if !dir.join("config.json").exists() {
        anyhow::bail!("model not found: '{model_name}' (checked {})\nRun 'rustylm list' to see available models.", dir.display());
    }

    print!("loading {} ... ", dir.display());
    io::stdout().flush()?;
    let engine = Engine::load(&dir, device)?;
    println!("{} on {}", engine.architecture(), engine.device());
    if engine.vram_bytes() > 0 {
        println!("{} MiB of weights in VRAM", engine.vram_bytes() >> 20);
    }

    if let Some(q) = prompt {
        return run(&engine, &q, &params);
    }

    loop {
        print!("\n> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 || matches!(line.trim(), "quit" | "exit" | "q") {
            break;
        }
        if !line.trim().is_empty() {
            run(&engine, line.trim(), &params)?;
        }
    }
    Ok(())
}

fn search_directories() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(c) = std::env::var("RUSTYLM_MODELS_DIR") { dirs.push(PathBuf::from(c)); }
    if let Ok(h) = std::env::var("HOME").map(PathBuf::from) {
        dirs.extend([h.join("Models"), h.join("models"), h.join(".rustylm/models"), h.join(".cache/huggingface/hub")]);
    }
    dirs.extend([PathBuf::from("./models"), PathBuf::from("./Models")]);
    dirs
}

fn cmd_list() -> anyhow::Result<()> {
    let mut found = Vec::new();
    for base in search_directories() {
        if let Ok(entries) = std::fs::read_dir(&base) {
            for e in entries.flatten().filter(|e| e.path().is_dir()) {
                let p = e.path();
                let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
                if p.join("config.json").exists() && !found.iter().any(|(n, _)| n == &name) {
                    found.push((name, p));
                } else if let Ok(snaps) = std::fs::read_dir(p.join("snapshots")) {
                    for s in snaps.flatten().filter(|s| s.path().join("config.json").exists()) {
                        if !found.iter().any(|(n, _)| n == &name) {
                            found.push((name.clone(), s.path()));
                        }
                    }
                }
            }
        }
    }

    if found.is_empty() {
        println!("No models found in ~/Models (or set RUSTYLM_MODELS_DIR).");
    } else {
        println!("{:<35} PATH", "NAME");
        for (name, path) in found {
            println!("{:<35} {}", name, path.display());
        }
    }
    Ok(())
}

fn run(engine: &Engine, question: &str, params: &Params) -> anyhow::Result<()> {
    let stats = engine.generate(question, params, |chunk| { print!("{chunk}"); io::stdout().flush().ok(); })?;
    println!("\n\n[{} tokens in {:.2}s | {:.1} tok/s]", stats.prompt_tokens + stats.generated, stats.prefill_secs + stats.decode_secs, stats.tokens_per_sec());
    Ok(())
}

fn default_device() -> Device {
    match std::env::var("RUSTYLM_DEVICE").as_deref() {
        Ok("cuda") => Device::Cuda,
        Ok("cpu") => Device::Cpu,
        _ => Device::Auto,
    }
}

fn help() -> anyhow::Result<()> {
    println!("Usage: rustylm [command] [flags]\n\nCommands:\n  run <model> [prompt]   Run inference (interactive or prompt)\n  list, ls               List available models\n  help                   Show this help");
    Ok(())
}

fn help_run() -> anyhow::Result<()> {
    println!("Usage: rustylm run [flags] MODEL [PROMPT]\n\nFlags:\n  -p, --prompt TEXT    answer once\n  -t, --temp F         temperature (default 0.7)\n  -n, --max-tokens N   max tokens (default 512)\n  -d, --device DEV     auto|cuda|cpu\n      --cuda / --cpu   shortcut");
    Ok(())
}

fn resolve_model_dir(raw: &str) -> PathBuf {
    let path = if let Some(stripped) = raw.strip_prefix("~/") {
        std::env::var("HOME").map_or_else(|_| PathBuf::from(raw), |h| PathBuf::from(h).join(stripped))
    } else {
        PathBuf::from(raw)
    };
    if path.exists() {
        return path;
    }
    search_directories().into_iter().map(|d| d.join(raw)).find(|p| p.exists()).unwrap_or(path)
}
