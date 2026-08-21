use std::io::{self, Write};
use std::path::PathBuf;

use rustylm_runtime::{Device, Engine, Params};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut model_arg = None;
    let mut device = Device::Auto;
    let mut question = None;
    let mut params = Params::default();

    if let Ok(env_dev) = std::env::var("RUSTYLM_DEVICE") {
        match env_dev.to_lowercase().as_str() {
            "cuda" => device = Device::Cuda,
            "cpu" => device = Device::Cpu,
            "auto" => device = Device::Auto,
            _ => {}
        }
    }

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-m" | "--model" => model_arg = args.next(),
            "-p" | "--prompt" => question = args.next(),
            "-t" | "--temp" => params.temperature = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.7),
            "-n" | "--max-tokens" => params.max_tokens = args.next().and_then(|v| v.parse().ok()).unwrap_or(512),
            "-d" | "--device" => {
                if let Some(d) = args.next() {
                    match d.to_lowercase().as_str() {
                        "cuda" => device = Device::Cuda,
                        "cpu" => device = Device::Cpu,
                        "auto" => device = Device::Auto,
                        other => anyhow::bail!("unknown device: '{other}', expected auto|cuda|cpu"),
                    }
                }
            }
            "--cuda" => device = Device::Cuda,
            "--cpu" => device = Device::Cpu,
            "-h" | "--help" => return help(),
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    let dir = resolve_model_dir(model_arg.as_deref());

    print!("loading {} ... ", dir.display());
    io::stdout().flush()?;
    let engine = Engine::load(&dir, device)?;
    println!("{} on {}", engine.architecture(), engine.device());
    if engine.vram_bytes() > 0 {
        println!("{} MiB of weights in VRAM", engine.vram_bytes() >> 20);
    }

    if let Some(q) = question {
        return run(&engine, &q, &params);
    }

    loop {
        print!("\n> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "quit" | "exit" | "q") {
            break;
        }
        run(&engine, line, &params)?;
    }
    Ok(())
}

fn run(engine: &Engine, question: &str, params: &Params) -> anyhow::Result<()> {
    let stats = engine.generate(question, params, |chunk| {
        print!("{chunk}");
        io::stdout().flush().ok();
    })?;
    println!(
        "\n\n[{} prompt tokens in {:.2}s | {} generated at {:.1} tok/s]",
        stats.prompt_tokens,
        stats.prefill_secs,
        stats.generated,
        stats.tokens_per_sec()
    );
    Ok(())
}

fn help() -> anyhow::Result<()> {
    println!(
        "rustylm [options]\n\
         \x20 -m, --model DIR        model directory\n\
         \x20 -p, --prompt TEXT      answer once and exit\n\
         \x20 -t, --temp F           sampling temperature (default 0.7)\n\
         \x20 -n, --max-tokens N     generation limit (default 512)\n\
         \x20 -d, --device DEV      device backend (auto, cuda, cpu)\n\
         \x20     --cuda / --cpu     shortcut for device flag\n\
         \x20 env RUSTYLM_DEVICE    default device setting (auto, cuda, cpu)"
    );
    Ok(())
}

fn resolve_model_dir(arg: Option<&str>) -> PathBuf {
    let home = std::env::var("HOME").ok().map(PathBuf::from);
    let models_dir = home.as_ref().map(|h| h.join("Models")).unwrap_or_else(|| PathBuf::from("./Models"));

    match arg {
        Some(raw) => {
            let path = expand_tilde(raw, home.as_ref());
            if path.exists() {
                return path;
            }
            let inside_models = models_dir.join(raw);
            if inside_models.exists() {
                return inside_models;
            }
            if !path.is_absolute() && !raw.starts_with('.') && !raw.starts_with('~') {
                inside_models
            } else {
                path
            }
        }
        None => {
            if models_dir.join("config.json").exists() {
                return models_dir;
            }
            if let Ok(entries) = std::fs::read_dir(&models_dir) {
                let mut subdirs = Vec::new();
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_dir() && p.join("config.json").exists() {
                        subdirs.push(p);
                    }
                }
                if subdirs.len() == 1 {
                    return subdirs.remove(0);
                }
            }
            models_dir
        }
    }
}

fn expand_tilde(raw: &str, home: Option<&PathBuf>) -> PathBuf {
    if raw == "~" {
        home.cloned().unwrap_or_else(|| PathBuf::from("~"))
    } else if let Some(stripped) = raw.strip_prefix("~/") {
        if let Some(h) = home {
            h.join(stripped)
        } else {
            PathBuf::from(raw)
        }
    } else {
        PathBuf::from(raw)
    }
}
