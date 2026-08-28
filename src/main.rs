use std::io::{self, Write};
use std::path::PathBuf;

use rustylm_runtime::{Device, Engine, Params};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let command = match args.next() {
        Some(cmd) => cmd,
        None => return help(),
    };

    match command.as_str() {
        "run" => cmd_run(args),
        "list" | "ls" => cmd_list(),
        "help" => match args.next().as_deref() {
            Some("run") => help_run(),
            Some("list") | Some("ls") => {
                println!("List available models\n\nUsage:\n  rustylm list");
                Ok(())
            }
            _ => help(),
        },
        "-h" | "--help" => help(),
        "-v" | "--version" | "version" => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        other => {
            anyhow::bail!("unknown command '{other}'. Run 'rustylm --help' for usage.");
        }
    }
}

fn cmd_run(mut args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let mut model_arg = None;
    let mut device = default_device();
    let mut question = None;
    let mut params = Params::default();
    let mut positional = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return help_run(),
            "-m" | "--model" => model_arg = args.next(),
            "-p" | "--prompt" => question = args.next(),
            "-t" | "--temp" => {
                params.temperature = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.7)
            }
            "-n" | "--max-tokens" => {
                params.max_tokens = args.next().and_then(|v| v.parse().ok()).unwrap_or(512)
            }
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
            other if other.starts_with('-') => anyhow::bail!("unknown flag for run: {other}"),
            _ => positional.push(arg),
        }
    }

    if model_arg.is_none() && !positional.is_empty() {
        model_arg = Some(positional.remove(0));
    }
    if question.is_none() && !positional.is_empty() {
        question = Some(positional.join(" "));
    }

    let Some(model_name) = model_arg else {
        anyhow::bail!(
            "missing model name. Usage: rustylm run <model> [prompt]\nRun 'rustylm list' to see available models."
        );
    };

    let dir = resolve_model_dir(&model_name);
    if !dir.exists() || !dir.join("config.json").exists() {
        anyhow::bail!(
            "model not found: '{model_name}' (checked {})\nRun 'rustylm list' to see available models.",
            dir.display()
        );
    }

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

fn cmd_list() -> anyhow::Result<()> {
    let mut search_dirs = Vec::new();
    if let Ok(home) = std::env::var("HOME") {
        let h = PathBuf::from(home);
        search_dirs.push(h.join("Models"));
        search_dirs.push(h.join("models"));
    }
    search_dirs.push(PathBuf::from("./models"));
    search_dirs.push(PathBuf::from("./Models"));

    let mut found = Vec::new();
    for base in search_dirs {
        if let Ok(entries) = std::fs::read_dir(&base) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() && p.join("config.json").exists() {
                    let name = p
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    if !found.iter().any(|(n, _)| n == &name) {
                        found.push((name, p));
                    }
                }
            }
        }
    }

    if found.is_empty() {
        println!("No models found in ./models or ~/Models");
    } else {
        println!("{:<30} PATH", "NAME");
        for (name, path) in found {
            println!("{:<30} {}", name, path.display());
        }
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

fn default_device() -> Device {
    if let Ok(env_dev) = std::env::var("RUSTYLM_DEVICE") {
        match env_dev.to_lowercase().as_str() {
            "cuda" => return Device::Cuda,
            "cpu" => return Device::Cpu,
            "auto" => return Device::Auto,
            _ => {}
        }
    }
    Device::Auto
}

fn help() -> anyhow::Result<()> {
    println!(
        "Usage:\n\
         \x20 rustylm [flags]\n\
         \x20 rustylm [command]\n\n\
         Available Commands:\n\
         \x20 run         Run a model\n\
         \x20 list, ls    List available models\n\
         \x20 help        Help about any command\n\n\
         Flags:\n\
         \x20 -h, --help      help for rustylm\n\
         \x20 -v, --version   version for rustylm\n\n\
         Use \"rustylm [command] --help\" for more information about a command."
    );
    Ok(())
}

fn help_run() -> anyhow::Result<()> {
    println!(
        "Run a model\n\n\
         Usage:\n\
         \x20 rustylm run [flags] MODEL [PROMPT]\n\n\
         Flags:\n\
         \x20 -p, --prompt TEXT      answer once and exit\n\
         \x20 -t, --temp F           sampling temperature (default 0.7)\n\
         \x20 -n, --max-tokens N     generation limit (default 512)\n\
         \x20 -d, --device DEV       device backend (auto, cuda, cpu)\n\
         \x20     --cuda / --cpu     shortcut for device flag\n\
         \x20 -h, --help             help for run"
    );
    Ok(())
}

fn resolve_model_dir(raw: &str) -> PathBuf {
    let home = std::env::var("HOME").ok().map(PathBuf::from);
    let path = expand_tilde(raw, home.as_ref());
    if path.exists() {
        return path;
    }
    let candidates = [
        home.as_ref().map(|h| h.join("Models")),
        home.as_ref().map(|h| h.join("models")),
        Some(PathBuf::from("./models")),
        Some(PathBuf::from("./Models")),
    ];
    for base in candidates.iter().flatten() {
        let inside = base.join(raw);
        if inside.exists() {
            return inside;
        }
    }
    path
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
