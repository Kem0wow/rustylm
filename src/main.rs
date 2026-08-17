use std::io::{self, Write};

use rustylm_runtime::{Device, Engine, Params};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "models/qwen2.5-1.5b-instruct".into());
    let mut device = Device::Auto;
    let mut question = None;
    let mut params = Params::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-m" | "--model" => dir = args.next().unwrap_or(dir),
            "-p" | "--prompt" => question = args.next(),
            "-t" | "--temp" => params.temperature = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.7),
            "-n" | "--max-tokens" => params.max_tokens = args.next().and_then(|v| v.parse().ok()).unwrap_or(512),
            "--cuda" => device = Device::Cuda,
            "--cpu" => device = Device::Cpu,
            "-h" | "--help" => return help(),
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    print!("loading {dir} ... ");
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
         \x20 -m, --model DIR      model directory\n\
         \x20 -p, --prompt TEXT    answer once and exit\n\
         \x20 -t, --temp F         sampling temperature (default 0.7)\n\
         \x20 -n, --max-tokens N   generation limit (default 512)\n\
         \x20     --cuda / --cpu   force a backend (default: auto)"
    );
    Ok(())
}
