//! `vole-field` command line.
//!
//! ```text
//! vole-field run        [--run-dir DIR] [--context N] [--future N] [--threads N] [--no-clean]
//! vole-field train      [--seed N] [--iterations N] [--batch N] [--burn-min N]
//!                       [--burn-max N] [--loss-steps N] [--lr F] [--out PATH]
//! vole-field producer   --run-dir DIR --checkpoint PATH --scene ID --context N
//!                       [--skip-entropyfs] [--report PATH]
//! vole-field request    --run-dir DIR --checkpoint PATH --mode baseline|raw|vole
//!                       --scene ID [--expect-scene ID] --context N --future N
//!                       --request NAME [--reps N] [--warmup] [--emit PATH]
//!                       [--vole-blob HEX] [--raw-file PATH] [--report PATH]
//! vole-field version
//! ```
//!
//! `producer` and `request` are the child modes: the top-level `run` orchestrates
//! them as separate processes and they are also usable by hand, which is how a
//! reviewer reproduces a single step of the experiment.

use std::path::PathBuf;
use std::process::ExitCode;

use vole_field::experiment::{
    emit_child_report, run_demo, run_producer, run_request, ProducerArgs, RequestArgs, RunOptions,
};
use vole_field::train::{self, TrainConfig};

/// `--flag value` reader over a slice of arguments.
fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn flag_u64(args: &[String], name: &str, default: u64) -> u64 {
    flag(args, name)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn flag_f64(args: &[String], name: &str, default: f64) -> f64 {
    flag(args, name)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn fail(context: &str, e: impl std::fmt::Display) -> ExitCode {
    eprintln!("vole-field: {context}: {e}");
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("run");
    let rest: Vec<String> = args.iter().skip(1).cloned().collect();

    match cmd {
        // ---------------------------------------------------------------- run
        "run" => {
            let defaults = RunOptions::default();
            let opts = RunOptions {
                run_dir: flag(&rest, "--run-dir")
                    .map(PathBuf::from)
                    .unwrap_or(defaults.run_dir),
                checkpoint: flag(&rest, "--checkpoint")
                    .map(PathBuf::from)
                    .unwrap_or(defaults.checkpoint),
                context_len: flag_u64(&rest, "--context", defaults.context_len as u64) as usize,
                future_len: flag_u64(&rest, "--future", defaults.future_len as u64) as usize,
                threads: flag_u64(&rest, "--threads", defaults.threads as u64) as usize,
                clean: !rest.iter().any(|a| a == "--no-clean"),
            };
            match run_demo(&opts) {
                Ok(r) => {
                    if r.pass {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::FAILURE
                    }
                }
                Err(e) => fail("run", e),
            }
        }

        // ------------------------------------------------------------ producer
        "producer" => match ProducerArgs::parse(&rest) {
            Ok(a) => match run_producer(&a) {
                Ok(r) => {
                    if let Err(e) = emit_child_report(&r) {
                        return fail("producer report", e);
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => fail("producer", e),
            },
            Err(e) => fail("producer arguments", e),
        },

        // ------------------------------------------------------------- request
        "request" => match RequestArgs::parse(&rest) {
            Ok(a) => match run_request(&a) {
                Ok(r) => {
                    if let Err(e) = emit_child_report(&r) {
                        return fail("request report", e);
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => fail("request", e),
            },
            Err(e) => fail("request arguments", e),
        },

        // --------------------------------------------------------------- train
        "train" => {
            let defaults = TrainConfig::default();
            let tc = TrainConfig {
                seed: flag_u64(&rest, "--seed", defaults.seed),
                iterations: flag_u64(&rest, "--iterations", defaults.iterations as u64) as usize,
                batch: flag_u64(&rest, "--batch", defaults.batch as u64) as usize,
                burn_min: flag_u64(&rest, "--burn-min", defaults.burn_min as u64) as usize,
                burn_max: flag_u64(&rest, "--burn-max", defaults.burn_max as u64) as usize,
                loss_steps: flag_u64(&rest, "--loss-steps", defaults.loss_steps as u64) as usize,
                lr: flag_f64(&rest, "--lr", defaults.lr),
                full_bptt: !rest.iter().any(|a| a == "--detach-burn"),
                loss: flag(&rest, "--loss")
                    .and_then(train::Loss::parse)
                    .unwrap_or(defaults.loss),
                scheduled_sampling_p_max: flag_f64(
                    &rest,
                    "--scheduled-sampling",
                    defaults.scheduled_sampling_p_max as f64,
                ) as f32,
                out: flag(&rest, "--out")
                    .map(PathBuf::from)
                    .unwrap_or(defaults.out),
            };
            eprintln!(
                "training: seed=0x{:016x} iterations={} batch={} window={}..={} loss_steps={} lr={} full_bptt={} out={}",
                tc.seed,
                tc.iterations,
                tc.batch,
                tc.burn_min,
                tc.burn_max,
                tc.loss_steps,
                tc.lr,
                tc.full_bptt,
                tc.out.display()
            );
            match train::run(&tc, false) {
                Ok(r) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&r).unwrap_or_else(|_| "{}".into())
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => fail("train", e),
            }
        }

        // ---------------------------------------------------------------- eval
        "eval" => {
            let checkpoint = flag(&rest, "--checkpoint")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("assets/tiny_convlstm.safetensors"));
            let context_len = flag_u64(&rest, "--context", 256) as usize;
            let future_len = flag_u64(&rest, "--future", 16) as usize;
            match train::evaluate(&checkpoint, context_len, future_len) {
                Ok(r) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&r).unwrap_or_else(|_| "{}".into())
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => fail("eval", e),
            }
        }

        // ------------------------------------------------------------- version
        "version" => {
            println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }

        other => fail(
            "arguments",
            format!(
                "unknown subcommand {other:?}; expected run|producer|request|train|eval|version"
            ),
        ),
    }
}
