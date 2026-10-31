//! gputrace CLI: orchestrates capture, replay, mutate, fuzz, validate, and
//! inspect against CUDA Driver API trace bundles.

mod capture;

use std::ffi::OsString;
use std::path::PathBuf;

use clap::builder::TypedValueParser;
use clap::{Parser, Subcommand};
use gputrace_core::Fidelity;

#[derive(Parser)]
#[command(
    name = "gputrace",
    about = "Capture, replay, mutate, and fuzz CUDA Driver API calls"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a target under the interposer and record a trace bundle.
    Capture {
        /// Confine the target to one GPU (sets CUDA_VISIBLE_DEVICES).
        ///
        /// Prefer a GPU that does not drive your display. Capturing a heavy
        /// workload on the display GPU can freeze the desktop or trigger an
        /// Xid reset that tears down every context on it, this capture
        /// included. See docs/choosing-a-capture-gpu.md.
        #[arg(long)]
        device: Option<String>,
        /// How much to record of each intercepted call.
        ///
        /// Each level adds to the one before it: metadata is the call, its
        /// thread, its ordering stamps, its scalar arguments and its return
        /// value; handles adds stable identity for handles and device
        /// pointers; replayable adds host->device buffers, launch parameters
        /// and module images; golden adds device->host outputs.
        #[arg(long, value_name = "LEVEL", default_value = "replayable", value_parser = fidelity())]
        fidelity: Fidelity,
        /// Shorthand for --fidelity golden: also capture device->host outputs.
        #[arg(long, conflicts_with = "fidelity")]
        golden: bool,
        /// Output trace bundle directory.
        #[arg(short, long, default_value = "trace")]
        out: PathBuf,
        /// The target command and its arguments.
        #[arg(last = true, required = true)]
        target: Vec<OsString>,
    },
    /// Faithfully re-issue a trace bundle against the real driver.
    Replay {
        /// Trace bundle directory.
        bundle: String,
    },
    /// Replay a bundle with one deterministic mutation spec applied.
    Mutate {
        /// Trace bundle directory.
        bundle: String,
        /// TOML mutation spec.
        #[arg(long)]
        spec: String,
    },
    /// Run a parallel fuzzing campaign over a seed bundle.
    Fuzz {
        /// Seed trace bundle directory.
        bundle: String,
        /// Number of parallel worker processes.
        #[arg(short, long, default_value_t = 1)]
        jobs: usize,
    },
    /// Golden-diff replay plus capture-overhead benchmark.
    Validate {
        /// Trace bundle directory.
        bundle: String,
    },
    /// Dump a bundle's manifest and reconstructed happens-before graph.
    Inspect {
        /// Trace bundle directory.
        bundle: String,
    },
}

fn fidelity() -> impl TypedValueParser<Value = Fidelity> {
    clap::builder::PossibleValuesParser::new(Fidelity::ALL.map(Fidelity::name))
        .try_map(|name| name.parse::<Fidelity>())
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Capture {
            device,
            fidelity,
            golden,
            out,
            target,
        } => {
            let capture = capture::Capture {
                out,
                device,
                fidelity: if golden { Fidelity::Golden } else { fidelity },
                target,
            };
            match capture.run() {
                Ok(never) => match never {},
                Err(e) => {
                    eprintln!("gputrace: {e}");
                    std::process::exit(2);
                }
            }
        }
        Command::Replay { .. } => unimplemented!("replay"),
        Command::Mutate { .. } => unimplemented!("mutate"),
        Command::Fuzz { .. } => unimplemented!("fuzz"),
        Command::Validate { .. } => unimplemented!("validate"),
        Command::Inspect { .. } => unimplemented!("inspect"),
    }
}
