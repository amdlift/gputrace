//! gputrace CLI: orchestrates capture, replay, mutate, fuzz, validate, and
//! inspect against CUDA Driver API trace bundles.

use clap::{Parser, Subcommand};

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
        /// Target GPU (passed through as CUDA_VISIBLE_DEVICES).
        #[arg(long)]
        device: Option<String>,
        /// Also capture device->host golden outputs (L3).
        #[arg(long)]
        golden: bool,
        /// Output trace bundle directory.
        #[arg(short, long, default_value = "trace")]
        out: String,
        /// The target command and its arguments.
        #[arg(last = true, required = true)]
        target: Vec<String>,
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

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Capture { .. } => unimplemented!("capture"),
        Command::Replay { .. } => unimplemented!("replay"),
        Command::Mutate { .. } => unimplemented!("mutate"),
        Command::Fuzz { .. } => unimplemented!("fuzz"),
        Command::Validate { .. } => unimplemented!("validate"),
        Command::Inspect { .. } => unimplemented!("inspect"),
    }
}
