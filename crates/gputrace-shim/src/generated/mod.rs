//! The generated pass-through surface: a shim for every `cu*` entry point
//! the CUDA driver headers declare, so no call is invisible before a deep
//! handler exists for it.
//!
//! `cuda_types`, `record_args` and `trampolines` are vendored bindgen output
//! (ADR-0005) — a downstream build needs neither `cuda.h` nor bindgen.
//! Regenerate them, never edit them:
//!
//! ```text
//! nix develop --command cargo run --manifest-path crates/xtask/Cargo.toml -- gen-trampolines
//! ```
//!
//! `docs/generated-surface.md` has the longer version, and `PROVENANCE.txt`
//! records which header the vendored output came from.

mod cuda_types;
pub(crate) mod hand_written;
mod record_args;
mod trampolines;
