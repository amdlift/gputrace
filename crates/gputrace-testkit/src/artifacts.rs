//! Locating the build outputs a capture test needs.
//!
//! `cargo test` builds no cdylib, so the first test that needs the interposer
//! or the stub drives a nested `cargo build` for them. That keeps
//! `cargo test --workspace` self-sufficient on a clean checkout.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use crate::TestkitError;

const PROFILE_DIR: &str = env!("GPUTRACE_PROFILE_DIR");
const CARGO_PROFILE: &str = env!("GPUTRACE_CARGO_PROFILE");
const CARGO_TARGET: &str = env!("GPUTRACE_CARGO_TARGET");

const INTERPOSER_PACKAGE: &str = "gputrace-shim";
const STUB_PACKAGE: &str = "gputrace-stub-cuda";
const TARGET_PACKAGE: &str = "gputrace-testkit";

pub fn interposer() -> Result<&'static Path, TestkitError> {
    Ok(built()?.interposer.as_path())
}

pub fn stub_libcuda() -> Result<&'static Path, TestkitError> {
    Ok(built()?.stub.as_path())
}

/// A fixture target by its `[[bin]]` name.
pub fn target_bin(name: &str) -> PathBuf {
    if let Err(e) = built() {
        panic!("{e}");
    }
    let path = Path::new(PROFILE_DIR).join(name);
    assert!(path.exists(), "no fixture target at {}", path.display());
    path
}

struct Built {
    interposer: PathBuf,
    stub: PathBuf,
}

fn built() -> Result<&'static Built, TestkitError> {
    static BUILT: OnceLock<Result<Built, String>> = OnceLock::new();
    BUILT
        .get_or_init(|| build().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| TestkitError::Build(e.clone()))
}

fn build() -> Result<Built, TestkitError> {
    let mut cargo = Command::new(env!("CARGO"));
    cargo
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["build", "--profile", CARGO_PROFILE])
        .args(["-p", INTERPOSER_PACKAGE, "-p", STUB_PACKAGE])
        .args(["-p", TARGET_PACKAGE]);
    if !CARGO_TARGET.is_empty() {
        cargo.args(["--target", CARGO_TARGET]);
    }

    let output = cargo
        .output()
        .map_err(|e| TestkitError::Build(format!("spawning cargo: {e}")))?;
    if !output.status.success() {
        return Err(TestkitError::Build(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }

    let profile = Path::new(PROFILE_DIR);
    let built = Built {
        interposer: profile.join("libgputrace_shim.so"),
        stub: profile.join("libgputrace_stub_cuda.so"),
    };
    for artifact in [&built.interposer, &built.stub] {
        if !artifact.exists() {
            return Err(TestkitError::Build(format!(
                "cargo build produced no {}",
                artifact.display()
            )));
        }
    }
    Ok(built)
}
