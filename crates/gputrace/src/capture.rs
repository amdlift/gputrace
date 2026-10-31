//! The `capture` subcommand: run a target under the interposer.
//!
//! The wrapper only arranges the target's environment and then becomes the
//! target, so the target's streams and exit status are its own. The bundle is
//! written from inside the target process by the interposer, which is why
//! nothing here waits for a child.

use std::convert::Infallible;
use std::env;
use std::ffi::{OsStr, OsString};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Overrides where the interposer is loaded from. Without it the interposer
/// is taken from beside the `gputrace` executable.
pub const INTERPOSER_ENV: &str = "GPUTRACE_INTERPOSER";
const INTERPOSER_FILE: &str = "libgputrace_shim.so";
const PRELOAD_ENV: &str = "LD_PRELOAD";
const VISIBLE_DEVICES_ENV: &str = "CUDA_VISIBLE_DEVICES";

// The CLI cannot link the interposer — it is a cdylib — so the variables it
// reads are named here too. They are a contract between the two crates.
const BUNDLE_DIR_ENV: &str = "GPUTRACE_OUT";
const FIDELITY_ENV: &str = "GPUTRACE_FIDELITY";

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("no target command to capture")]
    NoTarget,
    #[error("no interposer at {0}; point {INTERPOSER_ENV} at one")]
    NoInterposer(PathBuf),
    #[error("cannot locate the gputrace executable: {0}")]
    NoSelf(#[source] std::io::Error),
    #[error("cannot resolve the output directory {path}: {source}")]
    BadOut {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot run {target}: {source}")]
    Exec {
        target: String,
        #[source]
        source: std::io::Error,
    },
}

/// One capture: what to run, where the bundle goes, and how much to record.
pub struct Capture {
    pub out: PathBuf,
    pub device: Option<String>,
    pub golden: bool,
    pub target: Vec<OsString>,
}

impl Capture {
    /// The target, prepared to run under `interposer`.
    pub fn command(&self, interposer: &Path) -> Result<Command, CaptureError> {
        let (program, args) = self.target.split_first().ok_or(CaptureError::NoTarget)?;
        let out = std::path::absolute(&self.out).map_err(|source| CaptureError::BadOut {
            path: self.out.clone(),
            source,
        })?;

        let mut command = Command::new(program);
        command
            .args(args)
            .env(PRELOAD_ENV, preload(interposer))
            .env(BUNDLE_DIR_ENV, out)
            .env(
                FIDELITY_ENV,
                if self.golden { "golden" } else { "replayable" },
            );
        if let Some(device) = &self.device {
            command.env(VISIBLE_DEVICES_ENV, device);
        }
        Ok(command)
    }

    /// Become the target. Returns only when it could not be started.
    pub fn run(&self) -> Result<Infallible, CaptureError> {
        let mut command = self.command(&interposer()?)?;
        let source = command.exec();
        Err(CaptureError::Exec {
            target: self.target[0].to_string_lossy().into_owned(),
            source,
        })
    }
}

/// The interposer to preload: the `GPUTRACE_INTERPOSER` override when set,
/// otherwise the one shipped beside this executable.
fn interposer() -> Result<PathBuf, CaptureError> {
    let candidate = match env::var_os(INTERPOSER_ENV).filter(|path| !path.is_empty()) {
        Some(path) => PathBuf::from(path),
        None => {
            let exe = env::current_exe().map_err(CaptureError::NoSelf)?;
            exe.parent()
                .unwrap_or_else(|| Path::new("."))
                .join(INTERPOSER_FILE)
        }
    };
    if candidate.exists() {
        Ok(candidate)
    } else {
        Err(CaptureError::NoInterposer(candidate))
    }
}

/// Prepend the interposer to any `LD_PRELOAD` the caller already set, so a
/// target that needs other preloaded libraries still gets them.
fn preload(interposer: &Path) -> OsString {
    let mut preload = interposer.as_os_str().to_owned();
    if let Some(existing) = env::var_os(PRELOAD_ENV).filter(|value| !value.is_empty()) {
        preload.push(OsStr::new(":"));
        preload.push(existing);
    }
    preload
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAKE_INTERPOSER: &str = "/opt/gputrace/libgputrace_shim.so";

    fn capture() -> Capture {
        Capture {
            out: PathBuf::from("trace"),
            device: None,
            golden: false,
            target: vec![OsString::from("/bin/true")],
        }
    }

    fn env_of(command: &Command, key: &str) -> Option<OsString> {
        command
            .get_envs()
            .find(|(name, _)| *name == OsStr::new(key))
            .and_then(|(_, value)| value.map(ToOwned::to_owned))
    }

    fn command_of(capture: &Capture) -> Command {
        capture
            .command(Path::new(FAKE_INTERPOSER))
            .expect("command")
    }

    #[test]
    fn the_bundle_directory_is_resolved_before_the_target_can_change_directory() {
        let out = PathBuf::from(env_of(&command_of(&capture()), BUNDLE_DIR_ENV).expect("out"));
        assert!(out.is_absolute(), "{}", out.display());
        assert!(out.ends_with("trace"), "{}", out.display());
    }

    #[test]
    fn the_interposer_is_preloaded_ahead_of_the_callers_own_preloads() {
        let preload = env_of(&command_of(&capture()), PRELOAD_ENV).expect("preload");
        let preload = preload.to_string_lossy();
        assert_eq!(preload.split(':').next(), Some(FAKE_INTERPOSER));
    }

    /// The spec's default: a bundle is only worth keeping if it holds the
    /// payloads a replay needs.
    #[test]
    fn capture_records_a_replayable_bundle_unless_told_otherwise() {
        assert_eq!(
            env_of(&command_of(&capture()), FIDELITY_ENV).as_deref(),
            Some(OsStr::new("replayable"))
        );
    }

    #[test]
    fn golden_capture_asks_the_interposer_for_golden_fidelity() {
        let mut capture = capture();
        capture.golden = true;
        assert_eq!(
            env_of(&command_of(&capture), FIDELITY_ENV).as_deref(),
            Some(OsStr::new("golden"))
        );
    }

    #[test]
    fn a_chosen_device_is_the_only_one_the_target_can_see() {
        let mut capture = capture();
        capture.device = Some("1".to_owned());
        assert_eq!(
            env_of(&command_of(&capture), VISIBLE_DEVICES_ENV).as_deref(),
            Some(OsStr::new("1"))
        );
    }

    #[test]
    fn the_target_keeps_its_own_arguments() {
        let mut capture = capture();
        capture.target.push(OsString::from("--flag"));
        let command = command_of(&capture);
        assert_eq!(command.get_program(), OsStr::new("/bin/true"));
        assert_eq!(
            command.get_args().collect::<Vec<_>>(),
            [OsStr::new("--flag")]
        );
    }
}
