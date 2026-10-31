//! What the environment asked this capture to do.
//!
//! The CLI cannot link the interposer, so it hands its decisions over as
//! environment variables; these are the names both sides agree on and the
//! answers the rest of the recorder reads them as. Each is read once.

use std::env;
use std::sync::OnceLock;

use gputrace_core::{CaptureConfig, Fidelity};

/// Names the directory the interposer writes its trace bundle to. Unset or
/// empty means no capture: the shims forward and record nothing.
pub const BUNDLE_DIR_ENV: &str = "GPUTRACE_OUT";
/// Names how much each shim records: `metadata`, `handles`, `replayable`
/// or `golden`.
pub const FIDELITY_ENV: &str = "GPUTRACE_FIDELITY";
/// How CUDA confines a process to a subset of the GPUs. `gputrace capture
/// --device` sets it, and the manifest records what it named.
pub const VISIBLE_DEVICES_ENV: &str = "CUDA_VISIBLE_DEVICES";

pub(super) fn capture_config() -> CaptureConfig {
    CaptureConfig {
        backend: "cuda".to_owned(),
        fidelity: fidelity(),
        device: selected_device(),
        command: env::args_os()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect(),
    }
}

/// The GPU this capture was pointed at, as the launcher numbered it. CUDA
/// renumbers a confined process's devices from zero, so this is not the
/// ordinal the records carry — it is the one `--device` named.
fn selected_device() -> Option<u32> {
    sole_device(&env::var(VISIBLE_DEVICES_ENV).ok()?)
}

/// The one device `selection` names, or `None` when it names several, none,
/// or a GPU by something other than an ordinal.
fn sole_device(selection: &str) -> Option<u32> {
    let mut devices = selection.split(',');
    let only = devices.next()?.trim();
    devices.next().is_none().then(|| only.parse().ok())?
}

/// How much every shim in this process records, read once. Unset means the
/// capture is replayable: a bundle is only worth keeping if it holds the
/// payloads a replay needs.
pub(super) fn fidelity() -> Fidelity {
    static FIDELITY: OnceLock<Fidelity> = OnceLock::new();
    *FIDELITY.get_or_init(|| {
        env::var(FIDELITY_ENV)
            .ok()
            .and_then(|name| name.parse().ok())
            .unwrap_or(Fidelity::Replayable)
    })
}

/// Which GPU a capture was pointed at is only an answer when exactly one was
/// selected. The cases below do not reach the bundle from any fixture, so
/// they are pinned here.
#[cfg(test)]
mod sole_device_tests {
    use super::sole_device;

    #[test]
    fn one_selected_device_is_the_one_the_capture_was_pointed_at() {
        assert_eq!(sole_device("1"), Some(1));
        assert_eq!(sole_device(" 2 "), Some(2));
    }

    #[test]
    fn a_selection_of_several_devices_names_no_single_one() {
        assert_eq!(sole_device("0,1"), None);
    }

    /// CUDA also accepts GPU UUIDs, and an empty selection hides every GPU.
    #[test]
    fn a_selection_that_is_not_an_ordinal_is_not_guessed_at() {
        assert_eq!(
            sole_device("GPU-dead0000-beef-1234-5678-90abcdef1234"),
            None
        );
        assert_eq!(sole_device(""), None);
    }
}
