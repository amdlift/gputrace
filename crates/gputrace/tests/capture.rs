//! `gputrace capture` end to end: the wrapper runs a target under the
//! interposer, the bundle lands in `--out`, and the target is undisturbed.

use std::process::{Command, Output};

use gputrace_core::{Fidelity, TraceBundle};
use gputrace_testkit::{interposer, stub_libcuda, target_bin};
use tempfile::TempDir;

fn capture(out: &TempDir, args: &[&str], options: &[&str]) -> Output {
    interposer().expect("interposer is built");
    Command::new(env!("CARGO_BIN_EXE_gputrace"))
        .arg("capture")
        .args(options)
        .arg("--out")
        .arg(out.path().join("trace"))
        .arg("--")
        .arg(target_bin("capture_target"))
        .args(args)
        .env("GPUTRACE_LIBCUDA", stub_libcuda().expect("stub driver"))
        .output()
        .expect("run gputrace capture")
}

#[test]
fn capture_records_the_targets_driver_calls_into_the_out_directory() {
    let out = TempDir::new().expect("tempdir");
    let run = capture(&out, &[], &[]);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );

    let bundle = TraceBundle::open(out.path().join("trace")).expect("bundle");
    let functions: Vec<String> = bundle
        .threads()
        .iter()
        .flat_map(|summary| bundle.read_segment(summary.thread).expect("segment"))
        .map(|record| record.function)
        .collect();
    assert_eq!(functions, ["cuInit", "cuDriverGetVersion"]);
}

#[test]
fn the_targets_output_and_exit_status_pass_through_the_wrapper() {
    let out = TempDir::new().expect("tempdir");
    let run = capture(&out, &["7"], &[]);

    assert_eq!(run.status.code(), Some(7));
    assert_eq!(String::from_utf8_lossy(&run.stdout), "out:driver=13040\n");
    assert_eq!(String::from_utf8_lossy(&run.stderr), "err:ready\n");
}

/// Capturing against the GPU that drives the display can freeze the desktop,
/// so `--device` has to actually confine the target: the chosen GPU must be
/// the only one it can reach, not merely the one it is encouraged to prefer.
#[test]
fn the_chosen_device_is_the_only_gpu_the_target_can_see() {
    let out = TempDir::new().expect("tempdir");
    interposer().expect("interposer is built");
    let run = Command::new(env!("CARGO_BIN_EXE_gputrace"))
        .args(["capture", "--device", "1", "--out"])
        .arg(out.path().join("trace"))
        .arg("--")
        .arg(target_bin("stub_smoke"))
        .env("GPUTRACE_LIBCUDA", stub_libcuda().expect("stub driver"))
        .env("GPUTRACE_STUB_DEVICES", "2")
        .output()
        .expect("run gputrace capture");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "driver=13040\ndevices=1\nname=gputrace-stub-device-1\n"
    );

    let bundle = TraceBundle::open(out.path().join("trace")).expect("bundle");
    assert_eq!(
        bundle.manifest().capture.device,
        Some(1),
        "the manifest records which GPU the capture was pointed at"
    );
    assert_eq!(
        bundle
            .manifest()
            .environment
            .devices
            .iter()
            .map(|device| device.device)
            .collect::<Vec<_>>(),
        [0],
        "a confined target renumbers its one GPU to zero, and records say so"
    );
}

/// User Story 10: the fidelity is the researcher's to choose, not something
/// reachable only through an undocumented environment variable.
#[test]
fn each_fidelity_the_cli_offers_is_the_one_the_bundle_was_recorded_at() {
    for level in Fidelity::ALL {
        let out = TempDir::new().expect("tempdir");
        let run = capture(&out, &[], &["--fidelity", level.name()]);
        assert!(
            run.status.success(),
            "{level}: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        let bundle = TraceBundle::open(out.path().join("trace")).expect("bundle");
        assert_eq!(bundle.manifest().capture.fidelity, level);
    }
}

/// The spec's CLI contract keeps `--golden`, so it has to go on meaning the
/// same thing the long form does.
#[test]
fn golden_is_the_shorthand_for_the_golden_fidelity() {
    let out = TempDir::new().expect("tempdir");
    let run = capture(&out, &[], &["--golden"]);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let bundle = TraceBundle::open(out.path().join("trace")).expect("bundle");
    assert_eq!(bundle.manifest().capture.fidelity, Fidelity::Golden);
}

/// Heavy capture against the GPU that drives the display can freeze the
/// desktop or trigger an Xid reset. A researcher meets `--device` in the
/// flag's own help long before they open a document, so the warning has to
/// be there.
#[test]
fn the_device_flag_warns_about_capturing_on_the_display_gpu() {
    let help = Command::new(env!("CARGO_BIN_EXE_gputrace"))
        .args(["capture", "--help"])
        .output()
        .expect("run gputrace capture --help");
    let help = String::from_utf8_lossy(&help.stdout).into_owned();

    for mention in [
        "--device",
        "display",
        "Xid",
        "docs/choosing-a-capture-gpu.md",
    ] {
        assert!(help.contains(mention), "no {mention} in:\n{help}");
    }
}
