//! `gputrace capture` end to end: the wrapper runs a target under the
//! interposer, the bundle lands in `--out`, and the target is undisturbed.

use std::process::{Command, Output};

use gputrace_core::TraceBundle;
use gputrace_testkit::{interposer, stub_libcuda, target_bin};
use tempfile::TempDir;

fn capture(out: &TempDir, args: &[&str]) -> Output {
    interposer().expect("interposer is built");
    Command::new(env!("CARGO_BIN_EXE_gputrace"))
        .args(["capture", "--out"])
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
    let run = capture(&out, &[]);
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
    let run = capture(&out, &["7"]);

    assert_eq!(run.status.code(), Some(7));
    assert_eq!(String::from_utf8_lossy(&run.stdout), "out:driver=13040\n");
    assert_eq!(String::from_utf8_lossy(&run.stderr), "err:ready\n");
}
