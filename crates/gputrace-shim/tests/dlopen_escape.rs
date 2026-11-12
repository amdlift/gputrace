//! The third call path ADR-0003 did not close: a target that `dlopen`s the
//! driver and `dlsym`s its handle reaches the real entry point directly, and
//! nothing about that lookup passes through the interposer (#13, ADR-0007).
//!
//! These tests pin the escape as it is today. They are the regression guard
//! that will have to change if we ever close the path.

use gputrace_core::{CallRecord, TraceBundle};
use gputrace_testkit::{CaptureRun, Harness, target_bin};

fn run() -> CaptureRun {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("dlopened_driver"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());
    run
}

/// One `key=value` line of what the fixture observed.
fn observed<'a>(run: &'a CaptureRun, key: &str) -> &'a str {
    run.stdout()
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("no {key} in: {}", run.stdout()))
}

fn records(bundle: &TraceBundle) -> Vec<CallRecord> {
    bundle
        .threads()
        .iter()
        .flat_map(|summary| bundle.read_segment(summary.thread).expect("segment"))
        .collect()
}

#[test]
fn a_dlsym_on_the_drivers_own_handle_hands_back_the_driver_not_the_interposer() {
    let run = run();

    assert_eq!(observed(&run, "dlopen"), "true");
    assert_eq!(observed(&run, "dlsym"), "true");
    assert_eq!(
        observed(&run, "dlsym_is_interposer"),
        "false",
        "a lookup against a library handle searches that object, so LD_PRELOAD \
         has nothing to shadow"
    );
}

#[test]
fn the_call_reaches_the_driver_and_is_not_recorded() {
    let run = run();

    assert_eq!(observed(&run, "call"), "0");
    assert_eq!(
        observed(&run, "devices"),
        "1",
        "the target got the driver's own answer"
    );
    assert_eq!(
        run.stub_call_count("cuDeviceGetCount"),
        1,
        "the driver answered the escaped call"
    );

    let bundle = run.bundle().expect("bundle");
    let recorded = records(&bundle);
    assert!(
        recorded.iter().any(|record| record.function == "cuInit"),
        "the same target's global-scope call is captured, so capture was live: {recorded:?}"
    );
    assert!(
        !recorded
            .iter()
            .any(|record| record.function == "cuDeviceGetCount"),
        "the escaped call is absent from the bundle: {recorded:?}"
    );
}
