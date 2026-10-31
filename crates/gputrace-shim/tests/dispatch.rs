//! Capture through the dispatch table: a target that resolves a `cu*` entry
//! point with `cuGetProcAddress` instead of linking it is recorded just the
//! same (ADR-0003).

use gputrace_core::{CallRecord, TraceBundle};
use gputrace_testkit::{CaptureRun, Harness, target_bin};

fn run(fixture: &str) -> CaptureRun {
    let harness = Harness::new().expect("harness");
    let run = harness.run(target_bin(fixture), &[]).expect("run target");
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
fn a_call_resolved_through_the_dispatch_table_is_recorded() {
    let run = run("dispatch_table");

    assert_eq!(observed(&run, "lookup"), "0");
    assert_eq!(observed(&run, "status"), "0");
    assert_eq!(observed(&run, "call"), "0");
    assert_eq!(
        observed(&run, "version"),
        "13040",
        "the target must see the driver's own answer"
    );

    let bundle = run.bundle().expect("bundle");
    let record = records(&bundle)
        .into_iter()
        .find(|record| record.function == "cuDriverGetVersion")
        .expect("a cuDriverGetVersion record");
    assert_eq!(record.result, 0);
}

#[test]
fn the_cuda_11_arity_answers_with_the_same_trampoline() {
    let run = run("dispatch_table");

    assert_eq!(observed(&run, "v1"), "0");
    assert_eq!(
        observed(&run, "v1_agrees"),
        "true",
        "both arities the driver exports must be shimmed"
    );
}
