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

    assert_eq!(
        run.stdout(),
        "lookup=0 status=0\ncall=0\nversion=13040\n",
        "the target must see the driver's own answers"
    );

    let bundle = run.bundle().expect("bundle");
    let record = records(&bundle)
        .into_iter()
        .find(|record| record.function == "cuDriverGetVersion")
        .expect("a cuDriverGetVersion record");
    assert_eq!(record.result, 0);
}
