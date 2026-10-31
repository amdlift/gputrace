//! End-to-end capture: a target runs under the interposer with the stub
//! driver behind it, and the trace bundle is read back and asserted on.

use gputrace_core::{ArgValue, CallRecord, TraceBundle};
use gputrace_testkit::{Harness, target_bin};

fn records(bundle: &TraceBundle) -> Vec<CallRecord> {
    bundle
        .threads()
        .iter()
        .flat_map(|summary| bundle.read_segment(summary.thread).expect("segment"))
        .collect()
}

#[test]
fn an_intercepted_call_lands_in_the_bundle_with_its_return_value() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("stub_smoke"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());

    let bundle = run.bundle().expect("bundle");
    let records = records(&bundle);

    let init = records
        .iter()
        .find(|record| record.function == "cuInit")
        .expect("a cuInit record");
    assert_eq!(init.result, 0);
    assert_eq!(
        init.args,
        [gputrace_core::Argument::new("flags", ArgValue::Unsigned(0))]
    );
}
