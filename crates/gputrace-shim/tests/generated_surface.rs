//! The generated pass-through surface (ADR-0005): an entry point with no
//! hand-written shim is still recorded, and a hand-written shim keeps the
//! symbol to itself.

use gputrace_core::{ArgValue, Argument, CallRecord, TraceBundle};
use gputrace_testkit::{Harness, target_bin};

fn records(bundle: &TraceBundle) -> Vec<CallRecord> {
    bundle
        .threads()
        .iter()
        .flat_map(|summary| bundle.read_segment(summary.thread).expect("segment"))
        .collect()
}

fn capture(fixture: &str) -> Vec<CallRecord> {
    let harness = Harness::new().expect("harness");
    let run = harness.run(target_bin(fixture), &[]).expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());
    records(&run.bundle().expect("bundle"))
}

fn record(records: &[CallRecord], function: &str) -> CallRecord {
    records
        .iter()
        .find(|record| record.function == function)
        .unwrap_or_else(|| panic!("no {function} record in {records:?}"))
        .clone()
}

#[test]
fn an_entry_point_with_no_hand_written_shim_is_still_recorded() {
    let records = capture("stub_smoke");

    let get_name = record(&records, "cuDeviceGetName");
    assert_eq!(get_name.result, 0);
    assert_eq!(
        get_name
            .args
            .iter()
            .map(|arg| arg.name.as_str())
            .collect::<Vec<_>>(),
        ["name", "len", "dev"],
        "the generated shim records every argument the caller passed"
    );
    assert_eq!(get_name.args[1].value, ArgValue::Signed(64));
    assert_eq!(get_name.args[2].value, ArgValue::Signed(0));
}

#[test]
fn a_hand_written_shim_keeps_the_symbol_to_itself() {
    let records = capture("stub_smoke");

    // The two spellings are what tells them apart: the hand-written shim in
    // `handlers/init.rs` names the argument as the shim author did, the
    // generated one as the header does (`Flags`).
    assert_eq!(
        record(&records, "cuInit").args,
        [Argument::new("flags", ArgValue::Unsigned(0))],
        "the hand-written shim, not a generated pass-through, answered cuInit"
    );
    assert_eq!(
        record(&records, "cuDriverGetVersion")
            .args
            .first()
            .map(|arg| arg.name.as_str()),
        Some("driver_version"),
    );
}
