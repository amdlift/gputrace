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

#[test]
fn a_recorded_call_still_reaches_the_driver_and_returns_its_answer() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("stub_smoke"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());

    assert_eq!(
        run.stub_calls(),
        [
            "cuInit",
            "cuDriverGetVersion",
            "cuDeviceGetCount",
            "cuDeviceGetName"
        ],
        "the shim must forward, and must not disturb entry points it does not shim"
    );
    assert_eq!(
        run.stdout(),
        "driver=13040\ndevices=1\nname=gputrace-stub-device-0\n",
        "the target must see the driver's own answers"
    );
}

#[test]
fn records_carry_the_per_thread_order_the_calls_were_made_in() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("stub_smoke"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());

    let bundle = run.bundle().expect("bundle");
    let records = records(&bundle);

    let functions: Vec<&str> = records.iter().map(|r| r.function.as_str()).collect();
    assert_eq!(functions, ["cuInit", "cuDriverGetVersion"]);
    assert_eq!(
        records.iter().map(|r| r.thread_seq).collect::<Vec<_>>(),
        [0, 1]
    );
    assert!(
        records[0].timestamp_ns > 0 && records[1].timestamp_ns >= records[0].timestamp_ns,
        "monotonic stamps: {records:?}"
    );
    assert_eq!(bundle.threads().len(), 1, "one host thread, one segment");
}

#[test]
fn every_host_thread_gets_its_own_segment_and_its_own_sequence() {
    const THREADS: usize = 4;
    const CALLS_PER_THREAD: usize = 600;

    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("capture_threads"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());
    assert_eq!(run.stdout(), "agreed=4\n");

    let bundle = run.bundle().expect("bundle");
    assert_eq!(
        bundle.threads().len(),
        THREADS + 1,
        "one segment per thread"
    );

    let mut workers = 0;
    for summary in bundle.threads() {
        let segment = bundle.read_segment(summary.thread).expect("segment");
        assert_eq!(summary.records as usize, segment.len(), "manifest count");

        let expected = match segment.first().map(|record| record.function.as_str()) {
            Some("cuInit") => 1,
            _ => {
                workers += 1;
                CALLS_PER_THREAD
            }
        };
        assert_eq!(segment.len(), expected, "thread {:?}", summary.thread);
        assert_eq!(
            segment
                .iter()
                .map(|record| record.thread_seq)
                .collect::<Vec<_>>(),
            (0..expected as u64).collect::<Vec<_>>(),
            "each thread counts from zero on its own"
        );
    }
    assert_eq!(workers, THREADS);
    assert_eq!(
        run.stub_call_count("cuDriverGetVersion"),
        THREADS * CALLS_PER_THREAD
    );
}
