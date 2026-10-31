//! End-to-end capture: a target runs under the interposer with the stub
//! driver behind it, and the trace bundle is read back and asserted on.

use gputrace_core::{ArgValue, CallRecord, Fidelity, TraceBundle};
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

#[test]
fn the_manifest_names_what_was_captured() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("stub_smoke"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());

    let bundle = run.bundle().expect("bundle");
    let manifest = bundle.manifest();
    assert_eq!(manifest.schema_version, gputrace_core::SCHEMA_VERSION);
    assert_eq!(manifest.capture.backend, "cuda");
    assert_eq!(manifest.capture.fidelity, Fidelity::Metadata);
    assert_eq!(
        manifest.capture.command.first().map(String::as_str),
        target_bin("stub_smoke").to_str()
    );
    assert_eq!(
        manifest.threads.iter().map(|t| t.records).sum::<u64>(),
        2,
        "one summary row per thread, counting that thread's records"
    );
}

#[test]
fn an_interposer_with_nowhere_to_write_is_a_pass_through() {
    let harness = Harness::new()
        .expect("harness")
        .with_env("GPUTRACE_OUT", "");
    let run = harness
        .run(target_bin("stub_smoke"), &[])
        .expect("run target");

    assert!(run.status().success(), "target failed: {}", run.stderr());
    assert_eq!(
        run.stdout(),
        "driver=13040\ndevices=1\nname=gputrace-stub-device-0\n"
    );
    assert_eq!(run.stub_calls().len(), 4);
    assert!(
        run.bundle().is_err(),
        "no bundle directory should have been created"
    );
}

#[test]
fn golden_fidelity_is_recorded_in_the_manifest() {
    let harness = Harness::new()
        .expect("harness")
        .with_env("GPUTRACE_FIDELITY", "golden");
    let run = harness
        .run(target_bin("stub_smoke"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());

    assert_eq!(
        run.bundle().expect("bundle").manifest().capture.fidelity,
        Fidelity::Golden
    );
}

#[test]
fn a_shim_whose_entry_point_the_driver_lacks_reports_it_and_records_it() {
    let harness = Harness::new()
        .expect("harness")
        // A real library that exports no `cu*` at all, standing in for a
        // driver older than the surface the interposer was built against.
        .with_env("GPUTRACE_LIBCUDA", "libc.so.6");
    let run = harness
        .run(target_bin("absent_entry_point"), &[])
        .expect("run target");

    assert!(run.status().success(), "target failed: {}", run.stderr());
    assert_eq!(run.stdout(), "init=500\nversion=500\nread=-1\n");
    assert!(run.stub_calls().is_empty(), "nothing reached the stub");

    let bundle = run.bundle().expect("bundle");
    let results: Vec<i32> = records(&bundle).iter().map(|r| r.result).collect();
    assert_eq!(
        results,
        [500, 500],
        "a failed call is still a recorded call"
    );
}
