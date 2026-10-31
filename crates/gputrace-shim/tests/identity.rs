//! What a captured `HandleId` does and does not say about identity.
//!
//! The id is derived from the raw value the driver handed out, which is what
//! lets threads agree on it with no shared map (ADR-0004) — and what makes it
//! ambiguous the moment the driver hands the same address out twice. These
//! tests pin both halves: the collision in the trace, and the offline
//! resolver that separates the two objects behind it.

use gputrace_core::identity::{CallSite, HandleIdentities, Resolution};
use gputrace_core::{ArgValue, CallRecord, HandleRole, ThreadId, TraceBundle};
use gputrace_testkit::{CaptureRun, Harness, target_bin};

/// The stub only reuses a freed address when asked, so every other fixture
/// keeps the strictly increasing pointers it has always seen.
fn reusing_run(fixture: &str) -> CaptureRun {
    let harness = Harness::new()
        .expect("harness")
        .with_env("GPUTRACE_STUB_REUSE_ADDRESSES", "1");
    let run = harness.run(target_bin(fixture), &[]).expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());
    run
}

fn segments(bundle: &TraceBundle) -> Vec<(ThreadId, Vec<CallRecord>)> {
    bundle
        .threads()
        .iter()
        .map(|summary| {
            (
                summary.thread,
                bundle.read_segment(summary.thread).expect("segment"),
            )
        })
        .collect()
}

fn reported_pointers(run: &CaptureRun) -> Vec<u64> {
    run.stdout()
        .lines()
        .filter_map(|line| line.strip_prefix("device_ptr=0x"))
        .map(|hex| u64::from_str_radix(hex, 16).expect("device pointer"))
        .collect()
}

fn bindings(
    segments: &[(ThreadId, Vec<CallRecord>)],
    function: &str,
    role: HandleRole,
) -> Vec<(CallSite, gputrace_core::HandleId)> {
    segments
        .iter()
        .flat_map(|(thread, records)| {
            records
                .iter()
                .filter(|record| record.function == function)
                .flat_map(move |record| {
                    record
                        .handles
                        .iter()
                        .filter(move |binding| binding.role == role)
                        .map(move |binding| (CallSite::new(*thread, record.thread_seq), binding.id))
                })
        })
        .collect()
}

#[test]
fn a_reused_device_address_records_one_handle_id_for_two_allocations() {
    let run = reusing_run("reused_pointer");
    let pointers = reported_pointers(&run);
    assert_eq!(
        pointers.len(),
        2,
        "the fixture allocates twice: {}",
        run.stdout()
    );
    assert_eq!(
        pointers[0], pointers[1],
        "the stub must hand the freed address back for this to be the case under test"
    );

    let bundle = run.bundle().expect("bundle");
    let produced = bindings(&segments(&bundle), "cuMemAlloc_v2", HandleRole::Produced);
    assert_eq!(produced.len(), 2);
    assert_eq!(
        produced[0].1, produced[1].1,
        "the id is derived from the raw address, so two objects at one address share it"
    );
}

#[test]
fn the_two_allocations_at_one_address_resolve_to_two_objects() {
    let run = reusing_run("reused_pointer");
    let bundle = run.bundle().expect("bundle");
    let segments = segments(&bundle);
    let identities = HandleIdentities::from_segments(segments.clone());

    let copies = bindings(&segments, "cuMemcpyHtoD_v2", HandleRole::Consumed);
    assert_eq!(copies.len(), 2, "the fixture copies into each allocation");

    let objects: Vec<_> = copies
        .iter()
        .map(|&(site, handle)| match identities.object(site, handle) {
            Some(Resolution::Resolved(object)) => *object,
            other => panic!("one host thread totally orders these calls: {other:?}"),
        })
        .collect();

    assert_ne!(
        objects[0], objects[1],
        "a replay remapping by id alone would send the second copy to the first allocation"
    );
    assert_eq!(identities.lifetimes(copies[0].1), 2);
}

#[test]
fn each_copy_carries_the_payload_of_the_allocation_it_named() {
    let run = reusing_run("reused_pointer");
    let bundle = run.bundle().expect("bundle");
    let copies: Vec<_> = segments(&bundle)
        .into_iter()
        .flat_map(|(_, records)| records)
        .filter(|record| record.function == "cuMemcpyHtoD_v2")
        .collect();
    assert_eq!(copies.len(), 2);

    let payload = |record: &CallRecord| {
        let blob = &record
            .payloads
            .iter()
            .find(|payload| payload.role == "host_to_device")
            .expect("host_to_device payload")
            .blob;
        bundle.blobs().get(blob).expect("blob")
    };
    assert_eq!(payload(&copies[0]), [0x11; 8]);
    assert_eq!(payload(&copies[1]), [0x22; 8]);

    assert!(matches!(
        copies[0]
            .args
            .iter()
            .find(|arg| arg.name == "dstDevice")
            .map(|arg| &arg.value),
        Some(ArgValue::DevicePtr(_))
    ));
}
