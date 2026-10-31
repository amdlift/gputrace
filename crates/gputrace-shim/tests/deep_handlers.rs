//! The deep handlers for the memory and context subset: every produced
//! handle and device pointer carries a stable id, every consuming call names
//! it, and a host-to-device copy carries its payload into the blob store.

use gputrace_core::{ArgValue, BlobId, CallRecord, Fidelity, HandleId, HandleRole, TraceBundle};
use gputrace_testkit::{CaptureRun, Harness, target_bin};

fn records(bundle: &TraceBundle) -> Vec<CallRecord> {
    bundle
        .threads()
        .iter()
        .flat_map(|summary| bundle.read_segment(summary.thread).expect("segment"))
        .collect()
}

fn run(fixture: &str) -> CaptureRun {
    let harness = Harness::new().expect("harness");
    let run = harness.run(target_bin(fixture), &[]).expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());
    run
}

fn record(records: &[CallRecord], function: &str) -> CallRecord {
    records
        .iter()
        .find(|record| record.function == function)
        .unwrap_or_else(|| panic!("no {function} record"))
        .clone()
}

fn arg(record: &CallRecord, name: &str) -> ArgValue {
    record
        .args
        .iter()
        .find(|arg| arg.name == name)
        .unwrap_or_else(|| panic!("no {name} argument on {}", record.function))
        .value
        .clone()
}

fn bound(record: &CallRecord, role: HandleRole) -> HandleId {
    let mut matching = record
        .handles
        .iter()
        .filter(|binding| binding.role == role)
        .map(|binding| binding.id);
    let id = matching
        .next()
        .unwrap_or_else(|| panic!("{} binds no {role:?} handle", record.function));
    assert!(matching.next().is_none(), "{}", record.function);
    id
}

fn payload(record: &CallRecord, role: &str) -> (BlobId, u64) {
    let payload = record
        .payloads
        .iter()
        .find(|payload| payload.role == role)
        .unwrap_or_else(|| panic!("{} carries no {role} payload", record.function));
    (payload.blob.clone(), payload.len)
}

/// What `alloc_copy` prints: the device pointer the stub handed it.
fn reported_pointer(run: &CaptureRun) -> u64 {
    let line = run
        .stdout()
        .lines()
        .find_map(|line| line.strip_prefix("device_ptr="))
        .expect("the fixture reports its device pointer");
    u64::from_str_radix(line.trim_start_matches("0x"), 16).expect("device pointer")
}

const SENT: [u8; 8] = [0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03, 0x04];

#[test]
fn an_allocation_and_every_call_that_uses_it_name_one_handle() {
    let run = run("alloc_copy");
    let pointer = reported_pointer(&run);
    let bundle = run.bundle().expect("bundle");
    let records = records(&bundle);

    let alloc = record(&records, "cuMemAlloc_v2");
    assert_eq!(arg(&alloc, "dptr"), ArgValue::DevicePtr(pointer));
    assert_eq!(arg(&alloc, "bytesize"), ArgValue::Unsigned(8));
    let id = bound(&alloc, HandleRole::Produced);

    let copy = record(&records, "cuMemcpyHtoD_v2");
    assert_eq!(arg(&copy, "dstDevice"), ArgValue::DevicePtr(pointer));
    assert_eq!(bound(&copy, HandleRole::Consumed), id);

    let free = record(&records, "cuMemFree_v2");
    assert_eq!(arg(&free, "dptr"), ArgValue::DevicePtr(pointer));
    assert_eq!(bound(&free, HandleRole::Destroyed), id);
}

#[test]
fn a_host_to_device_copy_carries_its_payload_into_the_blob_store() {
    let run = run("alloc_copy");
    let bundle = run.bundle().expect("bundle");

    assert_eq!(bundle.manifest().capture.fidelity, Fidelity::Replayable);

    let copy = record(&records(&bundle), "cuMemcpyHtoD_v2");
    let (blob, len) = payload(&copy, "host_to_device");
    assert_eq!(len, SENT.len() as u64);
    assert_eq!(bundle.read_blob(&blob).expect("blob"), SENT);
}

/// Fidelity is what a deep handler costs: identity is cheap, a payload is a
/// copy of every buffer the target sends. A capture that asked for neither
/// must get neither.
#[test]
fn a_metadata_capture_records_the_call_without_its_identity_or_payload() {
    let harness = Harness::new()
        .expect("harness")
        .with_env("GPUTRACE_FIDELITY", "metadata");
    let run = harness
        .run(target_bin("alloc_copy"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());

    let bundle = run.bundle().expect("bundle");
    let records = records(&bundle);
    for function in ["cuMemAlloc_v2", "cuMemcpyHtoD_v2", "cuMemFree_v2"] {
        let record = record(&records, function);
        assert_eq!(record.result, 0);
        assert!(record.handles.is_empty(), "{function}");
        assert!(record.payloads.is_empty(), "{function}");
    }
    let blobs = std::fs::read_dir(bundle.path().join("blobs")).expect("blob store");
    assert_eq!(blobs.count(), 0, "nothing should have been stored");
}

#[test]
fn a_handles_capture_records_identity_without_the_payload() {
    let harness = Harness::new()
        .expect("harness")
        .with_env("GPUTRACE_FIDELITY", "handles");
    let run = harness
        .run(target_bin("alloc_copy"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());

    let records = records(&run.bundle().expect("bundle"));
    let copy = record(&records, "cuMemcpyHtoD_v2");
    assert_eq!(
        bound(&copy, HandleRole::Consumed),
        bound(&record(&records, "cuMemAlloc_v2"), HandleRole::Produced)
    );
    assert!(copy.payloads.is_empty());
}
