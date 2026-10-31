//! The deep handlers for the memory and context subset: every produced
//! handle and device pointer carries a stable id, every consuming call names
//! it, and a host-to-device copy carries its payload into the blob store.

use gputrace_core::happens_before::{HappensBefore, OpId, Reason};
use gputrace_core::{ArgValue, BlobId, CallRecord, Fidelity, HandleId, HandleRole, TraceBundle};
use std::collections::{HashMap, HashSet};

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

/// What `shared_alloc` prints: the two device pointers it allocated.
fn reported_pointers(run: &CaptureRun) -> (u64, u64) {
    let mut lines = run.stdout().lines();
    let mut pointer = |label: &str| {
        let line = lines.next().unwrap_or_default();
        let value = line
            .strip_prefix(label)
            .unwrap_or_else(|| panic!("expected {label}…, got {line:?}"));
        u64::from_str_radix(value.trim_start_matches("0x"), 16).expect("device pointer")
    };
    (pointer("shared="), pointer("private="))
}

fn copies_into(records: &[CallRecord], pointer: u64) -> Vec<&CallRecord> {
    records
        .iter()
        .filter(|record| {
            record.function == "cuMemcpyHtoD_v2"
                && record
                    .args
                    .iter()
                    .any(|arg| arg.name == "dstDevice" && arg.value == ArgValue::DevicePtr(pointer))
        })
        .collect()
}

/// The identity rule that makes a trace replayable at all: an allocation is
/// one handle no matter which host thread touches it. Nothing coordinates
/// between those threads, so this is the test that would catch an id handed
/// out by a per-thread counter.
#[test]
fn threads_that_share_an_allocation_record_it_as_one_handle() {
    let run = run("shared_alloc");
    let (shared, private) = reported_pointers(&run);
    let bundle = run.bundle().expect("bundle");
    let records = records(&bundle);

    let produced = |pointer| {
        let alloc = records
            .iter()
            .find(|record| {
                record.function == "cuMemAlloc_v2"
                    && record
                        .args
                        .iter()
                        .any(|arg| arg.value == ArgValue::DevicePtr(pointer))
            })
            .expect("an allocation of this pointer");
        bound(alloc, HandleRole::Produced)
    };
    assert_ne!(
        produced(shared),
        produced(private),
        "two allocations are two handles"
    );

    let copies = copies_into(&records, shared);
    assert_eq!(copies.len(), 4, "one copy per worker thread");
    for copy in copies {
        assert_eq!(bound(copy, HandleRole::Consumed), produced(shared));
    }
}

#[test]
fn identical_payloads_are_stored_once_however_many_threads_sent_them() {
    let run = run("shared_alloc");
    let (shared, private) = reported_pointers(&run);
    let bundle = run.bundle().expect("bundle");
    let records = records(&bundle);

    let blobs_of = |pointer| {
        copies_into(&records, pointer)
            .into_iter()
            .map(|copy| payload(copy, "host_to_device"))
            .collect::<Vec<_>>()
    };

    let shared_blobs = blobs_of(shared);
    assert_eq!(shared_blobs.len(), 4);
    let (blob, len) = shared_blobs[0].clone();
    assert!(
        shared_blobs.iter().all(|sent| *sent == (blob.clone(), len)),
        "four threads sent the same 16 bytes: {shared_blobs:?}"
    );
    assert_eq!(bundle.read_blob(&blob).expect("blob"), [0xa5u8; 16]);

    let private_blobs: HashSet<BlobId> = blobs_of(private)
        .into_iter()
        .map(|(blob, _)| blob)
        .collect();
    assert_eq!(private_blobs.len(), 4, "four threads sent four payloads");

    let stored = std::fs::read_dir(bundle.path().join("blobs")).expect("blob store");
    assert_eq!(
        stored.count(),
        5,
        "the store holds one file per distinct payload"
    );
}

/// The whole point of tracking context lifecycle deeply: in a process using
/// more than one GPU, every record says which GPU its thread was on, even
/// when the call itself names no device.
#[test]
fn making_a_context_current_moves_the_thread_to_that_contexts_device() {
    let harness = Harness::new()
        .expect("harness")
        .with_env("GPUTRACE_STUB_DEVICES", "2");
    let run = harness
        .run(target_bin("ctx_stack"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());

    let bundle = run.bundle().expect("bundle");
    let tagged: Vec<(String, Option<u32>)> = records(&bundle)
        .iter()
        .map(|record| (record.function.clone(), record.device))
        .collect();

    let expected = [
        ("cuInit", None),
        ("cuDeviceGet", None),
        ("cuCtxCreate_v4", Some(0)),
        ("cuDeviceGet", Some(0)),
        ("cuCtxCreate_v4", Some(1)),
        ("cuDriverGetVersion", Some(1)),
        ("cuCtxPushCurrent_v2", Some(0)),
        ("cuDriverGetVersion", Some(0)),
        ("cuCtxPopCurrent_v2", Some(0)),
        ("cuDriverGetVersion", Some(1)),
        ("cuCtxSetCurrent", Some(0)),
        ("cuDriverGetVersion", Some(0)),
        ("cuCtxDestroy_v2", Some(0)),
        ("cuCtxSetCurrent", Some(1)),
        ("cuCtxDestroy_v2", Some(1)),
    ]
    .map(|(function, device): (&str, Option<u32>)| (function.to_owned(), device));

    assert_eq!(tagged, expected);
}

#[test]
fn a_context_is_one_handle_from_the_call_that_made_it_to_the_one_that_ends_it() {
    let harness = Harness::new()
        .expect("harness")
        .with_env("GPUTRACE_STUB_DEVICES", "2");
    let run = harness
        .run(target_bin("ctx_stack"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());

    let bundle = run.bundle().expect("bundle");
    let records = records(&bundle);
    let of = |function: &str, role, nth: usize| {
        bound(
            records
                .iter()
                .filter(|record| record.function == function)
                .nth(nth)
                .unwrap_or_else(|| panic!("no {nth}th {function}")),
            role,
        )
    };

    let first = of("cuCtxCreate_v4", HandleRole::Produced, 0);
    let second = of("cuCtxCreate_v4", HandleRole::Produced, 1);
    assert_ne!(first, second);

    assert_eq!(of("cuCtxPushCurrent_v2", HandleRole::Consumed, 0), first);
    assert_eq!(of("cuCtxPopCurrent_v2", HandleRole::Consumed, 0), first);
    assert_eq!(of("cuCtxSetCurrent", HandleRole::Consumed, 0), first);
    assert_eq!(of("cuCtxDestroy_v2", HandleRole::Destroyed, 0), first);
    assert_eq!(of("cuCtxSetCurrent", HandleRole::Consumed, 1), second);
    assert_eq!(of("cuCtxDestroy_v2", HandleRole::Destroyed, 1), second);
}

/// CUDA ships every call under several spellings and a target binds whichever
/// its headers chose. A deep handler that covered only the current one would
/// leave an older or stream-ordered build recording nothing a replay can use.
#[test]
fn an_older_spelling_of_a_call_records_exactly_as_the_current_one_does() {
    let run = run("legacy_spellings");
    let pointer = reported_pointer(&run);
    let bundle = run.bundle().expect("bundle");
    let records = records(&bundle);

    let allocation = bound(&record(&records, "cuMemAlloc"), HandleRole::Produced);
    assert_eq!(
        bound(&record(&records, "cuMemFree"), HandleRole::Destroyed),
        allocation
    );

    for copy in ["cuMemcpyHtoD", "cuMemcpyHtoD_v2_ptds"] {
        let copy = record(&records, copy);
        assert_eq!(arg(&copy, "dstDevice"), ArgValue::DevicePtr(pointer));
        assert_eq!(bound(&copy, HandleRole::Consumed), allocation);
        let (blob, len) = payload(&copy, "host_to_device");
        assert_eq!(len, 16);
        assert_eq!(bundle.read_blob(&blob).expect("blob"), [0x5au8; 16]);
    }

    let context = bound(&record(&records, "cuCtxCreate_v2"), HandleRole::Produced);
    assert_eq!(
        bound(&record(&records, "cuCtxDestroy"), HandleRole::Destroyed),
        context
    );
    assert_ne!(context, allocation);
}

/// What a stream/event fixture prints: the raw handle behind a label.
fn reported(run: &CaptureRun, label: &str) -> u64 {
    let line = run
        .stdout()
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{label}=")))
        .unwrap_or_else(|| panic!("the fixture reports {label}"));
    u64::from_str_radix(line.trim_start_matches("0x"), 16).expect("handle")
}

fn handle_arg(record: &CallRecord, name: &str) -> HandleId {
    match arg(record, name) {
        ArgValue::Handle(id) => id,
        other => panic!(
            "{} records {name} as {other:?}, not a handle",
            record.function
        ),
    }
}

fn bound_all(record: &CallRecord, role: HandleRole) -> HashSet<HandleId> {
    record
        .handles
        .iter()
        .filter(|binding| binding.role == role)
        .map(|binding| binding.id)
        .collect()
}

fn nth(records: &[CallRecord], function: &str, index: usize) -> CallRecord {
    records
        .iter()
        .filter(|record| record.function == function)
        .nth(index)
        .unwrap_or_else(|| panic!("no {index}th {function} record"))
        .clone()
}

/// Streams and events are the handles the happens-before order is made of:
/// every call that touches one has to name it, or the edge between two
/// streams cannot be found again offline.
#[test]
fn every_stream_and_event_call_names_the_handles_it_relates() {
    let run = run("stream_event");
    let (producer, consumer) = (reported(&run, "producer"), reported(&run, "consumer"));
    let bundle = run.bundle().expect("bundle");
    let records = records(&bundle);

    let created = |index| {
        let create = nth(&records, "cuStreamCreate", index);
        let id = handle_arg(&create, "hStream");
        assert_eq!(
            bound_all(&create, HandleRole::Produced),
            HashSet::from([id])
        );
        id
    };
    let (produced, consumed) = (created(0), created(1));
    assert_ne!(produced, consumed, "{producer:#x} and {consumer:#x}");

    let start = handle_arg(&nth(&records, "cuEventCreate", 0), "hEvent");
    let finish = handle_arg(&nth(&records, "cuEventCreate", 1), "hEvent");
    assert_ne!(start, finish);

    let record_start = nth(&records, "cuEventRecord", 0);
    assert_eq!(handle_arg(&record_start, "hEvent"), start);
    assert_eq!(handle_arg(&record_start, "hStream"), produced);
    assert_eq!(
        bound_all(&record_start, HandleRole::Consumed),
        HashSet::from([start, produced])
    );

    let wait = record(&records, "cuStreamWaitEvent");
    assert_eq!(handle_arg(&wait, "hStream"), consumed);
    assert_eq!(handle_arg(&wait, "hEvent"), start);
    assert_eq!(
        bound_all(&wait, HandleRole::Consumed),
        HashSet::from([start, consumed])
    );

    assert_eq!(
        handle_arg(&nth(&records, "cuStreamSynchronize", 0), "hStream"),
        produced
    );
    assert_eq!(
        handle_arg(&record(&records, "cuEventSynchronize"), "hEvent"),
        finish
    );
    assert_eq!(
        bound_all(
            &record(&records, "cuEventDestroy_v2"),
            HandleRole::Destroyed
        ),
        HashSet::from([finish])
    );
    assert_eq!(
        bound_all(
            &record(&records, "cuStreamDestroy_v2"),
            HandleRole::Destroyed
        ),
        HashSet::from([consumed])
    );
}

/// The acceptance criterion of a capture that enforces no order (ADR-0004):
/// a known edge between two streams must come back out of the bundle. The
/// producer's copy *after* the event is the control — it is the one the
/// trace must refuse to order against the consumer.
#[test]
fn a_known_cross_stream_edge_is_reconstructable_from_the_bundle() {
    let run = run("cross_stream");
    assert_cross_stream_edge(
        &run,
        "cuMemcpyHtoDAsync_v2",
        "cuEventRecord",
        "cuStreamWaitEvent",
    );
}

/// CUDA ships each of these calls under a per-thread-default-stream
/// spelling too, and a target built with `--default-stream per-thread`
/// binds those instead. The same scenario through those spellings has to
/// reconstruct the same edge, or half the builds in the world capture an
/// order nobody can recover.
#[test]
fn the_per_thread_stream_spellings_reconstruct_the_same_edge() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("cross_stream"), &["ptsz"])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());
    assert_cross_stream_edge(
        &run,
        "cuMemcpyHtoDAsync_v2_ptsz",
        "cuEventRecord_ptsz",
        "cuStreamWaitEvent_ptsz",
    );
}

fn assert_cross_stream_edge(run: &CaptureRun, copy: &str, event_record: &str, wait: &str) {
    let bundle = run.bundle().expect("bundle");
    let order = HappensBefore::reconstruct(&bundle).expect("reconstruct the order");

    let copies: HashMap<u8, OpId> = records(&bundle)
        .iter()
        .filter(|record| record.function == copy)
        .map(|record| {
            let (blob, len) = payload(record, "host_to_device");
            let sent = bundle.read_blob(&blob).expect("blob");
            assert_eq!(len as usize, sent.len());
            let op = order
                .ops()
                .iter()
                .find(|op| op.function == record.function && op.thread_seq == record.thread_seq)
                .unwrap_or_else(|| panic!("no stream operation for {record:?}"));
            (sent[0], op.id)
        })
        .collect();
    assert_eq!(copies.len(), 3, "one copy per fill byte");
    let (before, after, waiter) = (copies[&0x11], copies[&0x22], copies[&0x33]);

    assert!(
        order.happens_before(before, waiter),
        "the event the consumer waited on was recorded after this copy"
    );
    assert!(
        !order.happens_before(after, waiter),
        "this copy was issued after the event was recorded, so nothing orders it"
    );

    let crossing: Vec<_> = order
        .edges()
        .iter()
        .filter(|edge| matches!(edge.reason, Reason::Event(_)))
        .collect();
    assert_eq!(crossing.len(), 1, "one event joins the two streams");
    let (record_op, wait_op) = (
        &order.ops()[crossing[0].from.0],
        &order.ops()[crossing[0].to.0],
    );
    assert_eq!(record_op.function, event_record);
    assert_eq!(wait_op.function, wait);
    assert_ne!(record_op.stream, wait_op.stream, "the edge crosses streams");
    assert_eq!(
        crossing[0].reason,
        Reason::Event(record_op.event.expect("an event"))
    );
}

/// What `module_launch` loads. The image is nul-terminated PTX, and the
/// ABI hands the driver nothing but that pointer, so the capture is only
/// right if it read exactly as far as the format says.
const PTX: &[u8] = b".version 8.0\n.target sm_86\n.visible .entry kernel_a() { ret; }\n.visible .entry kernel_b() { ret; }\n";

#[test]
fn a_loaded_module_image_is_captured_as_a_blob_the_load_record_names() {
    let run = run("module_launch");
    let bundle = run.bundle().expect("bundle");
    let records = records(&bundle);

    let load = record(&records, "cuModuleLoadData");
    let (blob, len) = payload(&load, "module_image");
    assert_eq!(len, PTX.len() as u64);
    assert_eq!(bundle.read_blob(&blob).expect("blob"), PTX);
}

/// A kernel is reached by name through the module that defines it, and the
/// name is the only thing in the call a replay could look the kernel up by.
#[test]
fn a_kernel_lookup_names_its_module_and_the_name_it_asked_for() {
    let run = run("module_launch");
    let bundle = run.bundle().expect("bundle");
    let records = records(&bundle);

    let module = bound(&record(&records, "cuModuleLoadData"), HandleRole::Produced);
    let looked_up = |index: usize| {
        let lookup = nth(&records, "cuModuleGetFunction", index);
        assert_eq!(handle_arg(&lookup, "hmod"), module);
        (arg(&lookup, "name"), lookup)
    };

    let (first_name, first) = looked_up(0);
    let (again_name, again) = looked_up(1);
    let (other_name, other) = looked_up(2);
    assert_eq!(first_name, ArgValue::Str("kernel_a".to_owned()));
    assert_eq!(again_name, ArgValue::Str("kernel_a".to_owned()));
    assert_eq!(other_name, ArgValue::Str("kernel_b".to_owned()));

    let kernel_a = bound(&first, HandleRole::Produced);
    assert_eq!(bound(&again, HandleRole::Produced), kernel_a);
    assert_ne!(bound(&other, HandleRole::Produced), kernel_a);
    assert_ne!(kernel_a, module, "a function is not its module");

    let missing = looked_up(3).1;
    assert_ne!(
        missing.result, 0,
        "the fixture asks for a kernel that is absent"
    );
    assert!(
        missing
            .handles
            .iter()
            .all(|binding| binding.role != HandleRole::Produced)
    );

    assert_eq!(
        bound(&record(&records, "cuModuleUnload"), HandleRole::Destroyed),
        module
    );
}

/// What `launch_params` packs into its `extra` buffer.
const PACKED: [u8; 12] = [7, 0, 0, 0, 0, 0, 0xc0, 0x3f, 0xef, 0xbe, 0xad, 0xde];

#[test]
fn a_launch_records_its_configuration_and_the_kernel_it_ran() {
    let run = run("launch_params");
    let bundle = run.bundle().expect("bundle");
    let records = records(&bundle);

    let kernel = bound(
        &record(&records, "cuModuleGetFunction"),
        HandleRole::Produced,
    );
    let stream = handle_arg(&record(&records, "cuStreamCreate"), "hStream");

    let launch = nth(&records, "cuLaunchKernel", 0);
    assert_eq!(handle_arg(&launch, "f"), kernel);
    assert_eq!(handle_arg(&launch, "hStream"), stream);
    assert_eq!(
        bound_all(&launch, HandleRole::Consumed),
        HashSet::from([kernel, stream])
    );
    for (name, value) in [
        ("gridDimX", 4),
        ("gridDimY", 2),
        ("gridDimZ", 1),
        ("blockDimX", 64),
        ("blockDimY", 8),
        ("blockDimZ", 1),
        ("sharedMemBytes", 256),
    ] {
        assert_eq!(arg(&launch, name), ArgValue::Unsigned(value), "{name}");
    }
}

/// The `extra` packing states the size of the argument buffer, so the whole
/// buffer is recoverable byte for byte.
#[test]
fn a_launch_packed_through_extra_carries_its_parameter_buffer() {
    let run = run("launch_params");
    let bundle = run.bundle().expect("bundle");

    let launch = nth(&records(&bundle), "cuLaunchKernel", 0);
    let (blob, len) = payload(&launch, "kernel_params");
    assert_eq!(len, PACKED.len() as u64);
    assert_eq!(bundle.read_blob(&blob).expect("blob"), PACKED);
}

/// `kernelParams` is a bare `void**`: no count, no element sizes, nothing
/// at the ABI level that says how far it is safe to read. Capture records
/// that the launch used it and refuses to invent a size.
#[test]
fn a_launch_through_the_bare_pointer_array_records_it_without_a_payload() {
    let run = run("launch_params");
    let array = reported(&run, "kernel_params");
    let bundle = run.bundle().expect("bundle");

    let launch = nth(&records(&bundle), "cuLaunchKernel", 1);
    assert_eq!(arg(&launch, "kernelParams"), ArgValue::HostPtr(array));
    assert_eq!(arg(&launch, "extra"), ArgValue::HostPtr(0));
    assert!(
        launch.payloads.is_empty(),
        "no size is recoverable, so nothing is stored"
    );
}
