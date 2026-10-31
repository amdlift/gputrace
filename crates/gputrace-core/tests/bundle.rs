use gputrace_core::{
    BundleWriter, CallRecord, CaptureConfig, Fidelity, SCHEMA_VERSION, ThreadId, TraceBundle,
};

#[test]
fn manifest_round_trips_through_a_written_bundle() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("trace");

    let writer = BundleWriter::create(
        &root,
        CaptureConfig {
            backend: "cuda".into(),
            fidelity: Fidelity::Replayable,
            device: Some(1),
            command: vec!["./vecadd".into(), "--n=1024".into()],
        },
    )
    .unwrap();
    writer.finish().unwrap();

    let bundle = TraceBundle::open(&root).unwrap();
    let manifest = bundle.manifest();

    assert_eq!(manifest.schema_version, SCHEMA_VERSION);
    assert_eq!(manifest.capture.backend, "cuda");
    assert_eq!(manifest.capture.fidelity, Fidelity::Replayable);
    assert_eq!(manifest.capture.device, Some(1));
    assert_eq!(manifest.capture.command, ["./vecadd", "--n=1024"]);
}

#[test]
fn call_records_round_trip_through_a_per_thread_segment() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("trace");

    let writer = BundleWriter::create(&root, cuda_capture()).unwrap();
    let records = vec![
        CallRecord::new("cuInit", 0, 1_000, 0),
        CallRecord::new("cuMemAlloc_v2", 1, 2_500, 0),
        CallRecord::new("cuMemFree_v2", 2, 9_750, 201),
    ];
    let mut segment = writer.open_segment(ThreadId(7)).unwrap();
    for record in &records {
        segment.write(record).unwrap();
    }
    segment.finish().unwrap();
    writer.finish().unwrap();

    let bundle = TraceBundle::open(&root).unwrap();
    assert_eq!(bundle.read_segment(ThreadId(7)).unwrap(), records);
}

fn cuda_capture() -> CaptureConfig {
    CaptureConfig {
        backend: "cuda".into(),
        ..Default::default()
    }
}
