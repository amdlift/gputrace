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

#[test]
fn every_captured_thread_gets_its_own_segment_listed_in_the_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("trace");

    let writer = BundleWriter::create(&root, cuda_capture()).unwrap();
    for (thread, calls) in [(ThreadId(11), 3u64), (ThreadId(12), 1), (ThreadId(13), 5)] {
        let mut segment = writer.open_segment(thread).unwrap();
        for seq in 0..calls {
            segment
                .write(&CallRecord::new(
                    format!("cuCall_t{}_{seq}", thread.0),
                    seq,
                    seq * 10,
                    0,
                ))
                .unwrap();
        }
        segment.finish().unwrap();
    }
    writer.finish().unwrap();

    let bundle = TraceBundle::open(&root).unwrap();
    let mut threads: Vec<_> = bundle.threads().to_vec();
    threads.sort_by_key(|t| t.thread);

    assert_eq!(
        threads.iter().map(|t| t.thread).collect::<Vec<_>>(),
        [ThreadId(11), ThreadId(12), ThreadId(13)]
    );
    assert_eq!(
        threads.iter().map(|t| t.records).collect::<Vec<_>>(),
        [3, 1, 5]
    );

    let segments: std::collections::BTreeSet<_> =
        threads.iter().map(|t| t.segment.clone()).collect();
    assert_eq!(segments.len(), 3, "each thread needs its own segment file");
    for segment in &segments {
        assert!(root.join(segment).is_file(), "missing segment {segment}");
    }

    for summary in &threads {
        let records = bundle.read_segment(summary.thread).unwrap();
        assert_eq!(records.len() as u64, summary.records);
        assert!(records.iter().all(|r| {
            r.function
                .starts_with(&format!("cuCall_t{}_", summary.thread.0))
        }));
    }
}
