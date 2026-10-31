use gputrace_core::{
    ArgValue, Argument, BundleWriter, CallRecord, CaptureConfig, Fidelity, HandleBinding, HandleId,
    HandleRole, PayloadRef, SCHEMA_VERSION, ThreadId, TraceBundle,
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

#[test]
fn identical_payloads_dedup_to_one_content_addressed_blob() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("trace");

    let writer = BundleWriter::create(&root, cuda_capture()).unwrap();
    let blobs = writer.blobs().clone();
    let first = blobs.put(b"h2d payload").unwrap();
    let again = blobs.put(b"h2d payload").unwrap();
    let other = blobs.put(b"ptx module image").unwrap();
    writer.finish().unwrap();

    assert_eq!(first, again, "same bytes must yield the same blob id");
    assert_ne!(first, other);

    let stored = std::fs::read_dir(root.join("blobs")).unwrap().count();
    assert_eq!(stored, 2, "identical payloads must share one file");

    let bundle = TraceBundle::open(&root).unwrap();
    assert_eq!(bundle.read_blob(&first).unwrap(), b"h2d payload");
    assert_eq!(bundle.read_blob(&other).unwrap(), b"ptx module image");
}

#[test]
fn manifest_is_human_readable_json_carrying_a_schema_version() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("trace");

    let writer = BundleWriter::create(
        &root,
        CaptureConfig {
            backend: "cuda".into(),
            fidelity: Fidelity::Golden,
            device: Some(0),
            command: vec!["./vecadd".into()],
        },
    )
    .unwrap();
    let mut segment = writer.open_segment(ThreadId(42)).unwrap();
    segment.write(&CallRecord::new("cuInit", 0, 1, 0)).unwrap();
    segment.finish().unwrap();
    writer.finish().unwrap();

    let text = std::fs::read_to_string(root.join("manifest.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();

    assert_eq!(json["schema_version"], SCHEMA_VERSION);
    assert_eq!(json["capture"]["backend"], "cuda");
    assert_eq!(json["capture"]["fidelity"], "golden");
    assert_eq!(json["capture"]["device"], 0);
    assert_eq!(json["threads"][0]["thread"], 42);
    assert_eq!(json["threads"][0]["records"], 1);
    assert_eq!(json["threads"][0]["segment"], "segments/thread-42.seg");

    assert!(
        text.contains("\n  \"schema_version\""),
        "manifest should be indented for reading by hand:\n{text}"
    );
}

#[test]
fn records_carry_arguments_handle_bindings_and_payload_refs() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("trace");

    let writer = BundleWriter::create(&root, cuda_capture()).unwrap();
    let payload: &[u8] = &[0x01, 0x02, 0x03, 0x04];
    let blob = writer.blobs().put(payload).unwrap();

    let record = CallRecord {
        args: vec![
            Argument::new("dstDevice", ArgValue::DevicePtr(0x7f00_0000)),
            Argument::new("ByteCount", ArgValue::Unsigned(4)),
            Argument::new("hStream", ArgValue::Handle(HandleId(9))),
        ],
        handles: vec![
            HandleBinding {
                id: HandleId(9),
                role: HandleRole::Consumed,
            },
            HandleBinding {
                id: HandleId(10),
                role: HandleRole::Produced,
            },
        ],
        payloads: vec![PayloadRef {
            role: "h2d".into(),
            blob: blob.clone(),
            len: payload.len() as u64,
        }],
        ..CallRecord::new("cuMemcpyHtoDAsync_v2", 5, 123_456, 0)
    };

    let mut segment = writer.open_segment(ThreadId(1)).unwrap();
    segment.write(&record).unwrap();
    segment.finish().unwrap();
    writer.finish().unwrap();

    let bundle = TraceBundle::open(&root).unwrap();
    let read = bundle.read_segment(ThreadId(1)).unwrap();
    assert_eq!(read, [record]);
    assert_eq!(
        bundle.read_blob(&read[0].payloads[0].blob).unwrap(),
        payload
    );
}

#[test]
fn many_threads_write_independent_segments_through_one_shared_blob_store() {
    const THREADS: u64 = 8;
    const CALLS: u64 = 50;

    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("trace");
    let writer = BundleWriter::create(&root, cuda_capture()).unwrap();

    std::thread::scope(|scope| {
        for thread in 0..THREADS {
            let writer = &writer;
            scope.spawn(move || {
                let shared = writer.blobs().put(b"module image shared by all").unwrap();
                let mut segment = writer.open_segment(ThreadId(thread)).unwrap();
                for seq in 0..CALLS {
                    let unique = writer
                        .blobs()
                        .put(format!("payload {thread}").as_bytes())
                        .unwrap();
                    segment
                        .write(&CallRecord {
                            payloads: vec![
                                PayloadRef {
                                    role: "module_image".into(),
                                    blob: shared.clone(),
                                    len: 26,
                                },
                                PayloadRef {
                                    role: "h2d".into(),
                                    blob: unique,
                                    len: 0,
                                },
                            ],
                            ..CallRecord::new("cuLaunchKernel", seq, seq, 0)
                        })
                        .unwrap();
                }
                segment.finish().unwrap();
            });
        }
    });
    writer.finish().unwrap();

    let bundle = TraceBundle::open(&root).unwrap();
    assert_eq!(bundle.threads().len() as u64, THREADS);
    for summary in bundle.threads() {
        assert_eq!(summary.records, CALLS);
        assert_eq!(
            bundle.read_segment(summary.thread).unwrap().len() as u64,
            CALLS
        );
    }

    let blobs: Vec<_> = std::fs::read_dir(root.join("blobs"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        blobs.len() as u64,
        THREADS + 1,
        "one shared module image plus one payload per thread, got {blobs:?}"
    );
    assert!(
        blobs.iter().all(|name| !name.starts_with('.')),
        "no staging files should survive: {blobs:?}"
    );
}
