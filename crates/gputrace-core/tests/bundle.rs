use gputrace_core::{BundleWriter, CaptureConfig, Fidelity, SCHEMA_VERSION, TraceBundle};

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
            ..Default::default()
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
