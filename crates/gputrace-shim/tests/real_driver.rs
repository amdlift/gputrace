//! The non-gating smoke test: one capture against a real `libcuda.so.1` on a
//! machine with a GPU.
//!
//! Every other capture test drives the stub, which is what makes the suite
//! deterministic and runnable with no GPU — and what leaves one thing
//! unproven, that the interposer forwards to the driver NVIDIA actually
//! ships. This covers that, and is `#[ignore]`d so it never runs in GPU-less
//! CI. See `docs/real-gpu-smoke.md` for how to run it.

use std::path::PathBuf;

use gputrace_core::{CallRecord, TraceBundle};
use gputrace_testkit::{Harness, target_bin};

/// Where the real driver is, overridable because a distribution that does not
/// put `libcuda.so.1` on the loader's search path (NixOS, containers) is
/// common enough to be worth one environment variable.
const REAL_LIBCUDA_ENV: &str = "GPUTRACE_REAL_LIBCUDA";
const CANDIDATES: [&str; 3] = [
    "/run/opengl-driver/lib/libcuda.so.1",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib64/libcuda.so.1",
];

fn real_libcuda() -> PathBuf {
    if let Some(named) = std::env::var_os(REAL_LIBCUDA_ENV) {
        return PathBuf::from(named);
    }
    CANDIDATES
        .iter()
        .map(PathBuf::from)
        .find(|path| path.exists())
        .unwrap_or_else(|| PathBuf::from("libcuda.so.1"))
}

fn records(bundle: &TraceBundle) -> Vec<CallRecord> {
    bundle
        .threads()
        .iter()
        .flat_map(|summary| bundle.read_segment(summary.thread).expect("segment"))
        .collect()
}

#[test]
#[ignore = "needs a real NVIDIA driver and a GPU; see docs/real-gpu-smoke.md"]
fn capturing_a_real_cuda_program_produces_a_bundle() {
    let harness = Harness::new().expect("harness").against(real_libcuda());
    let run = harness
        .run(target_bin("alloc_copy"), &[])
        .expect("run target");
    assert!(
        run.status().success(),
        "the target must still run normally under capture: {}",
        run.stderr()
    );
    assert!(
        run.stdout().contains("received=deadbeef01020304"),
        "the real driver must have round-tripped the buffer: {}",
        run.stdout()
    );

    let bundle = run.bundle().expect("a bundle");
    let records = records(&bundle);
    for expected in [
        "cuInit",
        "cuCtxCreate_v4",
        "cuMemAlloc_v2",
        "cuMemcpyHtoD_v2",
        "cuMemcpyDtoH_v2",
        "cuMemFree_v2",
    ] {
        assert!(
            records.iter().any(|record| record.function == expected),
            "no {expected} in a bundle of {} records",
            records.len()
        );
    }

    let environment = &bundle.manifest().environment;
    assert!(
        environment
            .driver_version
            .is_some_and(|version| version > 0),
        "the manifest must record the real driver's version: {environment:?}"
    );
    assert!(
        environment
            .devices
            .iter()
            .any(|device| device.compute_capability.is_some()),
        "the manifest must record a real GPU: {environment:?}"
    );
}
