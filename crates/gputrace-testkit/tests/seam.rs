use gputrace_testkit::{Harness, target_bin};

#[test]
fn a_targets_calls_reach_the_stub_driver() {
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
        ]
    );
}

#[test]
fn the_stub_answers_deterministically() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("stub_smoke"), &[])
        .expect("run target");

    assert_eq!(
        run.stdout(),
        "driver=13040\ndevices=1\nname=gputrace-stub-device-0\n"
    );
}

#[test]
fn the_stub_round_trips_a_copy_through_device_memory() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("alloc_copy"), &[])
        .expect("run target");

    assert!(run.status().success(), "target failed: {}", run.stderr());
    assert_eq!(
        run.stdout(),
        "device_ptr=0x10000000\nreceived=deadbeef01020304\n"
    );
    assert_eq!(
        run.stub_calls(),
        [
            "cuInit",
            "cuDeviceGet",
            "cuCtxCreate_v4",
            "cuMemAlloc_v2",
            "cuMemcpyHtoD_v2",
            "cuMemcpyDtoH_v2",
            "cuMemFree_v2",
            "cuCtxDestroy_v2",
        ]
    );
}
