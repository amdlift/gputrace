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
