//! The third call path ADR-0003 did not close: a target that `dlopen`s the
//! driver and `dlsym`s its handle reaches the real entry point directly, and
//! nothing about that lookup passes through the interposer (#13, ADR-0007).
//!
//! The stub is left out of `LD_PRELOAD` here, so it is loaded the way a real
//! `libcuda.so.1` is — by the target's own `dlopen`, into its own scope. The
//! escape is the dynamic linker's scoping rule, not an artefact of preloading
//! the stub beside the interposer.
//!
//! These tests pin the escape as it is today. They are the regression guard
//! that will have to change if we ever close the path.

use gputrace_testkit::{CaptureRun, Harness, stub_libcuda, target_bin};

fn run() -> CaptureRun {
    let harness = Harness::new()
        .expect("harness")
        .against(stub_libcuda().expect("stub libcuda"));
    let run = harness
        .run(target_bin("dlopened_driver"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());
    run
}

#[test]
fn a_dlsym_on_the_drivers_own_handle_hands_back_the_driver_not_the_interposer() {
    let run = run();

    assert_eq!(run.observed("dlopen"), "true");
    assert_eq!(run.observed("dlsym"), "true");
    assert_eq!(
        run.observed("dlsym_is_interposer"),
        "false",
        "a lookup against a library handle searches that object, so LD_PRELOAD \
         has nothing to shadow"
    );
}

#[test]
fn the_call_reaches_the_driver_and_is_not_recorded() {
    let run = run();

    assert_eq!(run.observed("call"), "0");
    assert_eq!(
        run.observed("devices"),
        "1",
        "the target got the driver's own answer"
    );
    assert_eq!(
        run.stub_call_count("cuDeviceGetCount"),
        1,
        "the driver answered the escaped call"
    );

    let records = run.records().expect("records");
    assert!(
        records.iter().any(|record| record.function == "cuInit"),
        "the same target's global-scope call is captured, so capture was live: {records:?}"
    );
    assert!(
        !records
            .iter()
            .any(|record| record.function == "cuDeviceGetCount"),
        "the escaped call is absent from the bundle: {records:?}"
    );
}
