//! Capture through the dispatch table: a target that resolves a `cu*` entry
//! point with `cuGetProcAddress` instead of linking it is recorded just the
//! same (ADR-0003).

use gputrace_core::{ArgValue, Argument, CallRecord, TraceBundle};
use gputrace_testkit::{CaptureRun, Harness, target_bin};

fn run(fixture: &str) -> CaptureRun {
    run_against(fixture, None)
}

/// Run `fixture`, optionally pointing the interposer at another backing
/// library than the stub driver.
fn run_against(fixture: &str, libcuda: Option<&str>) -> CaptureRun {
    let mut harness = Harness::new().expect("harness");
    if let Some(libcuda) = libcuda {
        harness = harness.with_env("GPUTRACE_LIBCUDA", libcuda);
    }
    let run = harness.run(target_bin(fixture), &[]).expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());
    run
}

/// One `key=value` line of what the fixture observed.
fn observed<'a>(run: &'a CaptureRun, key: &str) -> &'a str {
    run.stdout()
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("no {key} in: {}", run.stdout()))
}

fn records(bundle: &TraceBundle) -> Vec<CallRecord> {
    bundle
        .threads()
        .iter()
        .flat_map(|summary| bundle.read_segment(summary.thread).expect("segment"))
        .collect()
}

#[test]
fn a_call_resolved_through_the_dispatch_table_is_recorded() {
    let run = run("dispatch_table");

    assert_eq!(observed(&run, "lookup"), "0");
    assert_eq!(observed(&run, "status"), "0");
    assert_eq!(observed(&run, "call"), "0");
    assert_eq!(
        observed(&run, "version"),
        "13040",
        "the target must see the driver's own answer"
    );

    let bundle = run.bundle().expect("bundle");
    let record = records(&bundle)
        .into_iter()
        .find(|record| record.function == "cuDriverGetVersion")
        .expect("a cuDriverGetVersion record");
    assert_eq!(record.result, 0);
}

#[test]
fn the_cuda_11_arity_answers_with_the_same_trampoline() {
    let run = run("dispatch_table");

    assert_eq!(observed(&run, "v1"), "0");
    assert_eq!(observed(&run, "lookup_is_interposer"), "true");
    assert_eq!(
        observed(&run, "v1_agrees"),
        "true",
        "both arities the driver exports must be shimmed"
    );
}

#[test]
fn an_entry_point_the_interposer_does_not_shim_is_answered_by_the_driver() {
    let run = run("dispatch_table");

    assert_eq!(
        observed(&run, "unshimmed_agrees"),
        "true",
        "the dispatch table must answer what the exported-symbol path binds"
    );
    assert_eq!(observed(&run, "count_call"), "0");
    assert_eq!(observed(&run, "devices"), "1");
}

#[test]
fn a_lookup_the_driver_refuses_is_reported_as_the_driver_reported_it() {
    let run = run("dispatch_table");

    assert_eq!(observed(&run, "missing"), "500");
    assert_eq!(observed(&run, "missing_status"), "1");
    assert_eq!(observed(&run, "missing_null"), "true");

    assert_eq!(observed(&run, "too_old"), "500");
    assert_eq!(
        observed(&run, "too_old_status"),
        "2",
        "the driver's own reason must reach the caller"
    );
}

#[test]
fn resolving_the_dispatch_table_itself_answers_with_the_interposer() {
    let run = run("dispatch_table");

    assert_eq!(
        observed(&run, "dispatch_is_interposer"),
        "true",
        "a lookup of the base spelling must land on the interposer's own hook"
    );
}

#[test]
fn the_lookup_is_recorded_with_the_symbol_it_asked_for() {
    let run = run("dispatch_table");
    let bundle = run.bundle().expect("bundle");
    let records = records(&bundle);

    let lookup = records
        .iter()
        .find(|record| record.function == "cuGetProcAddress_v2")
        .expect("a cuGetProcAddress_v2 record");
    assert_eq!(lookup.result, 0);
    assert_eq!(
        lookup.args,
        [
            Argument::new("symbol", ArgValue::Str("cuDriverGetVersion".to_owned())),
            Argument::new("cudaVersion", ArgValue::Signed(13040)),
            Argument::new("flags", ArgValue::Unsigned(0)),
        ]
    );

    assert!(
        records
            .iter()
            .any(|record| record.function == "cuGetProcAddress"),
        "the CUDA 11 arity is recorded under its own name"
    );
}

#[test]
fn a_lookup_with_no_backing_driver_answers_without_a_pointer() {
    // A real library that exports no `cu*` at all, standing in for a driver
    // older than the dispatch table itself.
    let run = run_against("dispatch_table", Some("libc.so.6"));

    assert_eq!(observed(&run, "lookup"), "500");
    assert_eq!(observed(&run, "status"), "1");
    assert_eq!(
        observed(&run, "lookup_null"),
        "true",
        "the caller must not be handed an unwritten pointer"
    );
}
