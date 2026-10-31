//! The concurrency guarantees of the capture record path (ADR-0004): one
//! segment per host thread with correct attribution, per-thread counters and
//! monotonic stamps instead of a global order, device tagging, and a flush
//! that does not depend on a thread having exited first.

use gputrace_core::{CallRecord, ThreadId, TraceBundle};
use gputrace_testkit::{Harness, target_bin};

fn segments(bundle: &TraceBundle) -> Vec<(ThreadId, Vec<CallRecord>)> {
    bundle
        .threads()
        .iter()
        .map(|summary| {
            let records = bundle.read_segment(summary.thread).expect("segment");
            assert_eq!(
                summary.records as usize,
                records.len(),
                "the manifest must count what the segment holds"
            );
            (summary.thread, records)
        })
        .collect()
}

#[test]
fn a_thread_still_alive_at_exit_flushes_its_buffered_tail() {
    const THREADS: usize = 4;
    const CALLS_PER_THREAD: usize = 300;

    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("capture_exit_tail"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());
    assert_eq!(run.stdout(), "parked=4\n");

    let bundle = run.bundle().expect("bundle");
    let segments = segments(&bundle);
    assert_eq!(segments.len(), THREADS + 1, "one segment per thread");

    let mut parked = Vec::new();
    for (thread, records) in &segments {
        match records.first().map(|record| record.function.as_str()) {
            Some("cuInit") => assert_eq!(records.len(), 1, "the main thread, {thread:?}"),
            _ => parked.push(records.len()),
        }
    }
    assert_eq!(
        parked, [CALLS_PER_THREAD; THREADS],
        "a parked thread's partial buffer must still reach its segment"
    );
}
