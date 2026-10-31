//! The concurrency guarantees of the capture record path (ADR-0004): one
//! segment per host thread with correct attribution, per-thread counters and
//! monotonic stamps instead of a global order, device tagging, and a flush
//! that does not depend on a thread having exited first.

use std::collections::HashMap;

use gputrace_core::{CallRecord, ThreadId, TraceBundle};
use gputrace_testkit::{CaptureRun, Harness, target_bin};

/// Mirrors `capture_interleaved`: four threads, each round of calls closed by
/// a barrier, each thread ending on a different total.
const THREADS: usize = 4;
const ROUNDS: usize = 32;

/// The call total each worker thread reports for its own OS thread id.
fn reported_totals(run: &CaptureRun) -> HashMap<u64, usize> {
    let totals: HashMap<u64, usize> = run
        .stdout()
        .lines()
        .map(|line| {
            let (tid, calls) = line.split_once(' ').expect("tid and calls");
            (
                tid.trim_start_matches("tid=").parse().expect("tid"),
                calls.trim_start_matches("calls=").parse().expect("calls"),
            )
        })
        .collect();
    assert_eq!(totals.len(), THREADS, "{}", run.stdout());
    totals
}

fn interleaved_run(harness: &Harness) -> CaptureRun {
    let run = harness
        .run(target_bin("capture_interleaved"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());
    run
}

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

#[test]
fn every_concurrent_thread_gets_its_own_segment_with_its_own_calls() {
    let harness = Harness::new().expect("harness");
    let run = interleaved_run(&harness);
    let totals = reported_totals(&run);

    let bundle = run.bundle().expect("bundle");
    let segments = segments(&bundle);
    assert_eq!(segments.len(), THREADS + 1, "one segment per host thread");

    let mut attributed = 0;
    for (thread, records) in &segments {
        match totals.get(&thread.0) {
            Some(&calls) => {
                attributed += 1;
                assert_eq!(records.len(), calls, "worker {thread:?}");
                assert!(
                    records
                        .iter()
                        .all(|record| record.function == "cuDriverGetVersion"),
                    "worker {thread:?} picked up another thread's calls"
                );
            }
            None => assert_eq!(
                records
                    .iter()
                    .map(|record| record.function.as_str())
                    .collect::<Vec<_>>(),
                ["cuInit"],
                "the main thread's segment"
            ),
        }
    }
    assert_eq!(
        attributed, THREADS,
        "every reported OS thread id must name a segment"
    );
}

#[test]
fn each_thread_counts_from_zero_and_nothing_stamps_a_global_order() {
    let harness = Harness::new().expect("harness");
    let run = interleaved_run(&harness);

    let bundle = run.bundle().expect("bundle");
    let segments = segments(&bundle);

    for (thread, records) in &segments {
        assert_eq!(
            records
                .iter()
                .map(|record| record.thread_seq)
                .collect::<Vec<_>>(),
            (0..records.len() as u64).collect::<Vec<_>>(),
            "{thread:?} must count its own calls from zero"
        );
        assert!(
            records.iter().all(|record| record.timestamp_ns > 0),
            "{thread:?} has an unstamped record"
        );
        assert!(
            records
                .windows(2)
                .all(|pair| pair[1].timestamp_ns >= pair[0].timestamp_ns),
            "{thread:?} is not monotonically stamped"
        );
    }

    let total: usize = segments.iter().map(|(_, records)| records.len()).sum();
    let highest = segments
        .iter()
        .flat_map(|(_, records)| records.iter().map(|record| record.thread_seq))
        .max()
        .expect("records");
    assert!(
        highest + 1 < total as u64,
        "the highest counter ({highest}) reaches the {total} calls made, \
         so the records are numbered from one global sequence"
    );
}

#[test]
fn concurrent_calls_stay_interleaved_in_the_trace() {
    let harness = Harness::new().expect("harness");
    let run = interleaved_run(&harness);
    let totals = reported_totals(&run);

    let bundle = run.bundle().expect("bundle");
    let mut merged: Vec<(u64, ThreadId)> = segments(&bundle)
        .into_iter()
        .filter(|(thread, _)| totals.contains_key(&thread.0))
        .flat_map(|(thread, records)| {
            records
                .into_iter()
                .map(move |record| (record.timestamp_ns, thread))
        })
        .collect();
    merged.sort_by_key(|(timestamp, _)| *timestamp);

    let handovers = merged
        .windows(2)
        .filter(|pair| pair[0].1 != pair[1].1)
        .count();
    assert!(
        handovers >= ROUNDS,
        "merging by timestamp gave {handovers} thread handovers across \
         {ROUNDS} barrier rounds: the threads were recorded as if serialized"
    );
}

/// ADR-0004's two prohibitions — no lock and no global counter on the common
/// record path — are invisible from the bundle: a capture that serialized
/// every thread on a mutex would still produce correct records. This reads
/// the record path itself so the guarantee has a regression guard.
mod record_path {
    const RECORDER: &str = include_str!("../src/recorder.rs");

    /// Everything an intercepted call runs through, from entering a shim to
    /// handing a filled buffer to the writer.
    const ON_THE_RECORD_PATH: [&str; 15] = [
        "begin",
        "finish",
        "finish_deep",
        "close",
        "deep",
        "binds",
        "payload",
        "next_seq",
        "push",
        "flush",
        "hand_over",
        "claim",
        "publish",
        "with_local",
        "monotonic_ns",
    ];

    /// Every body of a function named `name`, brace-matched.
    fn bodies(name: &str) -> Vec<&'static str> {
        let opening = format!("fn {name}");
        let mut found = Vec::new();
        let mut rest = RECORDER;
        while let Some(at) = rest.find(&opening) {
            let signature = &rest[at..];
            if !signature[opening.len()..].starts_with(['(', '<']) {
                rest = &signature[opening.len()..];
                continue;
            }
            let open = signature.find('{').expect("a function body");
            let mut depth = 0usize;
            let close = signature[open..]
                .char_indices()
                .find_map(|(offset, character)| {
                    match character {
                        '{' => depth += 1,
                        '}' => depth -= 1,
                        _ => return None,
                    }
                    (depth == 0).then_some(open + offset)
                })
                .expect("a closed function body");
            found.push(&signature[open..=close]);
            rest = &signature[close..];
        }
        assert!(!found.is_empty(), "no fn {name} on the record path");
        found
    }

    #[test]
    fn takes_no_lock() {
        for name in ON_THE_RECORD_PATH {
            for body in bodies(name) {
                assert!(
                    !body.contains(".lock("),
                    "{name} locks on the record path, which serializes the \
                     workload capture exists to observe (ADR-0004):\n{body}"
                );
            }
        }
    }

    /// What a deep handler captures costs the recording thread a memcpy and
    /// nothing more. Hashing a payload and writing it to the blob store is
    /// the writer thread's work: doing it inline would put the target's own
    /// progress behind a filesystem write.
    #[test]
    fn stores_no_payload_inline() {
        for name in ON_THE_RECORD_PATH {
            for body in bodies(name) {
                for stored in ["blobs(", "blake3", "fs::"] {
                    assert!(
                        !body.contains(stored),
                        "{name} reaches the blob store on the record path, which \
                         puts the captured workload behind a filesystem write:\n{body}"
                    );
                }
            }
        }
    }

    #[test]
    fn touches_no_global_counter() {
        for name in ON_THE_RECORD_PATH {
            for body in bodies(name) {
                for counter in ["fetch_add", "fetch_sub", "fetch_update"] {
                    assert!(
                        !body.contains(counter),
                        "{name} advances a shared counter on the record path; \
                         ordering is per-thread and offline (ADR-0004):\n{body}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_two_device_run_is_distinguishable_in_the_bundle() {
    const DEVICES: usize = 2;
    const CALLS_PER_THREAD: usize = 300;

    let harness = Harness::new()
        .expect("harness")
        .with_env("GPUTRACE_STUB_DEVICES", "2");
    let run = harness
        .run(target_bin("capture_devices"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());

    let mut reported = run.stdout().lines();
    assert_eq!(reported.next(), Some("devices=2"));
    let devices: HashMap<u64, u32> = reported
        .map(|line| {
            let (tid, device) = line.split_once(' ').expect("tid and device");
            (
                tid.trim_start_matches("tid=").parse().expect("tid"),
                device
                    .trim_start_matches("device=")
                    .parse()
                    .expect("device"),
            )
        })
        .collect();
    assert_eq!(devices.len(), DEVICES, "{}", run.stdout());

    let bundle = run.bundle().expect("bundle");
    let mut tagged = Vec::new();
    for (thread, records) in segments(&bundle) {
        let Some(&device) = devices.get(&thread.0) else {
            assert!(
                records.iter().all(|record| record.device.is_none()),
                "the main thread holds no context, so it targets no device"
            );
            continue;
        };
        tagged.push(device);

        assert_eq!(
            records.len(),
            CALLS_PER_THREAD + 3,
            "create, the device query, {CALLS_PER_THREAD} version calls, destroy: {thread:?}"
        );
        assert_eq!(
            records.first().map(|record| record.function.as_str()),
            Some("cuCtxCreate_v4")
        );
        assert_eq!(
            records.last().map(|record| record.function.as_str()),
            Some("cuCtxDestroy_v2")
        );
        assert_eq!(
            records
                .iter()
                .filter(|record| record.device == Some(device))
                .count(),
            records.len(),
            "every call this thread made targeted device {device}, including \
             the ones that name no device themselves"
        );
    }

    tagged.sort_unstable();
    assert_eq!(
        tagged,
        [0, 1],
        "the two GPUs must be distinguishable from the bundle alone"
    );
}

#[test]
fn exiting_while_threads_are_recording_still_closes_a_consistent_bundle() {
    const CALLS_BEFORE_REPORTING: usize = 512;

    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("capture_busy_exit"), &[])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());
    assert_eq!(run.stdout(), "recording=4\n");

    let bundle = run.bundle().expect("bundle");
    let segments = segments(&bundle);
    assert_eq!(segments.len(), THREADS + 1, "one segment per host thread");

    for (thread, records) in &segments {
        if records.len() == 1 {
            continue;
        }
        assert!(
            records.len() >= CALLS_BEFORE_REPORTING,
            "{thread:?} reported {CALLS_BEFORE_REPORTING} calls but kept {}",
            records.len()
        );
        assert_eq!(records[0].thread_seq, 0, "{thread:?} lost its first calls");
        assert!(
            records
                .windows(2)
                .all(|pair| pair[0].thread_seq < pair[1].thread_seq),
            "{thread:?} was collected out of the order it recorded in"
        );
    }
}

#[test]
fn an_abrupt_exit_passes_through_unchanged_and_claims_no_complete_bundle() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("capture_busy_exit"), &["abrupt"])
        .expect("run target");

    assert_eq!(
        run.status().code(),
        Some(7),
        "capture must not change what the target's exit looks like: {}",
        run.stderr()
    );
    assert!(
        run.bundle().is_err(),
        "`_exit` runs no atexit handler, so the bundle was never finished and \
         must not read back as one that was"
    );
}
