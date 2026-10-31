//! Golden capture of the asynchronous read-back: the dominant shape of real
//! CUDA output, where the call that asks for the data is not the call that
//! can answer it.

use gputrace_core::{CallRecord, HandleId, HandleRole, PayloadRef, TraceBundle};
use gputrace_testkit::{CaptureRun, Harness, target_bin};

/// What `async_golden` copies to the device and reads back.
const EXPECTED: [u8; 8] = [0xc0, 0xff, 0xee, 0x00, 0x11, 0x22, 0x33, 0x44];

const READ_BACK: &str = "cuMemcpyDtoHAsync_v2";

fn run(settles: &str, fidelity: &str) -> CaptureRun {
    let harness = Harness::new()
        .expect("harness")
        .with_env("GPUTRACE_FIDELITY", fidelity);
    let run = harness
        .run(target_bin("async_golden"), &[settles])
        .expect("run target");
    assert!(run.status().success(), "target failed: {}", run.stderr());
    run
}

fn records(bundle: &TraceBundle) -> Vec<CallRecord> {
    bundle
        .threads()
        .iter()
        .flat_map(|summary| bundle.read_segment(summary.thread).expect("segment"))
        .collect()
}

fn golden(records: &[CallRecord]) -> Vec<(String, PayloadRef)> {
    records
        .iter()
        .flat_map(|record| {
            record
                .payloads
                .iter()
                .filter(|payload| payload.role == "golden_output")
                .map(|payload| (record.function.clone(), payload.clone()))
        })
        .collect()
}

fn read_back_allocation(records: &[CallRecord]) -> HandleId {
    records
        .iter()
        .find(|record| record.function == READ_BACK)
        .expect("a read-back record")
        .handles
        .iter()
        .find(|binding| binding.role == HandleRole::Consumed)
        .expect("the read-back names the allocation it read")
        .id
}

/// The three calls that can make an asynchronous read-back's destination
/// valid. Whichever one the target uses is the one that must carry the
/// golden value, because it is the first moment the bytes mean anything.
#[test]
fn whichever_call_blocks_for_an_async_read_back_carries_its_golden_value() {
    for (settles, blocking) in [
        ("stream", "cuStreamSynchronize"),
        ("event", "cuEventSynchronize"),
        ("context", "cuCtxSynchronize"),
    ] {
        let run = run(settles, "golden");
        assert_eq!(run.stdout(), "received=c0ffee0011223344\n");

        let bundle = run.bundle().expect("bundle");
        let records = records(&bundle);
        let captured = golden(&records);
        assert_eq!(
            captured.len(),
            1,
            "{settles}: one read-back, one golden value: {:?}",
            captured.iter().map(|(call, _)| call).collect::<Vec<_>>()
        );

        let (carried_by, payload) = &captured[0];
        assert_eq!(carried_by, blocking);
        assert_eq!(bundle.blobs().get(&payload.blob).expect("blob"), EXPECTED);
        assert_eq!(payload.len, EXPECTED.len() as u64);
        assert_eq!(
            payload.source,
            Some(read_back_allocation(&records)),
            "{settles}: the golden value must say which allocation it mirrors"
        );
    }
}

/// Nothing can be read at the copy itself: the driver has only queued it.
#[test]
fn the_async_read_back_itself_carries_no_answer() {
    let run = run("stream", "golden");
    let records = records(&run.bundle().expect("bundle"));
    let copy = records
        .iter()
        .find(|record| record.function == READ_BACK)
        .expect("a read-back record");
    assert!(copy.payloads.is_empty(), "{:?}", copy.payloads);
}

#[test]
fn a_replayable_capture_pays_nothing_for_an_async_read_back() {
    let run = run("stream", "replayable");
    let records = records(&run.bundle().expect("bundle"));
    assert!(golden(&records).is_empty());
}
