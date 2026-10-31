//! Reconstructing the happens-before partial order offline, which is the
//! whole of what capture's refusal to order anything costs (ADR-0004).

use gputrace_core::happens_before::{HappensBefore, Reason};
use gputrace_core::{ArgValue, Argument, CallRecord, HandleId, ThreadId};

const PRODUCER: HandleId = HandleId(0xa1);
const CONSUMER: HandleId = HandleId(0xa2);
const DONE: HandleId = HandleId(0xe1);

struct Trace {
    thread: ThreadId,
    records: Vec<CallRecord>,
}

impl Trace {
    fn new(thread: u64) -> Self {
        Self {
            thread: ThreadId(thread),
            records: Vec::new(),
        }
    }

    fn call(mut self, function: &str, args: &[(&str, HandleId)]) -> Self {
        let seq = self.records.len() as u64;
        let mut record = CallRecord::new(function, seq, 10 * (seq + 1), 0);
        record.args = args
            .iter()
            .map(|(name, id)| Argument::new(*name, ArgValue::Handle(*id)))
            .collect();
        self.records.push(record);
        self
    }
}

/// Stream `PRODUCER` does work, records `DONE`, then does more work;
/// stream `CONSUMER` waits on `DONE` and then does work of its own.
fn cross_stream() -> HappensBefore {
    let trace = Trace::new(7)
        .call("cuMemcpyHtoDAsync_v2", &[("hStream", PRODUCER)])
        .call("cuEventRecord", &[("hEvent", DONE), ("hStream", PRODUCER)])
        .call("cuMemcpyHtoDAsync_v2", &[("hStream", PRODUCER)])
        .call(
            "cuStreamWaitEvent",
            &[("hStream", CONSUMER), ("hEvent", DONE)],
        )
        .call("cuMemcpyHtoDAsync_v2", &[("hStream", CONSUMER)]);
    HappensBefore::from_segments([(trace.thread, trace.records)])
}

fn op(order: &HappensBefore, index: usize) -> gputrace_core::happens_before::OpId {
    order.ops()[index].id
}

#[test]
fn work_before_a_recorded_event_happens_before_work_after_the_wait_on_it() {
    let order = cross_stream();
    assert!(order.happens_before(op(&order, 0), op(&order, 4)));
}

/// The assertion that separates a reconstruction from a list of records:
/// work the producer issued *after* the event was recorded is ordered
/// against nothing in the consumer.
#[test]
fn work_issued_after_the_event_was_recorded_orders_nothing_in_the_other_stream() {
    let order = cross_stream();
    assert!(!order.happens_before(op(&order, 2), op(&order, 4)));
    assert!(!order.happens_before(op(&order, 4), op(&order, 0)));
}

#[test]
fn work_in_one_stream_is_ordered_by_the_order_it_was_issued_in() {
    let order = cross_stream();
    assert!(order.happens_before(op(&order, 0), op(&order, 2)));
    assert!(!order.happens_before(op(&order, 2), op(&order, 0)));
}

#[test]
fn the_event_is_what_the_cross_stream_edge_is_attributed_to() {
    let order = cross_stream();
    let crossing: Vec<_> = order
        .edges()
        .iter()
        .filter(|edge| matches!(edge.reason, Reason::Event(_)))
        .collect();
    assert_eq!(crossing.len(), 1);
    assert_eq!(crossing[0].reason, Reason::Event(DONE));
    assert_eq!(crossing[0].from, op(&order, 1));
    assert_eq!(crossing[0].to, op(&order, 3));
}

/// A host thread that blocks on one stream and then issues into another is
/// a second kind of edge, and it is in the trace just as plainly: the
/// per-thread sequence says what that thread did next.
#[test]
fn blocking_the_host_on_a_stream_orders_what_that_thread_issues_next() {
    let trace = Trace::new(7)
        .call("cuMemcpyHtoDAsync_v2", &[("hStream", PRODUCER)])
        .call("cuStreamSynchronize", &[("hStream", PRODUCER)])
        .call("cuMemcpyHtoDAsync_v2", &[("hStream", CONSUMER)]);
    let order = HappensBefore::from_segments([(trace.thread, trace.records)]);

    assert!(order.happens_before(op(&order, 0), op(&order, 2)));
    assert!(!order.happens_before(op(&order, 2), op(&order, 0)));
}

/// Nothing relates two streams that never exchanged an event, however the
/// wall clock interleaved them.
#[test]
fn two_unrelated_streams_are_unordered_in_both_directions() {
    let trace = Trace::new(7)
        .call("cuMemcpyHtoDAsync_v2", &[("hStream", PRODUCER)])
        .call("cuMemcpyHtoDAsync_v2", &[("hStream", CONSUMER)]);
    let order = HappensBefore::from_segments([(trace.thread, trace.records)]);

    assert!(!order.happens_before(op(&order, 0), op(&order, 1)));
    assert!(!order.happens_before(op(&order, 1), op(&order, 0)));
}
