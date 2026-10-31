//! The happens-before graph: the partial order across host threads and
//! streams, reconstructed from a captured trace rather than enforced while
//! capturing it (ADR-0004).
//!
//! Capture stamps each record with its thread's own sequence number and a
//! monotonic clock reading, and each stream or event call names the handles
//! it relates. That is enough to recover the order here: the recorded calls
//! say which work was issued into which stream, and which event carries one
//! stream's progress into another's.
//!
//! What this does *not* claim to order: a call that names no stream is not
//! in the graph at all, and a whole-context barrier (`cuCtxSynchronize`)
//! names no stream either, so it adds no edge. The graph is therefore a
//! lower bound on the real order — every edge it reports is real, and an
//! absent edge means the trace does not prove one.

use std::collections::HashMap;

use crate::bundle::{Result, TraceBundle};
use crate::trace::{ArgValue, Argument, CallRecord, EVENT_ARG, HandleId, STREAM_ARG, ThreadId};

/// One operation's place in the order. An index into [`HappensBefore::ops`],
/// which is in the order the host issued the operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpId(pub usize);

/// What an operation does to the order, beyond occupying a place in its
/// stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    /// Work issued into the stream, ordered only against that stream.
    Issued,
    /// `cuEventRecord`: the event comes to stand for everything issued into
    /// this stream so far.
    RecordsEvent,
    /// `cuStreamWaitEvent`: this stream proceeds no further until the event
    /// is reached.
    WaitsOnEvent,
    /// `cuStreamSynchronize`: the *host thread* proceeds no further until
    /// the stream drains.
    BlocksHost,
}

/// One call that named a stream, and therefore occupies a place in that
/// stream's order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamOp {
    pub id: OpId,
    pub function: String,
    pub kind: OpKind,
    pub stream: HandleId,
    pub event: Option<HandleId>,
    pub thread: ThreadId,
    pub thread_seq: u64,
    pub timestamp_ns: u64,
}

/// Why one operation is ordered before another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// Both were issued into the same stream, in this order.
    StreamOrder,
    /// One stream recorded the event; another waited on it.
    Event(HandleId),
    /// A host thread blocked on a stream, then issued the later operation.
    HostSync,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub from: OpId,
    pub to: OpId,
    pub reason: Reason,
}

/// The reconstructed partial order over a trace's stream operations.
pub struct HappensBefore {
    ops: Vec<StreamOp>,
    edges: Vec<Edge>,
    successors: Vec<Vec<usize>>,
}

impl HappensBefore {
    pub fn reconstruct(bundle: &TraceBundle) -> Result<Self> {
        let segments = bundle
            .threads()
            .iter()
            .map(|summary| Ok((summary.thread, bundle.read_segment(summary.thread)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::from_segments(segments))
    }

    pub fn from_segments(segments: impl IntoIterator<Item = (ThreadId, Vec<CallRecord>)>) -> Self {
        let mut ops: Vec<StreamOp> = segments
            .into_iter()
            .flat_map(|(thread, records)| {
                records
                    .into_iter()
                    .filter_map(move |record| stream_op(thread, record))
            })
            .collect();
        // No global counter exists by construction, so the issue order is
        // recovered from the monotonic clock, with the per-thread sequence
        // settling two calls that read the same nanosecond.
        ops.sort_by_key(|op| (op.timestamp_ns, op.thread, op.thread_seq));
        for (index, op) in ops.iter_mut().enumerate() {
            op.id = OpId(index);
        }

        let edges = edges(&ops);
        let mut successors = vec![Vec::new(); ops.len()];
        for edge in &edges {
            successors[edge.from.0].push(edge.to.0);
        }
        Self {
            ops,
            edges,
            successors,
        }
    }

    /// Every operation that named a stream, in the order the host issued
    /// them.
    pub fn ops(&self) -> &[StreamOp] {
        &self.ops
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Whether the trace proves `earlier` completed before `later` began.
    /// `false` means unordered, not "after": this is a partial order.
    pub fn happens_before(&self, earlier: OpId, later: OpId) -> bool {
        if earlier == later || earlier.0 >= self.ops.len() || later.0 >= self.ops.len() {
            return false;
        }
        let mut seen = vec![false; self.ops.len()];
        let mut frontier = vec![earlier.0];
        seen[earlier.0] = true;
        while let Some(at) = frontier.pop() {
            for &next in &self.successors[at] {
                if next == later.0 {
                    return true;
                }
                if !std::mem::replace(&mut seen[next], true) {
                    frontier.push(next);
                }
            }
        }
        false
    }
}

fn stream_op(thread: ThreadId, record: CallRecord) -> Option<StreamOp> {
    let stream = handle_arg(&record.args, STREAM_ARG)?;
    Some(StreamOp {
        id: OpId(0),
        kind: kind_of(&record.function),
        function: record.function,
        stream,
        event: handle_arg(&record.args, EVENT_ARG),
        thread,
        thread_seq: record.thread_seq,
        timestamp_ns: record.timestamp_ns,
    })
}

fn handle_arg(args: &[Argument], name: &str) -> Option<HandleId> {
    args.iter().find_map(|arg| match (&arg.name, &arg.value) {
        (found, ArgValue::Handle(id)) if found == name => Some(*id),
        _ => None,
    })
}

fn kind_of(function: &str) -> OpKind {
    if function.starts_with("cuEventRecord") {
        OpKind::RecordsEvent
    } else if function.starts_with("cuStreamWaitEvent") {
        OpKind::WaitsOnEvent
    } else if function.starts_with("cuStreamSynchronize") {
        OpKind::BlocksHost
    } else {
        OpKind::Issued
    }
}

fn edges(ops: &[StreamOp]) -> Vec<Edge> {
    let mut edges = Vec::new();
    let mut last_in_stream: HashMap<HandleId, usize> = HashMap::new();
    let mut last_on_thread_block: HashMap<ThreadId, usize> = HashMap::new();
    let mut last_record_of_event: HashMap<HandleId, usize> = HashMap::new();

    for (index, op) in ops.iter().enumerate() {
        if let Some(previous) = last_in_stream.insert(op.stream, index) {
            edges.push(edge(previous, index, Reason::StreamOrder));
        }
        if let Some(blocked) = last_on_thread_block.remove(&op.thread) {
            edges.push(edge(blocked, index, Reason::HostSync));
        }
        match op.kind {
            OpKind::RecordsEvent => {
                if let Some(event) = op.event {
                    last_record_of_event.insert(event, index);
                }
            }
            OpKind::WaitsOnEvent => {
                // CUDA's wait captures the event's *latest* recording at the
                // moment the wait is issued; a later recording of the same
                // event orders nothing against this wait.
                if let Some(event) = op.event
                    && let Some(&recorded) = last_record_of_event.get(&event)
                {
                    edges.push(edge(recorded, index, Reason::Event(event)));
                }
            }
            OpKind::BlocksHost => {
                last_on_thread_block.insert(op.thread, index);
            }
            OpKind::Issued => {}
        }
    }
    edges
}

fn edge(from: usize, to: usize, reason: Reason) -> Edge {
    Edge {
        from: OpId(from),
        to: OpId(to),
        reason,
    }
}
