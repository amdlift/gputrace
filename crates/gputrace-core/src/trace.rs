use serde::{Deserialize, Serialize};

/// How much is recorded per intercepted call. Capture defaults to
/// [`Fidelity::Replayable`]; [`Fidelity::Golden`] is opt-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fidelity {
    /// Function, thread, ordering stamps, scalar args, return value.
    Metadata,
    /// Adds stable identity for handles and device pointers.
    Handles,
    /// Adds payloads: host->device buffers, launch params, module images.
    Replayable,
    /// Adds device->host outputs as golden values for validation.
    Golden,
}

/// The host thread a segment belongs to. Capture attributes every record
/// to the thread that made the call; there is no global ordering (ADR-0004).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ThreadId(pub u64);

/// Stable id bound to a live handle or device pointer at capture time so
/// replay can remap it onto a freshly created one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HandleId(pub u64);

/// Content-addressed reference into the trace bundle's blob store.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlobId(pub String);

/// One intercepted call as written to a per-thread segment. Ordering is
/// per-thread plus a monotonic timestamp; the cross-thread happens-before
/// order is reconstructed offline, never enforced here.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CallRecord {
    pub function: String,
    pub thread_seq: u64,
    pub timestamp_ns: u64,
    pub result: i32,
}

impl CallRecord {
    pub fn new(
        function: impl Into<String>,
        thread_seq: u64,
        timestamp_ns: u64,
        result: i32,
    ) -> Self {
        Self {
            function: function.into(),
            thread_seq,
            timestamp_ns,
            result,
        }
    }
}
