use serde::{Deserialize, Serialize};

/// How much is recorded per intercepted call. Capture defaults to
/// [`Fidelity::Replayable`]; [`Fidelity::Golden`] is opt-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

impl Fidelity {
    /// Every level, in the order each adds to the one before it.
    pub const ALL: [Self; 4] = [
        Self::Metadata,
        Self::Handles,
        Self::Replayable,
        Self::Golden,
    ];

    /// How a level is spelled on the command line and in `GPUTRACE_FIDELITY`.
    /// The CLI and the interposer cannot share a type across the `cdylib`
    /// boundary, but they must share this spelling or the level the user
    /// asked for is not the level recorded.
    pub fn name(self) -> &'static str {
        match self {
            Self::Metadata => "metadata",
            Self::Handles => "handles",
            Self::Replayable => "replayable",
            Self::Golden => "golden",
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unknown fidelity {0:?}; expected metadata, handles, replayable or golden")]
pub struct UnknownFidelity(pub String);

impl std::str::FromStr for Fidelity {
    type Err = UnknownFidelity;

    fn from_str(name: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|level| level.name() == name)
            .ok_or_else(|| UnknownFidelity(name.to_owned()))
    }
}

impl std::fmt::Display for Fidelity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// What a record calls the stream the call was issued into.
///
/// Capture records CUDA's own spelling and the offline reconstruction finds a
/// call's stream by this name, so the two sides share one definition: were
/// they to drift, the happens-before graph would go quietly blind rather than
/// fail to build.
pub const STREAM_ARG: &str = "hStream";
/// What a record calls the event the call records or waits on.
pub const EVENT_ARG: &str = "hEvent";

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

/// A scalar argument as the application passed it. Deep handlers add the
/// payload behind a pointer separately, as a [`PayloadRef`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Argument {
    pub name: String,
    pub value: ArgValue,
}

impl Argument {
    pub fn new(name: impl Into<String>, value: ArgValue) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArgValue {
    Null,
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    DevicePtr(u64),
    HostPtr(u64),
    Handle(HandleId),
    Str(String),
    Blob(BlobId),
}

/// How one call relates to a handle: creating it, using it, or ending it.
/// Replay reads these to remap captured ids onto freshly created handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandleRole {
    Produced,
    Consumed,
    Destroyed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandleBinding {
    pub id: HandleId,
    pub role: HandleRole,
}

/// A large payload this call carried, stored once in the bundle's blob
/// store. `role` names what the payload is to the call (for example a
/// host-to-device buffer or a module image) and stays open-ended so deep
/// handlers can add kinds without a schema change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayloadRef {
    pub role: String,
    pub blob: BlobId,
    pub len: u64,
    /// The device allocation these bytes mirror, when the call carrying the
    /// payload is not the call that named it — a golden output taken at the
    /// synchronization that finally made an asynchronous read-back valid.
    #[serde(default)]
    pub source: Option<HandleId>,
}

/// One intercepted call as written to a per-thread segment. Ordering is
/// per-thread plus a monotonic timestamp; the cross-thread happens-before
/// order is reconstructed offline, never enforced here.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CallRecord {
    pub function: String,
    pub thread_seq: u64,
    pub timestamp_ns: u64,
    pub result: i32,
    pub args: Vec<Argument>,
    pub handles: Vec<HandleBinding>,
    pub payloads: Vec<PayloadRef>,
    /// The `CUdevice` this call targeted: the one the call itself names, else
    /// the one the calling thread's current context belongs to. `None` when
    /// the call is device-agnostic or no context is current, which is what
    /// keeps multi-GPU activity in one process distinguishable.
    pub device: Option<u32>,
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
            ..Default::default()
        }
    }
}
