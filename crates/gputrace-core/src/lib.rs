//! Backend-agnostic core: the trace representation and the capture/replay/
//! mutate operations defined independently of any accelerator API. CUDA is
//! the first concrete [`Backend`]; the trait keeps the rest vendor-neutral.

pub mod backend;
pub mod bundle;
pub mod trace;

pub use backend::Backend;
pub use bundle::{
    BlobStore, BundleError, BundleWriter, CaptureConfig, Manifest, SCHEMA_VERSION, SegmentWriter,
    ThreadSummary, TraceBundle,
};
pub use trace::{
    ArgValue, Argument, BlobId, CallRecord, Fidelity, HandleBinding, HandleId, HandleRole,
    PayloadRef, ThreadId,
};
