//! The capture record path (ADR-0004).
//!
//! Each host thread appends to a buffer it alone owns, hands full buffers to
//! a single background writer over a lock-free stack, and the writer drains
//! them into one per-thread segment of the trace bundle. Nothing on the
//! record path takes a lock or touches a global counter: ordering is
//! recovered offline from the per-thread sequence and the monotonic stamp.

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::env;
use std::ffi::c_int;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle, Thread};
use std::time::Duration;

use gputrace_core::{
    Argument, BundleWriter, CallRecord, CaptureConfig, Fidelity, SegmentWriter, ThreadId,
};

/// Names the directory the interposer writes its trace bundle to. Unset or
/// empty means no capture: the shims forward and record nothing.
pub const BUNDLE_DIR_ENV: &str = "GPUTRACE_OUT";
/// Names how much each shim records: `metadata`, `handles`, `replayable`
/// or `golden`.
pub const FIDELITY_ENV: &str = "GPUTRACE_FIDELITY";

const CHUNK_RECORDS: usize = 256;
const WRITER_IDLE_POLL: Duration = Duration::from_millis(2);

/// One intercepted call, from the moment the shim is entered to the moment
/// the real driver has answered.
pub struct CallScope {
    stamp: Option<Stamp>,
}

struct Stamp {
    function: &'static str,
    thread_seq: u64,
    timestamp_ns: u64,
}

/// Open a scope for `function`, stamping it with this thread's next sequence
/// number and a monotonic timestamp.
pub fn begin(function: &'static str) -> CallScope {
    let stamp = with_local(|local| {
        let thread_seq = local.next_seq();
        Stamp {
            function,
            thread_seq,
            timestamp_ns: monotonic_ns(),
        }
    });
    CallScope { stamp }
}

impl CallScope {
    /// Close the scope with what the driver returned and the arguments the
    /// caller passed, appending the record to this thread's buffer.
    pub fn finish(self, result: c_int, args: Vec<Argument>) {
        let Some(stamp) = self.stamp else {
            return;
        };
        let mut record =
            CallRecord::new(stamp.function, stamp.thread_seq, stamp.timestamp_ns, result);
        record.args = args;
        let mut record = Some(record);
        with_local(|local| {
            if let Some(record) = record.take() {
                local.push(record);
            }
        });
    }
}

/// Finish the bundle: flush this thread's buffer, stop the writer, and write
/// the manifest. Idempotent, and a no-op when capture never started.
pub fn shutdown() {
    let Some(capture) = CAPTURE.get().and_then(Option::as_ref) else {
        return;
    };
    with_local(|local| local.flush());
    capture.queue.shutdown.store(true, Ordering::Release);
    if let Some(writer) = capture.queue.writer.get() {
        writer.unpark();
    }
    let handle = capture
        .writer
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    if let Some(handle) = handle {
        let _ = handle.join();
    }
}

/// One thread's filled buffer, on its way to that thread's segment.
struct Chunk {
    thread: ThreadId,
    records: Vec<CallRecord>,
}

struct Node {
    chunk: Chunk,
    next: *mut Node,
}

/// A Treiber stack of filled buffers. Producers only ever compare-and-swap
/// the head, so a recording thread never waits on another thread.
struct Queue {
    head: AtomicPtr<Node>,
    writer: OnceLock<Thread>,
    shutdown: AtomicBool,
}

impl Queue {
    fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
            writer: OnceLock::new(),
            shutdown: AtomicBool::new(false),
        }
    }

    fn push(&self, chunk: Chunk) {
        let node = Box::into_raw(Box::new(Node {
            chunk,
            next: ptr::null_mut(),
        }));
        let mut head = self.head.load(Ordering::Relaxed);
        loop {
            unsafe { (*node).next = head };
            match self
                .head
                .compare_exchange_weak(head, node, Ordering::Release, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(current) => head = current,
            }
        }
        if let Some(writer) = self.writer.get() {
            writer.unpark();
        }
    }

    fn drain(&self) -> Vec<Chunk> {
        let mut node = self.head.swap(ptr::null_mut(), Ordering::Acquire);
        let mut chunks = Vec::new();
        while !node.is_null() {
            let popped = unsafe { Box::from_raw(node) };
            node = popped.next;
            chunks.push(popped.chunk);
        }
        // The stack hands buffers back newest-first; a thread's buffers must
        // reach its segment in the order it filled them.
        chunks.reverse();
        chunks
    }
}

struct Capture {
    queue: Arc<Queue>,
    writer: Mutex<Option<JoinHandle<()>>>,
}

static CAPTURE: OnceLock<Option<Capture>> = OnceLock::new();

fn capture() -> Option<&'static Capture> {
    CAPTURE.get_or_init(start).as_ref()
}

fn start() -> Option<Capture> {
    let dir = env::var_os(BUNDLE_DIR_ENV).filter(|dir| !dir.is_empty())?;
    let bundle = match BundleWriter::create(PathBuf::from(dir), capture_config()) {
        Ok(bundle) => bundle,
        Err(e) => {
            eprintln!("gputrace: cannot open the trace bundle: {e}");
            return None;
        }
    };

    let queue = Arc::new(Queue::new());
    let handle = {
        let queue = Arc::clone(&queue);
        thread::Builder::new()
            .name("gputrace-writer".to_owned())
            .spawn(move || drain_to_bundle(&queue, bundle))
            .ok()?
    };
    let _ = queue.writer.set(handle.thread().clone());
    register_exit_hook();

    Some(Capture {
        queue,
        writer: Mutex::new(Some(handle)),
    })
}

fn drain_to_bundle(queue: &Queue, bundle: BundleWriter) {
    let mut segments: HashMap<ThreadId, SegmentWriter> = HashMap::new();
    let outcome = (|| -> gputrace_core::bundle::Result<()> {
        loop {
            // Read the flag before draining: anything queued before shutdown
            // was requested is in the batch this pass takes.
            let shutting_down = queue.shutdown.load(Ordering::Acquire);
            let chunks = queue.drain();
            let idle = chunks.is_empty();
            for chunk in chunks {
                let segment = match segments.entry(chunk.thread) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(bundle.open_segment(chunk.thread)?),
                };
                for record in &chunk.records {
                    segment.write(record)?;
                }
            }
            if idle {
                if shutting_down {
                    return Ok(());
                }
                thread::park_timeout(WRITER_IDLE_POLL);
            }
        }
    })();

    let finished = outcome
        .and_then(|()| segments.into_values().try_for_each(SegmentWriter::finish))
        .and_then(|()| bundle.finish());
    if let Err(e) = finished {
        eprintln!("gputrace: capture incomplete: {e}");
    }
}

fn capture_config() -> CaptureConfig {
    CaptureConfig {
        backend: "cuda".to_owned(),
        fidelity: fidelity(),
        device: None,
        command: env::args_os()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect(),
    }
}

fn fidelity() -> Fidelity {
    match env::var(FIDELITY_ENV).as_deref() {
        Ok("handles") => Fidelity::Handles,
        Ok("replayable") => Fidelity::Replayable,
        Ok("golden") => Fidelity::Golden,
        _ => Fidelity::Metadata,
    }
}

struct Local {
    capture: &'static Capture,
    thread: ThreadId,
    seq: u64,
    records: Vec<CallRecord>,
}

impl Local {
    fn start() -> Option<Self> {
        Some(Self {
            capture: capture()?,
            thread: ThreadId(current_thread_id()),
            seq: 0,
            records: Vec::with_capacity(CHUNK_RECORDS),
        })
    }

    fn next_seq(&mut self) -> u64 {
        let seq = self.seq;
        self.seq += 1;
        seq
    }

    fn push(&mut self, record: CallRecord) {
        self.records.push(record);
        if self.records.len() >= CHUNK_RECORDS {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.records.is_empty() {
            return;
        }
        let records = std::mem::replace(&mut self.records, Vec::with_capacity(CHUNK_RECORDS));
        self.capture.queue.push(Chunk {
            thread: self.thread,
            records,
        });
    }
}

impl Drop for Local {
    fn drop(&mut self) {
        self.flush();
    }
}

thread_local! {
    static LOCAL: RefCell<Option<Local>> = RefCell::new(Local::start());
}

/// Run `f` against this thread's buffer, or do nothing when capture is off
/// or the thread-local has already been destroyed on the way out of a thread.
fn with_local<T>(f: impl FnOnce(&mut Local) -> T) -> Option<T> {
    LOCAL
        .try_with(|local| local.borrow_mut().as_mut().map(f))
        .ok()
        .flatten()
}

unsafe extern "C" {
    fn atexit(callback: extern "C" fn()) -> c_int;
    fn gettid() -> i32;
    fn clock_gettime(clock: i32, out: *mut Timespec) -> c_int;
}

extern "C" fn flush_at_exit() {
    shutdown();
}

fn register_exit_hook() {
    // The target process is what dies, so the bundle has to be finished from
    // inside it. glibc runs a thread's TLS destructors before its atexit
    // handlers, so every buffer this thread filled is already queued by then.
    unsafe { atexit(flush_at_exit) };
}

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

const CLOCK_MONOTONIC_RAW: i32 = 4;

fn current_thread_id() -> u64 {
    u64::from(unsafe { gettid() }.unsigned_abs())
}

fn monotonic_ns() -> u64 {
    let mut now = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if unsafe { clock_gettime(CLOCK_MONOTONIC_RAW, &mut now) } != 0 {
        return 0;
    }
    now.tv_sec.unsigned_abs() * 1_000_000_000 + now.tv_nsec.unsigned_abs()
}
