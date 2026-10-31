//! The capture record path (ADR-0004).
//!
//! Each host thread appends to a buffer it alone owns, hands full buffers to
//! a single background writer over a lock-free stack, and the writer drains
//! them into one per-thread segment of the trace bundle. Nothing on the
//! record path takes a lock or touches a global counter: ordering is
//! recovered offline from the per-thread sequence and the monotonic stamp.
//!
//! Buffers are reachable from outside their owning thread, so exiting does
//! not depend on every recording thread having exited first.

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
    ArgValue, Argument, BundleWriter, CallRecord, CaptureConfig, Fidelity, SegmentWriter, ThreadId,
};

/// Names the directory the interposer writes its trace bundle to. Unset or
/// empty means no capture: the shims forward and record nothing.
pub const BUNDLE_DIR_ENV: &str = "GPUTRACE_OUT";
/// Names how much each shim records: `metadata`, `handles`, `replayable`
/// or `golden`.
pub const FIDELITY_ENV: &str = "GPUTRACE_FIDELITY";

const CHUNK_RECORDS: usize = 256;
const WRITER_IDLE_POLL: Duration = Duration::from_millis(2);
/// How many times shutdown re-tries a buffer whose owner is mid-append.
const STEAL_ATTEMPTS: usize = 64;

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
        with_local(move |local| {
            let mut record =
                CallRecord::new(stamp.function, stamp.thread_seq, stamp.timestamp_ns, result);
            record.device = local.target_device(stamp.function, &args);
            record.args = args;
            local.push(record);
        });
    }
}

/// Finish the bundle: flush every thread's buffer, stop the writer, and
/// write the manifest. Idempotent, and a no-op when capture never started.
pub fn shutdown() {
    let Some(capture) = CAPTURE.get().and_then(Option::as_ref) else {
        return;
    };
    capture.flush_every_thread();
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

/// One thread's buffer, held where another thread can reach it.
///
/// The pointer is the buffer's ownership token: the owning thread swaps it
/// out, appends, and swaps it back, so exactly one party ever touches the
/// `Vec`. That keeps the record path wait-free — an uncontended exchange on
/// a line no other thread reads — while still letting shutdown collect what
/// a thread has buffered without waiting for that thread to exit.
struct Slot {
    thread: ThreadId,
    buffer: AtomicPtr<Vec<CallRecord>>,
}

// The box is the buffer's ownership token, not an extra container: the slot
// has to hold a pointer another thread can exchange, which a bare `Vec` is not.
#[allow(clippy::box_collection)]
impl Slot {
    fn new(thread: ThreadId) -> Self {
        Self {
            thread,
            buffer: AtomicPtr::new(Box::into_raw(Box::new(Vec::with_capacity(CHUNK_RECORDS)))),
        }
    }

    /// Take the buffer, leaving the slot empty. `None` means another party
    /// holds it right now.
    fn claim(&self) -> Option<Box<Vec<CallRecord>>> {
        // Acquire pairs with the `publish` that last put the buffer back, so
        // the records stored through that pointer are visible here.
        let buffer = self.buffer.swap(ptr::null_mut(), Ordering::Acquire);
        (!buffer.is_null()).then(|| unsafe { Box::from_raw(buffer) })
    }

    fn publish(&self, buffer: Box<Vec<CallRecord>>) {
        self.buffer.store(Box::into_raw(buffer), Ordering::Release);
    }

    /// Take whatever the owning thread has buffered so far, for a shutdown
    /// that cannot wait for that thread to exit. A slot is only ever null
    /// for the few instructions its owner spends appending, so a bounded
    /// yield is enough to catch a thread that is still recording.
    fn steal(&self) -> Option<Vec<CallRecord>> {
        for _ in 0..STEAL_ATTEMPTS {
            if let Some(buffer) = self.claim() {
                return Some(*buffer);
            }
            thread::yield_now();
        }
        None
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        let buffer = *self.buffer.get_mut();
        if !buffer.is_null() {
            drop(unsafe { Box::from_raw(buffer) });
        }
    }
}

struct Capture {
    queue: Arc<Queue>,
    writer: Mutex<Option<JoinHandle<()>>>,
    slots: Mutex<Vec<Arc<Slot>>>,
}

impl Capture {
    fn slots(&self) -> std::sync::MutexGuard<'_, Vec<Arc<Slot>>> {
        self.slots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Give a thread a slot of its own. Off the record path: once per thread.
    fn register(&self, thread: ThreadId) -> Arc<Slot> {
        let slot = Arc::new(Slot::new(thread));
        self.slots().push(Arc::clone(&slot));
        slot
    }

    fn deregister(&self, slot: &Arc<Slot>) {
        self.slots().retain(|known| !Arc::ptr_eq(known, slot));
    }

    /// Queue every thread's buffered tail, including threads that are still
    /// running: a thread that never exits before the process does would
    /// otherwise never run the destructor that flushes it.
    fn flush_every_thread(&self) {
        let slots = self.slots().clone();
        for slot in slots {
            let Some(records) = slot.steal() else {
                continue;
            };
            if records.is_empty() {
                continue;
            }
            self.queue.push(Chunk {
                thread: slot.thread,
                records,
            });
        }
    }
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
        slots: Mutex::new(Vec::new()),
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

/// What one call says about the device it targeted. A call that names a
/// device targets that one; everything else targets whatever its thread's
/// current context does, which is how a device-agnostic call in a multi-GPU
/// process is still attributable to a GPU.
enum TargetDevice {
    /// The call named a device and made it the calling thread's.
    Bound(u32),
    /// The call named a device without changing what its thread targets.
    Named(u32),
    /// The call ended the calling thread's context.
    Released,
    /// The call says nothing; it targets what its thread already did.
    Inherited,
}

fn target_device(function: &str, args: &[Argument]) -> TargetDevice {
    if function.starts_with("cuCtxDestroy") {
        return TargetDevice::Released;
    }
    match named_device(args) {
        Some(device) if binds_the_thread(function) => TargetDevice::Bound(device),
        Some(device) => TargetDevice::Named(device),
        None => TargetDevice::Inherited,
    }
}

fn binds_the_thread(function: &str) -> bool {
    function.starts_with("cuCtxCreate") || function.starts_with("cuDevicePrimaryCtxRetain")
}

/// The device ordinal a call passed by value. A `CUdevice` out-parameter is
/// recorded as a pointer, so a call that only asks for a device never looks
/// like one that targets it.
fn named_device(args: &[Argument]) -> Option<u32> {
    args.iter()
        .find_map(|arg| match (arg.name.as_str(), &arg.value) {
            ("dev" | "device", ArgValue::Signed(ordinal)) => u32::try_from(*ordinal).ok(),
            ("dev" | "device", ArgValue::Unsigned(ordinal)) => u32::try_from(*ordinal).ok(),
            _ => None,
        })
}

struct Local {
    capture: &'static Capture,
    slot: Arc<Slot>,
    seq: u64,
    device: Option<u32>,
}

impl Local {
    fn start() -> Option<Self> {
        let capture = capture()?;
        Some(Self {
            capture,
            slot: capture.register(ThreadId(current_thread_id())),
            seq: 0,
            device: None,
        })
    }

    /// The device to tag this call with, updating what the thread targets
    /// from here on. A teardown is still tagged with the device it ended.
    fn target_device(&mut self, function: &str, args: &[Argument]) -> Option<u32> {
        match target_device(function, args) {
            TargetDevice::Bound(device) => {
                self.device = Some(device);
                Some(device)
            }
            TargetDevice::Named(device) => Some(device),
            TargetDevice::Released => self.device.take(),
            TargetDevice::Inherited => self.device,
        }
    }

    fn next_seq(&mut self) -> u64 {
        let seq = self.seq;
        self.seq += 1;
        seq
    }

    fn push(&mut self, record: CallRecord) {
        let mut buffer = self
            .slot
            .claim()
            .unwrap_or_else(|| Box::new(Vec::with_capacity(CHUNK_RECORDS)));
        buffer.push(record);
        let full = buffer.len() >= CHUNK_RECORDS;
        let chunk =
            full.then(|| std::mem::replace(&mut *buffer, Vec::with_capacity(CHUNK_RECORDS)));
        self.slot.publish(buffer);
        if let Some(records) = chunk {
            self.hand_over(records);
        }
    }

    fn flush(&mut self) {
        let Some(mut buffer) = self.slot.claim() else {
            return;
        };
        let records = std::mem::take(&mut *buffer);
        self.slot.publish(buffer);
        if !records.is_empty() {
            self.hand_over(records);
        }
    }

    fn hand_over(&self, records: Vec<CallRecord>) {
        self.capture.queue.push(Chunk {
            thread: self.slot.thread,
            records,
        });
    }
}

impl Drop for Local {
    fn drop(&mut self) {
        self.flush();
        self.capture.deregister(&self.slot);
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
    // inside it. `_exit`, `abort` and a fatal signal all bypass `atexit`:
    // those runs lose the bundle by construction, which is why shutdown must
    // never be the only thing standing between the target and its own exit.
    unsafe { atexit(flush_at_exit) };
}

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

/// Linux's `CLOCK_MONOTONIC_RAW`: unslewed by NTP, and read from the vDSO,
/// so stamping every call costs no syscall (ADR-0004).
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

/// What a call says about its device decides a record's attribution in a
/// multi-GPU process. The cases below are ones no shimmed entry point
/// reaches yet, so they are pinned here rather than through the bundle.
#[cfg(test)]
mod target_device_tests {
    use super::{TargetDevice, target_device};

    use gputrace_core::{ArgValue, Argument};

    fn by_value(name: &str, ordinal: i64) -> Vec<Argument> {
        vec![Argument::new(name, ArgValue::Signed(ordinal))]
    }

    #[test]
    fn a_query_names_its_device_without_rebinding_the_calling_thread() {
        let args = by_value("dev", 1);
        assert!(matches!(
            target_device("cuDeviceGetAttribute", &args),
            TargetDevice::Named(1)
        ));
    }

    #[test]
    fn creating_a_context_binds_the_thread_to_the_device_it_names() {
        let args = by_value("dev", 1);
        assert!(matches!(
            target_device("cuCtxCreate_v4", &args),
            TargetDevice::Bound(1)
        ));
        assert!(matches!(
            target_device("cuDevicePrimaryCtxRetain", &args),
            TargetDevice::Bound(1)
        ));
    }

    #[test]
    fn asking_for_a_device_handle_is_not_targeting_one() {
        let args = vec![
            Argument::new("device", ArgValue::HostPtr(0x7fff_0000)),
            Argument::new("ordinal", ArgValue::Signed(1)),
        ];
        assert!(matches!(
            target_device("cuDeviceGet", &args),
            TargetDevice::Inherited
        ));
    }

    #[test]
    fn a_call_naming_no_device_targets_whatever_its_thread_does() {
        let args = vec![Argument::new("flags", ArgValue::Unsigned(0))];
        assert!(matches!(
            target_device("cuInit", &args),
            TargetDevice::Inherited
        ));
    }

    #[test]
    fn tearing_down_a_context_releases_the_binding_whatever_it_names() {
        let args = vec![Argument::new("ctx", ArgValue::HostPtr(0x1_0000))];
        assert!(matches!(
            target_device("cuCtxDestroy_v2", &args),
            TargetDevice::Released
        ));
    }
}
