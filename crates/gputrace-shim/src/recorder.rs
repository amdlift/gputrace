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
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::ffi::c_int;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle, Thread};
use std::time::Duration;

use gputrace_core::{
    ArgValue, Argument, BundleWriter, CallRecord, CaptureConfig, DeviceSummary, Environment,
    Fidelity, HandleBinding, HandleId, HandleRole, PayloadRef, SegmentWriter, ThreadId,
};

use crate::environment;

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
        self.close(result, DeepCall::plain(args));
    }

    /// Start recording what a deep handler saw beyond the scalar arguments.
    /// The scope decides here, once, how much of it is wanted, so a handler
    /// never consults the capture's fidelity itself.
    pub fn deep(&self, args: Vec<Argument>) -> DeepCall {
        DeepCall {
            fidelity: self.stamp.as_ref().map(|_| fidelity()),
            args,
            handles: Vec::new(),
            payloads: Vec::new(),
        }
    }

    pub fn finish_deep(self, result: c_int, deep: DeepCall) {
        self.close(result, deep);
    }

    fn close(self, result: c_int, deep: DeepCall) {
        let Some(stamp) = self.stamp else {
            return;
        };
        let DeepCall {
            args,
            handles,
            payloads,
            ..
        } = deep;
        with_local(move |local| {
            let mut record =
                CallRecord::new(stamp.function, stamp.thread_seq, stamp.timestamp_ns, result);
            record.device = local.target_device(stamp.function, &args);
            record.args = args;
            record.handles = handles;
            local.push(Pending { record, payloads });
        });
    }
}

/// A payload a handler copied out of the caller's memory, waiting to be
/// stored. Hashing it and writing it to the blob store is the writer
/// thread's work: the recording thread only memcpys, because the caller is
/// free to overwrite the buffer the moment the driver returns.
struct StagedPayload {
    role: &'static str,
    bytes: Vec<u8>,
}

/// One buffered call, with whatever the deep handler captured alongside it.
struct Pending {
    record: CallRecord,
    payloads: Vec<StagedPayload>,
}

/// What a deep handler captured beyond the scalar arguments: which handles
/// the call produced, used or ended, and which buffers it carried.
///
/// Everything below the capture's fidelity is dropped here rather than in
/// each handler, so asking for a payload that is not wanted costs no copy.
pub struct DeepCall {
    fidelity: Option<Fidelity>,
    args: Vec<Argument>,
    handles: Vec<HandleBinding>,
    payloads: Vec<StagedPayload>,
}

impl DeepCall {
    fn plain(args: Vec<Argument>) -> Self {
        Self {
            fidelity: None,
            args,
            handles: Vec::new(),
            payloads: Vec::new(),
        }
    }

    fn wants(&self, least: Fidelity) -> bool {
        self.fidelity.is_some_and(|have| rank(have) >= rank(least))
    }

    /// The call created `id`.
    pub fn produces(self, id: Option<HandleId>) -> Self {
        self.binds(id, HandleRole::Produced)
    }

    /// The call used `id` without creating or ending it.
    pub fn consumes(self, id: Option<HandleId>) -> Self {
        self.binds(id, HandleRole::Consumed)
    }

    /// The call ended `id`.
    pub fn destroys(self, id: Option<HandleId>) -> Self {
        self.binds(id, HandleRole::Destroyed)
    }

    fn binds(mut self, id: Option<HandleId>, role: HandleRole) -> Self {
        if let Some(id) = id.filter(|_| self.wants(Fidelity::Handles)) {
            self.handles.push(HandleBinding { id, role });
        }
        self
    }

    /// Capture `len` bytes of host memory at `src` as the call's `role`
    /// payload: what the caller handed the driver, which a replay has to
    /// hand it again.
    ///
    /// # Safety
    /// `src` must be readable for `len` bytes for the duration of the call.
    pub unsafe fn payload(self, role: &'static str, src: *const u8, len: usize) -> Self {
        unsafe { self.stage(Fidelity::Replayable, role, src, len) }
    }

    /// Capture `len` bytes of host memory at `src` as a golden value: what
    /// the driver answered, kept so a later replay can be diffed against it.
    /// Only a golden capture pays for it.
    ///
    /// # Safety
    /// `src` must be readable for `len` bytes for the duration of the call.
    pub unsafe fn output(self, role: &'static str, src: *const u8, len: usize) -> Self {
        unsafe { self.stage(Fidelity::Golden, role, src, len) }
    }

    /// # Safety
    /// `src` must be readable for `len` bytes for the duration of the call.
    unsafe fn stage(
        mut self,
        least: Fidelity,
        role: &'static str,
        src: *const u8,
        len: usize,
    ) -> Self {
        if len == 0 || src.is_null() || !self.wants(least) {
            return self;
        }
        self.payloads.push(StagedPayload {
            role,
            bytes: unsafe { std::slice::from_raw_parts(src, len) }.to_vec(),
        });
        self
    }
}

/// How much one fidelity records, as an order: each level adds to the one
/// before it.
fn rank(fidelity: Fidelity) -> u8 {
    match fidelity {
        Fidelity::Metadata => 0,
        Fidelity::Handles => 1,
        Fidelity::Replayable => 2,
        Fidelity::Golden => 3,
    }
}

/// Attribute this thread's later calls to `device`. A call that changes
/// which context is current says nothing about a device in its arguments,
/// so its handler names the device here instead.
pub fn bind_device(device: Option<u32>) {
    let _ = with_local(|local| local.device = device);
}

/// The device this thread's calls are currently attributed to.
pub fn bound_device() -> Option<u32> {
    with_local(|local| local.device).flatten()
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
    calls: Vec<Pending>,
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
    buffer: AtomicPtr<Vec<Pending>>,
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
    fn claim(&self) -> Option<Box<Vec<Pending>>> {
        // Acquire pairs with the `publish` that last put the buffer back, so
        // the records stored through that pointer are visible here.
        let buffer = self.buffer.swap(ptr::null_mut(), Ordering::Acquire);
        (!buffer.is_null()).then(|| unsafe { Box::from_raw(buffer) })
    }

    fn publish(&self, buffer: Box<Vec<Pending>>) {
        self.buffer.store(Box::into_raw(buffer), Ordering::Release);
    }

    /// Take whatever the owning thread has buffered so far, for a shutdown
    /// that cannot wait for that thread to exit. A slot is only ever null
    /// for the few instructions its owner spends appending, so a bounded
    /// yield is enough to catch a thread that is still recording.
    fn steal(&self) -> Option<Vec<Pending>> {
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
            let Some(calls) = slot.steal() else {
                continue;
            };
            if calls.is_empty() {
                continue;
            }
            self.queue.push(Chunk {
                thread: slot.thread,
                calls,
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
    let mut observed = Observed::default();
    let outcome = (|| -> gputrace_core::bundle::Result<()> {
        loop {
            // Read the flag before draining: shutdown queues every thread's
            // buffer and only then sets it, so anything that existed when the
            // process began exiting is in the batch this pass takes.
            let shutting_down = queue.shutdown.load(Ordering::Acquire);
            let chunks = queue.drain();
            let idle = chunks.is_empty();
            for chunk in chunks {
                let segment = match segments.entry(chunk.thread) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(bundle.open_segment(chunk.thread)?),
                };
                for mut call in chunk.calls {
                    observed.saw(&call.record);
                    for staged in std::mem::take(&mut call.payloads) {
                        let blob = bundle.blobs().put(&staged.bytes)?;
                        call.record.payloads.push(PayloadRef {
                            role: staged.role.to_owned(),
                            blob,
                            len: staged.bytes.len() as u64,
                        });
                    }
                    segment.write(&call.record)?;
                }
            }
            if shutting_down {
                // Threads that are still recording keep queueing buffers, so
                // draining until the queue runs dry would keep a process with
                // a busy thread from ever leaving `exit`.
                return Ok(());
            }
            if idle {
                thread::park_timeout(WRITER_IDLE_POLL);
            }
        }
    })();

    let finished = outcome
        .and_then(|()| segments.into_values().try_for_each(SegmentWriter::finish))
        .and_then(|()| bundle.finish_with(observed.describe()));
    if let Err(e) = finished {
        eprintln!("gputrace: capture incomplete: {e}");
    }
}

/// What the trace says about the GPUs it ran against, accumulated by the
/// writer thread as records pass through it.
///
/// The driver is asked about a device the first time a record names one:
/// on this thread, so no capturing thread waits for the answer, and while
/// the target is still running, rather than at exit when the driver may
/// already be tearing itself down.
#[derive(Default)]
struct Observed {
    driver_version: Option<u32>,
    asked_for_the_version: bool,
    devices: BTreeMap<u32, DeviceSummary>,
}

impl Observed {
    fn saw(&mut self, record: &CallRecord) {
        if !self.asked_for_the_version {
            self.asked_for_the_version = true;
            self.driver_version = environment::driver_version();
        }
        let Some(ordinal) = record.device else {
            return;
        };
        self.devices
            .entry(ordinal)
            .or_insert_with(|| DeviceSummary {
                ordinal,
                compute_capability: environment::compute_capability(ordinal),
                records: 0,
            })
            .records += 1;
    }

    fn describe(self) -> Environment {
        Environment {
            driver_version: self.driver_version,
            devices: self.devices.into_values().collect(),
        }
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

/// How much every shim in this process records, read once. Unset means the
/// capture is replayable: a bundle is only worth keeping if it holds the
/// payloads a replay needs.
fn fidelity() -> Fidelity {
    static FIDELITY: OnceLock<Fidelity> = OnceLock::new();
    *FIDELITY.get_or_init(|| match env::var(FIDELITY_ENV).as_deref() {
        Ok("metadata") => Fidelity::Metadata,
        Ok("handles") => Fidelity::Handles,
        Ok("golden") => Fidelity::Golden,
        _ => Fidelity::Replayable,
    })
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

    fn push(&mut self, call: Pending) {
        let mut buffer = self
            .slot
            .claim()
            .unwrap_or_else(|| Box::new(Vec::with_capacity(CHUNK_RECORDS)));
        buffer.push(call);
        let full = buffer.len() >= CHUNK_RECORDS;
        let chunk =
            full.then(|| std::mem::replace(&mut *buffer, Vec::with_capacity(CHUNK_RECORDS)));
        self.slot.publish(buffer);
        if let Some(calls) = chunk {
            self.hand_over(calls);
        }
    }

    fn flush(&mut self) {
        let Some(mut buffer) = self.slot.claim() else {
            return;
        };
        let calls = std::mem::take(&mut *buffer);
        self.slot.publish(buffer);
        if !calls.is_empty() {
            self.hand_over(calls);
        }
    }

    fn hand_over(&self, calls: Vec<Pending>) {
        self.capture.queue.push(Chunk {
            thread: self.slot.thread,
            calls,
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
