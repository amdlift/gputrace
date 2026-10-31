//! The stub's in-memory device model. Deterministic by construction: handles
//! and device pointers come from fixed, monotonically increasing sequences, so
//! a single-threaded target sees identical values on every run.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::c_void;
use std::sync::{LazyLock, Mutex, MutexGuard};

use crate::types::{CUdevice, CUdeviceptr};

const FIRST_HANDLE: usize = 0x0001_0000;
const HANDLE_STRIDE: usize = 0x100;
const FIRST_DEVICE_PTR: CUdeviceptr = 0x1000_0000;
const DEVICE_PTR_ALIGNMENT: CUdeviceptr = 0x1000;

pub struct State {
    pub initialized: bool,
    /// Every live context and the device it was created on, so the stub can
    /// answer `cuCtxGetDevice` for a process using more than one GPU.
    pub contexts: HashMap<usize, CUdevice>,
    pub allocations: BTreeMap<CUdeviceptr, Vec<u8>>,
    pub modules: HashMap<usize, Module>,
    pub functions: HashMap<usize, usize>,
    pub streams: HashSet<usize>,
    pub events: HashMap<usize, Option<u64>>,
    next_handle: usize,
    next_device_ptr: CUdeviceptr,
    next_record: u64,
}

thread_local! {
    static CURRENT_CONTEXT: Cell<usize> = const { Cell::new(0) };
}

#[derive(Default)]
pub struct Module {
    pub image: Vec<u8>,
    pub functions: HashMap<String, usize>,
}

impl Module {
    /// PTX declares each kernel with a `.entry` directive, so the stub can
    /// honestly answer `CUDA_ERROR_NOT_FOUND` for a name the image never
    /// defines instead of inventing a handle for anything asked of it.
    pub fn defines(&self, name: &str) -> bool {
        let needle = format!(".entry {name}");
        self.image
            .windows(needle.len())
            .any(|window| window == needle.as_bytes())
    }
}

impl State {
    pub fn fresh_handle(&mut self) -> usize {
        let handle = self.next_handle;
        self.next_handle += HANDLE_STRIDE;
        handle
    }

    /// A monotonic stamp per `cuEventRecord`. It stands in for the GPU clock
    /// so elapsed time between two recorded events is deterministic.
    pub fn next_record(&mut self) -> u64 {
        self.next_record += 1;
        self.next_record
    }

    pub fn allocate(&mut self, bytes: usize) -> CUdeviceptr {
        let base = self.next_device_ptr;
        let span = (bytes.max(1) as CUdeviceptr).next_multiple_of(DEVICE_PTR_ALIGNMENT);
        self.next_device_ptr += span;
        self.allocations.insert(base, vec![0; bytes]);
        base
    }

    /// The allocation backing `[ptr, ptr + len)`, or `None` when the span
    /// escapes every live allocation.
    pub fn span(&mut self, ptr: CUdeviceptr, len: usize) -> Option<&mut [u8]> {
        let (&base, block) = self.allocations.range_mut(..=ptr).next_back()?;
        let offset = (ptr - base) as usize;
        block.get_mut(offset..offset.checked_add(len)?)
    }
}

impl Default for State {
    fn default() -> Self {
        Self {
            initialized: false,
            contexts: HashMap::new(),
            allocations: BTreeMap::new(),
            modules: HashMap::new(),
            functions: HashMap::new(),
            streams: HashSet::new(),
            events: HashMap::new(),
            next_handle: FIRST_HANDLE,
            next_device_ptr: FIRST_DEVICE_PTR,
            next_record: 0,
        }
    }
}

pub fn state() -> MutexGuard<'static, State> {
    static STATE: LazyLock<Mutex<State>> = LazyLock::new(Mutex::default);
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// CUDA's current context is per host thread, and capture tests depend on that
/// so a multi-threaded target behaves the way the real driver makes it behave.
pub fn current_context() -> usize {
    CURRENT_CONTEXT.with(Cell::get)
}

pub fn set_current_context(handle: usize) {
    CURRENT_CONTEXT.with(|current| current.set(handle));
}

pub fn as_handle(handle: usize) -> *mut c_void {
    handle as *mut c_void
}
