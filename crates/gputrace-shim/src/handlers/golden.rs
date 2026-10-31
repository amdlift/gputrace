//! Where an asynchronous read-back's answer is captured.
//!
//! A synchronous `cuMemcpyDtoH` can be captured where it stands: the driver
//! has filled the destination by the time it returns. An asynchronous one has
//! not. Its destination holds whatever the caller last left there until some
//! later call blocks for the copy, and that later call is the only moment the
//! bytes mean anything — which is why `--golden` saw nothing at all for the
//! async-copy-then-synchronize pattern that dominates real CUDA output.
//!
//! So the copy stages its destination, and the synchronization that makes it
//! valid captures it, attributing the payload to the device allocation it
//! mirrors ([`PayloadRef::source`](gputrace_core::PayloadRef)).
//!
//! Staging is per host thread and takes no lock (ADR-0004), which bounds what
//! this can see: a read-back one thread issues and *another* synchronizes is
//! not captured. See `docs/golden-outputs.md`.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;

use gputrace_core::HandleId;

use crate::recorder::{self, DeepCall};

/// What a captured device-to-host result is called. The name a later
/// oracle's differential signal looks for (GLOSSARY: golden output).
pub(crate) const GOLDEN_OUTPUT: &str = "golden_output";

/// One read-back whose destination is not valid yet.
struct Staged {
    destination: *mut u8,
    len: usize,
    source: Option<HandleId>,
    stream: HandleId,
    issued: u64,
}

thread_local! {
    static PENDING: RefCell<Vec<Staged>> = const { RefCell::new(Vec::new()) };
    /// Counts the read-backs this thread issued, so an event can stand for
    /// the ones that preceded its recording and no others.
    static ISSUED: Cell<u64> = const { Cell::new(0) };
    /// The stream each event was last recorded on, and how far this thread
    /// had got when it was.
    static EVENT_MARKS: RefCell<HashMap<HandleId, (HandleId, u64)>> =
        RefCell::new(HashMap::new());
}

/// What a blocking call makes valid.
pub(crate) enum Settles {
    /// Everything issued into one stream.
    Stream(HandleId),
    /// Everything issued into the stream an event was recorded on, up to the
    /// point it was recorded.
    Event(HandleId),
    /// Everything this thread issued.
    Thread,
}

/// Hold `destination` until a later call blocks for the copy filling it.
///
/// # Safety
/// `destination` must stay readable for `len` bytes until that call.
pub(crate) unsafe fn stage(
    destination: *mut c_void,
    len: usize,
    source: Option<HandleId>,
    stream: HandleId,
) {
    if destination.is_null() || len == 0 || !recorder::wants_golden() {
        return;
    }
    let issued = ISSUED.with(|issued| {
        let next = issued.get() + 1;
        issued.set(next);
        next
    });
    PENDING.with_borrow_mut(|pending| {
        pending.push(Staged {
            destination: destination.cast::<u8>(),
            len,
            source,
            stream,
            issued,
        });
    });
}

/// Note that `event` now stands for everything issued into `stream` so far.
pub(crate) fn recorded(event: Option<HandleId>, stream: HandleId) {
    let Some(event) = event.filter(|_| recorder::wants_golden()) else {
        return;
    };
    let issued = ISSUED.with(Cell::get);
    EVENT_MARKS.with_borrow_mut(|marks| marks.insert(event, (stream, issued)));
}

/// Capture every staged read-back `what` has just made valid, onto the record
/// of the call that did the blocking.
///
/// # Safety
/// Must be called only after the real driver has answered that call, and only
/// when it answered successfully: until then the destinations hold nothing.
pub(crate) unsafe fn settle(deep: DeepCall, what: Settles) -> DeepCall {
    if !recorder::wants_golden() {
        return deep;
    }
    let settled = PENDING.with_borrow_mut(|pending| {
        let mut settled = Vec::new();
        pending.retain(|staged| {
            if !is_settled(staged, &what) {
                return true;
            }
            settled.push((staged.destination, staged.len, staged.source));
            false
        });
        settled
    });
    settled
        .into_iter()
        .fold(deep, |deep, (destination, len, source)| unsafe {
            deep.output(GOLDEN_OUTPUT, destination, len, source)
        })
}

fn is_settled(staged: &Staged, what: &Settles) -> bool {
    match what {
        Settles::Stream(stream) => staged.stream == *stream,
        Settles::Event(event) => EVENT_MARKS
            .with_borrow(|marks| marks.get(event).copied())
            .is_some_and(|(stream, mark)| staged.stream == stream && staged.issued <= mark),
        Settles::Thread => true,
    }
}
