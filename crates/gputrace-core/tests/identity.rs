//! Recovering which object a reused raw handle named, and admitting it when
//! the trace cannot say (ADR-0006).

use gputrace_core::identity::{CallSite, HandleIdentities, Resolution};
use gputrace_core::{CallRecord, HandleBinding, HandleId, HandleRole, ThreadId};

const BUFFER: HandleId = HandleId(0xb0);

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

    /// One call, stamped at `timestamp_ns` so threads can be interleaved the
    /// way the reconstruction will see them.
    fn call(mut self, function: &str, timestamp_ns: u64, role: HandleRole) -> Self {
        let seq = self.records.len() as u64;
        let mut record = CallRecord::new(function, seq, timestamp_ns, 0);
        record.handles = vec![HandleBinding { id: BUFFER, role }];
        self.records.push(record);
        self
    }

    fn site(&self, thread_seq: u64) -> CallSite {
        CallSite::new(self.thread, thread_seq)
    }

    fn segment(&self) -> (ThreadId, Vec<CallRecord>) {
        (self.thread, self.records.clone())
    }
}

#[test]
fn one_thread_reusing_an_address_keeps_its_two_objects_apart() {
    let trace = Trace::new(1)
        .call("cuMemAlloc_v2", 10, HandleRole::Produced)
        .call("cuMemcpyHtoD_v2", 20, HandleRole::Consumed)
        .call("cuMemFree_v2", 30, HandleRole::Destroyed)
        .call("cuMemAlloc_v2", 40, HandleRole::Produced)
        .call("cuMemcpyHtoD_v2", 50, HandleRole::Consumed)
        .call("cuMemFree_v2", 60, HandleRole::Destroyed);
    let identities = HandleIdentities::from_segments([trace.segment()]);

    assert_eq!(identities.lifetimes(BUFFER), 2);
    assert!(identities.ambiguities().is_empty());

    let first = identities
        .object(trace.site(1), BUFFER)
        .expect("first copy");
    let second = identities
        .object(trace.site(4), BUFFER)
        .expect("second copy");
    assert_ne!(first, second);
    assert!(matches!(first, Resolution::Resolved(_)));
    assert!(matches!(second, Resolution::Resolved(_)));
}

#[test]
fn a_handle_the_trace_never_saw_created_is_still_one_object() {
    let trace = Trace::new(1)
        .call("cuStreamWaitEvent", 10, HandleRole::Consumed)
        .call("cuLaunchKernel", 20, HandleRole::Consumed);
    let identities = HandleIdentities::from_segments([trace.segment()]);

    assert_eq!(identities.lifetimes(BUFFER), 1);
    assert!(identities.ambiguities().is_empty());
    assert!(matches!(
        identities.object(trace.site(0), BUFFER),
        Some(Resolution::Resolved(_))
    ));
}

/// The case the trace genuinely cannot settle: one thread releases the
/// address and another takes it, so a third thread's use of that address sits
/// between two lifetimes with nothing ordering it against either.
#[test]
fn a_use_racing_a_reuse_on_another_thread_is_reported_not_guessed() {
    let owner = Trace::new(1)
        .call("cuMemAlloc_v2", 10, HandleRole::Produced)
        .call("cuMemFree_v2", 30, HandleRole::Destroyed);
    let taker = Trace::new(2).call("cuMemAlloc_v2", 40, HandleRole::Produced);
    let user = Trace::new(3).call("cuMemcpyHtoD_v2", 35, HandleRole::Consumed);
    let identities =
        HandleIdentities::from_segments([owner.segment(), taker.segment(), user.segment()]);

    assert_eq!(identities.lifetimes(BUFFER), 2);
    let Some(Resolution::Ambiguous(candidates)) = identities.object(user.site(0), BUFFER) else {
        panic!("nothing in the trace orders this copy against either allocation");
    };
    assert_eq!(candidates.len(), 2);
    assert_eq!(identities.ambiguities(), [(user.site(0), BUFFER)]);
}

#[test]
fn a_call_naming_an_address_its_own_thread_already_released_fits_no_object() {
    let trace = Trace::new(1)
        .call("cuMemAlloc_v2", 10, HandleRole::Produced)
        .call("cuMemFree_v2", 20, HandleRole::Destroyed)
        .call("cuMemcpyHtoD_v2", 30, HandleRole::Consumed);
    let identities = HandleIdentities::from_segments([trace.segment()]);

    assert_eq!(
        identities.object(trace.site(2), BUFFER),
        Some(&Resolution::Unmatched)
    );
}
