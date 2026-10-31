//! Who a captured [`HandleId`] actually names, recovered offline.
//!
//! Capture derives an id from the raw value the driver handed out, which is
//! what lets threads agree on it with no shared map, no counter and no lock
//! on the record path (ADR-0004). The price is that the id is not injective
//! over a whole run: a driver that frees an allocation and hands the same
//! address back names two unrelated objects with one id, and a replay that
//! remapped by id alone would send the second object's work to the first.
//!
//! The answer is the same one capture gives ordering: record enough to
//! recover identity offline rather than enforce it while recording. Each
//! record already says which handles the call produced, consumed and
//! destroyed, so the lifetimes behind one raw id are recoverable by walking
//! those bindings — and where the trace does not prove which lifetime a call
//! named, this says so rather than guessing.
//!
//! What it can prove is bounded by what capture stamps. Two calls on one host
//! thread are totally ordered by their sequence numbers. Two calls on
//! different threads are not: `timestamp_ns` is read when the call *begins*,
//! so a free racing a reallocation on another thread can be stamped in the
//! opposite order to the one the driver actually applied. Those calls resolve
//! to [`Resolution::Ambiguous`] — see `docs/adr/0006`.

use std::collections::HashMap;

use crate::bundle::{Result, TraceBundle};
use crate::trace::{CallRecord, HandleId, HandleRole, ThreadId};

/// Where one call sits in the trace: the only coordinate capture records, and
/// a total order within a thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CallSite {
    pub thread: ThreadId,
    pub thread_seq: u64,
}

impl CallSite {
    pub fn new(thread: ThreadId, thread_seq: u64) -> Self {
        Self { thread, thread_seq }
    }

    /// Whether the trace proves this call finished before `later` began.
    /// Only a thread's own sequence says so; across threads the stamps are
    /// taken at call begin and prove nothing.
    fn provably_before(self, later: Self) -> bool {
        self.thread == later.thread && self.thread_seq < later.thread_seq
    }
}

/// One object behind a raw handle: its id, plus which of that id's lifetimes
/// this is. Generation 0 is the first lifetime the trace saw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectId {
    pub handle: HandleId,
    pub generation: u32,
}

/// Which object a call meant by the raw handle it named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    /// Exactly one lifetime of the raw handle fits.
    Resolved(ObjectId),
    /// Several fit and the trace orders this call against none of them, so
    /// which object was meant is not recoverable. Guessing one would be a
    /// replay silently targeting the wrong allocation.
    Ambiguous(Vec<ObjectId>),
    /// No lifetime fits: the call named a handle the trace had already seen
    /// destroyed, or had not yet seen created.
    Unmatched,
}

/// One lifetime of a raw handle, from the call that produced it to the call
/// that ended it. Either end is absent when the trace never saw it.
#[derive(Debug, Clone, Copy)]
struct Lifetime {
    produced: Option<CallSite>,
    destroyed: Option<CallSite>,
}

impl Lifetime {
    fn excludes(&self, site: CallSite) -> bool {
        self.produced
            .is_some_and(|produced| site.provably_before(produced))
            || self
                .destroyed
                .is_some_and(|destroyed| destroyed.provably_before(site))
    }
}

/// The lifetimes behind a trace's raw handle ids, and which one each call
/// named.
#[derive(Debug, Default)]
pub struct HandleIdentities {
    lifetimes: HashMap<HandleId, Vec<Lifetime>>,
    resolutions: HashMap<(CallSite, HandleId), Resolution>,
    ambiguous: Vec<(CallSite, HandleId)>,
}

impl HandleIdentities {
    pub fn resolve(bundle: &TraceBundle) -> Result<Self> {
        let segments = bundle
            .threads()
            .iter()
            .map(|summary| Ok((summary.thread, bundle.read_segment(summary.thread)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::from_segments(segments))
    }

    pub fn from_segments(segments: impl IntoIterator<Item = (ThreadId, Vec<CallRecord>)>) -> Self {
        let bindings = ordered_bindings(segments);
        let mut identities = Self::default();
        let mut generation_of = HashMap::new();
        for binding in &bindings {
            identities.open_or_close(binding, &mut generation_of);
        }
        for binding in &bindings {
            let resolution = match generation_of.get(&(binding.site, binding.handle)) {
                Some(&generation) => Resolution::Resolved(ObjectId {
                    handle: binding.handle,
                    generation,
                }),
                None => identities.fitting(binding),
            };
            if matches!(resolution, Resolution::Ambiguous(_)) {
                identities.ambiguous.push((binding.site, binding.handle));
            }
            identities
                .resolutions
                .insert((binding.site, binding.handle), resolution);
        }
        identities
    }

    /// Which object the call at `site` named by `handle`, or `None` when that
    /// call bound no such handle.
    pub fn object(&self, site: CallSite, handle: HandleId) -> Option<&Resolution> {
        self.resolutions.get(&(site, handle))
    }

    /// How many distinct objects the trace saw behind one raw id.
    pub fn lifetimes(&self, handle: HandleId) -> usize {
        self.lifetimes.get(&handle).map_or(0, Vec::len)
    }

    /// Every call the trace cannot attribute to a single object. Empty is the
    /// answer a consumer wants; a non-empty list is what a replay has to
    /// refuse rather than guess at.
    pub fn ambiguities(&self) -> &[(CallSite, HandleId)] {
        &self.ambiguous
    }

    /// Record what a producing or destroying binding does to a handle's
    /// lifetimes, and which generation that binding itself names.
    fn open_or_close(
        &mut self,
        binding: &Binding,
        generation_of: &mut HashMap<(CallSite, HandleId), u32>,
    ) {
        let lifetimes = self.lifetimes.entry(binding.handle).or_default();
        let generation = match binding.role {
            HandleRole::Produced => {
                lifetimes.push(Lifetime {
                    produced: Some(binding.site),
                    destroyed: None,
                });
                lifetimes.len() - 1
            }
            HandleRole::Destroyed => {
                match lifetimes.last_mut() {
                    Some(open) if open.destroyed.is_none() => open.destroyed = Some(binding.site),
                    // A handle the capture never saw created still has a
                    // lifetime; the trace just joined it late.
                    _ => lifetimes.push(Lifetime {
                        produced: None,
                        destroyed: Some(binding.site),
                    }),
                }
                lifetimes.len() - 1
            }
            HandleRole::Consumed => {
                if lifetimes.is_empty() {
                    lifetimes.push(Lifetime {
                        produced: None,
                        destroyed: None,
                    });
                }
                return;
            }
        };
        generation_of.insert(
            (binding.site, binding.handle),
            u32::try_from(generation).expect("a handle reused 4 billion times"),
        );
    }

    fn fitting(&self, binding: &Binding) -> Resolution {
        let candidates: Vec<_> = self
            .lifetimes
            .get(&binding.handle)
            .into_iter()
            .flatten()
            .enumerate()
            .filter(|(_, lifetime)| !lifetime.excludes(binding.site))
            .filter_map(|(generation, _)| {
                Some(ObjectId {
                    handle: binding.handle,
                    generation: u32::try_from(generation).ok()?,
                })
            })
            .collect();
        match <[ObjectId; 1]>::try_from(candidates) {
            Ok([only]) => Resolution::Resolved(only),
            Err(several) if several.is_empty() => Resolution::Unmatched,
            Err(several) => Resolution::Ambiguous(several),
        }
    }
}

struct Binding {
    site: CallSite,
    handle: HandleId,
    role: HandleRole,
}

/// Every handle binding in the trace, in the order the calls were issued.
///
/// There is no global counter to sort by, by construction (ADR-0004), so the
/// clock orders the threads against each other and each thread's own sequence
/// settles two calls that read the same nanosecond.
fn ordered_bindings(
    segments: impl IntoIterator<Item = (ThreadId, Vec<CallRecord>)>,
) -> Vec<Binding> {
    let mut stamped: Vec<_> = segments
        .into_iter()
        .flat_map(|(thread, records)| {
            records.into_iter().flat_map(move |record| {
                let site = CallSite::new(thread, record.thread_seq);
                let timestamp = record.timestamp_ns;
                record.handles.into_iter().map(move |binding| {
                    (
                        timestamp,
                        Binding {
                            site,
                            handle: binding.id,
                            role: binding.role,
                        },
                    )
                })
            })
        })
        .collect();
    stamped.sort_by_key(|(timestamp, binding)| (*timestamp, binding.site));
    stamped.into_iter().map(|(_, binding)| binding).collect()
}
