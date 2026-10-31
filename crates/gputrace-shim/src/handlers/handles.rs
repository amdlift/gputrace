//! Stable identity for the handles and device pointers a deep handler sees.
//!
//! An id is *derived* from the value the driver handed out, never allocated.
//! That is what lets a buffer allocated on one thread and used on another
//! record as one handle: every thread computes the same id from the same raw
//! value, so identity needs no shared map and no shared counter, and the
//! record path stays lock-free (ADR-0004).

use gputrace_core::HandleId;

/// What a raw driver value names. Two kinds can hand out the same bit
/// pattern, so the kind is part of what the id is derived from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HandleKind {
    Context,
    DevicePtr,
}

impl HandleKind {
    /// Distinct odd constants: mixing one in keeps a context and a device
    /// pointer of equal value apart.
    fn salt(self) -> u64 {
        match self {
            Self::Context => 0x9e37_79b9_7f4a_7c15,
            Self::DevicePtr => 0xc2b2_ae3d_27d4_eb4f,
        }
    }
}

/// The id standing for `raw`, or `None` when `raw` names nothing — a null
/// context or a null device pointer is an absence, not a handle.
pub(crate) fn handle_id(kind: HandleKind, raw: u64) -> Option<HandleId> {
    (raw != 0).then(|| HandleId(mix(raw ^ kind.salt())))
}

/// SplitMix64's finalizer: a bijection, so distinct inputs stay distinct.
fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{HandleKind, handle_id};

    /// The stub driver's sequences: handles from `0x10000` stride `0x100`,
    /// device pointers from `0x10000000` on a 4 KiB granularity.
    fn stub_handles() -> Vec<u64> {
        (0..1024).map(|n| 0x1_0000 + n * 0x100).collect()
    }

    fn stub_device_pointers() -> Vec<u64> {
        (0..1024).map(|n| 0x1000_0000 + n * 0x1000).collect()
    }

    #[test]
    fn one_raw_handle_always_has_one_id() {
        let first = handle_id(HandleKind::DevicePtr, 0x1000_0000);
        for raw in stub_handles() {
            handle_id(HandleKind::Context, raw);
        }
        assert_eq!(handle_id(HandleKind::DevicePtr, 0x1000_0000), first);
    }

    #[test]
    fn a_context_and_a_device_pointer_of_the_same_value_are_different_handles() {
        assert_ne!(
            handle_id(HandleKind::Context, 0x1_0000),
            handle_id(HandleKind::DevicePtr, 0x1_0000)
        );
    }

    #[test]
    fn distinct_handles_get_distinct_ids() {
        let ids: HashSet<_> = stub_handles()
            .into_iter()
            .map(|raw| handle_id(HandleKind::Context, raw))
            .chain(
                stub_device_pointers()
                    .into_iter()
                    .map(|raw| handle_id(HandleKind::DevicePtr, raw)),
            )
            .collect();
        assert_eq!(ids.len(), 2048);
    }

    #[test]
    fn a_null_handle_names_nothing() {
        assert_eq!(handle_id(HandleKind::Context, 0), None);
        assert_eq!(handle_id(HandleKind::DevicePtr, 0), None);
    }

    /// The identity rule capture depends on: a pointer allocated on one
    /// thread and used on another must record as one handle, with no shared
    /// map or counter in between (ADR-0004).
    #[test]
    fn threads_agree_on_the_id_of_a_handle_none_of_them_produced() {
        let raws = stub_device_pointers();
        let expected: Vec<_> = raws
            .iter()
            .map(|&raw| handle_id(HandleKind::DevicePtr, raw))
            .collect();

        std::thread::scope(|scope| {
            for _ in 0..8 {
                let (raws, expected) = (&raws, &expected);
                scope.spawn(move || {
                    let seen: Vec<_> = raws
                        .iter()
                        .map(|&raw| handle_id(HandleKind::DevicePtr, raw))
                        .collect();
                    assert_eq!(&seen, expected);
                });
            }
        });
    }
}
