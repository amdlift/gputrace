//! Which GPU a call targeted, and what the finished trace says about the GPUs
//! it ran against.
//!
//! Attribution is inferred from the call's own name and arguments and from
//! what its thread last bound, never by asking the driver: a `cuCtxGetDevice`
//! per record would serialize on driver state, which is the one thing capture
//! must not do (ADR-0004).

use std::collections::BTreeMap;

use gputrace_core::{ArgValue, Argument, CallRecord, DeviceSummary, Environment};

use crate::environment;

/// What one call says about the device it targeted. A call that names a
/// device targets that one; everything else targets whatever its thread's
/// current context does, which is how a device-agnostic call in a multi-GPU
/// process is still attributable to a GPU.
pub(super) enum TargetDevice {
    /// The call named a device and made it the calling thread's.
    Bound(u32),
    /// The call named a device without changing what its thread targets.
    Named(u32),
    /// The call ended the calling thread's context.
    Released,
    /// The call says nothing; it targets what its thread already did.
    Inherited,
}

pub(super) fn target_device(function: &str, args: &[Argument]) -> TargetDevice {
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

/// The `CUdevice` a call passed by value. A `CUdevice` out-parameter is
/// recorded as a pointer, so a call that only asks for a device never looks
/// like one that targets it.
fn named_device(args: &[Argument]) -> Option<u32> {
    args.iter()
        .find_map(|arg| match (arg.name.as_str(), &arg.value) {
            ("dev" | "device", ArgValue::Signed(device)) => u32::try_from(*device).ok(),
            ("dev" | "device", ArgValue::Unsigned(device)) => u32::try_from(*device).ok(),
            _ => None,
        })
}

/// What the trace says about the GPUs it ran against, accumulated by the
/// writer thread as records pass through it.
///
/// The driver is asked about a device the first time a record names one:
/// on this thread, so no capturing thread waits for the answer, and while
/// the target is still running, rather than at exit when the driver may
/// already be tearing itself down.
#[derive(Default)]
pub(super) struct Observed {
    driver_version: Option<u32>,
    asked_for_the_version: bool,
    devices: BTreeMap<u32, DeviceSummary>,
}

impl Observed {
    pub(super) fn saw(&mut self, record: &CallRecord) {
        if !self.asked_for_the_version {
            self.asked_for_the_version = true;
            self.driver_version = environment::driver_version();
        }
        let Some(device) = record.device else {
            return;
        };
        self.devices
            .entry(device)
            .or_insert_with(|| DeviceSummary {
                device,
                compute_capability: environment::compute_capability(device),
                records: 0,
            })
            .records += 1;
    }

    pub(super) fn describe(self) -> Environment {
        Environment {
            driver_version: self.driver_version,
            devices: self.devices.into_values().collect(),
        }
    }
}

/// What a call says about its device decides a record's attribution in a
/// multi-GPU process. The cases below are ones no shimmed entry point
/// reaches yet, so they are pinned here rather than through the bundle.
#[cfg(test)]
mod target_device_tests {
    use super::{TargetDevice, target_device};

    use gputrace_core::{ArgValue, Argument};

    fn by_value(name: &str, device: i64) -> Vec<Argument> {
        vec![Argument::new(name, ArgValue::Signed(device))]
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
