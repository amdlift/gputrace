//! The stub's in-memory device model. Deterministic by construction: handles
//! and device pointers are handed out from fixed, monotonically increasing
//! sequences so a single-threaded target produces identical values every run.

use std::sync::{Mutex, MutexGuard};

#[derive(Default)]
pub struct State {
    pub initialized: bool,
}

pub fn state() -> MutexGuard<'static, State> {
    static STATE: Mutex<State> = Mutex::new(State { initialized: false });
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}
