//! The hand-written shims, grouped by the part of the Driver API they cover.
//! Each module is a `shim!` block; see [`crate::shim`] for the pattern.

mod context;
mod golden;
mod handles;
mod init;
mod launch;
mod memory;
mod module;
mod stream;
