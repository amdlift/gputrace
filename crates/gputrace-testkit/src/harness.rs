//! The capture harness: one `Harness` owns a scratch directory, one `run`
//! launches a target under the interposer and collects the result.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use gputrace_core::TraceBundle;
use tempfile::TempDir;

use crate::{TestkitError, artifacts};

/// Names the library the interposer forwards to.
pub const LIBCUDA_ENV: &str = "GPUTRACE_LIBCUDA";
/// Names the directory the interposer writes its trace bundle to.
pub const BUNDLE_ENV: &str = "GPUTRACE_OUT";
/// Names the file the stub appends each answered call to.
pub const STUB_LOG_ENV: &str = "GPUTRACE_STUB_LOG";

pub struct Harness {
    workdir: Arc<TempDir>,
    runs: AtomicUsize,
    extra_env: Vec<(OsString, OsString)>,
}

impl Harness {
    pub fn new() -> Result<Self, TestkitError> {
        Ok(Self {
            workdir: Arc::new(TempDir::new()?),
            runs: AtomicUsize::new(0),
            extra_env: Vec::new(),
        })
    }

    /// Set an extra environment variable on every target this harness runs.
    pub fn with_env(mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) -> Self {
        self.extra_env
            .push((key.as_ref().to_owned(), value.as_ref().to_owned()));
        self
    }

    /// Run `target` under the interposer with the stub driver behind it.
    pub fn run(&self, target: impl AsRef<Path>, args: &[&str]) -> Result<CaptureRun, TestkitError> {
        let interposer = artifacts::interposer()?;
        let stub = artifacts::stub_libcuda()?;

        let slot = self.runs.fetch_add(1, Ordering::Relaxed);
        let run_dir = self.workdir.path().join(format!("run-{slot}"));
        let bundle = run_dir.join("trace");
        let stub_log = run_dir.join("stub-calls.log");
        std::fs::create_dir_all(&run_dir)?;

        // The interposer comes first so its exported `cu*` win, and the stub is
        // preloaded behind it because a fixture target links no libcuda at all:
        // its undefined `cu*` have to bind to something in the global scope.
        let mut preload = interposer.as_os_str().to_owned();
        preload.push(":");
        preload.push(stub.as_os_str());

        let output = Command::new(target.as_ref())
            .args(args)
            .env("LD_PRELOAD", preload)
            .env(LIBCUDA_ENV, stub)
            .env(BUNDLE_ENV, &bundle)
            .env(STUB_LOG_ENV, &stub_log)
            .envs(self.extra_env.iter().map(|(k, v)| (k, v)))
            .output()?;

        let stub_calls = std::fs::read_to_string(&stub_log)
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect();

        Ok(CaptureRun {
            _workdir: Arc::clone(&self.workdir),
            status: output.status,
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            stub_calls,
            bundle,
        })
    }
}

/// What one capture run produced.
pub struct CaptureRun {
    _workdir: Arc<TempDir>,
    status: ExitStatus,
    stdout: String,
    stderr: String,
    stub_calls: Vec<String>,
    bundle: PathBuf,
}

impl CaptureRun {
    pub fn status(&self) -> ExitStatus {
        self.status
    }

    pub fn stdout(&self) -> &str {
        &self.stdout
    }

    pub fn stderr(&self) -> &str {
        &self.stderr
    }

    /// The `cu*` entry points the stub driver answered, in call order.
    pub fn stub_calls(&self) -> &[String] {
        &self.stub_calls
    }

    /// How many times the stub answered `symbol`. Order is not meaningful
    /// across host threads, so a concurrent target asserts on counts.
    pub fn stub_call_count(&self, symbol: &str) -> usize {
        self.stub_calls
            .iter()
            .filter(|call| *call == symbol)
            .count()
    }

    pub fn bundle_path(&self) -> &Path {
        &self.bundle
    }

    /// The trace bundle the interposer wrote, for assertion.
    pub fn bundle(&self) -> Result<TraceBundle, TestkitError> {
        Ok(TraceBundle::open(&self.bundle)?)
    }
}
