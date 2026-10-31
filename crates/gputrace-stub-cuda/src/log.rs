//! Every entry point the stub answers is appended, one name per line, to the
//! file named by `GPUTRACE_STUB_LOG`. It is how a test observes that a
//! target's calls actually reached the stub rather than a real driver.

use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

pub const STUB_LOG_ENV: &str = "GPUTRACE_STUB_LOG";

fn log_path() -> Option<&'static PathBuf> {
    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| {
        env::var_os(STUB_LOG_ENV)
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
    })
    .as_ref()
}

pub fn note(symbol: &str) {
    let Some(path) = log_path() else { return };
    // O_APPEND keeps one line atomic, so concurrent host threads interleave
    // whole call names rather than fragments.
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(format!("{symbol}\n").as_bytes());
    }
}
