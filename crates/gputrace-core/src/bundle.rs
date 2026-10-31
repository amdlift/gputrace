use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::trace::Fidelity;

/// Bumped whenever the on-disk layout or record schema changes
/// incompatibly, so a consumer can refuse a bundle it cannot read.
pub const SCHEMA_VERSION: u32 = 1;

const MANIFEST_FILE: &str = "manifest.json";

#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("malformed manifest {path}: {source}")]
    Manifest {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

pub type Result<T> = std::result::Result<T, BundleError>;

fn io(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> BundleError {
    let path = path.into();
    move |source| BundleError::Io { path, source }
}

/// What the capture was asked to do. Round-trips through `manifest.json`
/// so a consumer can tell which fidelity produced the records it reads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CaptureConfig {
    pub backend: String,
    pub fidelity: Fidelity,
    pub device: Option<u32>,
    pub command: Vec<String>,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            backend: String::new(),
            fidelity: Fidelity::Replayable,
            device: None,
            command: Vec::new(),
        }
    }
}

/// The human-readable index of a trace bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub capture: CaptureConfig,
}

/// Writes a trace bundle directory.
pub struct BundleWriter {
    root: PathBuf,
    capture: CaptureConfig,
}

impl BundleWriter {
    pub fn create(root: impl AsRef<Path>, capture: CaptureConfig) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(io(&root))?;
        Ok(Self { root, capture })
    }

    pub fn path(&self) -> &Path {
        &self.root
    }

    pub fn finish(self) -> Result<()> {
        let manifest = Manifest {
            schema_version: SCHEMA_VERSION,
            capture: self.capture,
        };
        let path = self.root.join(MANIFEST_FILE);
        let json =
            serde_json::to_string_pretty(&manifest).map_err(|source| BundleError::Manifest {
                path: path.clone(),
                source,
            })?;
        fs::write(&path, format!("{json}\n")).map_err(io(&path))?;
        Ok(())
    }
}

/// Reads a trace bundle directory written by [`BundleWriter`].
pub struct TraceBundle {
    root: PathBuf,
    manifest: Manifest,
}

impl TraceBundle {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let path = root.join(MANIFEST_FILE);
        let json = fs::read_to_string(&path).map_err(io(&path))?;
        let manifest =
            serde_json::from_str(&json).map_err(|source| BundleError::Manifest { path, source })?;
        Ok(Self { root, manifest })
    }

    pub fn path(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
}
