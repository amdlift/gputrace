use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::trace::{CallRecord, Fidelity, ThreadId};

/// Bumped whenever the on-disk layout or record schema changes
/// incompatibly, so a consumer can refuse a bundle it cannot read.
pub const SCHEMA_VERSION: u32 = 1;

const MANIFEST_FILE: &str = "manifest.json";
const SEGMENTS_DIR: &str = "segments";

fn segment_path(root: &Path, thread: ThreadId) -> PathBuf {
    root.join(SEGMENTS_DIR)
        .join(format!("thread-{}.seg", thread.0))
}

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
    #[error("malformed segment {path}: {source}")]
    Segment {
        path: PathBuf,
        #[source]
        source: postcard::Error,
    },
    #[error("truncated segment {path}")]
    TruncatedSegment { path: PathBuf },
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

    /// Opens an independent writer for one host thread's records. Segment
    /// writers share no state, so a background drain thread can hold one per
    /// captured thread without any of them contending.
    pub fn open_segment(&self, thread: ThreadId) -> Result<SegmentWriter> {
        let path = segment_path(&self.root, thread);
        let dir = self.root.join(SEGMENTS_DIR);
        fs::create_dir_all(&dir).map_err(io(&dir))?;
        let file = File::create(&path).map_err(io(&path))?;
        Ok(SegmentWriter {
            path,
            out: BufWriter::new(file),
            scratch: Vec::new(),
        })
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

/// Appends one host thread's call records to its own segment file.
pub struct SegmentWriter {
    path: PathBuf,
    out: BufWriter<File>,
    scratch: Vec<u8>,
}

impl SegmentWriter {
    pub fn write(&mut self, record: &CallRecord) -> Result<()> {
        self.scratch.clear();
        postcard::to_io(record, &mut self.scratch).map_err(|source| BundleError::Segment {
            path: self.path.clone(),
            source,
        })?;
        let len = u32::try_from(self.scratch.len()).expect("call record exceeds 4 GiB");
        self.out
            .write_all(&len.to_le_bytes())
            .and_then(|()| self.out.write_all(&self.scratch))
            .map_err(io(&self.path))
    }

    pub fn finish(mut self) -> Result<()> {
        self.out.flush().map_err(io(&self.path))
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

    pub fn read_segment(&self, thread: ThreadId) -> Result<Vec<CallRecord>> {
        let path = segment_path(&self.root, thread);
        let bytes = fs::read(&path).map_err(io(&path))?;
        let mut records = Vec::new();
        let mut rest = bytes.as_slice();
        while !rest.is_empty() {
            let (header, body) = rest
                .split_at_checked(4)
                .ok_or_else(|| BundleError::TruncatedSegment { path: path.clone() })?;
            let len = u32::from_le_bytes(header.try_into().unwrap()) as usize;
            let (frame, tail) = body
                .split_at_checked(len)
                .ok_or_else(|| BundleError::TruncatedSegment { path: path.clone() })?;
            records.push(
                postcard::from_bytes(frame).map_err(|source| BundleError::Segment {
                    path: path.clone(),
                    source,
                })?,
            );
            rest = tail;
        }
        Ok(records)
    }
}
