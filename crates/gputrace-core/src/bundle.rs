use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::trace::{BlobId, CallRecord, Fidelity, ThreadId};

/// Bumped whenever the on-disk layout or record schema changes
/// incompatibly, so a consumer can refuse a bundle it cannot read.
pub const SCHEMA_VERSION: u32 = 1;

const MANIFEST_FILE: &str = "manifest.json";
const SEGMENTS_DIR: &str = "segments";
const BLOBS_DIR: &str = "blobs";

static STAGING_SEQ: AtomicU64 = AtomicU64::new(0);

fn segment_name(thread: ThreadId) -> String {
    format!("{SEGMENTS_DIR}/thread-{}.seg", thread.0)
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

/// The bundle's content-addressed payload store. Cloning yields another
/// handle to the same directory, so every capturing thread can share one
/// store without coordinating: identical payloads collapse onto one file.
#[derive(Debug, Clone)]
pub struct BlobStore {
    dir: Arc<PathBuf>,
}

impl BlobStore {
    fn at(root: &Path) -> Self {
        Self {
            dir: Arc::new(root.join(BLOBS_DIR)),
        }
    }

    fn create(root: &Path) -> Result<Self> {
        let store = Self::at(root);
        fs::create_dir_all(store.dir.as_path()).map_err(io(store.dir.as_path()))?;
        Ok(store)
    }

    /// Stores `payload` and returns its content hash. A payload already in
    /// the store is not rewritten.
    pub fn put(&self, payload: &[u8]) -> Result<BlobId> {
        let id = BlobId(blake3::hash(payload).to_hex().to_string());
        let path = self.dir.join(&id.0);
        if path.exists() {
            return Ok(id);
        }
        // Publish via rename so a reader never sees a half-written blob. The
        // staging name must be unique per attempt: two threads storing the
        // same payload would otherwise share it and the loser's rename would
        // find the file already consumed.
        let staging = self.dir.join(format!(
            ".{}.{}.partial",
            id.0,
            STAGING_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&staging, payload).map_err(io(&staging))?;
        fs::rename(&staging, &path).map_err(io(&path))?;
        Ok(id)
    }

    pub fn get(&self, id: &BlobId) -> Result<Vec<u8>> {
        let path = self.dir.join(&id.0);
        fs::read(&path).map_err(io(&path))
    }
}

/// One captured host thread: where its records live and how many there are.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadSummary {
    pub thread: ThreadId,
    /// Path of the thread's segment file, relative to the bundle root.
    pub segment: String,
    pub records: u64,
}

/// The human-readable index of a trace bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub capture: CaptureConfig,
    pub threads: Vec<ThreadSummary>,
}

/// Writes a trace bundle directory.
pub struct BundleWriter {
    root: PathBuf,
    capture: CaptureConfig,
    blobs: BlobStore,
    threads: Mutex<Vec<(ThreadId, Arc<AtomicU64>)>>,
}

impl BundleWriter {
    pub fn create(root: impl AsRef<Path>, capture: CaptureConfig) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(io(&root))?;
        let blobs = BlobStore::create(&root)?;
        Ok(Self {
            root,
            capture,
            blobs,
            threads: Mutex::new(Vec::new()),
        })
    }

    pub fn path(&self) -> &Path {
        &self.root
    }

    /// The payload store shared by every segment writer in this bundle.
    pub fn blobs(&self) -> &BlobStore {
        &self.blobs
    }

    /// Opens an independent writer for one host thread's records. Segment
    /// writers share no state, so a background drain thread can hold one per
    /// captured thread without any of them contending.
    pub fn open_segment(&self, thread: ThreadId) -> Result<SegmentWriter> {
        let dir = self.root.join(SEGMENTS_DIR);
        fs::create_dir_all(&dir).map_err(io(&dir))?;
        let path = self.root.join(segment_name(thread));
        let file = File::create(&path).map_err(io(&path))?;
        let records = Arc::new(AtomicU64::new(0));
        self.threads
            .lock()
            .expect("bundle thread registry poisoned")
            .push((thread, Arc::clone(&records)));
        Ok(SegmentWriter {
            path,
            out: BufWriter::new(file),
            scratch: Vec::new(),
            records,
        })
    }

    pub fn finish(self) -> Result<()> {
        let threads = self
            .threads
            .into_inner()
            .expect("bundle thread registry poisoned")
            .into_iter()
            .map(|(thread, records)| ThreadSummary {
                thread,
                segment: segment_name(thread),
                records: records.load(Ordering::Relaxed),
            })
            .collect();
        let manifest = Manifest {
            schema_version: SCHEMA_VERSION,
            capture: self.capture,
            threads,
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
    records: Arc<AtomicU64>,
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
            .map_err(io(&self.path))?;
        self.records.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.out.flush().map_err(io(&self.path))
    }
}

/// Reads a trace bundle directory written by [`BundleWriter`].
pub struct TraceBundle {
    root: PathBuf,
    manifest: Manifest,
    blobs: BlobStore,
}

impl TraceBundle {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let path = root.join(MANIFEST_FILE);
        let json = fs::read_to_string(&path).map_err(io(&path))?;
        let manifest =
            serde_json::from_str(&json).map_err(|source| BundleError::Manifest { path, source })?;
        let blobs = BlobStore::at(&root);
        Ok(Self {
            root,
            manifest,
            blobs,
        })
    }

    pub fn path(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn blobs(&self) -> &BlobStore {
        &self.blobs
    }

    pub fn read_blob(&self, id: &BlobId) -> Result<Vec<u8>> {
        self.blobs.get(id)
    }

    pub fn threads(&self) -> &[ThreadSummary] {
        &self.manifest.threads
    }

    pub fn read_segment(&self, thread: ThreadId) -> Result<Vec<CallRecord>> {
        let path = self.root.join(segment_name(thread));
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
