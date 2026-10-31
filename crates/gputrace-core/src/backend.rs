/// A concrete accelerator API family the interposer can target.
///
/// CUDA (Driver API) is the first and only implementation. Capture, replay,
/// and mutate are defined against this trait so the core stays vendor-neutral.
pub trait Backend {
    /// Short backend name, e.g. `"cuda"`.
    fn name(&self) -> &str;

    /// The runtime library whose real symbols the shim resolves,
    /// e.g. `"libcuda.so.1"`.
    fn library(&self) -> &str;
}
