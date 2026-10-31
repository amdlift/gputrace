# The real-GPU smoke test

Every gating capture test drives the stub driver (`GPUTRACE_LIBCUDA`), which is
what makes `cargo test --workspace` deterministic and runnable on a machine
with no GPU. One thing that leaves unproven: that the interposer forwards to,
and reads its manifest out of, the `libcuda.so.1` NVIDIA actually ships.

`crates/gputrace-shim/tests/real_driver.rs` covers it. It is `#[ignore]`d, so
it never runs in GPU-less CI and never gates a build.

## Running it

On a machine with an NVIDIA driver and a GPU:

```sh
cargo test -p gputrace-shim --test real_driver -- --ignored
```

It captures the `alloc_copy` fixture — `cuInit`, `cuCtxCreate_v4`,
`cuMemAlloc_v2`, a round trip through real device memory, `cuMemFree_v2` — and
asserts that the target still ran normally, that a bundle was produced holding
those records, and that the manifest names the driver's version and the GPU's
compute capability.

The driver is looked for at `/run/opengl-driver/lib/libcuda.so.1`, then the
usual Debian and Fedora paths, then `libcuda.so.1` on the loader's search path.
Point `GPUTRACE_REAL_LIBCUDA` at it if yours is somewhere else:

```sh
GPUTRACE_REAL_LIBCUDA=/opt/nvidia/libcuda.so.1 \
  cargo test -p gputrace-shim --test real_driver -- --ignored
```

## Hazard

This runs real work on a real GPU. Prefer a GPU that does not drive your
display — see `docs/choosing-a-capture-gpu.md`. The fixture is tiny (one 8-byte
round trip, no kernel launch), so the risk is small, but the habit matters for
everything built on top of capture.
