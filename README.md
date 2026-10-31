# gputrace

A security-research interposer for AI-accelerator APIs. It sits between an application and the accelerator's user-space driver, recording the API calls that cross that boundary — without distorting the thing that makes GPUs interesting: massive parallelism.

`gputrace capture` runs an unmodified program under an `LD_PRELOAD` interposer that shadows the CUDA Driver API, records every `cu*` call, then forwards it to the real driver so the program runs normally. The result is a portable, inspectable **trace bundle** that later phases replay, mutate, and fuzz.

## Status

**Capture is implemented.** Replay, mutate, fuzz, validate, and inspect are scaffolded CLI subcommands that return `unimplemented!()` — each is a follow-up spec consuming the bundle this phase produces.

CUDA is the first and only backend. The trace representation and operations are defined behind a vendor-neutral `Backend` seam so other accelerators can be added without reworking the core ([ADR-0001](docs/adr/0001-cuda-first-behind-vendor-neutral-seam.md)).

## Requirements

- **Linux** with the NVIDIA driver installed (`libcuda.so.1`). No CUDA toolkit needed to build or run the interposer.
- **Rust** 2024 edition.

The repo ships a Nix flake with the full dev environment, including the dev-only CUDA toolchain used to regenerate bindings:

```sh
nix develop
```

## Install

```sh
cargo build --release
```

This produces two artifacts you need together:

- `target/release/gputrace` — the CLI
- `target/release/libgputrace_shim.so` — the interposer

The CLI finds the interposer beside its own executable, so keep them in the same directory or point `GPUTRACE_INTERPOSER` at the `.so`.

## Quickstart

Capture a program:

```sh
gputrace capture --out trace -- ./my_cuda_app --its --own --args
```

Your program runs normally — stdout, stderr, and exit status pass through unchanged — and the trace lands in `trace/`:

```sh
$ ls trace
blobs/  manifest.json  segments/

$ jq '.capture, .environment.driver_version' trace/manifest.json
{
  "backend": "cuda",
  "fidelity": "replayable",
  "device": null,
  "command": ["./my_cuda_app", "--its", "--own", "--args"]
}
13040
```

Pick a GPU that does not drive your display, and capture device→host outputs as golden values:

```sh
gputrace capture --device 1 --golden --out trace -- ./my_cuda_app
```

> [!WARNING]
> Capturing a heavy workload on the GPU that drives your display can freeze the desktop or trigger an Xid reset that tears down every context on it, this capture included. Use `--device` to steer work to a secondary GPU. See [docs/choosing-a-capture-gpu.md](docs/choosing-a-capture-gpu.md).

## Usage

```
gputrace capture [OPTIONS] -- <TARGET>...
```

| Flag | Default | Description |
| --- | --- | --- |
| `--out <DIR>`, `-o` | `trace` | Trace bundle directory to write. |
| `--device <ID>` | all | Confine the target to one GPU (sets `CUDA_VISIBLE_DEVICES`). |
| `--fidelity <LEVEL>` | `replayable` | How much to record per call. |
| `--golden` | off | Shorthand for `--fidelity golden`. |
| `-- <TARGET>...` | required | The program to run and its arguments. |

### Fidelity levels

Each level adds to the one before it, trading bundle size and overhead against replay faithfulness.

| Level | Records |
| --- | --- |
| `metadata` | The call, its thread, its ordering stamps, its scalar arguments, its return value. |
| `handles` | Adds stable identity for handles and device pointers. |
| `replayable` | Adds host→device buffers, launch parameters, module images. |
| `golden` | Adds device→host outputs, for validating a later replay. |

### Environment variables

The CLI sets these for the target; set them yourself to drive the interposer directly.

| Variable | Purpose |
| --- | --- |
| `GPUTRACE_OUT` | Bundle directory. **Unset or empty means every shim is a pure pass-through** — capture is opt-in. |
| `GPUTRACE_FIDELITY` | `metadata` \| `handles` \| `replayable` \| `golden`. |
| `GPUTRACE_LIBCUDA` | Back the interposer with an alternate library instead of `libcuda.so.1`. This is the seam the test suite drives. |
| `GPUTRACE_INTERPOSER` | Path to `libgputrace_shim.so`, when it is not beside the CLI. |

## Trace bundle

A directory you can inspect without special tooling:

```
trace/
├── manifest.json              # schema version, driver version, compute capability,
│                              # capture config, per-thread and per-device summaries
├── segments/
│   └── thread-<tid>.seg       # one binary segment per host thread (postcard)
└── blobs/
    └── <blake3-hex>           # payloads, content-addressed so duplicates store once
```

## How it works

**Both resolution paths are closed.** The interposer exports the `cu*` symbols *and* hooks `cuGetProcAddress`, so neither a program that links the driver directly nor one that resolves entry points through the dispatch table (as the CUDA Runtime does) escapes capture ([ADR-0003](docs/adr/0003-dual-interception-symbols-and-cugetprocaddress.md)).

```
target calls cuMemAlloc_v2
  ├─ linked directly      ──> our exported symbol ──┐
  └─ via cuGetProcAddress ──> our trampoline      ──┤
                                                    ├─> record
                                                    └─> forward to the real driver
```

**682 `cu*` entry points** are covered: 676 auto-generated pass-through trampolines plus hand-written deep handlers for the security-relevant subset (memory, context, module load, kernel launch, stream and event lifecycle). The generated output is vendored, so downstream builds need no CUDA headers and no `bindgen` ([ADR-0005](docs/adr/0005-vendored-bindgen-output-for-header-free-builds.md)).

**Capture never serializes the workload it measures.** Each host thread records into a thread-local buffer drained by a background writer; no lock and no global sequence counter sit on the record path. Each record carries a per-thread counter and a `CLOCK_MONOTONIC_RAW` timestamp, and the cross-thread/stream happens-before partial order is reconstructed *offline* rather than enforced at capture ([ADR-0004](docs/adr/0004-no-global-ordering-at-capture.md)). Handle identity follows the same philosophy: raw handles are recorded with their lifetime bindings, then resolved into generation-qualified identities offline ([ADR-0006](docs/adr/0006-handle-identity-is-generation-qualified-offline.md)).

## Development

```sh
cargo test --workspace     # 119 tests, no GPU required
cargo clippy --workspace --all-targets
cargo fmt
```

Every gating test runs a target under the interposer against a **stub `libcuda`** — a small in-memory implementation of the `cu*` subset with deterministic behaviour — via `GPUTRACE_LIBCUDA`. No physical GPU is involved, so the suite is CI-safe and reproducible.

One test does use a real driver. It is `#[ignore]`d and never runs in GPU-less CI:

```sh
cargo test --workspace -- --ignored    # requires an NVIDIA GPU
```

See [docs/real-gpu-smoke.md](docs/real-gpu-smoke.md).

### Regenerating the `cu*` surface

A deliberate maintainer action, not a build step — re-run it when targeting a new driver version:

```sh
nix develop --command cargo run --manifest-path crates/xtask/Cargo.toml -- gen-trampolines
```

See [docs/generated-surface.md](docs/generated-surface.md).

### Layout

```
crates/
├── gputrace-core/      # trace bundle, happens-before, handle identity — backend-agnostic
├── gputrace-shim/      # the interposer (cdylib)
├── gputrace/           # the CLI
├── gputrace-stub-cuda/ # test-only stub driver
├── gputrace-testkit/   # test-only harness and fixtures
└── xtask/              # dev-time codegen (its own workspace)
```

## Roadmap

- [x] **Capture** — record Driver API calls into a concurrency-correct trace bundle
- [ ] **Replay** — faithfully re-issue a bundle against the real driver, remapping handles
- [ ] **Mutate** — replay with one deterministic, targeted alteration applied
- [ ] **Fuzz** — parallel campaigns over a seed bundle, judged by a crash/hang/fault/differential oracle
- [ ] **Validate** — golden-diff replay plus a capture-overhead benchmark
- [ ] **Inspect** — dump a bundle's manifest and reconstructed happens-before graph

### Known gaps

- `cuMemcpy`, `cuMemcpyAsync`, `cuMemcpy2D` and `cuMemcpy3D` capture no payload in either direction: unified addressing makes the transfer direction unknowable from the signature alone. See [docs/golden-outputs.md](docs/golden-outputs.md).
- `cuLaunchKernel`'s `kernelParams` is a bare `void**` with no element count or sizes anywhere in the ABI, so it is recorded by address only. The `extra` packing *is* captured byte for byte.
- Deep handlers for `cuGraph*` are a later phase; the schema reserves room for graph nodes.
- Capture is launch-only and single-process: attaching to a running process, and multi-process or distributed capture, are out of scope.

## Documentation

| Document | What it covers |
| --- | --- |
| [GLOSSARY.md](GLOSSARY.md) | The project's domain language — read this first |
| [AGENTS.md](AGENTS.md) | Conventions for agents working in this repo |
| [docs/adr/](docs/adr/) | Architecture decision records, with the reasoning and the alternatives rejected |
| [docs/choosing-a-capture-gpu.md](docs/choosing-a-capture-gpu.md) | The display-GPU hazard and how `--device` avoids it |
| [docs/generated-surface.md](docs/generated-surface.md) | Regenerating the vendored trampolines |
| [docs/golden-outputs.md](docs/golden-outputs.md) | What `--golden` captures, and what it does not |
| [docs/real-gpu-smoke.md](docs/real-gpu-smoke.md) | Running the non-gating real-GPU test |

## Contributing

Work is tracked as GitHub issues in this repo. Specs are written as issues with implementing tickets as sub-issues; see [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md) for the conventions.

Before opening a PR: `cargo test --workspace`, `cargo clippy --workspace --all-targets` and `cargo fmt --check` should all be clean, and new behaviour should be tested against the stub driver rather than a real GPU.

Decisions that constrain future work belong in an ADR under `docs/adr/`, and new domain terms belong in `GLOSSARY.md`.

## License

MIT — see [LICENSE](LICENSE).
