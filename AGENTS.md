# gputrace

An interposer that records and manipulates the CUDA Driver API calls an application makes. `capture` is built; replay, mutate, fuzz, validate and inspect are scaffolded.

## Working in this repo

### The test seam

Every gating test drives a stub `libcuda` through `GPUTRACE_LIBCUDA`, so the suite is deterministic and needs no GPU. Run a target under the interposer with `gputrace_testkit::Harness` and assert on the trace bundle it produces. The one test that uses a real driver stays `#[ignore]`d: `docs/real-gpu-smoke.md`.

### Adding a shim

One line per symbol in a `shim!` block under `crates/gputrace-shim/src/handlers/` records the call and forwards it. A deep handler hand-writes the same shape and claims its symbol with one line appended to `pass_through!` in `generated/hand_written.rs`, in the same commit. Cover the whole `_vN` / `_ptsz` / `_ptds` spelling family — the driver picks which twin a program reaches.

### The record path

ADR-0004 keeps locks, global counters, blob hashing and filesystem work off the path a shim takes, so capture does not serialize the parallelism it exists to study. `crates/gputrace-shim/tests/concurrency.rs` enforces this at the source level.

### Regenerating the `cu*` surface

A deliberate maintainer action, not a build step (ADR-0005): downstream builds compile vendored output with no CUDA headers and no `bindgen`, and a gating test fails if `bindgen` enters the root lockfile. See `docs/generated-surface.md`.

## Agent skills

### Issue tracker

Issues live as GitHub issues in `amdlift/gputrace`, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each label string equal to its name. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
