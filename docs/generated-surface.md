# Regenerating the generated shim surface

The interposer exports a pass-through shim for every `cu*` entry point
`cuda.h` declares, so no call is invisible before a deep handler exists for
it. Those shims are generated from the CUDA driver headers at dev time and
**vendored** into `crates/gputrace-shim/src/generated/` (ADR-0005): a
downstream build compiles plain Rust and needs neither CUDA headers nor
bindgen.

## The command

```
nix develop --command cargo run --manifest-path crates/xtask/Cargo.toml -- gen-trampolines
```

bindgen only parses `cuda.h` inside the devShell, which is where both
`libclang` (`rustPlatform.bindgenHook`) and the header (`cudaPackages.cuda_cudart`)
come from. Outside it there is no `clang` and no header, which is the same
state a downstream builder is in.

`--header <path/to/cuda.h>` targets another CUDA version; the default is the
header the devShell puts on the include path. `--out <dir>` writes elsewhere
than the vendored location. The tool runs `rustfmt` on what it writes, so a
regeneration that changes nothing produces no diff.

`crates/xtask` is its own workspace, deliberately: a workspace member would
put bindgen in the root lockfile, and a feature-gated `build.rs` could be
switched back on by `--all-features`. Neither is allowed to happen — the
test `nothing_in_the_workspace_depends_on_bindgen` fails if it does.

## What lands in the repo

| File | What it is |
| --- | --- |
| `generated/cuda_types.rs` | The CUDA ABI vocabulary (handles, enums, structs) the signatures are written in, vendored so downstream needs no `cuda.h`. |
| `generated/trampolines.rs` | One `pass_through!` line per entry point. |
| `generated/record_args.rs` | `RecordArg` for the argument types that are neither scalar nor pointer. |
| `generated/PROVENANCE.txt` | Which header, which CUDA version, which bindgen produced the above. |
| `generated/hand_written.rs` | **Not generated.** The exclusion list, see below. |

The vendored types are pinned to one CUDA version's ABI (`PROVENANCE.txt`
records which). Regenerate when retargeting a newer driver.

## Hand-written shims win

A hand-written shim and a generated one are both `#[unsafe(no_mangle)]`
exports of the same symbol, and two definitions of one symbol do not link.
`generated/hand_written.rs` lists the symbols that are shimmed by hand; the
generated definition for a listed name expands to nothing.

**Adding a deep handler is one line there**, in the same commit that adds the
handler:

```rust
(fn cuMemAlloc_v2 $($rest:tt)*) => {};
```

Nothing is regenerated for it, and `cargo build` tells you immediately if a
handler was added without its line — the duplicate symbol does not link.
