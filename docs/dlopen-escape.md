# The `dlopen` + `dlsym` escape

ADR-0003 claimed that closing two resolution paths — the exported `cu*`
symbols and the `cuGetProcAddress` dispatch table — left no call path out of
capture. It does not. A program that `dlopen`s `libcuda.so.1` and `dlsym`s
that handle gets the driver's own entry point, and neither path we close is
on the way. Issue #13 asked whether that is real and whether real CUDA stacks
do it. Both answers are yes, and the second is worse than the first: on the
stacks measured here it is the *primary* path, not a corner case.

## It escapes

`crates/gputrace-shim/tests/dlopen_escape.rs` is the demonstration. The
`dlopened_driver` fixture makes one call each way in one process: `cuInit`
bound from the global scope, and `cuDeviceGetCount` resolved from the
driver's own handle. The first is recorded; the second reaches the driver,
returns its answer, and leaves nothing in the bundle. `real_driver.rs` runs
the same fixture against NVIDIA's `libcuda.so.1` with the same result, so
this is the dynamic linker's scoping rule and not an artefact of the stub.

`dlsym(handle, name)` searches that object and its dependencies. `LD_PRELOAD`
interposition works by putting our object *earlier in the global scope*; a
lookup that never consults the global scope has nothing to shadow. The
`cuGetProcAddress` hook does not help either: the program never asks us for a
pointer, it asks the driver. This is the same mechanism `real.rs` uses to
reach the driver without recursing into itself — it was always available to
anything we capture.

## Real stacks use it, for nearly everything

Measured on NVIDIA 595.71.05 / CUDA 13.2, GTX 1660 Ti, with
`LD_DEBUG=symbols` (a lookup scoped to one file and never scanning the global
scope is a `dlsym` on a handle) and by capturing under `gputrace capture`.

**The CUDA Runtime does not link the driver.** `libcudart.so.13` carries no
`DT_NEEDED` on `libcuda.so.1`; nor does a statically-linked `nvcc` binary. It
`dlopen`s the driver on first use, then takes exactly two symbols off that
handle — `cuDriverGetVersion` and `cuGetProcAddress_v2` — and resolves
everything else through the dispatch table *it got from the driver*. Our
`cuGetProcAddress` hook is bypassed because the Runtime never obtained our
`cuGetProcAddress` to call. `libcudart.so.12` is the same with one symbol.

A trivial `cudaMalloc`/`cudaMemcpy`/`cudaFree` program, both `--cudart
static` and `--cudart shared`, run under `gputrace capture` against the real
driver: the program ran, round-tripped a buffer through device memory, and
**no trace bundle was written at all**. Zero records. The same CLI on the
`alloc_copy` fixture in the same session wrote a full bundle, so capture was
working.

**cuBLASLt takes the whole core surface by hand.** `libcublasLt.so.12`
`dlsym`s 32 driver entry points off the handle at load time, including
`cuLaunchKernel`, `cuLaunchKernelEx`, `cuModuleLoadData`, `cuModuleGetFunction`,
`cuMemcpyDtoH_v2`, `cuDevicePrimaryCtxRetain` and `cuCtxPushCurrent` — the
exact surface the deep handlers exist for.

**ollama.** `ollama 0.32.3` serving `qwen2.5:0.5b`, 100% on GPU, with the
interposer preloaded on the server so every `llama-server` child inherited
it. The model runner process produced **no segment at all**. The only records
in the bundle came from short-lived `llama-server --list-devices` discovery
processes: three calls each — `cuDeviceGet`, `cuDeviceGetAttribute`,
`cuMemGetAllocationGranularity`. Those three are captured because
`libggml-cuda.so` is the one object in that stack with a `DT_NEEDED` on
`libcuda.so.1`, for its virtual-memory calls, so they bind through the global
scope. Every kernel launch, every memcpy, all of the inference: invisible.

**PyTorch was not measured** — no CUDA-enabled PyTorch on this machine, and
building one was out of proportion to the question. It is a `libcudart` plus
`libcublas`/`libcublasLt` consumer, both of which were measured here and both
of which escape, so the expectation is a near-empty bundle. To settle it,
capture any `torch.cuda` script and count the records:

```sh
gputrace capture -o trace -- python -c \
  'import torch; torch.randn(1024, device="cuda").sum().item()'
```

## Reproducing

The two tests in the repo are the durable part; the rest is a shell session.

```sh
cargo test -p gputrace-shim --test dlopen_escape
cargo test -p gputrace-shim --test real_driver -- --ignored   # needs a GPU
```

To watch a library take symbols off the handle, build the trivial Runtime
program above and run it with `LD_DEBUG=symbols LD_DEBUG_OUTPUT=log`, then
look for `cu*` lookups whose only `lookup in file=` line names
`libcuda.so.1`: a global lookup lists every object in scope before it, a
`dlsym` on a handle lists one.

## The options, with costs

Not implemented here — #13 ends at a decision, and ADR-0007 records the one
taken.

**Interpose `dlsym` (and `dlvsym`).** Export our own `dlsym`; when the
requested name starts with `cu` and we shim it, hand back our trampoline
instead of the driver's. One hook closes both mechanisms observed above: the
Runtime's `cuGetProcAddress_v2` lookup comes back as our dispatch hook, which
ADR-0003's machinery already covers, and cuBLASLt's 32 direct lookups come
back as trampolines. Cost: `dlsym` is on every library's startup path, so the
hook must be a cheap prefix test and then a pass-through. The hazard the issue
names is real — our own resolver calls `dlsym`, so it would re-enter itself —
and wants a thread-local reentrancy guard rather than a handle comparison,
because `dlopen` of one path returns one refcounted handle and ours and the
target's are the same pointer. Nothing lands on the record path: this is
resolution-time work, so ADR-0004 is untouched. Portability is no worse than
what we already assume (`LD_PRELOAD` and glibc); `dlvsym` is a glibc
extension and needs its own export.

**Redirect the `dlopen` instead, with `LD_AUDIT`.** `la_objsearch` can
rewrite a search for `libcuda.so.1` to our own object, so the handle the
target gets *is* us and every `dlsym` on it resolves to a trampoline with no
`dlsym` hook at all. Cost: a second environment variable and a second
loading mechanism to get right, auditing applies process-wide and is easy to
get subtly wrong, and our resolver must still reach the real driver — which
works, because it opens a concrete path that `la_objsearch` can let through.
Worth keeping as the fallback if `dlsym` interposition proves unstable.

**Shim the Runtime API as well.** Rejected on its own terms: `nvcc` links
`cudart` statically by default, so the `cuda*` symbols are not in the dynamic
symbol table of a stock binary at all — the probe above proves it — and
ADR-0002 chose the driver surface deliberately.

**Do nothing and narrow the claim.** Costs nothing to build and costs the
project its motivating case: the stacks we most want to capture are exactly
the ones that escape. If we take this, the README and ADR-0003 have to say
so plainly, because "no call path escapes" would be false in front of
evaluators.
