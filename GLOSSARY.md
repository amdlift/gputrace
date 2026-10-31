# gputrace

A security-research interposer for AI-accelerator APIs. It sits between an application and the accelerator's user-space driver, recording and manipulating the API calls that cross that boundary. The first and only concrete backend targets NVIDIA CUDA; the core is kept backend-agnostic so other accelerators can be added later.

## Language

**Interposer**:
The library that sits between an application and the accelerator driver, intercepting each API call before it reaches the real driver. gputrace *is* an interposer.
_Avoid_: Hook library, proxy, wrapper, shim (reserve "shim" for a single intercepted function)

**Shim**:
The small per-function trampoline that stands in for one real driver symbol: it records the call, optionally alters it, then forwards to the real function. The interposer is a collection of shims.

**Backend**:
A concrete implementation of the interposer's core traits for one accelerator API family. CUDA (Driver API) is the first backend. The trace representation and the capture/replay/mutate operations are defined independently of any backend.

**Driver API**:
The low-level CUDA interface exposed by `libcuda.so` (the `cu*` symbols), installed with the NVIDIA driver itself. This is gputrace's interception surface, chosen over the higher-level Runtime API (`libcudart.so`, `cuda*`) because the Runtime API ultimately lowers onto it.

**Capture**:
Recording the sequence of API calls an application makes — function, arguments, return value, and (configurably) data payloads — into a trace, with enough fidelity and ordering information to replay it.

**Trace**:
The recorded artifact a capture produces: the ordered record of intercepted calls plus the data needed to reconstruct them. Consumed by replay, mutate, and fuzz.
_Avoid_: Log, recording, dump

**Replay**:
Re-issuing the calls recorded in a trace against the real driver, reconstructing handles and payloads so the run reproduces the original.
_Avoid_: Re-execute, playback

**Mutate**:
A single, deterministic, targeted alteration of a captured call applied during replay — changing an argument, corrupting a copied buffer, or overriding a return value / injecting an error. The primitive that fuzzing drives.

**Fuzz**:
An automated campaign that generates many mutations and runs them against an oracle (crash, hang, or driver-error detection) to discover vulnerabilities. Built on top of the mutate primitive plus a mutation strategy and an oracle.
_Avoid_: Using "fuzz" and "mutate" interchangeably — mutate is one alteration, fuzz is the campaign.

**Trace bundle**:
The on-disk form of a trace: a directory holding a `manifest.json`, one binary segment per captured host thread, and a content-addressed blob store for large payloads.

**Happens-before graph**:
The partial order of calls across threads and streams, reconstructed offline from recorded stream/event operations rather than enforced at capture time. The analysis artifact that makes a concurrent trace intelligible.
_Avoid_: Timeline, sequence (a trace has no single total order by design)

**Mutation spec**:
A declarative description of one mutation: a selector (which calls) paired with a transform (what to change). Authored by hand or emitted by the fuzzer; the unit that makes a mutated run reproducible.

## Fuzzing

**Oracle**:
The judge that decides whether a trial is a finding, from four signals: host-side crash, hang (timeout), device fault / unrecoverable context, or a differential against the golden output.

**Golden output**:
The device→host results captured at L3 fidelity from an unmutated run, used as the reference the oracle's differential signal compares against.

**Trial**:
A single fuzz execution: one trace replayed with one mutation spec in one isolated worker process.

**Finding**:
A trial the oracle flagged as interesting, deduplicated by signature (crashing function plus signal or `CUresult` class).
