# Golden outputs: what `--golden` captures, and what it cannot

`gputrace capture --golden` (equivalently `--fidelity golden`) keeps what the
device answered, so a later replay can be diffed against it. This is the list
of what that covers today and, more importantly, what it does not — a gap
stated plainly is worth more than a golden value that is silently absent or
silently wrong.

## Captured

| Call | Where the bytes are captured |
| --- | --- |
| `cuMemcpyDtoH`, `cuMemcpyDtoH_v2`, `cuMemcpyDtoH_v2_ptds` | the copy itself, after the driver has answered |
| `cuMemcpyDtoHAsync`, `cuMemcpyDtoHAsync_v2`, `cuMemcpyDtoHAsync_v2_ptsz` | the later call that blocks for the copy |

An asynchronous read-back has nothing to capture where it stands: the driver
has only queued the copy and the destination still holds whatever the caller
last left there. The copy therefore stages its destination, and the first of
these to make it valid carries the bytes:

- `cuStreamSynchronize{,_ptsz}` — every read-back queued into that stream.
- `cuEventSynchronize` — every read-back queued into the stream the event was
  recorded on, up to the moment it was recorded.
- `cuCtxSynchronize{,_v2}` — every read-back the calling thread queued.

The payload lands on the blocking call's record, with role `golden_output` and
`PayloadRef::source` naming the device allocation it mirrors. That `source` is
the tie back to the read-back, since the record carrying the bytes is not the
record that asked for them.

## Not captured

**Read-backs synchronized by another host thread.** Staging is per thread and
takes no lock, which is what keeps it off the serialized path ADR-0004 exists
to avoid. A read-back issued on thread A and synchronized only on thread B is
not captured. A thread that reads its own destination buffer necessarily
blocks for it first, so this costs nothing for a correctly synchronized
program; a program where it costs something has a data race anyway.

**A thread that stages a read-back under one context and then synchronizes a
different one.** `cuCtxSynchronize` settles everything the calling thread
staged. A thread juggling two contexts and synchronizing only one could have a
golden value taken slightly early.

**`cuMemcpy`, `cuMemcpyAsync`, `cuMemcpy2D*`, `cuMemcpy3D*`, `cuMemcpyPeer*`.**
Unified addressing is the reason: `cuMemcpy(dst, src, ByteCount)` takes two
`CUdeviceptr`s and the direction is not in the signature. Which of them is
host memory is a property of the allocation, answerable only by asking the
driver (`cuPointerGetAttribute`) — a round trip per copy on the record path,
which is exactly what ADR-0004 refuses. The same applies in reverse on the
host-to-device side for User Story 8: these calls record metadata and handle
bindings like any other, but neither their inputs nor their outputs are
captured as payloads. A replay reading such a bundle must treat these copies
as unfaithful rather than assume the buffer was empty. Closing this needs a
per-allocation memory-type cache keyed off the `cuMemAlloc*`/`cuMemHostAlloc*`
calls capture already deep-handles, which is a deliberate piece of work and
not a one-liner.

**Everything else that moves data device-to-host.** `cuMemcpyAtoH*`,
`cuMemcpyDtoHAsync` through a graph node, and the stream-capture/graph surface
generally are pass-through.
