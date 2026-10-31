# Choosing a GPU to capture on

**Do not capture on the GPU that drives your display.**

Capture forwards every `cu*` call to the real driver, so the target runs its
normal workload at close to normal speed. On the GPU that also renders your
desktop, that is enough to stall the compositor for seconds at a time, and a
workload that faults or runs long enough to hit the driver's timeout triggers
an **Xid** error. The driver's recovery from one is a GPU reset, which tears
down *every* context on that device — your desktop session, and the capture
in progress with it. A bundle lost that way is lost in the middle: `atexit`
never runs, so the manifest is never written.

Later phases make this sharper rather than softer. Mutation and fuzzing
deliberately feed the driver inputs it has never seen, so faults are the
expected outcome, not an accident.

## Pick a secondary GPU

List what the machine has:

```console
$ nvidia-smi --query-gpu=index,name,display_active --format=csv
index, name, display_active
0, NVIDIA GeForce RTX 4090, Enabled
1, NVIDIA RTX A4000, Disabled
```

Then capture on the one that is not driving a display:

```console
$ gputrace capture --device 1 --out trace -- ./my-workload
```

`--device` sets `CUDA_VISIBLE_DEVICES` for the target, which both **filters**
and **renumbers**: the target sees exactly that GPU, as ordinal `0`. That is
also what the trace bundle records — `manifest.json` holds the selection under
`capture.device` in your numbering, while `environment.devices[].ordinal` and
every `CallRecord.device` use the numbering the target itself saw.

## If the machine has only one GPU

There is no safe `--device` to pick. Capture from a console with the display
server stopped, or from an SSH session into a machine whose desktop is not
running. A headless host is the right place for anything beyond a short
capture.
