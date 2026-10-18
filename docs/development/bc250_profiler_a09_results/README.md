# A09: native Torch GPU profiler — 2026-10-18

The existing pinned Kineto backend builds against the installed native
rocprofiler-sdk. The former runtime explicitly set `USE_KINETO=0`; enabling it
in a separate Torch candidate supplies actual GPU tracing through
`torch.profiler`. No new architecture alias, matrix backend or hardware counter
support is introduced. Native Kineto/SDK source changes were unnecessary.

The user selected A09 ahead of A08. A08 remains unfinished; return to it before
N2.7. Earlier dependency-audit results remain historical evidence.

## Initialization order repaired after N2.12

The CPU-only-first→GPU defect found by N2.12 is now repaired and installed in
WS/pytorch-profiler-order. Native Torchb823a1b/final testsda453 retain available
ROCm GPU capability independently of first capture selection. Warm/cold
CPU-first, GPU-first and masked-device CPU captures, API7/core7 and installed
reactivation pass; only libtorch_cpu.so changes. The original A09 build remains
available for recovery. [Current evidence, source and recovery](../bc250_profiler_order_results/README.md).
The historical implementation/deployment sections below describe the original
A09 milestone; current flow/runtime manifests supersede their active-link claims.

## Implementation and build provenance

- Torch `4af4a1508a6ec0b7b2b29f5e0115f8477c4d1bb5` adds qualification tests only;
  native source base is unchanged `71a0426566bc9b952d9aa9b41f4a9d79ce46e0e8`.
- Kineto submodule `094d3c1d072362d0a919a77299459eee94f97931` chooses
  rocprofiler-sdk for this ROCm version. CMake cache confirms `USE_KINETO=1`,
  `KINETO_BACKEND=rocm` and `ROCPROF_LIBRARY=.../librocprofiler-sdk.so`.
  The legacy `HAS_ROCTRACER` macro/message is shared with the SDK implementation;
  it does not mean the legacy roctracer DSO was selected as the Kineto backend.
- TheRock runtime helpers `4c2edb785` provide the isolated build, qualification
  and reversible deployment. Build command remains
  `python -m pip install -e . -v --no-build-isolation`, MAX_JOBS12/link1,
  ccache20GiB, native gfx1013 and existing scalar/SDMA0/heap-off defaults.
- The existing source/build is preserved. Candidate
  `WS/pytorch-profiler-a09` copies the active patched source and object cache;
  unchanged `pytorch-sparse-followup/third_party` is bind-mounted read-only while
  building, saving disk space. Rebuilds must retain that mount. No source tree
  was reset, no package image replaced, no old cache/build/checkpoint removed.
  Build helpers/logs record exact mounts and the original CMake cache snapshot.
- Patch0008 and the verified incremental bundle on71a0426 preserve the tests;
  feature enablement is in the tracked A09 build helper. Full local Git is
  authoritative; the older three-file SDPA patch reconstruction gap remains.

## Bounded numerical and trace evidence

Seven native device tests require real execution and records:

| Case | Required evidence |
| --- | --- |
| FP32 and BF16 forward/backward | Exact profiled/unprofiled output and both gradients, CPU float64 references;13 GPU kernels/16 runtime calls/13 linked CPU events per case in candidate run |
| CPU/GPU correlation | Matching kernel/runtime correlation IDs, positive finite GPU durations; kernel starts within the synchronized CPU marker with a1ms allowance |
| Copies | Two GPU memcpy events with runtime correlation and exact pinned H2D/D2H round trip |
| Memory |4MiB allocation and matching free events, eight memory records; allocated-byte count returns to baseline |
| Scheduled/custom stream | Two active trace windows, eight kernels/window, numerical reference on a nondefault stream |
| CPU-only selection | CPU events present, no GPU kernel events |
| Bounded overhead | Three alternating small eager windows; raw wall samples include profiler setup/teardown, not kernel-only/application throughput |

The candidate's short-window overhead median ratio was~4.70x; this submillisecond
workload is dominated by profiler setup/teardown. Profiling adds cost and is
explicitly enabled by a context. This does not measure ordinary LoRA throughput
or promise a performance improvement. GPU timing and correlation were checked
only in these bounded current-boot fixtures; absolute trace-clock calibration,
reboot/reset behavior and physical counters remain separate work.

The first test harness counted both CPU `user_annotation` and
`gpu_user_annotation` ranges with the same name. Filtering the CPU marker by its
category fixed the assertion; both traces are retained. Numerical tolerances
were unchanged. A controller later expected a per-row `status` field absent from
the existing actual-model report; its root result and all three comparisons had
passed. The controller now verifies root status/count and acknowledged gates,
reusing the completed tests. Neither issue required a runtime source fix.

Candidate regressions pass: seven core groups,38 sparse backward tests,69 A05,
42 A06,36 A07,30 Torch convolution configurations, three actual2B nonzero-adapter
train/checkpoint cases and two exact CLI backend fresh/resume comparisons.
A06 retains exact BF16 gradients for all372 adapters at65/1024; A07 retains its
previous bounded gradient agreement and twelve head256 forward/recompute calls.
User checkpoints/data were not used or modified.

## Usage

Run this inside an ordinary `bc250-torch`/`bc250-train` Python workload after
loading the model and warming the selected operations:

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with record_function("training_step"):
        loss = model(**batch).loss
        loss.backward()
        torch.cuda.synchronize()

prof.export_chrome_trace("bc250-trace.json")
print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=15))
```

ROCm uses the `CUDA` activity name in the Torch API. Export to a persistent
project directory. The profiler is available to callers of the selected .104
Torch build; the LoRA trainer does not automatically start collecting traces.
For scheduled collection, use `on_trace_ready` to export each active window;
PyTorch's ordinary cycle-clearing notice remains visible. Stack and memory
collection can add further overhead.

## Remaining scope

Hardware counters, arbitrary simultaneous/external profiler sessions, compiled/
graph workloads, long traces, broad models/concurrency and sustained tracing
stability remain unqualified. SDK event availability is not hardware-counter
support. The historical Python CPU segfault remains unresolved and distinct from
the fixed VM-heap/CSR corruption. .105, BIOS, kernel and r2 fallback are unchanged.

## Installed deployment and recovery

[Deployment](deployment.json) passed four installed gates: seven profiler tests,
seven core groups, three actual2B model cases and two exact CLI backend resume
comparisons. [Final verification](installed-verification.json) confirms source,
Python-extension and library identities plus47 unchanged protected old/runtime/
SDK/application files. Only candidate `libtorch_cpu.so` and `libtorch_python.so`
differ from the previous native library cohort; GPU `libtorch_hip.so` is identical.
`readelf` confirms the direct [SDK dependency](native-dependencies.txt).

The active Torch link now points to `WS/pytorch-profiler-a09`; the complete old
`WS/pytorch-sparse-followup` remains. Backup old link/runtime metadata in
`WS/recovery/profiler-a09-before`, restore under `helpers/update_dev_prefix.py`.
The guarded deployment automatically restores that link and metadata on failure.
Images, ROCm DSOs, datasets, existing checkpoints, boot policy and .105 are unchanged.
The host usage file was backed up in the same recovery directory before updating.

[14 GPU guards](guards.json) all acknowledged600s deadlines with unchanged per-run
kernel logs. The only failed GPU test process was the retained marker-category
harness attempt; no driver/runtime defect was reproduced. No active agent test,
container or recovery deadline remains. Native build container
`bc250-profiler-a09-build` is retained stopped; no image commit was needed.

[Installed traces](traces), [candidate API](candidate.json),
[installed API](installed.json), [candidate regressions](candidate-regressions.json),
[model](model.json), [CLI](cli.json), [core](core.json), [source](source.json)
and [targeted lint](lint.log) preserve the evidence. Installed short-window
wall overhead median~4.63x; samples and scope are in installed.json. Nine configured
spin/lintrunner `-a` checks pass; no full-project CI/type claim.
Raw build/test/failed traces remain in `WS/logs/profiler-a09*` and ignored local
`kernel/artifacts/profiler-a09/remote-logs/`. Synthetic CLI checkpoints remain in
new log directories and were not copied over user checkpoints. Disk had~2.03GiB
available after qualification; check capacity before another native build.

