# A09/N4: CPU-first ROCm profiler initialization — 2026-10-18

Installed a bounded correction for the N2.12 tracing defect. A first CPU-only
capture no longer permanently disables GPU records in subsequent captures.
The default .104 Torch runtime, including bc250-torch/bc250-train callers,
now selects WS/pytorch-profiler-order. Normal computation and profiler opt-in
behavior are retained; no training/LoRA speedup is claimed.

Native Torch fixb823a1b2f327d5ba07af0796876f93e5308a1012; final test qualification
da45369 (two subsequent fixture corrections, native code unchanged). Build helper
TheRock669baa5d7, final controller/sealing39f3ed14b. [Final identities](final.json),
[installed result](installed.json), [bounded capture summary](summary.json).
Patch0009 is an incremental three-commit mbox on Torch4af4a15; pinned Kineto and
ROCm native sources are unchanged. Full local Git remains authoritative.

## Cause and correction

Torch prepareTrace passed the first capture's cpuOnly selection to libkineto_init.
Kineto stored that as controller capability and skipped ROCm logger registration.
Later requests found a registered profiler and never upgraded that capability.
N2.12 and fresh warm/cold CPU-first cases reproduce CPU events but zero GPU
kernels/copies; GPU-first control works.

For USE_ROCM only, first initialization now keeps GPU tracing capability when
ATen reports a visible GPU. The per-capture activity set still decides which
records are collected. With no visible GPU, CPU-only initialization remains.
This queries device availability and can initialize the HIP runtime/backend at
first profiling; the tested CPU-only first capture does not invoke Torch CUDA
lazy initialization. It does not promise that the HIP driver is untouched.
CUDA/non-ROCm branches are unchanged but were not rebuilt/tested on this board.
No architecture alias, GPU arithmetic/kernel change, external profiler override
or persistent controller replacement is introduced.

## Numerical and trace checks

Each ordered case uses a fresh process and two GPU capture windows separated
by CPU-selected captures. GPU windows execute FP32 matmul/sine and both input
gradients on a nondefault stream, plus a pinned256KiB H2D/D2H round trip.
Profiled output/gradients exactly equal the unprofiled GPU result; CPU float64
references retain atol2e-6/rtol2e-4. Kernels and copies must have positive finite
durations and runtime correlation IDs; linked CPU events are required.
CPU-selected captures execute GPU work too, requiring zero kernel/copy records.

| Case | Before | Candidate and installed |
| --- | --- | --- |
| Warm CPU-first→GPU |0 kernels/0 copies in requested GPU windows; defect reproduced |13 kernels/2 copies per GPU window; correlations and references pass |
| CPU capture before Torch GPU lazy init→GPU |same missing records |13 kernels/2 copies; CPU first leaves torch.cuda.is_initialized false |
| GPU-first→CPU→GPU |13 kernels/2 copies per GPU window |same records and references |
| No visible HIP GPU, CPU capture |not claimed as pre-fix coverage |CPU events,0 kernels/copies/runtime records; Torch lazy init stays false |
| Fresh installed reactivation |not applicable |CPU-first ordering,13 kernels/2 copies per GPU window |

The hidden-device fixture uses HIP_VISIBLE_DEVICES=999999, a valid nonmatching
ordinal on this single-GPU host. This simulates invisible hardware; it is not a
physical CPU-only host qualification. All three regular orders, hidden-device,
existing A09 API7 (FP32/BF16 gradients/copies/memory/schedule/streams/CPU selection)
and core7 (including attention/FFT gradients/RNG/consumers) pass against candidate
and installed runtime. No full-model convergence or broad concurrency claim.

A first16KiB copy fixture produced kernel records but no memcpy records; the
fixture was corrected to the already-qualified256KiB size without weakening
record requirements. Two hidden-device attempts used -1 visibility values that
the SDK parser rejects at Torch import; corrected after inspecting pinned
SDK/HIP parsers. Logs retained; no native patch change. These explicit SDK aborts
are distinct from the unresolved historical Python CPU execution segfault.

All21 process guards acknowledged600s:18 successful process gates, including
explicit pre-fix expected-defect cases, and three preserved failed fixtures.
Kernel logs are byte-identical across the entire batch. The first deployment's
final identity check rolled back successfully after all installed tests passed:
it incorrectly applied an old CPU-library digest to the intentionally switched
active alias. The corrected controller verifies that original identity through
the retained recovery link, verifies the installed candidate against qualified
hashes, reuses identical-binary passed gates and runs a fresh reactivation case.
The failed deployment report is retained. No numerical gate was skipped/relaxed.

Only libtorch_cpu.so differs among native DSOs; libtorch_hip.so is byte-identical.
104 original protected identities remain verified, with old runtime.json and
old active CPU-library alias checked through recovery paths. Targeted spin and
lintrunner -a checks pass; not full CI. No additional training/regression suite
was inferred from the unchanged GPU arithmetic DSO.

## Build, deployment and recovery

WS=/home/nik/bc250work/rocm-10.2-gfx1013. The separate candidate copies the qualified
A09 source/object cache independently, preserving all old builds/DSOs/caches.
Immutable pytorch-sparse-followup/third_party is mounted read-only. Required
out-of-place HIP generation runs first, then pip editable/no-build-isolation;
ccache20GiB, MAX_JOBS12/link1, native gfx1013 and all safe flags retained.
[Exact build command](build-command.json) and [log](build-build.log).

All actions use helpers/update_dev_prefix.py -- COMMAND under the exclusive
runtime lock; do not nest an ordinary launcher shared lock. Build helper:
`build_profiler_order_on_host.py`; qualification/deployment helper:
`qualify_profiler_order_on_host.py --phase candidate|install`; final seal:
`finalize_profiler_order_on_host.py`. Runtime checks use the existing guarded
validate_torch_a04_on_host.py with an explicit uninstalled candidate mount first.

Recovery WS/recovery/profiler-order-before contains exact old runtime metadata,
original Torch link text and a torch link resolving to the complete old
WS/pytorch-profiler-a09. Under the exclusive lock, atomically restore the saved
Torch target and exact runtime.json bytes; retain SOURCE_PROFILER_ORDER.json as
historical provenance. Existing helpers/deployment code show the tested automatic
rollback sequence. Prefix current/previous/rocm links, frozen release, ROCm DSOs/
kernel packs, runtime image, .105, BIOS/kernel/boot and r2 fallback are unchanged.
Only the development torch link and its corresponding runtime metadata change.
SOURCE_PROFILER_ORDER.json in WS/current records both candidate and recovery IDs.

Raw WS/logs/profiler-order-*; ignored local kernel/artifacts/profiler-order.
No running agent build/test/container/guard. Disk~10.13GiB free. The historical
Python CPU segfault stays unresolved and separate from fixed VM-heap/CSR.
External/simultaneous profilers, counters, daemon mode, compile/graphs, long
traces, cross-boot clocks and general overhead remain unqualified. Ordinary
workloads need no special initialization workaround after this installed fix.

Next N2.13: audit existing generic hipSPARSE/Torch CSR/BSR triangular vector and
multi-RHS paths, then CPU-reference bounded missing complex/index/layout/stream/
zero-pivot cases and repair only demonstrated defects. Keep legacy P2 real
csrsv16 evidence; general sparse factorization/spsolve B02 is a separate gap.
