# B07.1: CK grouped-convolution plugin admission

Completed 2026-10-29 on .104. Tools `a4bf75b8033b484b364be84a6233efe681203875`.
No production source/library/default changes. This is an admission audit and
isolated factory execution, not an installed MIOpen plugin qualification.

## Findings and implementation choice

| Boundary | Evidence | B07.2 consequence |
| --- | --- | --- |
| Build and default loader admission | Plugin CMake and loader whitelist exclude gfx1013; existing explicit `MIOPEN_CK_LIB_PATH` bypass remains | Use private path only; retain production admission |
| Hardware | gfx1013 scalar DL is available; native MFMA/WMMA unavailable | Reuse DL, keep matrix-only engines disabled |
| Existing forward factory | Same `DeviceGroupedConvFwdMultipleABD` FP32/2D/NHWGC/GKYXC/NHWGK interface used by MIOpen | Reuse existing code and installed instance archive |
| Macro boundary | Factory branch uses `DL_KERNELS`; config separately defines `CK_ENABLE_DL_KERNELS`; MIOpen helper does not explicitly propagate the former | Check actual private plugin compile flags; pass DL macro explicitly in isolated prototype |
| Loader contract | API5 requires66 symbols:6 common plus5 per each of12 solver groups | Supply full interface; unsupported directions/types must refuse explicitly |
| ABI | Extern-C names pass internal C++ `ProblemDescription`, `ExecutionContext`, `ConvSolution` objects | Build against exact qualified MIOpen headers/config/library; not a stable standalone C ABI |
| Existing N2.18 | Direct header, group-major layouts, plain/Add+ReLU forward FP32 | Preserve it; new probe covers plugin's interleaved factory layout and actual archive instances |
| Missing plugin coverage | No real plugin was built, loaded or selected by MIOpen | B07.2 tests must force and identify the solver; Torch fallback success alone is insufficient |

Nine source files are hash-identical locally and on .104; retained audit JSON
contains their paths/hashes and all required symbol names. The default build
exclusion is deliberate capability gating, not proof that a scalar plugin is
impossible. Adding an architecture name alone is insufficient.

## GPU evidence

Two native gfx1013 probe executables link the existing installed CK archive:
without `DL_KERNELS` the factory returns0 instances; with it the same FP32 factory
returns9 instances. The latter is not a plugin and does not change MIOpen routing.

In each daily/RCCL runtime profile,16 fixtures span groups1/3,1x1/3x3 filters,
batch2, per-group input/output channels4,5x7 images, contiguous interleaved layouts,
default/private streams, and dyadic/non-dyadic input values. Every admitted
instance runs:96 executions/profile, **192 accepted CPU-double comparisons**.
Dyadic outputs are exact; non-dyadic tolerance is unchanged from N2.18
(`2e-5 + 5e-5 * abs(reference)`), max absolute error `6.07623e-7`.
256-byte allocation guards and unchanged input/weight bytes pass.

The native Torch HIP backward check passes after each probe. Factory gates3+3
plus retained core7/daily5/RCCL6 total **24 checks**, five accepted600s recovery
guards. Earlier r1 completed its native probe but its Python parser rejected CK
diagnostics preceding JSON. r1 logs/build remain; parser now retains stdout and
reads the final result line. No reference/tolerance change; six total guards ACK.
Only r2 is the accepted numerical qualification.

Builds use installed headers/archive, ccache20GiB/no reset, OMP4, four CPUs,
RAM10GiB/memory+swap12GiB, read-only sources/prefix, no GPU/network,600s bound.
No CK/ROCm/Torch library rebuild, cache reset, installation or prefix change.
3649 protected hashes/35 links/four FFT AOT DB paths remain unchanged;136 retained
workstation backing paths match. Same r2 boot, unchanged kernel logs, no owned
jobs/containers/armed guards. .105 and user applications were not accessed.
29 local tests/critical Ruff/Black/clang-format/Python compilation/shell/diff pass;
full upstream CI not run. Free30,202,425,344B. No performance, backward-plugin,
FP16/BF16 plugin, arbitrary-layout/model, graph or general correctness claim.
Historical Python CPU fault remains unresolved/distinct from fixed VM-heap damage.

## Next and recovery evidence

**B07.2:** create a private matching-build forward-only FP32 DL plugin with all
required exports. Refuse unsupported solver groups/types, preserve default
admission and installed packages. Check private discovery, kernel identity,
workspace/argument contracts and forced MIOpen solver execution against CPU
references, plus retained HIP-direct/Torch forward/backward. Do not install until
that scope passes. A separately bounded performance comparison is required before
claiming benefit or choosing automatic default routing.

Qualification controller r1PID10147 exits1(parser), accepted r2PID10922 exits0;
evidence export exits0. Host `WS/logs/ck-plugin-b071-qualification-r2/final.json`
and `WS/logs/ck-plugin-b071-r2-controller.log`; builds at
`WS/tmp/ck-plugin-b071/build-r{1,2}`, export at `WS/tmp/ck-plugin-b071/export-r1`.
`WS=/home/nik/bc250work/rocm-10.2-gfx1013`. Source/library trees and defaults remain
intact; isolated files need no rollback of the active prefix.

Local `kernel/artifacts/ck-plugin-b071/` retains source audit, verification,
cleanup,109 raw files/7,008,402 bytes and both probe binaries. Evidence archive
1,560,256 bytes, SHA256
`bdd63ea9687bfe26f96ea253845fabfe9f906c91da2be4427ed360e7653c3319`.
