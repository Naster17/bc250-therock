# B07.2: private MIOpen FP32 DL forward plugin

Completed 2026-10-29 on .104. Source
`2976e7d30bfdccb2fb79809b8c58a6b5722a0822` over installed libraries `8389197c1`;
tools `528088c26`, controller environment correction `608ac7b86`.
**The candidate is isolated, not installed or admitted by default.**

## Implementation

`projects/miopen/src/ck_impl/bc250_dl_private.cpp` reuses the matching upstream
forward implementation, invokers and installed scalar DL instance archive.
A restricted API5 facade supplies all66 required symbols. Unsupported eleven
solver groups report false applicability/empty lists; workspace and solution
requests explicitly return INVALID_VALUE, with null arguments returning BAD_PARAM.
Forward admission restricts packed2D FP32/default-or-NHWC layouts and refuses
casts, mixed types, TF32 and deterministic descriptors. The tested route is the
public Immediate API with alpha1/beta0. Other scalar contracts remain unqualified.

The host CK feature remains off. Only the candidate uses a copied generated
config with CK templates on, explicit `DL_KERNELS`, C++20 and matching beta API/
existing dependency headers. There are no CK-conditional object-layout changes
in the matching included public/internal headers found during inspection; this
still requires the exact qualified C++ ABI and is not cross-version portability.
Host-hidden `TensorDescriptor::GetCastType()` is supplied using its exact matching
one-line `tensor.cpp` definition, hidden locally rather than changing the host
library. Independent ELF symbol inspection confirms that accessor and renamed
upstream helpers are not exported; all66 required exports are present.

Raw DL factory instances omit transfer parameters from their type strings,
producing duplicate plugin IDs. The `MIOPEN_BC250_DL_PRIVATE` branch in the existing
forward source retains the first instance per ID, yielding six unique IDs.
Production builds keep their original factory. This deliberately narrows the
private candidate; it is not a universal CK naming or kernel-selection repair.

## Qualification

| Check | Accepted result |
| --- | --- |
| Forced public MIOpen solver137 | 32 calls each daily/RCCL =64 CPU-double comparisons |
| Actual GPU route | Logs name `ConvHipImplicitGemmGroupFwdXdlops` and actual `DeviceGroupedConvFwdDl...` kernels; historical solver name does not imply matrix instructions |
| Geometry | Packed NCHW/NHWC, groups1/3,1x1/3x3, batch2,5x7, per-group C=K=4; default/private streams and dyadic/non-dyadic inputs |
| Numerical bound | Dyadic exact; unchanged N2.18 `2e-5 + 5e-5*abs(reference)` otherwise; max abs `6.07623e-7` |
| Memory contracts | 256-byte input/output/workspace allocation guards; immutable input/weight bytes |
| Private-path absence | Fresh subprocesses refuse forced solver compilation in32 cases/profile =64 |
| Direct facade contracts | 304 recorded checks across both profiles: FP16/TF32/null/casted descriptors and all unsupported groups, including unchanged failed outputs |
| Torch coexistence | Eight convolution output/all-first-gradient fixtures versus CPU FP64, plus exact native arithmetic backward; ordinary Torch can choose retained solvers |
| Retained runtime | Core7/CPU ONNX4/daily5/RCCL6 |
| Total | Private8+8 plus retained22 =38 checks; six accepted600s recovery guards |

Twelve total guards acknowledged, including prior failed attempts. r1 ABI gate
found duplicate IDs. r2/r3 reached a forced GPU invocation but the test mistakenly
required workspace scratch bytes to remain unchanged; r3 preserves native logs.
The corrected test checks workspace allocation guards. r4 private gates passed,
then core was launched through `bc250-torch` rather than required `bc250-train`
and lacked transformers. Final r5 passes all gates; no numerical/reference/
tolerance relaxation. No installed package or user application change was needed.

Eight bounded build attempts are retained: r1 C++17/includes, r2 beta/sqlite,
r3/r4 hidden-accessor link failures (r4 unchanged retry); r5/r6 built;
r7 private ID filter; final r8 workspace-test correction. Builds use four CPUs,
OMP4/one compiler job, RAM10GiB/memory+swap12GiB, existing ccache20GiB/no reset,
read-only source/prefix, no GPU/network and600s bound. No native host library,
Torch, image or kernel rebuild.3663 protected hashes/35 links/four AOT DB paths
are unchanged;136 retained workstation backing paths match. Same r2 boot and
unchanged kernel logs; no owned jobs/containers/armed guards. .105 and user
applications untouched.29 local tests, selected critical Ruff/Black/new-source
clang-format/Python compilation/shell/diff checks pass; full upstream CI not run.

## Private use and remaining limits

Candidate: `WS/tmp/ck-plugin-b072/build-r8/libMIOpenCKGroupedConv_gfx1013.so`.
Guarded tests pass `--env MIOPEN_CK_LIB_PATH=WS/tmp/ck-plugin-b072/build-r8` to the
ordinary launcher, and explicitly request public Immediate solver137.
This is an experimental matching-build fixture, not a release/user-default
recommendation. Omitting the private path preserves default admission and its
forced-solver refusal. No prefix rollback is needed; installed files are intact.

**Next B07.3:** harden scalar/argument admission and qualify each selected kernel
with stride/dilation/tail geometries. Resolve non-default alpha/beta semantics
before broader use. Current integration tests do not force every exposed kernel
ID; prior B07.1 factory tests are separate evidence. Only then select paired
existing HIP-direct timing and possible opt-in packaging. No speedup, plugin
backward, other dtype/layout/concurrency/graph or universal correctness claim.
The historical Python CPU fault remains unresolved/distinct from fixed VM heap.

## Evidence

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`.
Accepted buildr8PID29981/qualificationr5PID31501/export exited0. Host logs:
`WS/logs/ck-plugin-b072-qualification-r5/final.json` and matching
`WS/logs/ck-plugin-b072-r5-controller.log`; export `WS/tmp/ck-plugin-b072/export-r1`.
All attempts, binaries, overlay configs and caches remain. Free30,153,154,560B.

Local `kernel/artifacts/ck-plugin-b072/`:283 raw files/34,976,217 bytes, source/
binary/hash/guard/profile logs, independent local verification and cleanup.
Evidence archive7,298,658 bytes, SHA256
`9e2aaabae96bd8386cfb2ba65e3de62408a1bd7fd5cd121461baf64627e91ecc`.
