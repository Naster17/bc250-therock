# B06.3: strict C++17 MIGraphX reduction header repair

The isolated MIGraphX source now guards `reduce::subwave` and its `wave` alias
with the same `MIGRAPHX_HAS_DPP` condition as their implementation. This fixes
header parsing with DPP disabled; it does not implement a new subwave fallback.
The existing scalar/LDS block reducer remains available, and requesting the
unavailable subwave API with DPP disabled is deliberately rejected. The default
DPP path and arithmetic are unchanged. No backend/provider was installed.

Source: MIGraphX `35b40656d` over pinned upstream `95672916e` (`rocm-10.1`).
Tools: TheRock `a03d0e38a`. The isolated rocm-cmake dependency remains `6a7c5b7`;
existing TheRock sources, native Torch and runtime manifests are unchanged.

## Executed scope

The actual HIPRTC client reproduces the old C++17/Werror failure for DPP0,
while old DPP1 and patched DPP1/DPP0 compile successfully. Patched JIT modules
are loaded and executed on native gfx1013 wave32 for reduction lengths 33,
129 and 513. Results and input/output guards match independent CPU-double sums.
The client uses the project's strict warning family, without suppressing the
C++20 extension diagnostic. This is a direct HIPRTC probe, not a linked MIGraphX
program/graph or execution-provider test.

The 71-check native gate also retains both B06.2 header variants (30 checks),
adds 30 default-DPP subwave comparisons (widths 1/2/4/8/16/32, lengths
1/17/32/33/65), and executes the new upstream-style no-DPP block regression.
AOT builds use C++17 with `-Werror=c++20-extensions`; extracted gfx1013 objects
contain no MFMA/WMMA in these tested kernels. CPU-double references and guards
are exact on the binary-representable fixtures; no tolerances were relaxed.
Retained core 7, ONNX audit 4, daily consumer 5 and RCCL consumer 6 checks bring
93 checks per full pass. Two passes completed; the final pass explicitly routes
HIPRTC scratch to `WS/tmp/migraphx-b063/runtime-r2`.

The committed unit test has a whitespace-only clang-format reflow after the
native build; the preserved tested and committed hashes and whitespace identity
are in `test-format.json`. The reducer header is byte-identical to the staged
and tested source. 29 local stack tests, Python compilation, Black, shell syntax
and the new unit's clang-format check pass. Ruff was unavailable locally; no
full upstream CI claim.

## Feature admission and storage

The unchanged architecture functions are extracted from the pinned source,
compiled as C++17 and executed for gfx1013 plus five control architectures.

| gfx1013 predicate | Result | Qualification |
| --- | --- | --- |
| FP8 FNUZ / FP8 OCP / MX | false | Remain excluded |
| MIGraphX hipBLASLt | false | Initial backend uses rocBLAS |
| BF16 intrinsics | true | Existing heuristic only; graph/backend admission remains unqualified |

No BF16, FP8, CK, rocMLIR, ONNX parser, Python binding or GPU EP is enabled by
these checks. The BF16 result is an admission risk, not a reproduced numerical
defect in this batch or a statement about the existing Torch BF16 path.

The provisional full-build admission budget is 12 GiB for build/staging, 2 GiB
for cache growth and 2 GiB of ordinary available rescue space (16 GiB total).
This is a planning ceiling/floor, not a measured MIGraphX footprint. Current
ordinary space is approximately 1 GiB; filesystem-reserved free blocks are
reported separately and excluded from this budget. No reserve policy, retained
artifact or cache was changed/deleted. The full graph build is not admitted.
Build probes use four CPUs, RAM 10 GiB, memory+swap 12 GiB, shared ccache with
20 GiB environment limit and no statistics reset, no GPU and no network.

## Evidence and remaining work

Raw logs: `WS/logs/migraphx-b063-build-r{1,2,3,4}`,
`WS/logs/migraphx-b063-qualification-r{1,2}` and
`WS/logs/migraphx-b063-r{1,2}-{native,core,onnx,daily,rccl}`.
Local archive/verification/source patch bundle: `kernel/artifacts/migraphx-b063`.
The first build's test wrapper omitted a return after renaming `main`; corrected
before GPU execution. The standalone JIT host-client initially lacked the AMD
platform define; corrected before GPU execution. A later controller's stale
baseline path refused before GPU work. Original attempts/logs are retained.

Protected runtime hashes/links/AOT databases and kernel logs are verified by
the qualification/export records. All completed GPU guards acknowledged; no
runtime/kernel/boot/default/.105/user-application change. Historical Python CPU
segfault remains unresolved and distinct from the fixed VM-heap corruption.

**Next B06.4:** establish durable build storage satisfying the ordinary-space
budget without deleting retained artifacts or consuming rescue reserves, then
build the isolated minimal FP32 MIGraphX GPU backend (HIPRTC/rocBLAS/MIOpen;
Lt/CK/MLIR/ONNX/TF/Python off). Qualify actual pointwise/reduction/GEMM/convolution
graphs against independent references before considering any provider integration.
