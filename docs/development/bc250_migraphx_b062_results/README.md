# B06.2: MIGraphX source and native-header admission audit

A generic/scalar MIGraphX path is feasible enough to continue porting: unchanged
pinned pointwise/reduction headers compile for native gfx1013, pass30 GPU/CPU-reference
checks, and a minimal GPU-backend CMake configuration resolves existing rocBLAS,
MIOpen and HIPRTC with an isolated pinned ROCm CMake dependency. **No complete
MIGraphX backend or ONNX GPU provider is built, installed or qualified.** The current
ONNX Runtime remains CPU inference. [Verified identities](verification.json).
Tooling revision: `a8a6d2327286eb3bb4c8e93e4c507558de735e01`.

## Sources and applicability

Unmodified [MIGraphX rocm-10.1](https://github.com/ROCm/AMDMIGraphX/tree/95672916ed289d4be9d230a9be732e1bbad97e8c),
commit`95672916ed289d4be9d230a9be732e1bbad97e8c`, is staged locally at
`therock/sources/bc250-migraphx`. The upstream release list/refs inspected2026-10-24
has10.1 and10.0 tags, no10.2 release tag. This is a source-admission experiment
against our ROCm10.2 stack, not an assertion of release/binary compatibility.
The tag is lightweight; commit/TLS provenance is pinned, no local signature
qualification claimed. All4,780 tracked source files match the staged snapshot.

MIGraphX's own requirements pin [ROCm CMake6a7c5b7](https://github.com/ROCm/rocm-cmake/tree/6a7c5b73b8882c74f8f7060e2633f230dabb7b63).
It is staged as `therock/sources/bc250-migraphx-rocm-cmake`; the existing
`therock/base/rocm-cmake` remains unchanged. The initial existing dependency lacks
`rocm_add_version_resource`, so configure refuses. Selecting the isolated pinned
dependency resolves this; minimal GPU configure then completes. No stub/no-op
function or global dependency replacement is introduced.
[Source findings](source-audit.json).
[Upstream build guidance](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/develop/install/install-migraphx.html)
and [cross-compilation behavior](https://rocmdocs.amd.com/projects/AMDMIGraphX/en/develop/dev/cross_compilation.html)
were checked against the actual pinned source rather than used as admission proof.

| Area | Observed disposition |
| --- | --- |
|Device/wave | Live HIP device properties are used; cross-compile gfx10 policy is32. Native probe requires gfx1013:xnack- and warpSize32; no spoof/override. |
|Pointwise/reductions | Direct HIP headers exist independently of CK/rocMLIR; native AOT probe passes with DPP1 and scalar/LDS DPP0. No matrix instructions in either inspected device object. |
|GEMM | Start with existing qualified rocBLAS; MIGraphX's hipBLASLt whitelist excludes gfx1013. Our separate scalar hipBLASLt port is not automatic admission into MIGraphX. |
|Convolution | Minimal configuration resolves MIOpen and find2/find-mode APIs; graph execution through MIGraphX remains untested. |
|CK/rocMLIR/AMDMLSS | Disabled in the isolated profile; not installed/admitted by this experiment. No blanket assertion that every compiler path needs matrix hardware. |
|BF16 | Predicate currently excludes only gfx1030 and returns true for gfx1013; this needs dedicated graph/storage/backend validation. It is an admission risk, not a reproduced numerical failure. Initial full-graph scope should be FP32. |
|FP8/MX | Existing architecture predicates exclude gfx1013. No instruction admission. |
|DPP-disabled strict JIT | reduce.hpp contains an unguarded subwave_reduce reference when its declaration is excluded. C++17 accepts it as an extension in permissive AOT; `-Werror=c++20-extensions` refuses. The real JIT requests C++17/Werror, so DPP0 AOT numerics do not qualify strict JIT. Default DPP1 passes that diagnostic. |
|ONNX/Python | Disabled in minimal configure. Installed Python ONNX/protobuf wheels do not provide the native C++ Protobuf/protoc/Abseil and binding prerequisites. GPU ONNX provider/plugin integration is separate. |

## Build and correctness evidence

The owned probe uses the original `pointwise` and `block_reduce` headers, not
reimplemented device algorithms. Only the generated-style tensor argument shape
specializations and host fixture/reference code are owned here. Launch:128 threads,
one block, compile-time wave32. FP32/FP16 3×17 pointwise inputs use row stride23,
row-broadcast bias, output stride29, guard prefixes/suffixes/padding and unchanged
input checks. FP32 reductions use lengths1/17/31/32/33/63/64/127/128/129/257/513;
CPU-double oracles and guards match exactly. Inputs use exactly representable
binary fractions; this is not broad random/statistical or model qualification.

15 composites per variant =30 native header checks. Retained Torch core7,
ONNX package/provider audit4, daily consumers5 and opt-in RCCL consumers6 bring
acceptance to52 checks/five guarded gates. First gradients remain qualified only
by the retained Torch tests, not MIGraphX autograd or the header probe.
Device metadata records gfx1013; disassembly has no MFMA/WMMA opcodes in these
probe kernels. This does not characterize all MIGraphX kernels or emitted JIT code.

Existing rootful build imagefa3e8c25 is reused with no GPU/network, four CPU quota,
RAM10GiB/memory+swap12GiB, shared20GiB ccache/no stats reset. Two executables and
their device objects are built; no full backend build. Initial inspection wrongly
sent a HIP bundle to llvm-objdump; corrected by extracting its gfx1013 device object
with clang-offload-bundler. Original object/log/source retained. A wrapper path
expression also refused before a build; corrected. No numeric assertion/tolerance
change. DPP0's C++17 warning remains deliberately visible and is confirmed by the
separate strict compile.29 local BC250 tests and selected Ruff checks pass.

Minimal configuration: GPU_TARGETS=gfx1013, GPU/HIPRTC/rocBLAS/MIOpen ON;
hipBLASLt/CK/rocMLIR/AMDMLSS/Eigen/ONNX/TensorFlow/Python/tests OFF. It completes
using existing nlohmann-json/msgpack/SQLite staged dependencies and the isolated
ROCm CMake source. Configuration is not compilation/linking or graph correctness.
All generated/scratch files are under workspace tmp; original sources are unmodified.

Ordinary available disk measured1,234,112,512B; total free including filesystem
reserved space26,614,734,848B. No reserve policy was changed. A full build's
footprint has not been measured or budgeted, so this batch stops at configure and
bounded probes; it is not declared physically impossible. Preserve SSH/boot reserve
and existing artifacts when selecting a full-build workspace/budget.

## Preservation, artifacts and next action

Daily runtimea9a5afe3/profiledb556925/image1bb4ac0e/native Torcha524, all ROCm/HIP
binaries, math/SDMA0/heap-off/N3.2, boot controls and user applications remain
unchanged.489 protected hashes/four AOT DBs/links retained. Same r2bootc6a50cfa;
per-gate/batch/export kernel logs identical, allfive guards ACK. No active prefix
change, package install, source reset, prior build/cache/checkpoint/image/container
deletions, `.105` contact or user-application inspection/test occurs. New stopped
build/configure containers and scratch/build evidence remain. Historical Python
CPU segfault remains unresolved and distinct from fixed VM-heap/CSR/hostcall defects.

Raw reports: `WS/logs/migraphx-b062-qualification-r1/final.json`, build-r1/r2,
configure-r1, probe-r1. Native executables/device objects:
`WS/tmp/migraphx-b062/20261024-r2/tmp`; configure:
`WS/tmp/migraphx-b062/20261024-configure/tmp/configure`.
Local artifacts:`kernel/artifacts/migraphx-b062`; [archive.json](archive.json).
Archive8,720,498B SHA256`28b8390813ccdb72afe0c1fef2e7ee08c3810d93a88ddedfe95afab1ee9ac540`;
69 raw logs1,525,263B, eight tools, pinned source archives/full source hash map,
executables/ISA/CMake caches andfive guards verified. Local first-build source
variants are retained separately. Both pinned Git repositories have independent
bundles validated by fresh clone/connectivity checks:
[source bundle verification](source-bundle-verification.json). MIGraphX's initial
shallow bundle failed reconstruction despite bundle-verify succeeding; fetching
its parent history and recreating the bundle resolves that. The source revision/
working tree did not change. Complete bundle28,499,973B; isolated CMake419,273B. No own jobs/containers/armed guards running.
Memory snapshot is synchronized to `.104` only.

**Next B06.3:** fix and test the DPP-disabled header's strict C++17 declaration
boundary, retain default DPP behavior, establish gfx1013 feature-admission tests,
and select an explicit durable disk/reserve budget before a minimal FP32 GPU graph
backend build. Do not install a provider or admit BF16/FP8/CK/MLIR from this audit.
