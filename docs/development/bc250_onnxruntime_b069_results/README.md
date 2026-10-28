# B06.9: ONNX Runtime MIGraphX provider admission

Completed 2026-10-28. ORT 1.31.0 is pinned at
`402979ac1bef89dcd55add41eaa7277c407599eb` in
`sources/bc250-onnxruntime`. Source/API/dependency admission and an isolated native
SDK boundary probe pass. **The ORT provider itself has not been built, loaded or
installed.** Existing CPU ORT, registered MIGraphX SDK and Torch defaults remain.

## Source and compatibility findings

The matching [pinned provider build](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/cmake/onnxruntime_providers_migraphx.cmake)
builds a shared provider linking `migraphx::c` and `hip::host`, matching the
B06.7 CMake target repair. Its
[execution boundary](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/providers/migraphx/migraphx_execution_provider.cc)
parses serialized ONNX, wraps device pointers and runs programs on ORT's HIP
stream, copying outputs not bound as explicit parameters. Fast math is disabled;
FP16/BF16/INT8/FP8 conversion and exhaustive tuning default off, but environment
variables can enable them. An isolated candidate must explicitly retain FP32-only
admission. Existing CPU ORT does not gain a GPU provider by finding this SDK.

ORT requires CMake 3.28 and C++20. The installed SDK's public C++ wrapper fails
Clang C++20 parsing at two unqualified `module` return declarations. This is a
hand-written wrapper; only the C header/implementation are generated. Source fix:

```cpp
migraphx::module get_main_module();
migraphx::module create_module(const std::string& name);
```

Functional repair `144f6db5caf5590e41233bb1a94f152659d02f4a`; final MIGraphX source
`134ee25f67007523bd03374c4b723fe075a3b77f` additionally gates the new C++20 regression
on compiler support. Test validates both return types and constructs both module
handles. No C ABI/library math change. The installed SDK remains at4797a6112;
a verified private header overlay is used for these tests and the next candidate.
Do not edit registered SDK files or its pinned index in place.

The qualified build image lacks `clang-scan-deps`; disable
`CMAKE_CXX_SCAN_FOR_MODULES` for this non-module C++20 build. It is not necessary
to disable C++20 itself. Local source instructions and the repository's
[ort-build guidance](../../../sources/bc250-onnxruntime/.github/skills/ort-build/SKILL.md)
were consulted; no ORT source modifications were made.

## Dependency and resource plan

`kernel/artifacts/onnxruntime-b069/{dependencies,candidate-plan}.json` records
upstream URLs/SHA1 pins and the chosen configuration. Dependency bytes have not
been fetched or an ORT CMake configuration generated yet; SHA256 verification and
an offline mirror are the first B06.10 prerequisite.

| Dependency | Pinned ORT requirement / disposition |
| --- | --- |
| Protobuf / Abseil | 33.6 / 20250814.0, separate from MIGraphX's private30.0 / 20250512.0 and Python protobuf7.36.2 |
| ONNX C++ | 1.22.0; preserve upstream ORT patches, separate from installed Python ONNX1.23.2 |
| pybind11 | 3.0.2, separate from MIGraphX's private2.12.0 |
| FlatBuffers / JSON | 23.5.26 / 3.11.3, preserve ORT patches |
| Other candidate dependencies | Eigen commit1d8b82b, RE2 2024-07-02, date3.0.1, MP11 boost1.82.0, GSL4.2.1, SafeInt3.0.28, cpuinfo66ee79c |
| Excluded initially | Training, CUDA/TensorRT, DNNL/XNNPACK, mimalloc, OpenMP, telemetry, benchmarks and upstream unit-test binaries |

MIGraphX's ONNX types use `onnx_for_migraphx`; its ONNX DSO nevertheless exports
195 Protobuf/Abseil definitions. ORT's provider version script hides definitions
other than `GetProvider`. This inventory is not a reproduced collision or proof
of all loader interactions: inspect the actual candidate ELF and run fresh
Torch/ORT/MIGraphX import orders. A provider shared library uses ORT's internal
bridge ABI; do not drop an arbitrary provider DSO into the installed CPU wheel.
The candidate must have a matching isolated ORT core/binding/provider set.

Provisional full-build admission:24GiB ordinary free,20GiB growth cap,4GiB rescue
floor; private2GiB ccache, four CPUs/jobs4, link pool1, RAM10GiB/mem+swap12GiB,
read-only source/build root, no GPU/network during compilation and two-hour phase
bound. These are conservative limits, not a measured full ORT footprint. Native
source staging verifies11,379 files and admits40,796,561,408 ordinary free bytes.
No reserve-policy change or deletion. Next build must recheck current space and
monitor its actual growth before proceeding.

## Qualification and retained evidence

| Admission evidence | Result |
| --- | --- |
| Pinned-source assertions | 12 pass: version, CMake/C++ baseline, target/API/parser/stream/copy/default/fallback/visibility controls |
| Public C ABI declarations | 168 discovered `migraphx_status` functions compile/link against existing library; not 168 executed operations |
| Header regression | Standalone C++17 and C++20 binaries compile/link/run with repaired header and unchanged SDK libraries |
| Native device-pointer async probe | Eight cases per daily/RCCL cohort: pointwise and constant MatMul/ReLU, changed inputs, default/private streams, CPU-double and retained CPU ORT references |
| Retained consumers | core7, CPU ONNX4, daily5, RCCL6 |
| Accepted GPU batch | 38 checks/six ACK guards, including16 graph evaluations |
| Earlier fixture attempt | Same16 GPU evaluations plus retained core/ONNX pass, then duplicate log label refuses before remaining consumers; four guards ACK |

The probe uses parser buffer/external-data/input-shape options, FP32 explicit HIP
allocations and output binding, `run_async`, fallback device-to-device output
copy, stream synchronization, unchanged device/host inputs and allocation guards.
No tolerance relaxation: `2e-5 + 2e-5*abs(reference)`. The two small models establish
an SDK boundary, not ORT graph partition/allocator/scheduler/provider execution,
concurrent sessions, dynamic-shape, all-dtype/model, autograd or speed support.

All1,695 native protected hashes,29 recorded links/four AOT DBs match. Full installed
SDK identity passes separately;136 workstation-backed retained paths match.
Kernel logs unchanged through export/cleanup on r2boot
`83f6cb1b-4540-4149-839e-a6cf7cb80b63`. No active runtime/image/SDK/default/boot/user
application modification or .105 contact. No owned jobs/containers/armed guards
remain. Historical Python CPU fault remains unresolved, distinct from fixed
VM-heap corruption.

Accepted paths on .104 (`WS=/home/nik/bc250work/rocm-10.2-gfx1013`):

- Build: `WS/tmp/onnxruntime-b069/probe-r4/final.json` (PID10099 exited0).
- Qualification: `WS/logs/onnxruntime-b069-qualification-r2/final.json`, controller
  `WS/logs/onnxruntime-b069-qualification-r2-controller.log` (PID11488 exited0).
- Source: `WS/tmp/onnxruntime-b069/inputs`, stage PID12693 exited0.
- Export: `WS/tmp/onnxruntime-b069/export-r1`, PID13014 exited0.

Failed build-r1/r2 attempts retain missing scan-deps and missing sibling ccache
helper/read-only-default-cache errors. Corrected mounts include all setup helper
files and fail immediately on setup errors. Build-r3 reproduces the actual
C++20 wrapper defect. Build-r4 passes. These are compile/harness failures, not GPU
arithmetic failures. A prior scratch reconstruction failed during Git pack/index
changes; retained logs exist. Independent complete bundle-chain reconstruction
with automatic maintenance disabled passes full fsck at134ee25f6; no original
repository/history was reset.

Local artifacts: `kernel/artifacts/onnxruntime-b069`. Evidence1,440,485B SHA
`eb10bae63e6ab08dd09c54a0866ae74fe362bc72ea93076d72b942572f30a4d9`;
196 raw files5,500,940B. Exact ORT snapshot279,492,314B SHA
`e4243f26f7b6fd6be09fabd08a77161da719cdded1cbbc696458e3e2aeb3e51b`,
all11,379 files verified locally and remotely. Local checkout is shallow and has
three recorded uninitialized gitlinks; no signed-tag or full ORT history claim.
The native CMake plan uses pinned archive dependencies; Emscripten/fuzzer gitlinks
are outside this candidate. New repair bundle reconstructs over B06.7 and earlier
complete MIGraphX bundles. Local29 unit tests, Black, py_compile, C++ formatting
and diff checks pass. Ruff unavailable/full upstream CI not run.

Next B06.10: fetch/hash the selected dependency archives, create an offline mirror,
then configure/build a matching isolated ORT core/Python/MIGraphX provider with
private repaired header. Preserve CPU wheel, installed SDK and daily/RCCL. Only
after actual provider assignment/no-silent-fallback and independent-reference
checks should any versioned activation or installation be considered.
