# B06.10: isolated ONNX Runtime MIGraphX provider

Completed 2026-10-29. A matching ORT1.31.0 core, CPython3.12 binding and MIGraphX
provider were built and numerically executed on .104. Source
`b81a7866e5a6eff964b9e411535b4c9198651a9f` is three narrow compiler fixes over
`402979ac1bef89dcd55add41eaa7277c407599eb`. Tools milestone
`9791d4134b5cec52bdc2ed878873ef2b41681b05`.

**Candidate only.** Installed CPU ORT, runtime image/manifests, daily/RCCL defaults,
registered B06.8 SDK and .105 remain unchanged. `bc250-torch` does not acquire an
ORT GPU provider from this build. Relocation, packaging and versioned opt-in
activation are the next batch.

## Build and source changes

All13 B06.9 dependency archives were fetched locally, checked against upstream
SHA1 and pinned by SHA256 in
[the offline dependency manifest](../../../build_tools/bc250/onnxruntime_b0610_dependencies.json).
Offline CMake uses that mirror with package discovery disabled for FetchContent.
No package upgrade or dependency installation into the runtime image occurred.
A private build venv reads the existing NumPy2.5.1 site-packages explicitly;
nested `--system-site-packages` alone did not expose the outer venv's NumPy.

The private SDK copy uses the B06.9 repaired C++20 header. Registered SDK bytes
remain4797a6112; no active SDK index was modified. Original ORT source copy
`source-r1`, failed configure/build directories and caches remain. Patched source
`source-r2` verifies all11,379 file hashes against original plus the three-file
repair manifest. Runtime/native backends and GPU arithmetic were not rebuilt.

Strict Clang24 diagnostics required:

- Abseil's system-header option, plus a recorded CMake hook marking its exact
  include tree `SYSTEM`. ORT's include-copy rules otherwise lost that property.
  ORT `-Werror` remains enabled. The previously failing target passed a smoke
  build before restarting the full build.
- An explicit `uint32_t` cast for calibration-cache bits, preserving the previous
  unsigned conversion behavior. Calibration parsing/quantization is not newly
  qualified or hardened by this cast.
- Explicit `[=, this]` capture and removal of a redundant virtual destructor in a
  final provider class. Its base has a protected nonvirtual destructor. No new
  ownership/lifetime behavior or public provider interface is introduced.

Configuration still excludes training, GPU low-precision conversion, exhaustive
tuning, CUDA/TensorRT, DNNL/XNNPACK, telemetry, DLPack and upstream unit-test
binaries as recorded. Native CPU fallback remains available in a separately
requested CPU session; GPU test sessions explicitly disallow assigning it.

Four CPUs/jobs4, link pool1, RAM10GiB/mem+swap12GiB, private2GiB ccache/no reset,
read-only source/root and no build GPU/network. Each phase monitors ordinary free
space (4GiB floor), a20GiB loss cap and a two-hour deadline. Initial admission
requires24GiB; entry~32.6GiB, export32,344,141,824B. No reserve policy or old-artifact
cleanup. Full build reuses `task-r2`; retries compile affected objects rather
than restarting source trees.

## Qualification

118 checks/seven guards acknowledged;36 actual GPU graph evaluations:

| Checks | Scope |
| ---: | --- |
|32×3 | ORT-first, Torch-first and MIGraphX-first fresh imports, daily and RCCL cohorts |
|7 | Retained Torch outputs/first gradients/RNG/FFT consumers |
|4 | Retained installed CPU ONNX runtime audit |
|5 / 6 | Retained daily/RCCL consumers |

Each candidate pass checks six static FP32 IR10/opset18 graphs: pointwise,
reduction, dense, transpose/MatMul, convolution and serialized initializers, two
changed inputs each. Outputs, shapes/dtypes and unchanged host inputs match
retained independent FP64 and installed-CPU-ORT references at unchanged
`2e-5 + 2e-5*abs(reference)` tolerance. Wrong dtype/shape/missing-input refusals,
candidate CPU inference after GPU and exact Torch HIP first backward also pass.

Sessions select `MIGraphXExecutionProvider`, disable runtime fallback, and set
`session.disable_cpu_ep_fallback=1`. All18 retained per-model profiles contain
actual GPU-provider node events with no CPU-assigned nodes. Native device is
`gfx1013:xnack-`, SDMA stays0, low-precision conversion and exhaustive tuning are
explicitly disabled in options/environment. Loaded modules resolve to matching
candidate core/binding/shared-provider paths and the registered native SDK.

ELF inspection of copied candidate modules finds no exported Protobuf/Abseil
definitions. This and three import orders bound the B06.9 loader concern; they do
not establish all loader/plugin combinations. No numeric failure/tolerance
relaxation, GPU timeout or kernel-log drift occurred. All2,028 protected native
paths,29 links/four AOT DBs and136 retained workstation-backed paths match; full
installed SDK identity and all patched source hashes pass separately. Same r2boot
`83f6cb1b-4540-4149-839e-a6cf7cb80b63`. No owned jobs/containers/armed guards remain.

No broader model/dtype/opset, dynamic-shape, concurrency/allocator stress,
MIGraphX/ORT autograd, model performance or universal correctness claim. Historical
Python CPU fault remains unresolved and distinct from fixed VM-heap corruption.

## Evidence and next action

On .104 (`WS=/home/nik/bc250work/rocm-10.2-gfx1013`):

- `WS/tmp/onnxruntime-b0610/task-r2/build`: matching candidate.
- `WS/tmp/onnxruntime-b0610/build-r5/final.json`: accepted build; PID8939 exited0.
- `WS/logs/onnxruntime-b0610-qualification-r1/final.json`: accepted118; controller
  `WS/logs/onnxruntime-b0610-qualification-r1-controller.log`, PID9659 exited0.
- `WS/tmp/onnxruntime-b0610/export-r1`: evidence/verification; PID11602 exited0.
- Original failures retained: configure-r1 missing NumPy; build-r2/r3 Abseil
  diagnostics (option alone insufficient); build-r4 provider compiler diagnostics.
  Configure-r2/r3/r4 and smoke-r4 pass. No failed GPU gate.

Local `kernel/artifacts/onnxruntime-b0610` contains evidence, raw logs/profiles,
ELF verification, source repair/bundle, dependency pins, build snapshot and
preservation/cleanup receipts. Evidence3,223,004B SHA
`b7ff9d388611f0db8be2f66133dd42e9f4f702276e43064afe191ce1554a7d25`;
224 raw files17,064,377B. Raw candidate snapshot29,115,179B is retained for forensic
replay; it is not a relocation-qualified release. Local325 candidate file hashes
match; shallow clone/fsck and repair-bundle verification pass. Original full ORT
snapshot and shallow history are retained; no full-history/signature claim.

Pinned lintrunner0.13.1/adapters0.14.1/Ruff0.12.12/clang-format20.1.8 were installed
and initialized in an owned local lint venv. ORT changed-source lint passes.
29 retained local tests, selected critical Ruff checks for new tools, Black,
py_compile, shell syntax and diff checks pass. Full upstream CI was not run.

The unpackaged build emits a missing `build_and_package_info` package-metadata
warning. It does not invalidate these numerical checks, but should be resolved
and its identity checked during B06.11 packaging. Do not mask it in user scripts.

Next B06.11: package the matching candidate with sources/licenses/identity and
loader metadata; resolve package identity, qualify extraction/relocation and
versioned explicit activation/rollback with retained daily/RCCL checks. Preserve
installed CPU ORT, SDK/defaults and .105; no automatic provider/default switch.
