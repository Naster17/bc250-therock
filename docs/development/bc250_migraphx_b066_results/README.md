# B06.6: isolated Python MIGraphX bindings on gfx1013

All 169 checks across seven guarded gates pass. Python 3.12 bindings now execute
six owned FP32 ONNX graphs using the isolated native gfx1013 candidate. This is
not an active SDK installation or an ONNX Runtime GPU provider. Existing daily
Torch/CPU ORT/RCCL runtimes and defaults are unchanged; .105 was not contacted.

MIGraphX remains `622ea3f2ad8f1baa8a44edb4f1e07cbe6e532c83`; no new component
source repair was needed. Its committed requirement pins pybind11 2.12.0 at
`3e9dfa2866941655c56877882565e7577de6fc7b`. A separate clean repository and
262-file native snapshot preserve the existing Torch/profiler pybind sources.
The complete 9,152,172-byte Git bundle passed fresh reconstruction and full fsck;
SHA256 `6e7a1be57ffe2fb4ad1e936595cb3e8f2e89a0649830acaa29993f567e0c80bf`.
It remains locally in `kernel/artifacts/migraphx-b066`, avoiding unnecessary
copies into the low-space board root.

## Executed scope

Fresh child processes import MIGraphX/CPU ONNX Runtime/Torch in three orders:
MIGraphX first, ORT first and Torch first. Each passes 49 composites:

- Candidate extension location/native gfx1013 identity and CPU-only ORT provider admission.
- Six retained IR10/opset18 FP32 models: broadcast Add/ReLU, row ReduceSum,
  dense MatMul/bias/ReLU, transposed MatMul, Conv/bias/ReLU and serialized FP32
  initializer weight/bias MatMul/ReLU. Both Python file and buffer parse routes
  execute on the GPU; their uncompiled IR, parameter/output shapes and types match.
- Uncompiled file JSON/msgpack and buffer serialization preserve IR equality.
  The buffer-parsed graph compiles with automatic copies, `fast_math=False` and
  exhaustive tuning off. Lowering contains the expected GPU code-object/GEMM/
  convolution/literal operations. Two changed inputs match retained independent
  NumPy FP64 and explicit CPU ORT outputs; host inputs remain unchanged.
- Compiled file and buffer reloads execute against the final changed-input
  references. A separately compiled file-parsed graph uses explicit HIP input/
  output copies with automatic copies disabled and matches both references.
- Unknown operators and malformed wire data refuse through file and buffer paths
  with the expected diagnostic. CPU ORT inference and a Torch HIP quadratic
  first-gradient check pass after the MIGraphX work.

This is 30 GPU graph evaluations per import order, 90 overall. The existing core7,
ONNXaudit4, daily5 and RCCL6 gates bring 169 across seven acknowledged guards.
All 665 protected hashes, four AOT DBs and recorded links match; per-gate and
final kernel logs are unchanged on r2 boot
`e955931f-5edb-42e5-8f14-eafc2fbaf361`. No reboot or recovery-policy change.
Comparisons retain `2e-5 + 2e-5*abs(CPU_double)`; no math/tolerance/whitelist or
BF16/FP8/MX/CK/MLIR admission was relaxed. The CPU oracles are the accepted B06.5
fixtures, not newly generated self-references.

These are module import-order checks followed by the specified workloads,
not every ordering of initialized runtimes. Compiled serialization is qualified
only for this same-build candidate/device and these static graphs; no portable
cross-version/device MXR claim. Embedding/loader support libraries compile, but
a C++ embedded-Python application was not executed. No MIGraphX graph autograd,
all-model/IR/opset/dtype, GPU allocation canary or performance claim. The historical
Python CPU fault remains unresolved and distinct from fixed VM-heap corruption.

## Build, usage and replay

The extension `migraphx.cpython-312-x86_64-linux-gnu.so` and Python support libraries
build with Python 3.12.3/C++17/O2/PIC and B06.5's private static Abseil/Protobuf.
This qualifies the selected glibc container environment, not native musl-host
Python or another Python minor version. The private pybind install contains
headers/CMake files and does not replace any installed Python package.

New outputs use local `tmp/migraphx-b064/20261025-storage/durable/b066`, temporarily
exposed on .104 as `WS/tmp/migraphx-b064/20261025-storage/mount/b066` (STORAGE).
`WS` is `/home/nik/bc250work/rocm-10.2-gfx1013`. Original builds/caches remain via
reflink copies; native sources and B06.5 installed dependencies are read-only.
Four CPUs/jobs, link pool one, RAM10GiB/mem+swap12GiB, no build GPU/network,
read-only build root and private2GiB ccache/no reset. Existing external16GiB
admission/14GiB growth/root768MiB floor/two-hour phase bound remain in force.

Reconnect B06.4's documented temporary transport, then use the committed host
helpers `build_migraphx_b066_on_host.py --phase python --attempt r2`,
`qualify_migraphx_b066_on_host.py --attempt r2`, and
`export_migraphx_b066_on_host.py`. These describe the accepted run, whose directories already exist. Prepare new
output paths, matching build/qualification bindings and a current preservation
baseline before replaying; do not overwrite retained evidence. Build phase
scripts are immutable snapshots. The accepted build logs are `STORAGE/logs-python-r2`;
qualification is `WS/logs/migraphx-b066-qualification-r2`. Tools checkpoint
`8692aa5fe`, accepted fixture/retry revision `0bfe2858ceb6e2f7a580a9e2b70a6bc5be41992f`.

For automatic copies, compile with `offload_copy=True, fast_math=False` and pass
NumPy inputs to `program.run`. With `offload_copy=False`, upload inputs and allocate
**all** remaining compiled parameters from `program.get_parameter_shapes()`,
including `main:#output_0`; retain those arguments during execution and use
`migraphx.from_gpu` on results. Upstream `test/py/test_gpu_offload.py` follows this
contract. Do not expose a raw GPU argument through NumPy's host buffer interface.
The owned validator demonstrates the complete tested path and loader isolation.

Two failed attempts are retained. The first build controller had an unnecessary
nonexistent Protobuf source mount and refused before compilation. The first GPU
fixture passed pointwise references/compiled serialization but omitted the explicit
output allocation and received `Parameter not found: main:#output_0`. Corrected
r2 follows the existing API; no source/arithmetic/tolerance workaround. That guard
also acknowledged, with no remaining container or kernel-log change. An old
qualifier's exploratory `--help` invocation refused its obsolete hash baseline
before GPU work; current preflight is imported directly. System Python lacked
Black; the existing tool venv supplies it. 29 local stack tests, Black, Python
compilation and shell syntax pass. Ruff is unavailable; full upstream CI not run.

**Next B06.7:** assemble a versioned relocatable opt-in SDK archive with source,
license and loader metadata, then qualify relocation against the existing runtime
image before considering installation. Meet the ordinary root-space budget;
retain active runtimes, .105 and boot fallbacks. ORT GPU EP and broader dtype/
model/fusion/performance scope remain separate admissions.

## Final seal

Local evidence is `kernel/artifacts/migraphx-b066/evidence.tar.gz`, 21,375,875
bytes, SHA256 `79738ca55dec9686b95d9303a01a4de084ac6398be6e34edeaf64dd430129c41`.
Local verification extracts 239 raw files/9,029,640 bytes and verifies accepted
results, guards, helper bytes and archive identity. The failed guard is archived
separately and acknowledged. All 4,782 MIGraphX/262 pybind source files match.
Temporary mount/relay and owned jobs/containers/guards are closed; sources,
images, builds, caches and backing data remain. No reserve policy/deletion change.
Build~463MiB, private headers908KiB/cache51MiB; export ordinary root866,148,352B
and external114,335,145,984B. Root space constrains any future installation.
The archived exporter printed a legacy B065 success label; its structured
metadata/paths identify B06.6. The committed print-only label correction is syntax
checked and changes no build or numerical behavior.
