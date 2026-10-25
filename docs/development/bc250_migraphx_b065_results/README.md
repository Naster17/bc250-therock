# B06.5: isolated native ONNX parser on gfx1013

All 68 checks/six guarded gates and final preservation/export checks passed. This candidate adds native C++ ONNX parsing to the B06.4
FP32 backend. It is not installed in the active runtime. Python bindings and the
ONNX Runtime GPU provider remain unqualified and unavailable in the daily image.
The existing CPU ONNX Runtime and Torch defaults are unchanged.

MIGraphX source remains `622ea3f2ad8f1baa8a44edb4f1e07cbe6e532c83`;
no further MIGraphX source patch was needed. The private parser dependencies
follow its committed requirements:

| Source | Pin | Revision |
| --- | --- | --- |
| Abseil | 20250512.0 | `bc257a88f7c1939f24e0379f14a3589e926c950c` |
| Protobuf | v30.0; `libprotoc 30.0` | `d295af5c3002c08e1bfd9d7f9e175d0a4d015f1e` |

Protobuf's annotated tag object is `2c82161c69d52248d7dd34d0fba499e2743d411f`.
Pins were resolved from the upstream Git repositories; no signature qualification
is claimed. Complete Git bundles for both dependencies passed fresh clone,
checkout and connectivity checks. Existing profiler Abseil 20260107.1 and the
Torch protobuf source are preserved. These native dependencies are separate
from the existing Python protobuf wheel.

## Executed scope

Six owned models use ONNX IR 10/opset 18, standard FP32 graph inputs and outputs:
broadcast Add/ReLU, row ReduceSum, MatMul/bias/ReLU, transposed MatMul,
Conv/bias/ReLU, and a linear/ReLU model with serialized FP32 weight/bias
initializers. The last case exercises actual initializer decoding and GPU
literal upload. Each compiled graph executes with two changed input sets.

CPU ONNX Runtime explicitly selects only `CPUExecutionProvider`, disables
fallback and matches independent NumPy FP64 oracles for 12 evaluations.
The C++ parser's file and buffer routes produce equal uncompiled IR for all
six models. The file-parsed graphs compile for native gfx1013 wave32 and their
outputs match both CPU-double and CPU ORT results. Parsed/runtime output shapes,
dtypes and host input guards are checked. Required lowering is actually present:
`gpu::code_object`, `gpu::gemm`, `gpu::convolution`, and initializer `gpu::literal`.
Explicit HIP copies connect host inputs/outputs. This does not qualify GPU
allocation canaries or buffer lifetime beyond the exercised calls.

Unknown operator and malformed wire-data fixtures are refused through both
file and buffer routes with the expected parser diagnostics. Exceptions with
unrelated messages are test failures. This is four refusal cases, not a general
security/fuzzing qualification.

The native parser gate has 24 composites. Retained B06.4 hand-built graphs 22,
core 7, CPU ONNX audit 4 and daily/RCCL consumers 5/6 bring 68 across six guarded
gates. FP32 comparisons retain `abs(error) <= 2e-5 + 2e-5*abs(CPU_double)`;
`fast_math=false` and exhaustive tuning off. No arithmetic, dtype policy or
tolerance was relaxed. The prior five-model/10-reference fixture set is retained;
the accepted `models-r2` set adds serialized weights and 12 CPU references.

The loader resolves candidate parser/GPU/core libraries and the selected native
ROCm libraries. Protobuf/Abseil are static dependencies hidden by the parser's
existing export/link policy; no new shared protobuf replaces the Python wheel.
MIOpen fusion remains excluded by MIGraphX's existing gfx1013 whitelist decision;
convolution and separate pointwise kernels execute instead. No whitelist,
BF16/FP8/MX/CK/rocMLIR admission or model performance claim was added.

## Storage, builds and evidence

B06.4's documented temporary SSHFS transport is reused. New files are isolated
under local `tmp/migraphx-b064/20261025-storage/durable/b065`, exposed on .104 as
`WS/tmp/migraphx-b064/20261025-storage/mount/b065` (STORAGE). The old B06.4 build
and cache were copied with reflinks into the new task, preserving the original.
Native read-only source snapshots are under `WS/tmp/migraphx-b065/{abseil,protobuf}`;
MIGraphX uses the retained B06.4 native input snapshot. Root headroom remains low;
all build outputs, installed private dependencies and caches use external storage.
No retained source/build/cache/image was deleted and no reserve policy changed.

Four CPUs/jobs, link slot one, RAM 10 GiB, memory+swap 12 GiB; network off,
no GPU exposed to builds, read-only container root, C++17/O2/PIC and separate
2 GiB ccache paths without resets. Admission requires 16 GiB external ordinary
space; the controller stops its own job at 14 GiB growth, root ordinary space
below 768 MiB or two hours. The private dependency install prefix is
`STORAGE/dependencies`, not `/opt` or a Python environment.

Run the committed `build_migraphx_b065_on_host.py` phases `abseil`, `protobuf`,
then `parser`, with fresh attempt names; it snapshots its shell script before
execution. Accepted logs are `STORAGE/logs-abseil-r1`, `logs-protobuf-r2` and
`logs-parser-r1`. `generate_migraphx_b065.py` prepares CPU model references;
`qualify_migraphx_b065_on_host.py` runs public launchers under the existing
shared prefix lock and recovery procedure. `export_migraphx_b065_on_host.py`
checks preservation and records the source/build/model/guard evidence.
Temporary relay replay/cleanup is described in the B06.4 report.

Two setup failures are retained. Initial protobuf extraction raced archive
production and hit EOF before any protobuf build; completed archive SHA checking
and re-extraction succeeded. Later, updating the live shell script while its
phase was running truncated the final `protoc --version` command; compilation
and install had finished. The accepted incremental retry uses an immutable
phase script and reports `libprotoc 30.0`, with no rebuild or source change.
The initial CPU-only model launch was resumed after a background TTY stop;
subsequent launches use null stdin. None was a GPU numerical failure.
The disabled-rocMLIR unused-helper and host `--rocm-path` warnings remain;
no warning suppression was introduced. Full upstream CI was not run.

**Next B06.6:** audit/build the pinned Python bindings in an isolated candidate,
then qualify Python ONNX inference, serialized-program roundtrips and import
ordering with the existing Torch/CPU ORT environment before considering an
installed SDK package. Keep the active runtime and .105 unchanged. ORT GPU EP,
graph autograd, other dtypes/IR/opsets, fusion admission and model speed remain
separate work. The unknown historical Python CPU segfault remains unresolved
and distinct from the fixed VM-heap corruption.


## Final seal and handoff

All 629 protected hashes/four AOT DBs/links match. Six recovery guards acknowledged;
kernel logs match through export and unmount. The board had rebooted before this
batch: all current evidence is on r2 boot
`e955931f-5edb-42e5-8f14-eafc2fbaf361`; no cross-boot dmesg equality is claimed.
No agent reboot, root-reserve policy change, active prefix/image update or .105
contact occurred. Temporary mount, relay, test containers and owned jobs are
closed; backing artifacts, source snapshots, caches and stopped build containers
remain. Source verification covers 4,782 MIGraphX, 1,532 Abseil and 3,052 Protobuf
files. Existing source revisions and user-owned applications remain untouched.

Code revisions: TheRock checkpoint `b6a629170`, accepted fixture/qualification
`4711e9892f08599ef3db13ba1cbd2f90554795a4`. 29 local stack tests, Black,
Python compilation, shell syntax and C++ formatting/syntax checks pass. Ruff was
unavailable; full upstream CI was not run.

Local evidence: `kernel/artifacts/migraphx-b065/evidence.tar.gz`,
67,909,794 bytes; SHA256
`a44098301bfa85484798d9695a6656b360161389ce157c72f8b2260ac10ce6c8`.
118 extracted raw files total 2,346,845 bytes. Complete dependency Git bundles
(17,724,835 and 174,300,444 bytes) are retained locally, with fresh-reconstruction
verification in `source-bundles.json`; they are not copied into the low-space
.104 root filesystem. Native parser build approximately 451 MiB, private
installed dependencies 49 MiB, dependency builds 6.6/69 MiB and caches 19/50 MiB
remain on durable external storage. Export measured root ordinary headroom
877,867,008 bytes and external ordinary headroom 114,507,022,336 bytes before
writing the archive. Root storage remains a constraint for any future installed
SDK image; this batch performs no installation into the daily image.
