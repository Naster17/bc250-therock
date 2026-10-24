# B06.1: modern ONNX export and CPU runtime

The .104 runtime now provides explicit ONNX export and `CPUExecutionProvider`
inference alongside the existing native Torch GPU stack. This adds an export/
CPU-inference path; no ONNX GPU execution provider is installed or qualified.
[Usage](../bc250_torch_usage.md#onnx-export-and-explicit-cpu-inference-b061-104).
Tooling/source revision: `6ae431c9a2ebbe745af54306701ad42261779afc` (full revision in verification.json).

## Applicability and pinned dependencies

The patched Torch source at a524980 imports
`onnxscript._framework_apis.torch_2_11`; that exact API is present and tested in
the selected wheel. Its `dynamo=True` path uses `torch.export`; the old automatic
TorchScript fallback option is removed in this source. Upstream documents
[ONNX/ONNX Script exporter dependencies](https://docs.pytorch.org/docs/main/onnx_export.html).

ONNX Runtime's [ROCm provider documentation](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html)
states that the provider was removed in1.23 and directs AMD users to
[MIGraphX](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html).
The current profile has no qualified MIGraphX build or registered provider. This batch selects
the CPU wheel and does not revive the retired EP, install an incompatible AMD
wheel, spoof an architecture or claim that MIGraphX cannot be ported later.
The [provider API documentation](https://onnxruntime.ai/docs/api/python/api_summary.html)
explains provider selection and CPU data handling. Sources checked2026-10-24.

| Package | Selected version |
| --- | --- |
|onnx |1.23.2 |
|onnxscript |0.7.2 |
|onnx-ir |1.1.0 |
|onnxruntime |1.31.0, CPU wheel |
|ml-dtypes |0.6.0 |
|protobuf |7.36.2 |
|flatbuffers |25.12.19 |

[Exact wheel URLs, hashes, ABI tags, declared dependencies](../../../build_tools/bc250/onnx_b061_packages.json).
These seven packages were absent from the base image. Existing NumPy2.5.1,
SymPy1.14.0, packaging26.2 and typing_extensions4.16.0 stay unchanged; no dependency
upgrade or native Torch/ROCm rebuild occurs. Wheel licensing metadata is preserved.
Installation is offline, hash-required and `--no-deps`; stage uses CPUquota2/
RAM1GiB/no network or pull/180s image deadline. Ccache/native source builds are
unneeded. The new image retains the complete old W8-image layer prefix.

## Qualification

181 candidate +181 ordinary installed checks,22 accepted guarded gates; the early
export probe additionally passes33 checks with its source retained:

| Checks per phase | Coverage |
| ---: | --- |
|8 | Both cohorts: seven pinned packages, unchanged existing numerical dependencies, exporter API/W8/FFT/sparse imports, provider inventory, native GPU OrtValue allocation refusal by the CPU wheel. |
|66 | Both cohorts: dense/CNN/gated+LayerNorm, exported from CPU and gfx1013 FP32 tensors with modern dynamo/opset18; checked standard ONNX graphs/symbolic batch, three batch sizes, CPU-double Torch/ONNX NumPy reference/GPU eager comparisons, explicit host→Torch GPU downstream first gradients, actual CPU node profiles, dtype/shape/missing-input refusals. |
|4 | ORT session initialized before the first owned Torch GPU workload, both cohorts: CPU identity result, native GPU GEMM/input first gradient and FFT references. Torch was not preimported at script entry in all four order gates. |
|11 | Daily/opt-in training consumers and native backend identity. |
|71 | Retained installed W8 recipe: CPU references/first gradients/refusals/private stream/trace. |
|7 | Retained core GPU first derivatives/RNG/FFT and training consumers. |
|14 | Retained compiled dense outputs/first gradients versus CPU references. |

All ONNX inference sessions explicitly select CPU and disable runtime fallback.
Actual ORT profile nodes identify `CPUExecutionProvider`. The wheel also advertises
AzureExecutionProvider, which is excluded from these sessions and unqualified. Available provider names
and the CPU build's native GPU-allocation refusal are retained. This is direct
evidence of CPU inference, not an implicit GPU claim. Exporting a CUDA/HIP Torch
model produces a graph with host initializers; it does not select a GPU ONNX EP.

The owned FP32 fixture dimensions are small: dense inputsB×8; CNNB×2×8×8;
gated/normalizationB×5×8; batches1/3/4. Forward tolerances rtol3e-4/atol2e-5;
downstream HIP leaf-gradient rtol2e-6/atol1e-6. ONNX checker full validation and
NumPy ReferenceEvaluator provide an additional independent runtime. The Torch
export range1–4 is not enforced by ONNX symbolic dimensions in ORT. No numeric
assertion/tolerance is relaxed. These are inference and downstream Torch-gradient
checks, not autograd through ONNX, ONNX training, model convergence/quality,
large models, quantized W8 export, other dtypes or performance qualification.

The first qualification invocation retained an old report-directory name and
refused before any GPU gate; it did not overwrite prior B05.6 evidence. The name
was corrected; refusal/source are preserved.
No package/export/numeric production defect was found.29 local BC250 tests and
selected Ruff checks pass. All23 guards ACK; kernel logs unchanged per gate and
through candidate/installed/export. No own tests/containers/armed guards remain.
Historical Python CPU segfault remains unresolved, distinct from fixed VM-heap/
CSR/hostcall defects.

## Reproduction, deployment and recovery

Fetch the pinned wheels locally with
`python3 build_tools/bc250/fetch_onnx_b061.py --output <wheel-directory>`.
Existing wheel bytes are verified/preserved. Stage them as
`WS/onnx-b061-input/wheels` and stage the committed tools/manifest in `WS/helpers`.
Stage builds an isolated image and daily/RCCL candidate prefixes. Source trees,
builds/caches/checkpoints/releases/images/links remain; `.105` and user applications
are untouched. No kernel/BIOS/default boot, math SDPA, SDMA0, heap-off, N3.2,
daily NCCL-off or explicit bounded RCCL policy change.

Candidate and ordinary tests use the daily shared lock. New ONNX/W8 suites use
`bc250-torch`; retained training-dependent fixtures use `bc250-train`. Internal
candidate tests select isolated releases. Only after candidate success are both
runtime manifests replaced under the existing daily EX updater; ordinary installed
acceptance follows outside EX, then locked acceptance. The controller restores
old manifests under EX if installed testing fails. The two-file journal handles
Python exceptions but is not a host-crash-atomic transaction.

Old manifests are in `WS/recovery/onnx-b061-before`; prior imagee2821cf8 and both
candidate prefixes remain. Identity-guarded rollback (retained, not exercised):

```sh
WS=/home/nik/bc250work/rocm-10.2-gfx1013
doas python3 "$WS/helpers/update_dev_prefix.py" -- \
  python3 "$WS/helpers/deploy_onnx_b061_on_host.py" --rollback
```

This restores the preceding W8-package daily/profile metadata exactly.
[Source/runtime/image/package/preservation identities](verification.json),
[archive digest](archive.json). Raw reports:
`WS/logs/onnx-b061-{candidate,installed}-r1/final.json`; local raw artifacts:
`kernel/artifacts/onnx-b061`. Archive includes seven hashed source/manifest files,
wheels/licenses/build context, checked models/export reports/ORT profiles,
guards/commands/exit/kernel logs and both recovery manifests. Archive35,255,852B
SHA256`fb55f104f7c83fea3812c3a09acf05de56983df4636c0ae28e984accf3029e44`;
284 raw log/fixture files5,296,329B.477 protected hashes/four AOT DBs/links retain
expected identities with manifest changes/recovery explicitly accounted for.
Image1bb4ac0e; daily runtimea9a5afe3/profiledb556925; native Torcha524 remains.
Memory snapshot is
synchronized to `.104` only; no own jobs remain at handoff.

**Next B06.2:** inspect pinned MIGraphX source/dependencies and gfx1013 instruction/
backend applicability, including generic/scalar possibilities and build headroom,
before selecting any isolated GPU-provider experiment. Current ORT remains CPU;
no architecture override, unsupported matrix admission or GPU EP claim.
