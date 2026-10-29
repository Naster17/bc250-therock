# B06.12: modern Torch exports through public GPU inference

Completed 2026-10-29 on .104. Qualification tools `75da6725bd2566828ace1074c5d24c7e0e6347ad`;
evidence export `e9e6c29a8`. Installed ORT source `b81a7866e5a6eff964b9e411535b4c9198651a9f`,
registered MIGraphX SDK, native ROCm/Torch and both runtime profiles remain unchanged.

## Result and scope

Six fresh graphs cover owned dense, CNN and gated/LayerNorm models, each exported
from CPU and native gfx1013 GPU model state with modern `torch.onnx.export`
(`dynamo=True`, opset18). Static batch3 outputs are IR10/opset18 as produced;
no IR rewriting or opset substitution. Two deterministic changed inputs per graph
are checked against Torch CPU FP64, eager HIP FP32, retained CPU ORT1.31 and
ONNX NumPy ReferenceEvaluator. Tolerance remains `rtol=3e-4, atol=2e-5`.
Model state is unchanged and no export-time parameter gradients are created.

Generation30 + public `bc250-ort` daily30/RCCL30 + retained core7/CPU ONNX4/
daily5/RCCL6 = **112 checks**, seven acknowledged recovery guards. The consumer
runs perform24 GPU graph evaluations;12 actual per-model profiles assign all
recorded node events to MIGraphX. CPU fallback is disabled at session creation
and Python fallback is disabled afterward. Generation also performs12 eager GPU
forwards. Wrong shape/type/missing-input refusals, native architecture, loaded
package/provider origins, explicit CPU execution after GPU and exact native
Torch HIP backward pass. This last check is runtime coexistence, not ONNX autograd.

3642 protected hashes,35 links/four AOT DB paths remain unchanged;136 retained
workstation backing paths match. Same r2 boot, unchanged kernel logs, no remaining
owned jobs/containers/armed guards. Ordinary free30,209,069,056B. No native rebuild,
installation, prefix/default change or cache reset was needed. .105 and user
applications were not accessed.29 local stack tests, selected critical Ruff,
Black, Python compilation and diff checks pass; full upstream CI was not run.

## Practical B06 gap review

| Capability | Current evidence | Decision |
| --- | --- | --- |
| Modern static FP32 Torch export to GPU inference | Six fresh CPU/GPU-origin graphs, four reference routes, both profiles | Qualified within these recorded shapes/operators |
| Explicit ORT GPU provider and package recovery | B06.11 installation/relocation/identity/rollback; B06.12 node profiles | Keep opt-in `bc250-ort`; ordinary CPU ORT preserved |
| Native MIGraphX C/C++/Python SDK | B06.5–B06.9 bounded graphs/device-pointer/stream/ABI evidence | Retain registered SDK; do not repeat as missing |
| Dynamic/symbolic shapes, other opsets/operators/models | Not covered by this static set | Separate requirement and reference fixtures before expansion |
| FP16/BF16/quantized graph execution | Low-precision conversion disabled in public cohort | No dtype/performance claim; separate numerical and hardware audit |
| Concurrency, wider allocator/stream contracts | Earlier bounded SDK boundary tests are not general ORT coverage | Separate bounded task if required |
| ONNX graph training/autograd | This route is explicit inference; separate Torch backward still works | No automatic Torch dispatch or ONNX autograd claim |
| Performance | Correctness gates do not measure representative throughput | No speedup claim |

No additional mandatory implementation defect was found in this tested path.
Park optional B06 expansion until a concrete workload selects it. Next **B07.1**
is an admission audit of MIOpen CK grouped-convolution plugin ABI/instances and
native scalar/DL options versus already qualified N2.18/HIP-direct alternatives.
The current plugin build list excludes gfx1013; adding its name cannot supply
missing kernels or unsupported matrix instructions. Audit first, then select
whether a bounded port adds useful coverage. This is a plan, not B07 qualification.
The historical Python CPU fault remains unresolved and separate from repaired
VM-heap corruption.

## Use and evidence

Continue using [the explicit provider/session and recovery instructions](../bc250_onnxruntime_b0611_results/README.md).
For modern exports, keep the produced graph intact and validate its model contract
before GPU use. Successful export alone does not establish supported GPU execution.

Reproduce the owned gate only on a free .104 using the recorded lock/guard helpers:
`helpers/qualify_onnxruntime_b0612_on_host.py --attempt r2` selects new paths and
retains r1; it does not install or change runtime packages. Export helper currently
seals accepted r1 explicitly. Raw evidence is retained locally under
`kernel/artifacts/onnxruntime-b0612/raw/` and on .104 at
`WS/logs/onnxruntime-b0612-qualification-r1/final.json`, with fixtures at
`WS/tmp/onnxruntime-b0612/generation-r1/models`. `WS` is
`/home/nik/bc250work/rocm-10.2-gfx1013`.

Evidence archive813709 bytes, SHA256
`f5a882a28504f7e0058fd9ae4982f7bf13ad307a58da5ad7e36d92b208f967b1`;
179 raw files/3,670,773 bytes, including graphs, weights, input/reference bytes,
export reports, profiles, guard records, commands and source helpers. Controller
PID5491 and evidence export exited0. Local hash/profile/reference-file verification
passes; matching verification/cleanup receipts are retained beside the archive.
