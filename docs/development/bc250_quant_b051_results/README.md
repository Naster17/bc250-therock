# B05.1: native low-bit audit and isolated frozen W8 recipe

Tooling revision:`c4383972c8b59f06e436f84bcb57c203cd11c523`.
This batch selects and qualifies a bounded per-output-channel INT8 weight-only
recipe on `.104`. It is an explicitly imported prototype, not an installed
package or a model conversion. Daily/opt-in Torch/ROCm, images and defaults are
unchanged; `.105` and user applications were not inspected or modified.
[Verified identities/results](verification.json).

## Existing paths and observed defects

| Path | Pinned source/runtime evidence | Decision |
| --- | --- | --- |
| A08 `torch._int_mm` | Existing qualified signed INT8 × INT8 → INT32 scalar Lt path | Retain; not a weight-only quantization recipe or integer autograd |
| `_weight_int8pack_mm` | Existing generic HIP kernel in `aten/src/ATen/native/cuda/int8mm.cu`; contiguous FP32/FP16/BF16 CPU references pass | Reuse only with normalized layouts and explicit first-gradient wrapper |
| Strided activation | Native kernel reads row-major storage;15/15 output elements differ in the safe bounded transpose fixture | Demonstrated native bug, **not fixed** in this batch |
| Strided FP32 scales | `.to(float)` preserves the strided view, kernel reads contiguous storage;12/15 elements differ | Demonstrated native bug, **not fixed** in this batch |
| Native INT8 input derivative | Backward explicitly raises `derivative for aten::_weight_int8pack_mm is not implemented` | Preserve native refusal; prototype supplies its own input derivative |
| Packed native INT4 | Source has `TODO: Support RDNA`, CDNA2 architecture gate and matrix-instruction implementation; gfx1013 conversion refuses | No gate bypass or architecture spoofing; scalar packed INT4 remains separate future work |
| TorchAO / bitsandbytes | Neither installed in the reviewed runtime | No installation, external kernel or model-recipe qualification claim |
| Triton | Pinned3.8.0 available | Availability alone does not qualify third-party low-bit kernels; selected recipe uses existing generic HIP |

Active native Torch is`2b449c4`, version2.13.0a0+gfx1013. The qualified patched
local Torch source is14cfef2; both INT8/INT4 source hashes match the active source
exactly. Full hashes and native schema/autograd source provenance are recorded.
No Torch/library/kernel source edit or native rebuild occurred.

The initial audit incorrectly expected missing backward to leave a null gradient;
the actual explicit exception stopped the probe. The collector was corrected to
record that refusal, without changing backend behavior or numerical tolerances.
Both probe versions, the failed report and recovery guard are retained. The
corrected audit passes seven contracts/findings; reproduced wrong results are
reported as defects, not numerical passes.

## Prototype contract

`build_tools/bc250/bc250_quant_b051.py` provides `pack(frozen_weight)` and
`linear(input, packed_weight, bias=None)`:

- Symmetric per-output-channel INT8 quantization: FP32 max-absolute row scale
  times the FP32 reciprocal127, round-to-even and clamp[-127,127]. Zero rows use
  scale1; nonzero scale underflow is refused. Pack accepts explicitly frozen
  FP16/BF16/FP32 HIP weights and retains no floating base tensor.
- Forward normalizes activation/value/scale layouts and calls the existing
  native weight-only kernel. Output retains the explicit input dtype, including
  inside AMP; bias is added **after** the native output cast. This is not an
  automatic `F.linear` autocast replacement.
- Input backward uses `(upstream.float()*scales) @ integer_weight`, expressed
  through the same native kernel with contiguous transposed INT8 values and
  unit scales. No full floating base weight is materialized. Bias/owned adapter
  first gradients follow normal autograd. Quantization/scales/base weights are
  frozen; integer/scale derivatives, higher derivatives and training the base
  are not supported.
- Dimensions<=4096, at most4,194,304 weight elements,1,048,576 activation/output
  elements and134,217,728 multiply-accumulates per call. These are operand/work
  limits, not a total memory/allocator bound. Finite-data/scale validation
  synchronizes with the host; numerical tensor computation remains on GPU.

The prototype does not replace native operators, install TorchAO/bitsandbytes,
implement NF4/packed INT4, convert checkpoints or claim complete QLoRA support.
Callers retaining the original floating weight also retain its memory.

## Qualification and measured limits

**110 checks across six accepted gates**:

- Corrected native audit7, prototype71, core7, compiled dense14, daily launch5,
  opt-in RCCL launch6. All use owned fixtures through ordinary launchers.
- Prototype36 composites cover three dtypes/four geometries/three layouts,
  independent scalar CPU packing and CPU-double output/input/bias gradients,
  padding/unchanged storage and logical payload sizes. Additional checks cover
  ties/zero rows, all256 signed byte values, private stream handoff, six exact
  finite differences, two owned low-rank adapter composites, AMP dtype policy,
  absent higher derivatives,25 refusals, native forward/backward tracing and
  two bounded timing fixtures. This is not model quality or universal gradients.
- The trace contains exactly two native `weight_int8pack_mm_kernel` GPU events
  for forward/backward. Host control and finite-check scalar synchronization
  remain; no full activation/weight CPU arithmetic fallback is used.
-377 protected hashes/four AOT DBs/links unchanged. Seven600s guards including
  the initial failed probe ACK; all per-gate and whole-batch kernel logs match.
  Same r2 boot`5c04afbb-ff2f-4f4b-9eac-8238bb9288fc`, observed new since B03.8; reboot cause not investigated and no agent reboot.29 local BC250 tests and selected Ruff
  checks/format pass; no full upstream CI or broad stability claim.

Three alternating pairs of three eager forward calls, resident inputs, FP32
recipe versus a resident dequantized FP32 `F.linear` baseline:

| M,N,K | Recipe median ms | Dense median ms | Fresh packed payload | FP16 weight bytes |
| --- | ---: | ---: | ---: | ---: |
|8,128,256 |0.26045 |0.08222 |33,280 |65,536 |
|32,256,256 |0.25816 |0.08134 |66,560 |131,072 |

The recipe is about3.2× slower in these small fixtures, including its validation
and host synchronization. Fresh packed logical weight payload is about1.97×
smaller than FP16/BF16; this excludes source weights retained by callers,
activations, temporary backward transpose/scaled gradients and allocator costs.
No peak-memory, model-quality, training speed or general performance improvement
is established. The prototype stays isolated.

## Reproduce and preserve

On .104, `WS=/home/nik/bc250work/rocm-10.2-gfx1013`. The helper module is in
`WS/helpers`; fixtures import it explicitly. No site-package or runtime manifest
change is needed. For a stack-owned experiment importing the helper:

```sh
bc250-torch --mount-ro "$WS/helpers" --env PYTHONPATH="$WS/helpers" /path/to/owned_experiment.py
```

`qualify_quant_b051_on_host.py` preserves the B03.8 baseline under the daily
shared lock, incorporates the accepted audit/prototype gates, and runs four
ordinary-launch regressions. Do not wrap ordinary launchers in the exclusive
update lock. Export uses `update_dev_prefix.py -- python3 .../export_quant_b051_on_host.py`
under the exclusive daily lock after tests stop; no prefix contents are changed.
All helpers use single-use labels/directories and preserve failed evidence.

Raw `WS/logs/quant-b051-*`; final
`WS/logs/quant-b051-qualification-r1/final.json`. Local
`kernel/artifacts/quant-b051` contains the verified475,903-byte archive
SHA256`6051eee04658afdb81fd017087ae01f2b268843303202965a57906447ebb2f41`,
63 raw logs/1,497,225bytes, all five helper sources and the initial collector,
native source evidence and seven guard records. No native build/ccache operation,
cache cleanup or source reset occurred. Historical unknown Python CPU faults
remain unresolved and distinct from fixed VM-heap/CSR/hostcall defects.

**Next B05.2:** repair the demonstrated native INT8 activation/scale layout
handling in qualified Torch, preserve the native unsupported-backward boundary,
build an isolated candidate with recorded resources/ccache, and rerun independent
references before any activation. Keep daily Gloo and opt-in RCCL payloads,
`.105`, user applications, sources/builds/caches and fallbacks intact.
