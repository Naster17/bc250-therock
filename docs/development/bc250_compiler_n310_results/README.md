# N3.10: model-independent compiled training coverage

Completed2026-10-20. Code `7c8156c57`. The user's broad ROCm/PyTorch goal takes
priority over further Qwen-specific tuning: Qwen/LoRA remains one fixture, not
the support boundary. This batch replaces the proposed Qwen-only MLP check
with generic CNN/dense/gated/BatchNorm compiler qualification. Existing P1
pointwise/indexing/one-layer MLP/norm and small dynamic math-attention evidence
was inspected first and remains separate.

No numerical defect reproduced; **no production Torch/ROCm implementation was
changed**. The improvement is reproducible coverage of previously unqualified
composed training paths, not a new speed claim or proof of universal support.

## Contracts and results

Four standalone train-mode modules, no model packages/weights/user data:

| Module | Operation boundary | Largest CPU-reference relativeL2 | GPU trace kernels |
| --- | --- | --- | --- |
| Dense | Residual two-linear SiLU + affine LayerNorm | 0.004123 | 17 |
| Gated | Gated SiLU/up/down projections + residual RMS normalization | 0.006393 | 22 |
| CNN | Group2/dilation2 Conv2d + GroupNorm + GELU + adaptive pooling + linear | 0.008807 | 52 |
| BatchNorm | Training affine BatchNorm2d + SiLU + pooling + linear | 0.004749 | 10 |

FP32 plus FP16/BF16 **autocast with FP32 master parameters**; two shapes each,
contiguous/last-dimension-strided MLP input and contiguous/channels-last CNN/BN.
MLP shapes[2,9,17]/[3,7,17], image shapes[2,4,7,9]/[3,4,9,11]. These deliberately
exercise tails/layouts/state, not large-model throughput.

48 ordinary eager/compiled cases compare full outputs, input and every parameter
gradient against an independently executed CPU FP64 model starting with identical
FP32 values. Upstream gradients are nonuniform and quantized to actual output
dtype. CPU FP64 finite differences validate three oracle coordinates per family
(12 probes). AMP CPU references use FP64 mathematical intermediates, not bitwise
emulation of every low-precision rounding; explicit error bounds apply.
Initial unchanged limits: FP32 relativeL2<0.0005, FP16<0.006, BF16<0.04, alongside
per-element atol/rtol and finite/nonzero gradient checks. No tolerance was relaxed.

Every ordinary case independently replays its own GPU gradient through CPU FP32
AdamW, comparing parameters and step/exp_avg/exp_avg_sq; largest parameter error
1.49011612e-8. This tests the ordinary optimizer update, **not a compiled optimizer**.
All three BatchNorm buffers, including running mean/variance and integer batch
counter, match the CPU reference within the stated buffer contract.

Real `torch.compile(fullgraph=True,dynamic=False)` delegates a recording backend
to Inductor; eager fallback is not accepted.24 captured graphs/314 generated
kernels reported by Inductor metrics (not314 distinct operators or launches).
Each precision compiles both shapes separately; this is not dynamic-shape reuse.
Four additional warmed compiled/profiled forward/backward executions recheck
CPU-reference values and show101 native GPU kernel records, including Triton
kernels. FX code and four Chrome traces are preserved. Profiler overhead and
physical clocks are not performance measurements here.

**56 composite checks pass:**48 output/all-gradient/buffer/CPU-AdamW cases,
four finite-difference oracle groups and four traced CPU-reference replays.
Total52 GPU forward/backward executions,48 optimizer updates. The reused host
controller retains legacy sequence_length/audited_shape fields; they do not
describe these inputs. Per-case result shapes above are authoritative. Black/Ruff-F/
syntax/diff checks pass. No native source edit/build/reinstall required.

## Policy and preservation

The pinned Inductor source `torch/_inductor/utils.py:is_big_gpu` explicitly
excludes HIP major10 from max-autotune GEMM (its source comment calls the
older-model exclusion arbitrary). The emitted warning is retained. That policy
is not proof that every scalar tuning strategy is physically impossible.
This batch does not alter the guard, enable matrix instructions, qualify
max-autotune or claim Tensor Core/MFMA/WMMA support. Ordinary supported Inductor
and native BLAS routes remain the tested path.

Run through the exclusive prefix lock and existing600s recovery guards:

```
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_compiler_n310_on_host.py --attempt r1
```

`WS` is `/home/nik/bc250work/rocm-10.2-gfx1013`. Four guards acknowledge;
all313 protected identities/four FFT AOT databases/links retained, whole-batch
kernel log identical. Existing JIT caches are used/extended, never cleared;
four CPU/compiler threads retained. No active-prefix install/switch or app edit.

New boot observed **at entry**:7.1.5-bc250rocm-r2,
`48bdf727-b2f1-4d7b-aba5-2a0b775f1387`; no agent reboot. Native runtime62ab2437c,
rocBLAS8389197c1/Torchdc8fea0/source2b449c4/other libraries319a235d6 unchanged.
N3.2 preset/math defaults retained; experimental vocabulary/square overrides
inactive. No deletion/.105/BIOS/fallback change. Disk28,975,644,672B free observed;
no storage reclamation performed in this batch. No own running job/container/
guard deadline; root SSH closed. Historical Python CPU fault remains unresolved,
separate from fixed VM-heap/CSR and historical N3.8 host OOM.

All56 raw regular files (1,304,438B) copied and SHA256-verified locally at
`kernel/artifacts/training-n310/remote`; compact evidence/verification beside
this report. Raw `WS/logs/training-n310-*`; preservation baseline
`training-n310-qualification-r1/final.json`. Reused resource preflight additionally
writes `training-n39-resource-compiler-n310-r1.json`; its bytes are copied into
the N3.10 resource record. Previous evidence/caches/checkpoints remain.

Next **B04.1:** inspect matching Torch ABI/package/build configuration and existing
media/RPP evidence, select pinned compatible torchvision source, build isolated
native gfx1013 extensions on .104, and qualify actual image operators (for
example NMS/ROI Align forward/backward) with CPU references before deployment.
Do not substitute RPP coverage for torchvision or install a different Torch wheel.
Torchaudio follows separately; BF16 FFT/general sparse solve/distributed/low-bit/
modern CK plugin gaps and broader N2/N4–N8 remain open. Qwen is optional testing,
not the prerequisite for these broader library/API expansions.
