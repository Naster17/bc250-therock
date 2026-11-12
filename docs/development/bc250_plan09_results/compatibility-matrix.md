# BC-250 model/environment compatibility matrix

PLAN-09 living record of which package combinations and kernel routes are
**actually tested/qualified** on the native gfx1013 stack. This is not a promise
list: a combination appears here only with bounded evidence. Architecture names
alone do not establish support; each model is admitted by the owned admission
fixture (below) plus its recorded route/fallback behavior.

- Native provider: patched Torch/Triton/ROCm for `gfx1013:xnack-`.
- Release: `10.2-gfx1013-20261111-138a66f52` (active on .104 and .105).
- Status key: **qualified** = bounded evidence on the release; **frozen** =
  pinned fixture scope, retained; **absent/unqualified** = not installed or not
  proven; **pending** = admission run not yet recorded.

## 1. Native provider (shared by every environment)

| Component | Identity | Source |
| --- | --- | --- |
| Torch | `2.13.0a0+gfx1013`, rev `a5249802372882dedccb91d3fd4d73be9fcdf2bd` | D61 SBOM |
| Triton | rev `ee65ff3b85f99d52e678b252a99c8425fa46e329`, wheel `2d65d6fe…` | D61 SBOM |
| ROCm libraries / systems | `8389197c…` / `1f9609fd…` | D61 SBOM |
| Runtime profiles | `daily` (`86d94f43…`), `rccl-single` (`f183bd3f…`) | D59/D61 |
| Target arch | `gfx1013:xnack-` (no MFMA/WMMA; SDMA off) | D61 known-limitations |

No environment below replaces this provider. Package overlays/venvs link to it;
they do not freeze or substitute the native Torch/ROCm binaries.

## 2. Qualified Python package environments

Three distinct consumer environments are qualified. They are **not
interchangeable**: installing a newer Transformers into one does not qualify the
other's helpers or a new model's kernels.

### 2a. Frozen training-consumer fixtures (`bc250-train`)

Authoritative installed set, read from the release
`python-consumers/packages.json` on .104 (2026-11-12):

| Package | Version | Scope / evidence |
| --- | --- | --- |
| transformers | 5.2.0 | D26–D31 Trainer/FSDP/PEFT fixtures |
| accelerate | 1.12.0 | D28 full-model save/load route |
| peft | 0.18.1 | D28 canonical LoRA adapter export/replay |
| torch | 2.13.0a0+gfx1013 | native provider |
| triton | 3.8.0 | native provider wheel |
| numpy | 2.5.1 | base |
| huggingface-hub | 1.4.1 | support |
| safetensors | 0.8.0 | support |
| tokenizers | 0.22.2 | support |

- `training.env`: `PYTHONPATH=…/python-consumers/site-packages`,
  `HF_HUB_DISABLE_XET=1`, `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=8`.
  `runtime-constraints.txt` pins torch/triton/numpy/click.
- Qualified scope: owned GPT2 causal-LM (tied head), FP32/BF16/FP16 FSDP policies,
  frozen-base + native/PEFT LoRA adapters, non-reentrant activation checkpointing,
  bitwise checkpoint save/resume. Evidence: D26/D27/D28/D29/D30/D31 reports.
- The Trainer helpers **pin** these versions and refuse on mismatch; a source or
  package change requires helper requalification (see §6).

### 2b. lora-smoke application environment (L01/L02, `bc250-distributed`)

Verified installed in `~/bc250-projects/lora-smoke/env` on .104 (2026-11-12);
`python-origin.json` records Python 3.12.3, base `/opt/torch-venv/bin/python`,
torch 2.13.0a0+gfx1013 from the native provider (`/workspace/pytorch-work/torch`).

| Package | Version | Scope / evidence |
| --- | --- | --- |
| transformers | 5.22.1 | L01/L02 two-host LoRA |
| accelerate | 1.15.0 | L01/L02 |
| peft | 0.21.2 | L01/L02 |
| trl | 1.17.1 | L01/L02 |
| datasets | 5.1.0 | L01/L02 |
| torch | 2.13.0a0+gfx1013 | native provider (linked, not in app venv) |

- Qualified scope: native DDP two-host LoRA on Qwen3.5-2B (full base per board,
  FP32 LoRA + AdamW, RCCL over Ethernet), shifted-target loss normalization,
  sealed-checkpoint replay (L02). Evidence: BC250_LORA_TWO_HOST.md, L01/L02.
- This is an **application** environment, separate from the frozen fixtures. The
  newer application Trainer is not the source-pinned `BC250Trainer`; the old
  accumulation-divisor correction is deliberately not applied to it.
- Not a general Qwen-family or two-board speedup guarantee.

### 2c. Qwen3.5 training overlay (A06/A07, separate Transformers 5.19.0)

| Package | Version | Scope / evidence |
| --- | --- | --- |
| transformers | 5.19.0 | Qwen3.5 hybrid linear/full-attention ops overlay |
| torch | 2.13.0a0+gfx1013 | native provider |

- A **third** Transformers version, distinct from both the frozen 5.2.0 consumers
  and the 5.22.1 lora-smoke app. `qwen35_training_ops.install()` refuses unless
  `transformers.__version__ == "5.19.0"` and `model_type == "qwen3_5_text"`.
- Qualified scope: owned gated-delta (A06) + tiled/hybrid attention (A07) +
  frozen causal-conv ops on the hybrid Qwen3.5 architecture. Toggles
  `BC250_QWEN35_TRAINING_OPS=0` / `BC250_GATED_DELTA=0` restore references.
- The overlay runtime lives under the excluded user application; owned fixtures
  must lift the random-init config branch rather than import that module
  (the D62 admission fixture does this via its `qwen35-hybrid-tiny` registry
  entry, which records the architecture absent when 5.19.0 is not present).

## 3. Consumer wheels (release `python-consumers`)

| Package | Version | Source revision |
| --- | --- | --- |
| torchvision | 0.28.0+gfx1013 | `8fb87713…` |
| torchaudio | 2.11.0+gfx1013 | `1ae7f5d8…` |
| torchcodec | 0.17.0+gfx1013 | `3eecace0…` |

ONNX stack: onnx 1.23.2, onnxscript 0.7.2, onnx-ir 1.1.0, onnxruntime 1.31.0,
ml-dtypes 0.6.0, protobuf 7.36.2, flatbuffers 25.12.19 (D61 SBOM). ONNX Runtime
GPU requires an explicit registered execution provider (`bc250-ort`).

## 4. Kernel routes: qualified vs absent

| Route | Status | Notes |
| --- | --- | --- |
| Attention — math/eager SDPA | qualified | reference bring-up path; FP32/BF16/FP16 |
| Attention — tiled/hybrid | bounded | dropout/compile/graph boundaries remain; not a default |
| Conv (MIOpen) | qualified | PLAN-02 FP16 FP32-accumulation repair; versioned cache |
| Gated-delta | bounded | A06 owned fixtures; not a general FLA guarantee |
| Quantization — W8 | frozen | `bc250_quant 0.1.0`, explicit opt-in, frozen layouts |
| Quantization — NF4 / packed INT4 / full QLoRA | absent/unqualified | TorchAO/bitsandbytes not installed; native direct backward refuses |
| Compiler (torch.compile) | bounded | dense/gated/CNN/batchnorm fixtures (D59); eager is the reference |

A fallback can be correct yet slow. Warning suppression is not a backend fix.

## 5. New-model admission (reusable owned fixture)

Admission criteria (developer guide §10), operationalized by the D62 fixture
`admit_model_d62.py` (owned, random-init from config, no downloaded weights, no
dependency on excluded applications):

1. import/load on the native provider;
2. GPU forward (+ logits/loss vs an independent CPU/FP64 reference where feasible);
3. GPU backward with finite/nonzero expected gradients;
4. an optimizer step;
5. bitwise checkpoint save/resume and exact resumed continuation;
6. record the **actual** attention/conv/delta/quant routes exercised and any
   fallback, plus dtype/shape/mask/attention-implementation/seed/tolerance.

Admission is bounded validation, not universal correctness or model-quality
proof. Forward support alone is insufficient for a training claim.

### Admission records

| Model (owned config) | Env | Routes observed | Result | Evidence |
| --- | --- | --- | --- | --- |
| _pending D63 run_ | | | | |

## 6. Requalification rules

- A native Torch/ROCm **source change** requires requalification of the affected
  frozen helpers and any admitted model that uses the changed path.
- A **package version change** in either environment requires re-running the
  affected fixtures/admission; the frozen helpers refuse on pin mismatch.
- No model-family promise from an architecture name alone; each model is admitted
  individually with its recorded routes/fallbacks.
- Installing a newer Transformers does not automatically qualify the frozen
  helpers or a new model's kernels.

## Sources

D61 SBOM (`kernel/artifacts/release-d61/sbom.json`); D26–D31 reports; L01/L02
(`BC250_LORA_TWO_HOST.md`); PLAN-02 repair; B05/B05.6 quant; A06 gated-delta;
developer guide §9–10. Admission entries added by D62/D63.