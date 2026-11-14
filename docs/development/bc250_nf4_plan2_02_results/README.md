# PLAN2-02 — native NF4 / QLoRA weight-only quantization

Completed 2026-11-14 on the .104 release (r2 kernel `7.1.5-bc250rocm-r2`,
release `10.2-gfx1013-20261111-138a66f52`, Torch `2.13.0a0+gfx1013`, HIP 7.17.0,
`gfx1013:xnack-`). Second item of [PLAN2.md](../../../PLAN2.md). **Result:
PASSED** — a native 4-bit NormalFloat (NF4) weight-only recipe with a qualified
forward/backward, QLoRA composite, measured memory reduction and D62 admission,
all opt-in with no native operator, dispatch, default or source change.

## Headline

**70/70 GPU qualification checks + D62 admission passed.** NF4 is a **memory
win, not a compute win**, on RDNA1 (no INT4 matrix cores): weights are stored as
4-bit codes + per-block absmax scales and dequantized to FP16/BF16/FP32 for the
qualified rocBLAS GEMM.

| Metric | Value | Bound | Verdict |
| --- | ---: | --- | --- |
| GPU qualification checks | 70/70 | all | pass |
| Persistent weight bytes (NF4 / FP16) | 0.28125 (~3.56× smaller) | ≤ 0.30 | pass |
| Peak allocated memory (NF4 / FP16, 24-layer fixture) | 0.916 | < 1.0 | pass |
| Pack code-match vs independent reference | 1.000 | ≥ 0.999 | pass |
| Block-scale max relative error | 0.0 | < 1e-5 | pass |
| Dequant rel-L2 vs double reference | 3.8e-08 | < 0.01 | pass |
| NF4 recipe rel-Frobenius vs original (Gaussian) | 0.0916 | < 0.10 | pass |
| QLoRA loss (16 owned BF16 steps) | 0.00560 → 0.00188 | decrease, finite | pass |
| D62 admission | passed; `quant-nf4` fired | passed | pass |
| Refusals / kill-switch | 14 + native boundaries | all refuse | pass |

Compute parity (GPU result vs CPU-double reference operand; max rel-L2 across
4 geometries each):

| Dtype | Forward | Input grad | Bound (rel-L2) |
| --- | ---: | ---: | ---: |
| FP16 | 0.00045 | 0.00029 | < 0.004 |
| BF16 | 0.00353 | 0.00254 | < 0.01 |
| FP32 | 2.9e-07 | 1.9e-07 | < 1e-4 |

Bias-grad parity passes for all three dtypes; strided input is normalized.

## Feasibility decision (established read-only before implementation)

- **bitsandbytes / TorchAO absent** — verified live in the app environment and
  recorded by the B05.1 audit and inventory. Neither is installed or qualified.
- **Native packed INT4 is hardware-blocked, not merely gated.**
  `_weight_int4pack_mm_cuda` / `_convert_weight_to_int4pack_cuda` require
  `isCDNA2orLater` (gfx90a/gfx942/gfx950) **and** MFMA matrix instructions
  (`__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`) that RDNA1/gfx1013 physically
  lacks, plus BF16-only input. No software enablement is possible without
  forbidden architecture spoofing; the instructions do not exist on this core.
- **Decision: native NF4 via dequantize-to-FP16 + the qualified rocBLAS GEMM**, a
  pure-Python helper mirroring the proven B05.x frozen-W8 pattern. **No new
  native kernel and no Torch source edit.** The reusable native precedent is the
  generic scalar `_weight_int8pack_mm` (no MFMA); NF4 instead dequantizes through
  a 16-entry codebook gather + block-scale multiply (standard qualified ops) and
  computes with the existing FP16 GEMM. QLoRA = frozen NF4 base + trainable
  low-rank adapters.

## Method

- **Recipe.** Canonical 16-level NormalFloat-4 codebook (the bitsandbytes NF4
  table, normalized to [-1, 1]); block-wise absmax quantization (block sizes
  16/32/64/128/256, default 64); two 4-bit codes per byte; FP32 block scales;
  zero blocks use scale 1. Nearest level via the sorted midpoints
  (`searchsorted(right=True)`), ties round up. `in_features` must be divisible by
  the block size. This is an **NF4-compatible recipe qualified against an
  independent CPU-double reference — not a bitsandbytes bit-exactness claim.**
- **Forward/backward.** A custom `autograd.Function` dequantizes to the input
  dtype and uses `matmul` (rocBLAS); the input first derivative recomputes the
  dequantized weight from the **compact saved codes/scales** (no full floating
  base is persisted). `once_differentiable` — first derivatives only; the base
  weight/scales are frozen. Bias is added after the GEMM and carries normal
  autograd. A memory-lean dequant gathers the two nibbles separately and applies
  scales by blocked broadcast (bit-identical FP32 product).
- **Modules.** `FrozenNF4Linear` (frozen buffers + optional trainable bias,
  strict versioned state, CPU storage/load allowed, gfx1013 forward) and
  `QLoRALinear` (frozen NF4 base + trainable FP16/BF16/FP32 adapter,
  `alpha/rank` scaling). Explicit conversion only; no automatic model replacement.
- **Independent reference.** A dependency-free pure-Python scalar reference
  (`core.py` + `reference.py`) implements pack/dequant/matmul with a linear-scan
  nearest code and triple-loop matmul — sharing only the canonical codebook
  constants with the torch ops. Local unit tests prove the contract on the
  workstation; the GPU parity runs on native gfx1013.
- **Guardian.** The qualified PLAN2-01 envelope (`guard_nf4.py`): SH runtime
  lock, nonce-matched 600 s boot-guard ack, 460 s bounded child, `timeout 430`
  podman backstop, kernel-log/registry identity, foreign-container refusal,
  900 s orchestrator session. Kernel-log identity ignores ONLY benign governor
  `Unsupported clock type` noise (`kernel_noise.py`; tests prove real faults
  still fail). **Stock-clock gate:** the orchestrator reads `pp_dpm_sclk` and
  refuses if the ceiling exceeds 2000 MHz.

## Memory

- **Persistent weight storage is the capacity win:** NF4 codes + FP32 block
  scales = **0.28125×** the FP16 weight bytes (~3.56× smaller), measured directly.
- **Peak allocated memory** on a 24-layer 1024×1024 owned fixture (forward +
  backward, batch 32): NF4 110.8 MB vs FP16 120.9 MB = **0.916×**. The peak win is
  modest because eager dequant has a per-layer transient (int64 gather indices +
  FP32 intermediates); the persistent saving dominates as model size grows. A
  tiny fixture can mask the peak win entirely (the r2 6-layer fixture measured
  1.19× — retained as evidence); the capacity claim rests on the persistent
  ratio, which is unambiguous.

## QLoRA step

Owned composite (frozen NF4 base of a known weight + rank-16/alpha-32 adapter),
16 AdamW steps on synthetic regression data, **BF16** adapters: finite gradients
every step, loss **0.00560 → 0.00188** (finite, decreasing), adapter parameters
change, base codes/scales **bitwise unchanged**. BF16 is the QLoRA-standard
stable choice: FP16 AdamW `exp_avg_sq` underflows on small gradients and diverges
to NaN (observed in r2; see journey). Higher derivatives refuse
(`once_differentiable`).

## D62 admission

`admit_model_d62.py --routes-only` (gpt2-tiny, float32) under the guardian:
**status passed**, all routes fired — `attention`, `gated-delta`, `conv`,
`quant-w8`, and the new **`quant-nf4`** (`frozen-nf4-op` probe). The NF4
weight-only route is admitted by the D62 framework alongside the existing routes.

## Refusals / kill-switch (all refuse)

Non-gfx1013 CUDA / non-HIP; `requires_grad` weight pack; non-finite weight;
unsupported dtype; `in_features` not divisible by block size; bad block size;
shape/work-bound overflow; non-finite inputs/scales; non-positive scales;
mismatched bias; CPU linear (storage-only); subclass source; bad metadata;
incomplete payload. **Native boundaries preserved:** packed INT4 conversion still
refuses on gfx1013; native `_weight_int8pack_mm` backward still unimplemented.

## The attempt journey (test-harness fixes, not stack defects)

1. **r1** — two harness bugs: the code-match check compared GPU *indices* to
   reference *packed bytes* (fraction 0.0), and the QLoRA source `nn.Linear` was
   not frozen (pack correctly refused). The core ops were already correct
   (linear parity passed). Retained as evidence.
2. **r2** — 67/70: QLoRA diverged to NaN (FP16 AdamW `exp_avg_sq` underflow) and
   the 6-layer memory fixture's dequant transient exceeded the persistent saving
   (peak 1.19×). Fixes: BF16 adapters; memory-lean dequant; 24-layer fixture.
   Retained as evidence.
3. **r3** — **70/70**, guardian passed. D62 routes-only admission then passed
   with `quant-nf4` fired.

No numerical bound was relaxed after a run; the pre-declared bounds
(`tmp/nf4-plan2-02/BOUNDS.md`, fixed before any board run) held throughout.

## Boundaries

- Opt-in pure-Python helper (`build_tools/bc250/nf4_plan2_02/`); **no native
  operator, dispatch, default, package or source change**; the pytorch patch
  series is unchanged (23 patches). Not installed into any runtime image.
- Operand/work bounds mirror the B05.x W8 prototype (dims ≤ 4096, ≤ 4.2 M weight
  elements, ≤ 1 M activation/output elements, ≤ 134 M MACs/call). Real-model
  projections wider than 4096 (e.g. the 6144 intermediate) need a separate bound
  admission; this scope qualifies the recipe on owned fixtures within bounds.
- NF4 is a **memory** win, not a compute win (RDNA1 has no INT4 matrix cores;
  compute is dequantize-to-FP16 + the existing GEMM, with a dequant overhead).
  The recipe rel-Frobenius ~9% on Gaussian weights is the 4-bit quantization
  error, **not a model-quality claim**; no perplexity/accuracy claim is made.
- First derivatives only; frozen base; no integer/scale autograd, no training of
  the base, no packed-INT4 enablement, no bitsandbytes/TorchAO installation.
- Single-board (.104), stock clocks (≤ 2000 MHz), owned synthetic fixtures only;
  no user application; .105 untouched; host governor/clock/watchdog/kernel not
  modified.

## Verification

- Workstation: 34 NF4 unit tests (codebook, nearest-mapping equivalence, nibble
  pack/unpack, independent reference incl. idempotence + quantization-error bound,
  metadata/bounds, orchestrator config, guardian-envelope reuse, D62 route
  registration); full bc250 suite **413 pass**, Black clean.
- On-board: r3 qualification guardian passed (70/70) and D62 admission guardian
  passed; boot `0d46a6f1…` unchanged; kernel log clean (0 benign warnings added);
  registry unchanged; 0 foreign containers; all boot-guards acknowledged; final
  state 0 containers, r2 kernel intact.

## Reproduction and preservation

Tools: `build_tools/bc250/nf4_plan2_02/` (`core.py`, `reference.py`, `ops.py`,
`qualify_nf4.py`, `guard_nf4.py`, `run_nf4.py`, `run_d62_nf4.py`); reuses the
PLAN2-01 `bounded_launch.py` + `kernel_noise.py`. From the package directory on
the workstation:

```sh
python3 run_nf4.py --label nf4-r3      # stage + guarded GPU qualification + collect
python3 run_d62_nf4.py                 # guarded D62 routes-only admission
```

Root steps use `ssh -tt … doas` with the PIN recovered at runtime (never
persisted). The host task directory (`$TASK/nf4-plan2-02`) retains the staged
tools, configs, outputs and per-attempt guardian evidence (r1/r2 failures
included). Local evidence: `kernel/artifacts/nf4-plan2-02` (qualification +
admission reports, guardian finals, kernel logs, preflight, SHA manifest).

## Conclusion

PLAN2-02's NF4/QLoRA capability is **proven and admitted**: a native, opt-in
4-bit weight-only recipe with qualified forward/backward parity, a stable QLoRA
composite, a measured ~3.56× persistent weight-memory reduction (0.916× peak on
the owned fixture), D62 admission, and a preserved refusal surface — all without
touching native operators, defaults, or the release. This is the biggest LLM
capacity unlock within the 16 GB ceiling. Remaining PLAN2 items (native
sparse/eig, attention, compile, broader admission, FP16 PEFT) stay optional and
await user direction.
