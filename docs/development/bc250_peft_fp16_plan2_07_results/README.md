# PLAN2-07 — FP16 PEFT qualification (frozen-helper contracts)

**PASSED 2026-11-14** on .104 (release `10.2-gfx1013-20261111-138a66f52`,
Torch `2.13.0a0+gfx1013`, HIP 7.17.0, peft 0.21.2 / transformers 5.22.1 from the
owned env, boot `0d46a6f1` unchanged). **Seventh and final PLAN2 item.** Scope
(user-confirmed): owned-fixture FP16 PEFT proof mirroring the frozen-helper
precision contracts. **No change to the existing frozen FSDP helpers**
(D28/D30/D31 untouched, requalification rules respected); **no app/source/
default/package/release change** (pytorch series stays 23 patches).

## The gap closed

The frozen-helper PEFT chain covered FP32 (D28) and BF16 (D30/D31). The base
`BC250FSDPTrainer` has an FP16 path (`fsdp_fp16_parameter_policy`: FP16 working +
FP32 master/reduce + native scaler) **but requires all-trainable params — no
PEFT/frozen**; the BF16 PEFT trainer explicitly refuses FP16. So **FP16 PEFT
(frozen base + trainable LoRA, FP16 working + FP32 master + GradScaler) had no
qualified path.** This batch qualifies it on an owned fixture.

## Verdict — 12/12 checks

Fixture mirrors the D28 owned identity: GPT2-tiny (vocab 97, n_embd 32, n_layer 2,
n_head 4, n_inner 64, eager, dropout 0, cache False) = **21280 frozen base
params** + real `peft` LoRA rank 4 / alpha 8 / zero-init B on the 8 Conv1D
projections (c_attn/c_proj/c_fc) = **3072 trainable adapter params**.

| Check | Result |
| --- | --- |
| load-param-counts | 21280 frozen / 3072 trainable |
| precision-contract | FP32 master/frozen, FP16 autocast working, GradScaler |
| forward-vs-cpu-fp64 | relative **0.0** (bitwise) |
| backward-grads-vs-cpu-fp64 | global rel **7.3e-04**, worst per-param **9.3e-04** (bounds 0.02/0.03) |
| frozen-base-no-grads | absent |
| fp16-vs-fp32-relationship | loss rel **0.0** |
| root-clip-vs-cpu | rel **0.0** (bound 1e-5) |
| grad-scaler-inf-skip | step skipped, scale **65536→32768** |
| multi-update-training | loss 4.6025→4.5601 (8 updates, decreasing), base bitwise unchanged |
| resume-bitwise-state | exact |
| resume-exact-continuation | **bitwise** (3-step loss sequence + adapter/optimizer fingerprint) |

## FP16-specific findings

- **GradScaler is required for FP16** (unlike BF16, which uses no scaler). The
  inf/nan safety works: a forced inf gradient (injected into the scaled grad
  before `unscale_`, where GradScaler computes `found_inf`) skips the optimizer
  step and halves the scale.
- **FP32 master weights/optimizer state avoid the pure-FP16 AdamW underflow**
  found in PLAN2-02 (FP16 `exp_avg_sq` underflow → NaN). With FP32 masters the
  FP16 PEFT path trains stably and replays bitwise.
- FP16 (10 mantissa bits) is at least as tight as the qualified BF16 path:
  forward loss matches CPU-FP64 bitwise (the CE loss reduces in FP32), adapter
  gradients ~7e-4 global relative.

## Method

`build_tools/bc250/peft_fp16_plan2_07/` qualifies the FP16 PEFT policy on the
owned fixture: FP16 autocast working + FP32 master/frozen/reduce + GradScaler +
named LoRA + AdamW + root clip + full save (adapter/optimizer/scaler/RNG) +
bitwise replay. It does **not** import or modify the frozen FSDP helpers; it is a
single-process owned-fixture proof of the FP16 precision contract, complementing
the distributed D28/D30/D31 qualifications. peft 0.21.2 / transformers 5.22.1
(owned env) are recorded; the frozen FSDP helpers remain pinned to their qualified
package pair (transformers 5.2.0 / accelerate 1.12.0 / peft 0.18.1), untouched.

## Attempt journey (harness fixes, retained)

fp-r1: `rel_l2` device mismatch (cuda grads vs cpu reference) — fixed to move to
CPU. fp-r2: GradScaler inf-skip test injected inf *after* `unscale_` (GradScaler
computes `found_inf` *during* `unscale_`) — fixed to inject before unscale. fp-r3
passed 12/12. No numeric bound or envelope parameter relaxed. 11 harness tests,
suite 489, Black clean.

## Envelope and safety

Qualified PLAN2-01..06 guardian (SH lock, nonce-acked 600 s boot-guard, 460 s
child, `timeout 430` backstop, kernel_noise filter, registry identity,
foreign-container refusal) + stock-clock gate ≤2000 MHz. Boot `0d46a6f1`
unchanged; kernel logs identical (0 benign added); registry unchanged; 0 foreign
containers; boards idle pre/post; .105 untouched; host governor/clock/watchdog/
kernel NOT modified. No default/package/release/source change; not installed into
any image.

## Bounds and remaining scope

Owned tiny GPT2-tiny + rank-4 LoRA, single-process; not a distributed FSDP FP16
PEFT qualification (the frozen FSDP helpers remain FP32/BF16; an FP16 FSDP PEFT
trainer is a separate future scope). No model-quality/convergence claim; the
8-update loss decrease is a sanity signal, not a training result. Larger models,
real datasets, and the distributed FP16 PEFT path remain separate.

## Reproduction

```bash
python3 build_tools/bc250/peft_fp16_plan2_07/run_peft_fp16.py --label fp-r3
```

## Evidence

`kernel/artifacts/peft-fp16-plan2-07/`: BOUNDS.md, qualify-fp-r1…r3 reports,
guardians/preflights/kernel logs, orchestrator logs. Host task dir
`/home/nik/bc250work/peft-fp16-plan2-07` retained. Publication: bc250-therock
tools+report (no source change), umbrella curated PLAN2-07 + compatibility matrix.

**With PLAN2-07 complete, all seven forward-plan items (PLAN2-01…07) are done.**