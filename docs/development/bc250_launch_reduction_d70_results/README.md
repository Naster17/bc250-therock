# D70 — LoRA step decomposition and launch-reduction feasibility

Completed 2026-11-13 on the .104 release (app environment, ordinary qualified
callers under L01/D67-pattern guardians). First bounded scope of the broader
launch-reduction direction after D69: **quantify where the ~36 s LoRA update
actually goes, and prove whether HIP graph capture / torch.profiler are
feasible on this stack**, before building any attention/conv route.
Measurement only — no application source, entry point, output or checkpoint
is modified; no user data is read; no default/package/release change.

**Headline results.** (1) The 1024 LoRA step is **GEMM-compute-bound, not
launch-bound**: the GPU is ~100% busy (36.3 s device time per 36.1 s step,
157,675 kernel launches), and ~89% of device time is Tensile FP16 GEMM
kernels running the LoRA projections at very low efficiency (dominant
skinny-tile family: 14.5 ms/call). The optimizer is 0.1%, the eager
attention core 0.3%, conv+chunk ~3%. (2) The ~9× step growth at 4096 is
**not attention either**: the linear-attention mixer grows 10.8× (65.4% of
the step), MLP 7.1×, conv 20×; the eager core grows ~L² but is only 0.5% of
the 4096 step. (3) **HIP graph capture is feasible and bitwise-exact** on
this stack — including D68 Triton kernels and gradient checkpointing
(2.1–3.4× on launch-bound tiny fixtures) — but its payoff on the real model
is bounded by the non-GEMM share. **The evidence-backed next lever is GEMM
efficiency for the LoRA shapes (then native conv), not attention routing or
graph capture.** Both decompositions reproduce the D69 reference steps
within 2% (1024: 0.999×; 4096: 1.017×) with identical loss trajectories.

## Why (D69 finding)

D69 qualified the native gated-delta route for the app's Transformers 5.22.1
but measured only ~1–2% step gain (1024: 1.014×, 4096: 1.020×): the chunk op
is a small share of the whole LoRA step. The step was attributed
*qualitatively* to eager attention, the conv reference, whole-model
launch/dispatch (24 layers × 4 microbatches × checkpointing) and the
134M-parameter optimizer. PLAN-10 discipline ("pick the demonstrated dominant
cost") requires a quantitative decomposition before choosing the next
implementation scope, plus a feasibility verdict on graph capture — the lever
that attacks launch overhead across the whole model at once.

## Implementation (owned tools, `bc250-therock` fork)

- `decompose_lora_d70.py`: owned wrapper around the unchanged
  `train_two_host.py` entry point (D69 wrapper pattern). Transparent
  CUDA-event span instrumentation: class-level phase patches
  (`Trainer.training_step`, `Accelerator.backward`, `Accelerator.clip_grad_norm_`),
  an instance-level patch on the prepared optimizer's `step` (FP16 scaler +
  AdamW together), module hooks on every `Qwen3_5DecoderLayer` (by
  `block_type`) and its MLP, and transparent wraps around the module-level
  `causal_conv1d_fn` / `torch_chunk_gated_delta_rule` / eager attention
  functions. Spans are finalized after the entry point returns — no extra
  synchronization is inserted into the measured steps. Disjoint per-update
  budget: mixer/full_attention, mixer/linear_attention, mlp, fwd_other,
  bwd_other, clip, optimizer, host_gap; fn/* sub-spans split original forward
  vs checkpoint recompute. Kill switch `BC250_D70_DECOMPOSE=0`;
  `BC250_D70_PROFILE=1` additionally profiles exactly one steady update.
- `probe_launch_d70.py`: owned tiny-fixture feasibility probe (no app):
  torch.profiler CUDA activity, HIP graph capture of a launch-bound MLP
  fwd+bwd, tiny Qwen3.5 fwd+bwd (eager ATen path and D68 fused Triton path),
  capture with gradient checkpointing, `make_graphed_callables`, and an
  instrumentation selftest (hook coverage + budget closure) before any big
  run. Blocked captures are findings, parity violations are failures.
- `launch_reduction_d70/`: D67-pattern guardian (logic byte-identical to the
  qualified D69 guardian), identical bounded launcher, orchestrator with
  resume, pure summarizer.

Pre-declared bounds (before any run): instrumented steady step within ±5% of
the D69 reference 36.166 s; final loss |Δ| ≤ 1e-3 vs 2.4717 (hooks are
math-neutral); graph replay parity rel-L2 ≤ 1e-5 (FP32); selftest budget
residual ≤ 10% of the tiny-step wall.

## Feasibility probe (app environment, guardian-bounded)

Environment: torch 2.13.0a0+gfx1013, HIP 7.17.0, transformers 5.22.1,
gfx1013:xnack-. **All seven sections passed** (final evidence run 4; earlier
defects below):

| Section | Verdict | Replay parity | Eager → replay |
| --- | --- | ---: | ---: |
| torch.profiler CUDA activity | works | — | census available |
| Graph: 64-layer MLP fwd+bwd | works | bitwise 0 | 12.01 → 5.63 ms (2.13×) |
| Graph: tiny Qwen3.5 eager (ATen) | works | bitwise 0 | 28.49 → 13.62 ms (2.09×) |
| Graph: tiny Qwen3.5 + D68 fused (Triton) | works | bitwise 0 | 29.28 → 11.83 ms (2.47×) |
| Graph: + gradient checkpointing | works | bitwise 0 | 52.45 → 15.26 ms (3.44×) |
| `make_graphed_callables` | works | bitwise 0 | — |
| Instrumentation selftest | works | residual 0.12% | leftover = 8 (as diagnosed) |

**HIP graph capture is feasible on this stack** — including the D68 Triton
kernels (75 native calls, zero fallbacks inside the captured region) and
non-reentrant gradient checkpointing, with bitwise-identical loss and
gradients on replay. The launch-bound tiny fixtures gain 2.1–3.4× from
capture alone. `make_graphed_callables` (the practical per-layer integration
path) works once prior eager autograd graphs are released.

Checkpointing hook semantics established by diag1 (retained evidence): the
recomputed forward fires module forward-PRE hooks but is interrupted before
the POST hooks; full backward hooks fire exactly once per microbatch and
cover the recompute (selftest: only 1.6 ms of 38.5 ms backward outside the
layer backward spans). The recorder encodes these semantics (LIFO pairing
tolerates the dangling recompute pres; leftovers counted).

## LoRA step decomposition (owned D67 data, 1024 tokens, 6 updates)

D67 settings (rank-128/alpha-256 all-linear FP16 eager + gradient
checkpointing, global batch 4, seed 42, constant LR), reference path (no
routing), hash-verified owned synthetic data (c70264c1…). Reproduction:
steady **36.118 s/update** vs D69 36.166 s (**0.999×**, within ±5%); final
loss **2.47170** (D69: 2.4717, |Δ| ≤ 1e-3); peak 6.17 GiB (D69: 6.16);
4380 spans, 0 unpaired hooks, 1152 dangling recompute pres exactly as
diagnosed, exact uninstall. GPU-timeline coverage 1.0002; CPU wall closure
residual 0.000 s on every update.

GPU timeline per update (steady median of updates 2–6; disjoint event spans):

| Category | s/update | Share |
| --- | ---: | ---: |
| mixer/linear_attention (18 layers) | 19.755 | 54.7% |
| mlp (24 layers) | 11.740 | 32.5% |
| mixer/full_attention (6 layers) | 2.111 | 5.8% |
| bwd_other (glue/recompute remainder) | 1.505 | 4.2% |
| fwd_other (embed/norms/head/loss) | 0.967 | 2.7% |
| optimizer (FP16 scaler + AdamW) | 0.030 | 0.1% |
| clip | 0.007 | 0.0% |

CPU wall structure (overlaps GPU execution; the pipeline is asynchronous):
microbatch calls 22.40 s, inter-micro gaps 10.36 s, post-clip gap 3.34 s
(the audit's finite-gradient check syncs and drains the GPU), pre-optimizer
0.01 s, clip+optimizer walls 0.017 s. The GPU timeline is ~100% covered, so
the GPU — not the host — is the pacing item.

Sub-attribution (fn spans, forward + checkpoint recompute per update):
conv 0.227 + 0.213 s, chunk op 0.396 + 0.399 s, eager attention core
0.058 + 0.058 s. **Inside the 19.8 s linear-attention mixer, conv+chunk are
only ~1.2 s** — the rest is LoRA projections, norms and elementwise glue.
The eager attention *core* at 1024 is 0.12 s (0.3% of the step); the 2.1 s
full-attention mixer is dominated by its projections/rope/gating.

## One-update kernel census (torch.profiler, update 2 of 2)

Kernel events only (parent aten:: ops carry the same device time and are
reported separately to avoid double-counting): **157,675 kernel launches,
160 distinct kernels, 36.32 s device-busy time** in the profiled update
(42.13 s wall with profiler overhead; the unprofiled steady step is 36.1 s —
the GPU is ~100% busy). Top kernels:

| Kernel | Calls | Device s | Note |
| --- | ---: | ---: | --- |
| Cijk…HHS…MT64x1x64_SE…GSU3 | 1176 | 17.07 | 14.5 ms/call, skinny-tile FP16 GEMM |
| Cijk…HHS…MT64x32x8_SN (Bljk) | 3928 | 11.03 | FP16 GEMM |
| Cijk…HHS…MT64x32x8_SN (Ailk) | 3024 | 2.08 | FP16 GEMM |
| Cijk…HHS…MT64x1x64_SE…GSU15 | 72 | 1.26 | skinny-tile FP16 GEMM |
| elementwise_manual_unroll<128,8> | 936 | 0.75 | |
| Cijk…HSS…MT64x32x8_SE | 32 | 0.54 | FP32-out GEMM (addmm) |

GEMM kernels total ≈ **32.3 s ≈ 89% of device-busy time** (aten::mm alone:
8248 calls, 30.5 s device time). bmm: 19,056 calls / 1.71 s; copy_:
61,202 calls / 1.47 s (dtype/layout casts); the remainder is elementwise.

## Analysis and target ranking (1024)

**The 1024 LoRA step is GEMM-compute-bound, not launch-bound.** The GPU is
~100% busy, and ~89% of its time is Tensile FP16 GEMM kernels executing the
LoRA-adapted projections at very low efficiency: the dominant skinny-tile
family (MT64x1x64, split-K) averages 14.5 ms/call — for rank-128 LoRA GEMMs
that is tens of GFLOPS effective, orders of magnitude below the device's
capability. The 158k launches/update are numerous, but their overhead is
already hidden behind GPU execution; the host gaps overlap device work.

This refines (and partly corrects) D69's qualitative attribution:

- "launch/dispatch dominates" → the *device* is saturated with slow GEMMs;
  launch reduction alone cannot move the 1024 step much;
- "the 134M-param optimizer dominates" → refuted: 30 ms/update (0.1%);
- "eager attention dominates" → not at 1024 (core 0.3%, mixer 5.8%); it is
  the O(L²) growth term for longer sequences (see 4096 below);
- the chunk op and conv are ~3% of the step combined — D68's op-level 1.8×
  correctly translated to only ~1–2% step gain.

Ranked levers (evidence-backed at 1024 AND 4096):

1. **GEMM efficiency for the LoRA shapes** — ~89% of 1024 device time, and
   the superlinear growth term at 4096 (mixer 10.9×, MLP 7.2× for 4× the
   tokens). Concrete candidates, each a separate bounded scope: qualify the
   explicit CK FP16 path (B07.13 `bc250-ck`) on these shapes; Tensile
   solution selection/tuning for skinny-N FP16 GEMMs; batching/fusing the
   LoRA A/B GEMMs. A 2× GEMM improvement would cut the 1024 step ~45% and
   attack the dominant 4096/8192 cost.
2. **Native depthwise causal conv** — the conv reference grows 20× from
   1024 to 4096 (0.44 → 8.95 s, 2.7% of the 4096 step) and keeps growing;
   a qualified trainable-weight conv route (D66/D68 deferred it) is the
   second long-sequence lever.
3. **Graph capture** — feasible and bitwise-exact (probe), but at both
   lengths it can only remove host overhead that is already overlapped plus
   the ~11% non-GEMM device time; its value grows if GEMM time shrinks.
   Integration with LoRA/checkpointing/optimizer state is a separate scope.
4. **Fused/tiled attention routing** — demoted by the evidence: the eager
   core is 0.3% (1024) / 0.5% (4096) of the step. It scales as L² and would
   matter at much longer sequences or larger head counts, but it is not the
   lever that restores 8192 feasibility here.
5. **chunk fusion (D68 headroom)** — ~1–3% of the step; minor.

## 4096 cold-update decomposition (1 update, owned data 8bbd13bb…)

Reproduction (final run 6): **329.770 s** vs D69 324.172 s (**1.017×**,
within ±5%); coverage 0.9999; CPU closure 0.000 s; 730 spans, 0 unpaired,
192 dangling recompute pres (4 micro × 48 pairs, as diagnosed); peak
6.99 GiB (D69: 6.975); cold first-update loss 2.90242. Same caveat as
D67/D69: cold first update, so first-call kernel-selection/allocator
effects are included — the D69 reference was equally cold, making the
reproduction like-for-like. Cold variability is visible between the two
4096 runs (r5: 325.884 s, r6: 329.770 s; chunk fn 3.61 vs 5.93 s) while the
loss is bit-identical — the math is deterministic, the cold timing is not.

| Category | s/update | Share | Growth vs 1024 |
| --- | ---: | ---: | ---: |
| mixer/linear_attention | 215.122 | 65.2% | **10.9×** |
| mlp | 84.618 | 25.7% | 7.2× |
| mixer/full_attention | 16.107 | 4.9% | 7.6× |
| bwd_other | 6.400 | 1.9% | 4.3× |
| fwd_other | 6.264 | 1.9% | 6.5× |
| optimizer | 1.228 | 0.4% | 41× (still tiny) |
| clip | 0.007 | 0.0% | — |

fn spans (fwd + recompute): conv 4.48 + 4.46 = **8.95 s (20.3× growth)**,
chunk 4.28 + 1.65 = 5.93 s (7.5×; r5 measured 3.61 s / 4.5× — cold
variability), eager core 0.86 + 0.86 = **1.72 s (14.8× ≈ L², but only 0.5%
of the step)**. CPU structure: micro walls 270.8 s, inter-micro gaps 43.3 s,
post-clip 13.8 s — again overlapped with a ~100%-covered GPU timeline.

**The ~9× step growth from 1024 to 4096 is NOT attention.** The eager
attention core scales quadratically as theory predicts, but it is 0.3% of
the 1024 step and 0.5% of the 4096 step. The growth is dominated by the
linear-attention mixer (10.9× for 4× the tokens — superlinear LoRA
projection GEMM cost, consistent with shape-dependent Tensile solution
selection and memory effects) and MLP (7.2×), with the conv reference
growing 20× (0.44 → 8.95 s). Extrapolating the mixer term, 8192 remains
far beyond the envelope primarily because of GEMM/conv scaling — attention
routing alone would not restore 8192 feasibility. (Growth factors compare
a cold 4096 update against steady 1024; they are indicative attribution,
not a controlled scaling study.)

## Defects found and fixed while qualifying

1. **Probe tiny model defaulted to SDPA (r1):** the app loads with
   `attn_implementation="eager"`; the from-config tiny model defaulted to
   sdpa, so the eager function wrap never fired and the graph sections
   captured the wrong path. Fixed: set `config._attn_implementation="eager"`
   (the 5.22.1 constructor does not accept the keyword).
2. **`make_graphed_callables` stale-stream block (r1):** keeping the prior
   eager loss/output alive left stale AccumulateGrad default-stream
   references that invalidate capture. Fixed: release the eager graph before
   capture (fixture defect, not a stack limitation — manual capture worked).
3. **Selftest coverage expectations (r1):** assumed recompute forward POST
   hooks fire; diag1 proved they are interrupted under checkpointing on this
   stack (backward hooks fire once and cover recompute). Expectations and
   recorder pairing updated to the measured semantics; fire counters and
   pending-leftover tracking added.
4. **AdamW class patch vs accelerate (r2):** the decomposition run completed
   all 6 updates cleanly (36.99 s/it, loss 2.4717 identical, peak 6.17 GiB)
   but the exact-identity uninstall refused: accelerate re-patches the inner
   optimizer's instance `step` around every FP16 scaler step
   (`accelerate/optimizer.py`), which destabilizes a class-level AdamW patch.
   Fixed: instance-level patch on the prepared `AcceleratedOptimizer.step`
   (which accelerate never touches), restored by identity-checked deletion.
   The r2 training log is retained as transparency evidence.
5. **Async-pipeline budget and census double-counting (r3):** the first
   aggregation added a wall-derived host gap to the GPU event spans as if
   disjoint (shares summed to 138%): the pipeline is asynchronous, so host
   gaps overlap GPU execution. Fixed: two complementary views (GPU timeline
   coverage + CPU wall structure with inter-micro / pre-optimizer / post-clip
   gaps, closure residual 0). The first census summed kernel events AND
   their parent aten:: ops (both carry device time), reporting 72.9 s
   "device time" for a 42 s update. Fixed: separate by `device_type`
   (kernel-only busy time 36.3 s; op sum reported separately). r3 outputs
   retained as evidence; all final evidence re-run with the fixed tools (r4).
6. **Single-update progress print (r5):** the 4096 case completed its update
   and wrote full evidence, but the wrapper's final progress print called
   `steady_median` (which excludes the first update) on a one-update run and
   raised, failing the guardian exit. Fixed with a `progress_label` helper
   (cold label for single-update runs) + regression test; the r5 output is
   retained and the case re-run cleanly (r6).

## Boundaries

- Single-board (.104) measurement scope; no implementation of attention/conv
  routing or graph integration in this batch — those are separate bounded
  scopes chosen from this evidence.
- Tiny-fixture graph speedups are feasibility evidence, not a production
  speed guarantee; full-model capture with LoRA/optimizer/dynamic shapes is
  untested here.
- Span semantics: forward spans cover the original forward; backward spans
  include checkpoint recompute (verified); host_gap is wall-derived.
- No app/release/default change; installed native module untouched; D67
  owned data reused by SHA-256 (no user data).

## Verification

- Workstation: 19 new unit tests (aggregation closure, steady median,
  summarizer incl. cold-4096, guardian-reuse identity, orchestrator
  configs); full bc250 suite **304 pass**, Black clean.
- On-board: probe / decomposition / census passed with the final tools
  (run 4); 4096 cold-update decomposition run 5; evidence
  `kernel/artifacts/launch-reduction-d70/` (per-file SHA-256
  `artifacts.json`); host task directory
  `/home/nik/bc250work/launch-reduction-d70/` retained for replay. Board
  clean pre/post every phase (guardian kernel-log/registry/foreign-container
  gates, nonce-acked boot guards, boot `a3142efc…` unchanged).
