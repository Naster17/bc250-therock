# D69 — native gated-delta route portability to Transformers 5.22.1

Completed 2026-11-13 on the .104 release (app environment, ordinary qualified
callers under L01/D67-pattern guardians). Ports the D66 native-routing
discipline to the application's newer Transformers 5.22.1 environment and
re-checks D67's long-sequence LoRA infeasibility with the D68 fused op.
**Result: the port is qualified and clean (parity, anchors, refusals, 864/144
native calls with zero fallbacks, exact uninstall), but the step-level gain is
only ~1–2% (1024: 1.014×, 4096: 1.020×) and 8192 remains infeasible within the
qualified 460 s envelope — the LoRA step is launch-bound across the whole
model, not just the gated-delta op.** No application source, entry point,
output or checkpoint is modified; no user data is read; no default/package/
release change.

## Why a port was needed (read-only reconstruction)

Transformers 5.22.1 changed the Qwen3.5 gated-delta reference in two ways:

1. the 5.2.0 **63-iteration Python inverse loop is gone** — the non-export
   path now uses batched `torch.linalg.solve_triangular` (the loop survives
   only in the `is_torchdynamo_exporting()` branch);
2. `Qwen3_5GatedDeltaNet.forward` calls the **module-level global**
   `torch_chunk_gated_delta_rule` (kernel-hub decorated), not the per-instance
   attribute the D66 helper patches.

What remains launch-bound is the **sequential Python chunk scan** — the same
structure the installed native op's default path uses. Consequences:

- the D66 instance patch does not apply to 5.22.1;
- routing to the *installed* native op is near-parity (both ATen-solve +
  Python-scan; op parity measured bitwise 0 for BF16/FP16);
- the **D68 fused candidate** (`use_fused_solve_scan`, fused inverse + fused
  scan from two chunks) is the actual acceleration path for 5.22.1.

## Implementation (owned tools, `bc250-therock` fork)

- `route_qwen35_gated_delta_d69.py`: module-scope route. Patches
  `modeling_qwen3_5.torch_chunk_gated_delta_rule` with a gated wrapper;
  exact-identity uninstall. Gates: transformers == 5.22.1, actual gfx1013,
  app config `model_type == qwen3_5_text` verified before install, kill switch
  `BC250_NATIVE_GATED_DELTA=0`, eager first-order, FP32/BF16/FP16, geometry
  (≤128 head width, ≤1M padded tokens), no initial state. 5.22.1 forwards
  `cu_seqlens` + TransformersKwargs on every call: a pure classifier drops the
  control flags the torch reference itself ignores (`num_items_in_batch`,
  `output_*`, `return_dict` — recorded as ignored), and falls back (counted)
  on packed sequences (`cu_seqlens`/`seq_idx`/`cu_seq_lens_k`/`max_length_*`/
  `position_ids` non-None), `is_causal=False`, or unknown kwargs. Two explicit
  native targets: `installed` (release op) and `fused` (staged D68 candidate;
  refused unless the candidate has the fused path).
- `d69_train_wrapper.py`: owned wrapper around the unchanged
  `train_two_host.py` entry point — verifies the app config model type,
  installs the route, forwards argv verbatim, always uninstalls and writes
  `route-record-rank<N>.json` (native calls, fallbacks, ignored control keys).
  Knobs: `BC250_D69_ROUTE=0` (reference), `BC250_D69_NATIVE=installed|fused`.
- `qualify_route_d69.py`: app-environment qualification fixture (op parity vs
  the 5.22.1 reference, CPU-FP64 recurrence anchor, tiny owned Qwen3.5 text
  model parity under both targets, refusals/fallback counters).
- `route_port_d69/`: D67-pattern guardian (audit optional for parity phases),
  bounded launcher (identical envelope), orchestrator, summarizer.

Pre-declared bounds (before any run): op rel-L2 FP32 1e-5 / low 1e-2 (D66 op
bounds); CPU/FP64 recurrence anchor at the A06 tolerances; tiny-model parity
at the D66 model bounds + CPU/FP64 gradient anchor 5e-3.

## Qualification (app environment, guardian-bounded)

Fixture `qualify_route_d69.py` under the D67-pattern guardian (SH lock, 600 s
nonce-acked boot guard, 460 s bounded child, kernel-log/registry identity,
foreign-container refusal). Environment: transformers **5.22.1**, torch
2.13.0a0+gfx1013, triton 3.8.0; candidate module `06c5a8e1…` (D68), installed
native module `21398daf…` (unchanged). **All four sections passed.**

- **Op parity vs the 5.22.1 reference** (output + final state + 5 input grads,
  with initial state; geometries tiny/long256/2B-heads-256/2B-heads-1024):
  FP32 ≤ 7.3e-7 rel-L2 (bound 1e-5); **BF16/FP16 bitwise 0** — the 5.22.1
  reference and the native reference-ordered preparation are identical at low
  precision. The fused candidate matches within the same bounds.
- **CPU/FP64 token-recurrence anchor** (1,65,2,7,9, fixed upstream grads):
  worst tolerance fraction FP32 0.002 / BF16 0.053 / FP16 0.041 of the A06
  tolerances, across reference, installed and fused.
- **Tiny owned Qwen3.5 text model** (4 layers, 3 linear + 1 full attention,
  random weights, seq 64 → one chunk): FP32 worst grad rel-L2 **1.25e-3** ≤
  2e-3 for both targets (identical — at C=1 the fused flag keeps the default
  path, as designed); BF16/FP16 **bitwise 0** vs the reference. Independent
  CPU/FP64 gradient anchor: reference 1.86e-3, installed/fused **1.17e-3**
  (bound 5e-3) — the native path is closer to FP64 than the reference.
- **Refusals (7 checks):** kill switch, wrong-version gate, bad target, fused
  target requires the D68 candidate, exact uninstall identity (double
  uninstall and foreign-patch refusal), counted fallbacks (packed cu_seqlens
  forwards to the reference, which itself refuses packed sequences without
  FLA; initial-state; geometry; dtype), control-flag ignore-and-record
  (`num_items_in_batch`), and native dispatch bitwise-equal to a direct fused
  call.

## LoRA benchmark re-check (owned D67 data, global batch 4)

Reuses D67's hash-verified owned synthetic data (no user data), the qualified
`train_two_host.py` benchmark mode, rank-128/alpha-256 all-linear FP16 eager +
gradient checkpointing, dropout 0.05, constant LR, seed 42 — the D67 settings.
Every routed case's `route-record-rank0.json` proves the path: **864 native
calls per case** (6 updates × 4 microbatches × 18 linear layers × forward +
checkpoint recompute), **zero fallbacks**, exact uninstall.

**1024 tokens (6 updates, first excluded; steady median of 5):**

| Case | Steady s/update | tokens/s | vs ref | Final loss | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference (5.22.1) | 36.166 | 113.3 | 1× | 2.4717 | 6.16 |
| route → installed native | 36.189 | 113.2 | 1.000× | 2.4717 | 6.16 |
| route → D68 fused | 35.670 | 114.8 | 1.014× | 2.4727 | 6.16 |

The reference reproduces D67's 36.262 s within 0.3%. Routing to the installed
native op is exactly parity (both are ATen-solve + Python-scan; low-precision
preparation is bitwise identical, hence the identical loss trajectory). The
fused path is fully active (its slightly different rounding shows in the loss)
but the step gain at 1024 is marginal: **the chunk op is a small share of the
whole LoRA step** — the rest (eager full attention, the conv reference,
whole-model launch/dispatch overhead across 24 layers × 4 microbatches ×
checkpointing, and the 134M-parameter optimizer) dominates. This refines D67's
attribution: the gated-delta reference is launch-bound *as an op* (D68 measured
1.8× there), but fixing it alone does not move the 1024 step much.

**4096 tokens (cold first update, 1 update per case):**

| Case | First update s | tokens/s | vs ref | Peak GiB |
| --- | ---: | ---: | ---: | ---: |
| reference (5.22.1) | 324.172 | 50.5 | 1× | 6.975 |
| route → D68 fused | 317.671 | 51.6 | **1.020×** | 6.974 |

The reference reproduces D67's 326.822 s within 0.8%. The fused case routed
cleanly (144 native calls = 1 update × 4 micro × 18 layers × fwd+recompute,
zero fallbacks).

**8192 tokens (feasibility re-check, fused route):** the first update did
**not** complete within the qualified 460 s guardian child budget; the
guardian stopped it safely (nonce-acked boot guard, kernel log and registry
unchanged, no foreign containers, board boot unchanged, route uninstalled
with 18 native calls recorded before the stop). **8192 remains infeasible
within the qualified safety envelope**, now with the fused route active —
the D67 infeasibility is only marginally improved because the chunk op is not
the dominant step cost.

### Conclusion (honest, bounded)

The route port is **qualified and clean**: parity within pre-declared bounds,
CPU-FP64 anchors, 7 refusal checks, 864/144 native calls with zero fallbacks,
exact uninstall, app entry point/environment/release untouched, boards clean.
But the **step-level gain from fusing the gated-delta op is only ~1–2%** at
1024/4096 (1.014× / 1.020×), and 8192 stays beyond the envelope. The op-level
1.8× (D68) is real, yet the chunk op is a small share of the whole LoRA step:
eager full attention, the conv reference, whole-model launch/dispatch overhead
(24 layers × 4 microbatches × gradient checkpointing) and the 134M-parameter
optimizer dominate. This **refines D67's attribution** — "launch-bound
gated-delta" is true at the op level, but the app step is launch-bound across
the whole model, not just that op. The next lever for long-sequence LoRA is
broader launch reduction (qualified fused/tiled attention, conv, or graph
capture) — separate bounded scopes, not this port.

## Defects found and fixed while qualifying

1. **Fixture unpack (r1):** op-parity zipped the gradient tuple as one element
   (`'tuple' object has no attribute 'detach'`); fixed.
2. **cu_seqlens gate (r1→r2):** 5.22.1 always passes `cu_seqlens=None`; the
   first wrapper counted any kwarg key as a fallback, so model calls never
   routed. Fixed: None is the unpacked contract; non-None falls back.
3. **TransformersKwargs control flags (r2→r4):** the model/Trainer also
   forwards `num_items_in_batch`/`output_*`/`use_cache` control flags, which
   the torch reference ignores in the chunk body. Fixed with the explicit
   classifier (ignore-and-record the control flags; value-sensitive and
   unknown kwargs fall back). The residual keys were identified from the
   wrapper's own `fallback_kwarg_keys` evidence (`['cu_seqlens', 'use_cache']`).
4. **Refusal-test expectations (r2/r3):** demanded bitwise equality between
   the native op and the reference (they are different FP32 roundings, parity
   1.5e-7) — fixed to the pre-declared rel-L2 bound; and expected the packed
   fallback to return a result, while the 5.22.1 torch reference itself
   *refuses* packed sequences without flash-linear-attention — the wrapper
   correctly forwards and lets the reference raise; the test now expects that.
5. **runpy sys.path gap (r4→r5):** `runpy.run_path` does not put the script's
   directory on `sys.path` the way direct execution does, so the app entry
   point could not import its sibling `bc250_lora_support`. The wrapper now
   inserts PROJECT explicitly (replicating direct-execution semantics). The
   orchestrator gained resume support (skip passed phases, root-clean failed
   evidence/output dirs) so the parity phase was not re-run.

## Boundaries

- Single-board (.104) scope; the two-board DDP re-check is a separate
  follow-up. Eager first-order only; recurrent/decode and trainable-weight
  conv not routed; packed sequences fall back.
- The route is opt-in tooling around the unchanged app entry point; the
  installed release module and the app environment packages are untouched.
- Bounded owned measurements — not a production, convergence or general-model
  speed guarantee.

## Verification

- Workstation: classifier/gate/wrapper/summarizer unit tests (full bc250 suite
  **285 pass**), Black clean.
- On-board: parity fixture passed (run 4); LoRA cases passed (run 5: 1024
  ref/installed/fused, 4096 ref/fused) with the 8192 feasibility case safely
  stopped at the guardian budget; evidence
  `kernel/artifacts/route-port-d69/` (28 files + per-file SHA-256
  `artifacts.json`: parity report, per-case audits/route records/guardian
  finals/gpu logs, orchestrator log); host task directory
  `/home/nik/bc250work/route-port-d69/` retained for replay. Board clean
  pre/post every phase (guardian kernel-log/registry/foreign-container gates,
  nonce-acked boot guards, boot `a3142efc…` unchanged, 0 containers at
  handoff). D67 owned data reused by SHA-256 (no user data).
