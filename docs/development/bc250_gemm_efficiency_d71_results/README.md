# D71 — GEMM efficiency for the LoRA shapes

Completed 2026-11-13 on the .104 release (app environment, D67-pattern
guardians). First bounded scope of the evidence-ranked #1 lever from D70.
This batch censused the exact GEMM layouts, swept the existing admitted
Tensile solutions per layout, baked off hipBLASLt, A/B-tested a candidate
override preset on the whole LoRA step, and took a per-shape in-model
profile. Measurement and qualification only — no release/preset deployment,
no application source change, no user data.

**Headline: the easy GEMM win does not exist on this stack.** FP16 solution
selection has no headroom (2 admitted solutions/layout, all within noise;
candidate preset adds 0 rows; whole-step A/B bit-identical, 1.001×);
hipBLASLt is 1.7–3.6× slower. The per-shape profile independently confirms
D70 (GEMM ~90% of device time, 32.8 s/update) but corrects the framing: the
cost is the **large base-model projections** (35 ms/call in-model), not the
cheap LoRA adapters (0.25 ms/call), and isolated hot-cache fixtures
understate the in-model cost 4× — so the slowdown is memory/state-dependent,
not a bad-solution choice. **Recommendation revised: stop FP16 solution
tuning; the tractable levers are native conv, graph capture, and the non-GEMM
remainder; a GEMM breakthrough needs dedicated microarchitectural R&D.**

## Method (the qualified N32 pattern, reused)

N32 (2026-10-19) established the exact method for this stack: one logical
mm can map to several rocBLAS layouts; the default solution selection can
be catastrophically bad for a layout (N32 measured 173.8 ms default vs
4.84 ms for an existing admitted solution, 35.9×, on the BF16 vocab GEMM);
the fix is a per-layout `solution_index` override CSV, identity-gated by
the launcher (`ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH`, opt-out
`BC250_GEMM_TUNING=0`). The release already carries that mechanism with one
qualified BF16 row; the FP16 LoRA GEMMs are untouched by it.

D71 tools (owned, `bc250-therock`):

- `census_gemm_d71.py`: TorchDispatchMode census of every mm/bmm/addmm in
  one LoRA update — exact sizes, strides, dtypes, counts, plus the
  torch→rocBLAS column-major layout conversion (N32 convention).
  Dispatch-instrumented: counts/layouts only, never timing.
- `bench_gemm_d71.py`: private-handle `rocblas_gemm_ex_get_solutions`
  enumeration (≤32 candidates/layout, top 10 layouts by count, mm/addmm
  batch 1), every candidate checked against an independent CPU-double
  reference, canary guards, input preservation, 6 alternating rounds × 3
  calls paired timing of default vs best. Plus a torch-level hipBLASLt
  AB/BA bake-off (fresh subprocesses, `TORCH_BLAS_PREFER_HIPBLASLT=0/1`)
  on the top shapes. No global state change, no new kernels.
- `gemm_efficiency_d71/`: preset generator (release rows preserved verbatim
  + winners ≥1.05×), D69/D70-identical guardian, orchestrator with resume,
  pure summarizer. Whole-step A/B: reference (launcher automatic preset)
  vs candidate (manual override CSV = release rows + FP16 winners), 6
  updates each, D67 settings on hash-verified owned data.

Pre-declared bounds (before any run): candidate parity vs CPU-double
rtol 0.015 / atol 0.05 / rel-L2 < 0.004 (the unchanged N32 contract);
whole-step reference within ±5% of D69 36.166 s; candidate final loss
|Δ| ≤ 0.01 vs reference (reduction order changes — not bitwise); finite
gradients, no skipped steps.

## Census (1024, one update)

**27,240 GEMM calls, 71 distinct layouts.** The top mm layouts are the LoRA
projections: `1024×2048 @ 2048×128` (rank-128 A, 1296 calls, rocBLAS TN
M128 N1024 K2048), the B variants (648 each), merged QKV `…@2048×6144`
(528), plus the large vocab-side GEMMs (M32768, 14 calls each, ~23 ms/call).
The numerous batch-16 bmm layouts (64–128 dims) are the chunk-op internals
(~1.7 s device time in D70) — censused, not swept (the override mechanism's
batched coverage is unqualified).

## Solution bake-off (40 mm/addmm layouts, private handle)

**Result: essentially zero FP16 solution-selection headroom.**

- Every layout has exactly **2 admitted Tensile solutions** in the gfx1013
  FP16 library (contrast N32's BF16 case: 14/14/8). Best-vs-default paired
  ratios are 0.96–1.05× across all 40 layouts — within run-to-run noise
  (an apparent 1.20× qkv win in the first partial run did not reproduce:
  0.96× in the full paired protocol). **No winner ≥1.05× with meaningful
  call count; the candidate preset adds 0 rows.**
- All candidates pass the unchanged N32 parity contract (CPU-double
  rtol 0.015 / atol 0.05 / rel-L2 < 0.004, canaries, input preservation).
- **torch.mm ≈ private-handle default on the same tensors** (gap 0.81–1.16,
  mostly ~1.0): no dispatch-level penalty reproducible in isolation.
- **hipBLASLt is slower**: AB/BA fresh-process ratios 0.28–0.60× on the top
  three shapes (rocBLAS 0.21–0.45 ms vs hipBLASLt 0.68–0.82 ms). rocBLAS
  remains the right backend; `TORCH_BLAS_PREFER_HIPBLASLT` is not a lever.
- **Fixture-vs-model gap (the decisive open question):** the fixtures
  project ~12–13 s of GEMM device time per update (count × paired median),
  while D70's in-model kernel census attributed ~32 s. Isolated GEMMs of the
  exact census layouts are fast; the in-model cost is not reproduced by
  them. A per-shape in-model profile (next section) resolves where the
  in-model time actually sits.

## Whole-step A/B (candidate preset)

Reference (launcher automatic preset): steady **36.196 s** (1.0008× D69
36.166 s), final loss 2.4716983. Candidate preset = release rows + **0
winners** (mechanism sanity check): steady **36.158 s** (**1.001×**),
final loss **bit-identical** (Δ = 0.0), finite gradients, no skipped steps,
peak 6.16 GiB. The override mechanism is validated end-to-end (mount, env,
identity gate) but buys nothing for FP16 — as the sweep predicted.

## Per-shape in-model profile (one microbatch of update 2, ×4 per update)

Hardened one-microbatch window (r7, after the r6 host incident). Stream
aggregation of `prof.events()`:

- **GEMM device time: 8.20 s/window → 32.8 s/update.** All-kernel device
  time: 9.07 s/window → **36.3 s/update ≈ the 36.1 s step (GPU ~100% busy).**
  GEMM is **~90%** of device time — **D70's GEMM-bound attribution is
  independently confirmed.**
- Top GEMM ops by in-model device time (per call):

| Logical shape | ms/call (in-model) | calls/update | s/update |
| --- | ---: | ---: | ---: |
| `[1024,6144]@[6144,2048]` (large projection) | **34.98** | 456 | ~15.9 |
| `[1024,2048]@[2048,6144]` | 8.76 | 624 | ~5.5 |
| `[653,2048]@[2048,32768]` (vocab/logits) | 30.27 | 56 | ~1.7 |
| `[1024,6144]@[6144,128]` | 3.28 | 456 | ~1.5 |
| `[1024,2048]@[2048,2048]` | 2.74 | 504 | ~1.4 |
| `[1024,2048]@[2048,128]` (LoRA A) | **0.25** | 1560 | ~0.4 |

- The dominant kernel is `Cijk_Ailk_Bjlk_HHS_MT64x1x64…GSU3` at **14.55
  ms/call** — exactly D70's dominant kernel.

**Two corrections to the earlier framing:**

1. **The cost is the large base-model projections, not the LoRA adapters.**
   The rank-128 LoRA A/B GEMMs are cheap (0.25 ms/call, ~0.4 s/update); the
   ~16 s/update dominant term is the base `[1024,6144]@[6144,2048]`
   projection. "GEMM efficiency for the LoRA shapes" really means the base
   projections the adapters attach to.
2. **Isolated hot-cache fixtures are not decision-grade here.** The same
   logical GEMM runs at 35 ms/call in-model but 8.8 ms in the private-handle
   fixture (4×); the tiny LoRA GEMM is *faster* in-model (0.25 vs 0.52 ms).
   The fixture-vs-model gap is data/state-dependent (cold activations,
   memory locality, and/or the port's backward `fp16_alt_impl` flag, which
   the fixtures and the override CSV do not exercise). The solution sweep
   correctly found no headroom *among the admitted solutions it could
   measure*; it cannot see the in-model slowdown mechanism.

## Findings and recommendation

**The easy GEMM win does not exist on this stack.** FP16 solution selection
is exhausted (2 admitted solutions/layout, all within noise; preset adds 0
rows; whole-step A/B bit-identical), and hipBLASLt is slower. D70's
GEMM-bound attribution is confirmed (~90% of device time), but the cost is
the large base projections running far slower in-model than in isolation —
a memory/state-dependent effect, not a bad-solution choice.

**Re-ranked recommendation (revises D70's "GEMM efficiency first"):**

1. **Do not pursue further FP16 solution-selection tuning** — D71 proved it
   has no headroom; the N32-style override mechanism is validated but empty
   for FP16.
2. **If the GEMM cost is to be attacked, it is a deeper R&D scope**, not a
   tuning win: first resolve the in-model-vs-fixture gap (test the backward
   `fp16_alt_impl` flag path and memory-locality effects on the large
   projections), then consider memory-traffic reduction (fusing the LoRA
   update into the base GEMM to avoid materializing intermediates). High
   effort, uncertain payoff.
3. **Higher-ROI bounded levers now:** native depthwise causal conv (D70:
   20× growth 1024→4096), graph-capture integration for the ~10% non-GEMM
   device time (D70: proven feasible, bitwise-exact), and the D68 headroom.
   These are smaller but tractable; the GEMM term is structural.

The honest bottom line: the LoRA step is dominated by base-model projection
GEMMs whose in-model efficiency is poor and not fixable by solution
selection; the tractable near-term gains are conv, graph capture, and the
non-GEMM remainder, while a GEMM breakthrough requires dedicated
microarchitectural investigation.

## Defects found and fixed while qualifying

1. **Census app-config not staged (r1):** the orchestrator only staged the
   step-case configs; the census guardian failed on the missing config.
   Fixed: stage the census config before its phase.
2. **SystemExit swallowed the census output (r2):** the app entry ends in
   `sys.exit()`; the census wrapper's post-run write was skipped (the
   guardian passed on the audit alone). Fixed: capture SystemExit (D69/D70
   wrapper pattern) and make resume require the census output file, not
   just a passed guardian.
3. **PosixPath in hipBLASLt worker args (r3):** JSON serialization failed
   after the 10-layout sweep completed. Fixed: str/Path conversion. The r3
   partial results also showed an apparent 1.20× qkv win that the full
   paired protocol (r4) did not reproduce (0.96×) — retained as evidence
   that single-round comparisons are not decision-grade.
4. **Launcher mount syntax (r4):** `--mount src:dst:ro` is not the
   launcher's contract; it takes `--mount-ro <path>` (same-path read-only).
   The candidate case failed pre-launch; fixed and re-run (r5).
5. **Profiler OOM (r5):** `key_averages(group_by_input_shape=True)` on
   ~300k shape-tagged events OOM-killed the 10 g container (exit 137) after
   the app had finished. Fixed: stream-aggregate `prof.events()` into a
   per-layout dict (O(distinct layouts) memory); re-run (r6).
6. **Audit schema mismatch (r5):** the summarizer assumed
   `finite_grads`/`skipped`; the app audit uses
   `finite_gradients`/`optimizer_skipped`. Fixed with a regression test
   before the final collect.
7. **Host thrash + guardian SIGHUP incident (r6):** the streaming-aggregation
   profile still drove the host into severe memory pressure (SSH banner
   timeouts; ping alive). The orchestrator's 620 s session timeout closed the
   SSH connection, SIGHUP-killing the guardian mid-cleanup; because the
   guardian writes the boot-guard ack only in its `finally`, the ack was never
   written and the boot-guard child died with it, so the designed emergency
   reboot did not fire. An orphaned profile container held ~10 GB. **Recovery:
   the user power-cycled .104** (boot `a3142efc…` → `78b56954…`, r2 kernel,
   release intact, 0 containers, GPU devices present). No release/deployment
   change had been made, so nothing to roll back; no user data touched.
   Hardening (r7): (a) profile exactly ONE microbatch of update 2 (¼ the
   events; per-update figures × grad_accum); (b) a `timeout 430` backstop wraps
   podman in bounded_launch so a container can never outlive a killed guardian
   (BusyBox integer-seconds verified); (c) orchestrator session timeout raised
   620 → 900 s. The resource envelope (4 CPU / 10 g / 12 g / 512 PIDs) is
   unchanged.

## Boundaries

- Single-board (.104) measurement + qualification; the candidate preset is
  a task-local file, NOT deployed — deployment under the EX lock with
  identity gates and recovery is a separate scope.
- bmm (strided-batched) layouts are censused but not swept (the override
  mechanism's batched coverage is unqualified); FP16 backward GEMMs carry
  the port's alt-impl flag — the whole-step A/B measures the net effect.
- No app source/entry/output/checkpoint change; no user data; no
  default/package/release change; installed modules unchanged.

## Verification

- Workstation: 22 new unit tests (layout conversion, census aggregation,
  preset generation, summarizer, guardian-reuse identity, hardened-launcher
  backstop, orchestrator configs); full bc250 suite **327 pass**, Black clean.
- On-board (final r7): all five guardians passed (census, bake-off,
  step-ref, step-cand, profile-shapes). Reference 36.196 s (1.0008× D69),
  candidate 36.158 s (1.001×, loss Δ=0). Per-shape profile: GEMM 32.8 s/update,
  all-kernel 36.3 s/update. Evidence `kernel/artifacts/gemm-efficiency-d71/`
  (per-file SHA-256 `artifacts.json`); host task directory
  `/home/nik/bc250work/gemm-efficiency-d71/` retained for replay.
- Board: .104 power-cycled during the batch after the r6 incident (boot
  `a3142efc…` → `78b56954…`, r2 kernel `7.1.5-bc250rocm-r2`, release
  `10.2-gfx1013-20261111-138a66f52` intact); final state 0 containers,
  54.4 °C, 66.6 G free, GPU devices present. .105 untouched throughout.
