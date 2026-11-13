# D68 — fused native solve/scan for the gfx1013 gated-delta chunk op

Completed 2026-11-13 on the .104 release (first bounded scope of the deeper
bottleneck follow-up after D66). The native `chunk_gated_delta_rule`'s remaining
ATen solve/scan cost is root-caused by section attribution, and an **opt-in
fused path** (`use_fused_solve_scan=True`) is qualified: a fused in-register
unit-triangular inverse plus a fused linear-recurrence chunk scan, both with
analytic backward, applied selectively from two chunks up (measured crossover).
**Op-level 1.13–1.81× where applied (up to 1.94× across runs); model-step
1.15–1.17× at seq 256; no regression at single-chunk geometries; default path
bitwise unchanged.** No default, package, release or installed-module change.

## Root cause (profile r1)

D66 closed the launch-bound frozen-loop gap, but A06 recorded only ~7% native
win at B1/H16/D128/L1024 because "the remaining ATen solve/scan dominates
complete-operator time". The owned profile fixture
(`profile_gated_delta_d68.py`, sha `c7746f6e…` at r1) attributes the native op
with an exact-replica forward CUDA-event breakdown (verified against the
installed op, rel-L2 ≤ 1e-6) plus isolated forward+backward microbenchmarks of
the two batched `solve_triangular` calls and the sequential Python chunk scan:

| Geometry (chunks) | Full fwd+bwd | Solves | Scan | Combined |
| --- | ---: | ---: | ---: | ---: |
| a06-long B1 H16 D128 L1024 (16) | 29.20 ms | 3.34 ms (11.4%) | **20.35 ms (69.7%)** | 81.1% |
| a06-short L65 (2) | 6.51 ms | 0.91 ms (14.0%) | 2.85 ms (43.7%) | 57.8% |
| d66-benchmark B2 H2 D16 L32 (1) | 4.84 ms | **2.33 ms (48.2%)** | 1.55 ms (32.0%) | 80.1% |
| d66-long256 L256 (4) | 8.23 ms | 0.68 ms (8.2%) | 5.72 ms (69.5%) | 77.7% |

(FP32; BF16/FP16 within ~1 ms.) The scan is launch-bound autograd through the
Python loop (~1.27 ms/chunk fwd+bwd for ~2 MFLOP of actual GEMM work); the
batched ATen solves dominate only at the tiny single-chunk geometry.

## Implementation (patched Torch source, additive opt-in)

`torch/nn/bc250_gated_delta.py` (candidate sha `06c5a8e1…`, installed module
remains `21398daf…`):

- `_tril_inverse` Triton kernel + `_FusedTrilInverse` autograd Function: one
  program per 64×64 matrix, 64-step in-register forward substitution producing
  `(I + tril(A,-1))^-1` (diagonal treated as 1, upper ignored — exactly
  `solve_triangular(unitriangular=True)`). Analytic backward
  `grad_A = -tril(M^T dM M^T, -1)`. The two solves become one inverse + two
  batched matmuls.
- `_chunk_scan_fwd`/`_chunk_scan_bwd` Triton kernels + `_FusedChunkScan`
  Function: the scan is reformulated as the column-separable linear recurrence
  `S_{i+1} = (d_i I - k_i^T w_i) S_i + k_i^T u_i`,
  `out_i = (q_i - attn_i w_i) S_i + attn_i u_i`. The M/N/P/R precompute stays in
  autograd-visible ATen; one kernel per (batch·head, value-block) owns a
  [Dk,BV] state block across all chunks (BV ≤ 64), saving every entering state.
  Backward runs the adjoint recurrence `G_i = M_i^T G_{i+1} + P_i^T dout_i` in
  reverse chunk order and emits dM/dN/dP/dR via batched matmuls against the
  saved states; initial-state gradient supported.
- Both kernels follow the port's qualified Triton convention (elementwise/
  reduction style, no `tl.dot`, `num_warps=4`, `enable_fp_fusion=False`,
  wave32), like A05/A06.
- `use_fused_solve_scan=False` keyword on `chunk_gated_delta_rule`; default
  branch keeps the exact operation order (bitwise-invariant, checked).
- **Selectivity (measured, pre-declared):** `FUSED_SCAN_MIN_CHUNKS = 2` — the
  fused path applies only when the sequence spans ≥ 2 chunks; single-chunk
  calls keep the default path. `FUSED_SCAN_STATE_BYTES_LIMIT = 256 MiB` saved-
  state cap with explicit refusal (backward adds transient buffers of similar
  size). FP16/BF16 keep reference-ordered ATen preparation unchanged.

### Why the threshold (decomposition r2/r3)

The fixture's decomposition section times four variants (default /
inverse-only / scan-only / both) with independent fusion switches:

| Geometry (C) | inverse-only | scan-only | both |
| --- | ---: | ---: | ---: |
| a06-long (16) FP32 | 1.05× | 1.56× | **1.81×** |
| a06-short (2) FP32 | 1.02× | 1.16× | 1.20× |
| d66-benchmark (1) FP32 | 0.93× | 0.90× | 0.87× |
| d66-long256 (4) FP32 | 0.98× | 1.56× | 1.55× |

(BF16/FP16 similar; r2/r3/r6 consistent within run variance; full tables in
the reports.) At C=1 **every** fusion
variant regresses — the reformulation adds more launches than the single loop
iteration costs. From C=2 up, the combined path wins at every measured
geometry. One pre-declared constant captures this without overfitting; the
~5% residual at C=4 (scan-only beats both there) is recorded as remaining
headroom for finer selectivity, not chased in this scope.

## Qualification (final r6; candidate `06c5a8e1…`, fixture `d30c9310…`)

**Extended A06 battery on the candidate: 72/72 passed** (`test_bc250_gated_delta.py`,
sha `7dd17cec…`): the full original contract plus fused variants of the
CPU-FP64 token-recurrence test (3 dtypes × 2 normalize × 3 lengths, with
initial state + state gradients, A06 tolerances unchanged), fused-vs-default
equivalence at 3 geometries incl. C=5/D128 (pre-declared rel-L2 bounds FP32
1e-5 / low 1e-2), fused low-precision reference-preparation check, and fused
refusals (non-bool flag, 256 MiB state cap).

**Op parity (fused vs default, output + final state + 6 gradients):**

| Geometry | FP32 | BF16 | FP16 | Bound |
| --- | ---: | ---: | ---: | --- |
| a06-long | 1.41e-6 | 8.27e-5 | 2.14e-5 | 1e-5 / 1e-2 |
| a06-short | 1.49e-6 | 5.94e-5 | 1.69e-5 | 1e-5 / 1e-2 |
| d66-benchmark | 0 (default path) | 0 | 0 | — |
| d66-long256 | 4.77e-7 | 9.21e-5 | 2.16e-5 | 1e-5 / 1e-2 |

**Independent CPU/FP64 recurrence anchor** (1,65,2,7,9, with initial state,
fixed upstream grads, both paths): worst tolerance fraction FP32 **0.003**,
BF16 **0.056**, FP16 **0.038** of the A06 tolerances — the fused path is a
valid rounding of the same math, not a structural gradient defect.

**Op timing (A06 protocol: 2 warmup, 6 alternating rounds, fwd + all input
grads + state), default → fused, final r6:**

| Geometry | FP32 | BF16 | FP16 |
| --- | --- | --- | --- |
| a06-long | 26.17 → 14.49 ms (**1.81×**) | 26.76 → 15.16 (1.76×) | 26.84 → 15.14 (1.77×) |
| a06-short | 6.07 → 5.18 ms (1.17×) | 6.83 → 6.02 (1.13×) | 6.93 → 6.01 (1.15×) |
| d66-benchmark | 4.57 → 4.30 ms (1.06×) | 5.00 → 5.03 (0.99×) | 4.99 → 5.00 (1.00×) |
| d66-long256 | 8.09 → 5.12 ms (1.58×) | 8.83 → 6.08 (1.45×) | 8.79 → 6.08 (1.45×) |

Run-to-run variance: r3 measured a06-long FP32 28.31 → 14.60 ms (1.94×);
the default path's IQR reaches ~11% on some runs (fused ≤ ~5%, usually <1%).
The benchmark geometry is at parity by design (single chunk → default path).

**Model level (owned qwen35-hybrid-tiny, D66 routing with the fused flag
injected, D64 protocol, final r6):**

Parity vs frozen (pre-declared D66 bounds):

| Seq | Dtype | Loss Δ | Logits rel-L2 | Worst grad rel-L2 | Bound (grad) |
| --- | --- | ---: | ---: | ---: | --- |
| 32 (C=1, default path) | FP32 | 0.0 | 1.49e-7 | 8.43e-4 | 2e-3 |
| 32 | BF16 | 3.91e-5 | 2.67e-3 | 1.62e-2 | 0.04 |
| 32 | FP16 | 1.00e-5 | 3.72e-4 | 1.36e-2 | 0.04 |
| 256 (C=4, fused active) | FP32 | 0.0 | 1.37e-7 | 1.52e-3 | 2e-3 |
| 256 | BF16 | 1.10e-5 | 2.51e-3 | 1.68e-2 | 0.04 |
| 256 | FP16 | 4.77e-7 | 3.46e-4 | 1.01e-2 | 0.04 |

seq256 FP32 independent CPU/FP64 gradient anchor: frozen max rel-L2
**1.45e-3**, fused **2.07e-4** (bound 5e-3) — the fused path is ~7× closer to
the FP64 reference than the frozen fallback; both are valid roundings.

Step timing (D64 protocol; `scan_calls` proves which path ran):

| Seq | Variant | FP32 | BF16 | FP16 |
| --- | --- | ---: | ---: | ---: |
| 32 | frozen | 37.67 ms | 40.03 ms | 40.30 ms |
| 32 | routed (D66) | 15.43 ms | 18.02 ms | 18.34 ms |
| 32 | routed_fused | 15.30 ms (1.01×) | 17.83 ms (1.01×) | 18.04 ms (1.02×) |
| 256 | frozen | 42.99 ms | 46.05 ms | 46.33 ms |
| 256 | routed (D66) | 19.05 ms | 22.55 ms | 22.82 ms |
| 256 | routed_fused | **16.55 ms (1.15×)** | **19.33 ms (1.17×)** | **19.68 ms (1.16×)** |

(routed_fused ratios vs routed; seq256 fused vs frozen = 2.60/2.38/2.35×.
seq32 fused keeps the default path: `scan_calls=0`, no regression; seq256
fused `scan_calls=19` per case, routed control `scan_calls=0`.)

**Refusals:** non-bool flag, state cap, and **candidate default path bitwise
equal to the installed module** (outputs, states, all gradients).

**Board cleanliness:** 0 containers / 0 GPU clients before/after every run;
54 °C throughout; boot `a3142efc…` unchanged; ordinary `bc250-train` caller
under the immutable release (SH runtime lock taken by the caller); fixtures
mounted, nothing installed; new Triton kernels JIT into the release's designed
rw user-cache (`/workspace/user-cache/triton`), runtime payload untouched.

## Defects found and fixed while qualifying

1. **Battery unpack bug (r1):** `test_fused_matches_default` unpacked a
   5-tuple into 4 names — 6 errors, retained in `runner-qualify-r1.log`;
   fixed and re-run (r2: 69/69, r3: 72/72 with the added C=5 geometry).
2. **Fixture CPU-anchor bug (r1):** the oracle detached its inputs, so the
   reference had no `grad_fn` — section failed before measuring; fixed (r2
   passes with the margins above). The A06 battery's own oracle was unaffected.
3. **Single-chunk regression (r1/r2):** the unconditional fused path was
   0.79–0.94× at C=1 (op and model level). Diagnosed by the decomposition
   section, fixed with the pre-declared `FUSED_SCAN_MIN_CHUNKS = 2`
   selectivity; r3 confirms 1.00–1.02× there.
4. **Model-timing variant leak (r3/r4, root-caused r5):** the seq256 "routed"
   control ran the fused path (`scan_calls=19`, byte-identical peaks). The
   inject/restore trace showed all three dtype installs of a `routed_fused`
   variant happening **before** the per-variant `finally` uninstalled them:
   install #2 captured the already-patched `routing._native_call` as its
   "original", so the last uninstall "restored" `fused_native` — a nested-
   capture leak of the global injection. Fixed by uninstalling each dtype's
   record immediately after its case plus a variant-level save/restore;
   parity gained the same per-case `finally` and seq256 cases. diag1/diag2
   (retained) proved the fused path itself delivers 1.16× at seq256 on one
   instance; the Torch module was never at fault.

## Scope and boundaries

- Bounded owned measurement + qualification on .104; **not** a production,
  model-quality, convergence or general-model speed guarantee.
- Eager first-order only; no compile/graph, no higher-order gradients; the
  recurrent/decode path and trainable-weight conv remain unrouted (D66).
- The fused path is **opt-in and selective**; the installed release module is
  unchanged (`21398daf…`), the default path is bitwise invariant, and no
  consumer picks up the fused path without explicitly passing the flag.
- Remaining headroom (separate scopes): finer per-geometry selectivity (~5%
  at C=4), trainable-weight native conv, low-precision cast overhead,
  larger-geometry admission, and the D67 long-sequence LoRA reference path
  (which runs transformers 5.22.1, outside the D66 route's 5.2.0 gate).

## Verification

- Workstation: 8 numpy-float64 math tests (forward equivalence vs the token
  recurrence at 2e-12 across 4 geometries; finite-difference checks of the
  inverse and adjoint-scan backward formulas) + 12 fixture-logic tests; full
  bc250 suite **269 passed**; Black clean.
- On-board: extended A06 battery 72/72 on the candidate (r2/r3/r6);
  qualification fixture all sections passed (r2 pre-threshold, r3 threshold,
  r6 final with leak-fixed variant isolation); evidence
  `kernel/artifacts/gated-delta-d68/` (profile r1, qualify r1–r6, battery rows
  r2/r3/r6, runner logs incl. r4/r5 leak diagnosis, diag1/diag2,
  `artifacts.json` per-file SHA-256). Host task directory
  `/home/nik/bc250work/gated-delta-d68/` retained for replay.
- Publication: bc250-torch fork + regenerated 23-patch complete series +
  umbrella CI manifest (see flow.md cursor for revisions and CI run).
