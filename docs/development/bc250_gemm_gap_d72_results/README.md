# D72 — GEMM in-model-vs-fixture gap diagnostic

Completed 2026-11-13 on the .104 release (D67-pattern guardians, D71-hardened
envelope). First bounded scope of the evidence-ranked GEMM-gap question left
open by D71. Measurement only — no model training, no full-model profile
(avoids the D71 r6 host-thrash vector), no source/default/release/preset
change, no user data.

**Headline: none of the three proposed cheap explanations reproduces the 4×
gap.** D71 found the dominant base projection runs ~35 ms/call in-model but
~8.8 ms in an isolated hot-cache fixture, and listed three candidate causes
(the backward `fp16_alt_impl` flag, thermal/clock throttling, memory locality)
without testing any of them. D72 tests all three with isolated microbenchmarks
and **rules every one out**: the same GEMM runs ~7.8–9.4 ms under *every*
isolated condition (forward flag, backward flag, 36 s sustained load, cold
pooled inputs, freshly written inputs). The gap is therefore **not a per-GEMM
property** — it is a systemic in-model execution effect that isolated fixtures
cannot reproduce. Per-GEMM tuning (flag / solution / locality) cannot fix it.

## Why this scope

D71's private-handle sweep always passed `flags=0` (the forward path). The port
sets `rocblas_gemm_flags_fp16_alt_impl` (0x4) on **backward** fp16 GEMMs
(`aten/src/ATen/cuda/CUDABlas.cpp`, gated by `ROCmBackwardPassGuard`), which the
fixture never exercised — so the backward path was an untested, plausible
cause. The N4 telemetry audit had also left the physical attained clock
uncalibrated, leaving throttling open. D72 closes both questions cheaply,
before any commitment to conv kernel development or deeper GEMM R&D.

## Method (owned fixtures, reuse of the qualified N32/D71 pattern)

New tool `probe_gap_d72.py` (owned, `bc250-therock`), three modes, all reusing
the D71 private-handle `Gemm` (`bench_gemm_d71`) and the unchanged N32 parity
contract (CPU-double rtol 0.015 / atol 0.05 / rel-L2 < 0.004, canary guards,
input preservation). The dominant layout is taken from D71's per-shape
in-model profile (decision-grade device time), not a geometry proxy — the two
large projections tie on K×N (2048×6144 == 6144×2048). Inputs: the archived
D71 census + profile evidence (owned, no user data).

- **alt-impl**: for the top 8 census mm layouts, enumerate the admitted Tensile
  solutions and parity-check + paired-time (6 alternating rounds) the default
  solution under `flags=0` (forward) vs `flags=0x4` (backward `fp16_alt_impl`).
- **sustained**: the dominant GEMM (forward path) in a 36 s loop of bounded,
  synchronized ~2 s windows (calibrated call count), recording per-window
  ms/call plus best-effort read-only hwmon clock/temp. Throughput-over-time is
  the primary throttle signal; the clock reads corroborate.
- **locality**: the dominant GEMM with hot reused tensors vs a cold pool of 8
  input pairs (cycled) vs freshly written inputs (`copy_` immediately before
  the call), paired timing.

Pre-declared attribution thresholds (before any run): a ratio ≥ 1.5 is
"meaningful" (above D71's 0.96–1.05 noise band), ≥ 2.5 "dominant"; the primary
cause is the largest ratio ≥ 1.5, else `not_reproduced_by_isolated_fixtures`.

## Results

Dominant layout: `[1024,6144]@[6144,2048]` (rocBLAS T/N, M2048 N1024 K6144),
D71 in-model **34.98 ms/call**.

| Hypothesis | Measurement | Ratio | Verdict |
| --- | --- | ---: | --- |
| Backward `fp16_alt_impl` flag | fwd 9.217 ms vs bwd 9.446 ms (default); identical solution set `[-318,-319]` for both flags; best bwd/fwd 0.871× | **1.025×** | ruled out |
| Thermal / clock throttling | 21 windows × 221 calls over 36 s; head 7.825 ms, tail 7.841 ms (min 7.783 / max 7.873); clock flat **2000 MHz** throughout; temp ≤ 77 °C | **1.002×** | ruled out |
| Memory locality (cold pool) | hot 7.828 ms vs cold 7.826 ms | **0.9997×** | ruled out |
| Memory locality (fresh write) | hot 7.828 ms vs fresh 8.017 ms | **1.024×** | ruled out |

All 8 swept mm layouts show backward/forward ratios 0.987–1.025× (noise). The
fixture runs the dominant GEMM at ~7.8–9.4 ms in every condition; the in-model
cost is 35 ms — a **3.7–4.5× gap that survives every isolated test**.

**Attribution: `not_reproduced_by_isolated_fixtures` (primary cause: none).**

## Interpretation

The gap is a property of the **full-model execution context**, not of the GEMM
kernel, its solution, its flag, the clock, or input cache state. The remaining
candidate explanations (all deeper R&D, none tested here) are systemic:

- **Working-set / TLB pressure**: the fixture touches one ~37 MB tensor pair;
  in-model the GEMM runs amid the whole 2 B-parameter model + activations +
  optimizer state, so physical-page locality / TLB behavior differs even though
  the logical tensors are "hot".
- **Kernel concurrency / queueing**: in-model the GEMM is interleaved with
  ~157 k other launches per update; the profiler's device-time attribution may
  include queue/wait time that an isolated synchronized fixture excludes.
- **Profiler attribution**: D71's 35 ms is profiler self-device time; whether it
  equals pure kernel execution in the presence of concurrent work is unverified.

None of these is addressable by per-GEMM tuning. This **reinforces D71's bottom
line**: the GEMM term is structural; the tractable near-term levers remain
native conv, graph capture, and the D68/non-GEMM remainder. A GEMM breakthrough
requires dedicated microarchitectural investigation of the systemic effect
above — and D72 narrows that investigation by eliminating the cheap causes.

## Defect found and fixed while qualifying

**Sustained single-window bug (r1):** the first sustained loop queued async
calls for 2 s of *wall* time without bounding queue depth; the first window
built a large backlog whose single `synchronize()` consumed the whole 36 s
budget, yielding **one window** and a trivial `degradation_ratio=1.0` (head ==
tail). Not a throttle measurement. Fixed (r2): calibrate a bounded call count
per ~2 s window (`window_call_count`, clamped 50–2000) and synchronize each
window, giving 21 real windows. Added a regression test; the r1 result is
retained as labelled evidence (`sustained-r1-windowbug.json`,
`summary-r1-windowbug.json`). The alt-impl and locality modes were unaffected
(they use bounded 3-call synchronized measurements).

## Boundaries

- Single-board (.104) measurement; isolated GEMM microbenchmarks only — no
  model training, no full-model profile, no preset deployment.
- The systemic in-model explanations (working-set/TLB, concurrency, profiler
  attribution) are identified but NOT tested — they are a separate, deeper
  scope, admitted only on a concrete need for long-sequence throughput.
- No app source/entry/output/checkpoint change; no user data; no
  default/package/release change; installed modules unchanged. The
  `fp16_alt_impl` flag is measured, not modified.

## Verification

- Workstation: 23 new unit tests (layout selection, profile-based dominant,
  flag constant, bounded window sizing, attribution bands, guardian-reuse
  identity, hardened backstop, orchestrator configs); full bc250 suite
  **349 pass**, Black clean.
- On-board: all three guardians passed (alt-impl, sustained r2, locality).
  Evidence `kernel/artifacts/gemm-gap-d72/` (per-file SHA-256 `artifacts.json`,
  plus labelled r1 window-bug evidence); host task directory
  `/home/nik/bc250work/gemm-gap-d72/` retained for replay.
- Board: boot `78b56954…` unchanged across all phases; kernel log + distributed
  registry unchanged; 0 foreign containers; all three boot-guards
  **acknowledged** (nonce-matched acks); final state 0 containers, 58.85 °C,
  GPU devices present. .105 untouched throughout.
