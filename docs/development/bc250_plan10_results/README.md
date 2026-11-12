# PLAN-10 — measured training performance (D64 + D65)

Completed 2026-11-12. PLAN-10 replaces correctness-only evidence with a **bounded
steady-state training-performance measurement** of a fixed owned workload on the
native gfx1013 provider. Deliverables: the owned benchmark fixture
`build_tools/bc250/benchmark_training_d64.py` (D64) and a qualified GPU run on the
.104 release (D65). Evidence: `kernel/artifacts/plan10-d65/`.

This is a bounded measurement of **tiny owned random-init models**, not a
production-speed, model-quality, or general two-board speedup claim.

## Methodology

Reuses the D62 owned model registry/builders and the qualified timing protocol:
- **Fixed workload**: batch 2 × sequence 32 (64 tokens/step), AdamW **lr=0** holds
  weights fixed during timing (optimizer state still updates). Sequence is capped
  at the smallest admitted context (gpt2-tiny `n_positions=32`).
- **AMP pattern**: FP32 master weights + autocast compute in the target dtype;
  GradScaler for FP16. (Casting params to FP16 produces FP16 grads GradScaler
  refuses — see defects.)
- **Timing**: cold first step recorded separately; 3 warmup steps; then 5
  synchronized intervals of 3 steps each → per-step ms summarized by **median +
  IQR**; `tokens/s = tokens × 1000 / median_ms`.
- **Numerical checks**: finite first/timed loss, no AMP overflow skip
  (`scaler.get_scale()` not decreased), and loss stability (`assert_close` to the
  first loss, rtol/atol 1e-4, valid because lr=0).
- **Memory**: `torch.cuda.max_memory_allocated()` after reset.
- **Utilization**: `gpu_busy_percent` returns **EOPNOTSUPP** on this board, so
  direct GPU utilization is unavailable; the report records that boundary plus
  hwmon temperature (53→54 °C across the run) and the DPM governor (`auto`).

## Results — .104 release, 15/15 cases passed

torch 2.13.0a0+gfx1013, hip 7.17.0, transformers 5.2.0 / accelerate 1.12.0 /
peft 0.18.1, `gfx1013:xnack-`. Fixture sha256 `b4adb7f9…` bound into the report.
Step median ms / tokens·s⁻¹ (IQR% in the JSON):

| Model (params) | Backend | FP32 | BF16 | FP16 |
| --- | --- | --- | --- | --- |
| gpt2-tiny (21280) | eager | 7.48 / 8556 | 8.72 / 7341 | 8.98 / 7131 |
| gpt2-tiny | sdpa | 7.65 / 8363 | 9.02 / 7099 | 9.05 / 7071 |
| qwen3-tiny (312320) | sdpa | 11.23 / 5697 | 12.98 / 4931 | 13.27 / 4825 |
| qwen3-tiny | tiled | 11.04 / 5799 | 12.51 / 5117 | 12.85 / 4979 |
| qwen35-hybrid-tiny (109684) | sdpa | 35.74 / 1791 | 38.24 / 1674 | 39.04 / 1640 |

Peak allocated memory ~67–73 MB across cases. IQR 0.4–4.3% (stable). Board clean:
0 containers before/after; no source/provider/prefix/default change, no reboot.

## Bottlenecks and regressions (recorded before optimizing)

1. **Low precision is slower on these tiny models.** BF16/FP16 AMP steps are
   ~10–20% slower than FP32 (e.g. gpt2 eager 7.48 → 8.72/8.98 ms). The models are
   memory-bound and tiny, so autocast cast overhead dominates and there is no
   compute win. Low precision is not a speedup here.
2. **Native tiled ≈ math SDPA at this size.** qwen3 tiled is marginally faster
   than sdpa (FP32 11.04 vs 11.23 ms; BF16 12.51 vs 12.98 ms) — competitive, not a
   large win on tiny shapes.
3. **Hybrid gated-delta/conv reference path is the dominant cost.** qwen35-hybrid
   is ~3–5× slower than gpt2/qwen3 (35.7 ms vs 7.5/11.2 ms FP32) because the
   frozen 5.2.0 environment runs the reference (torch) gated-delta/conv path; the
   native op overlay (5.19.0-gated) is the acceleration route, not present here.
4. **Large first-step JIT cost for the hybrid.** qwen35-hybrid BF16/FP16 cold
   first step is ~16.7 s (Triton JIT/autotune) vs ~38 ms steady-state; FP32 first
   step ~0.43 s. This is an admission/warmup cost, not steady-state throughput.

## Defects found and fixed while qualifying the fixture

1. **Sequence overflow segfault**: the initial SEQ=64 exceeded gpt2-tiny
   `n_positions=32`, indexing the position embedding out of bounds and faulting
   the GPU (exit 139). Fixed to SEQ=32; added incremental per-case report writes
   so a fault preserves completed measurements.
2. **FP16 GradScaler refusal**: casting model params to FP16 produced FP16 grads
   that `scaler.unscale_` refuses. Fixed to the qualified AMP pattern (FP32 master
   weights + autocast compute).

## Verification

- 8 new local tests (`bc250_benchmark_d64_test.py`); full bc250 suite **198 passed
  / 84 subtests**; Black clean (ruff unavailable on the workstation).
- D65: 15/15 cases passed; fixture hash bound into the report; pre/post container
  counts 0; temperature stable; run via the ordinary `bc250-train` caller under
  the immutable release (no mutable runtime lock).

## Scope and remaining

PLAN-10 is complete for its bounded scope: a fixed owned workload measured with
warmup/repeated intervals, tokens/s, step time, memory, numerical checks, and an
honest utilization boundary; FP32/BF16/FP16 and eager/sdpa/tiled compared;
bottlenecks recorded before any optimization. **Single-board** scope; a distributed
(two-board) performance benchmark is a follow-up (L01 was correctness-first, not a
speedup claim). Optimizing the recorded bottlenecks (hybrid native-op overlay,
low-precision cast overhead) is separate future work, not part of this measurement.

With PLAN-10 done, **all ten original work packages are complete**. Publication of
the release still requires explicit user instruction.