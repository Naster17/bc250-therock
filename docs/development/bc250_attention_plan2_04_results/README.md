# PLAN2-04 — attention speed qualification (opt-in gfx1013 tiled SDPA)

**PASSED 2026-11-14** on .104 (release `10.2-gfx1013-20261111-138a66f52`,
Torch `2.13.0a0+gfx1013`, boot `0d46a6f1` unchanged). Fourth PLAN2 item.
Scope (user-confirmed): **qualify + document the EXISTING opt-in tiled SDPA**
(`torch.nn.attention._gfx1013`, sha256 `d61ac116d82e…`, installed == source).
No Torch source change, no default change, pytorch series stays 23 patches.

## Verdict

- **63/63 qualification checks** (r2 47/47 part 1, r3 16/16 part 2): parity vs
  GPU math SDPA and an independent CPU-FP64 anchor, 13 refusals, checkpoint /
  Transformers / autocast / empty-dim integration.
- **168/168 benchmark checks** (r6 130/130 part 1, r7 38/38 part 2): 41 paired
  math-vs-tiled fixtures, 0 unstable, drift gate 1.25 everywhere.
- **Causal attention: tiled is a real forward speed win** — 1.5–2.9× faster
  forward (tile skipping), fwd+bwd up to 2.0× at D64; D128 GQA fwd+bwd mixed
  (0.67–1.4×). **Full (non-causal) attention: tiled is slower** (fwd+bwd
  0.19–0.69×) but uses **12–86× less peak memory** — the long-seq capacity
  unlock. Backward (recomputed, scalar) is the weak point everywhere.
- Boundaries resolved as documented refusals: dropout / compile / mask-grad /
  higher-order / D>256 / seq>4096. Informational probes: CUDA-graph capture of
  the tiled forward **works and replays bitwise-exact** on a small fixture
  (graph qualification remains PLAN2-05 scope); math `no_grad`+GQA works.

## What existed and was reused

- Tiled SDPA: three Triton scalar kernels (online softmax, FP32 accumulation,
  recomputed backward, GQA without atomics), from batches bda0831/43cb299/
  4ee6571/71a0426 (2026-10-15…18). Opt-in only; math stays default. BF16 D>128
  training uses the documented hybrid (math FP32 forward + tiled backward).
- Evidence gap closed: A07's "BF16 slower" timings were all D=256 **hybrid**;
  pure-tiled FP16/BF16 D≤128 had never been paired against math at long seq.
- Guardian envelope reused byte-identical from PLAN2-01/02/03 (`guard_nf4.py`,
  `bounded_launch.py`, `kernel_noise.py`); orchestrator pattern from
  `run_nf4.py`/`run_sparse.py`. Transformers for the integration check comes
  from the owned lora-smoke env site-packages (soak pattern; no torch/triton
  there, release torch stays authoritative).

## Qualification (pre-declared bounds, tmp BOUNDS.md, never relaxed)

Parity rel-L2 vs CPU-FP64 (output + dQ/dK/dV), worst over 39 cases
(8 small/mid + 5 long geometries × FP32/FP16/BF16; masks incl. fully-masked
rows, cross-attention M≠N, GQA, head256):

| Dtype | Worst rel-L2 | Bound |
| --- | ---: | ---: |
| FP32 | 7.6e-07 | 3e-5 |
| FP16 | 4.3e-04 | 5e-3 |
| BF16 | 3.4e-03 | 3e-2 |

Tiled-vs-math `assert_close` (rtol/atol 5e-4/3e-5, 8e-3/3e-3, 5e-2/3e-2) passed
on every case. 13/13 refusals raise the documented exception (dropout,
mask-grad, causal+mask, GQA-without-flag, nonfinite scale, D=257, seq=4097,
FP64, dtype mismatch, CPU, wrong rank, higher-order, compile-fullgraph →
dynamo `Unsupported`). Integration: non-reentrant checkpoint parity FP16/BF16;
tiny Llama `bc250_tiled` + checkpointing FP32 worst grad rel-L2 7.2e-07
(rtol 2e-3), BF16 7.6e-03 (rtol 5e-2), losses match CPU-double math;
autocast FP16/BF16 dtype + grad flow + parity; empty M/N → zero out/grads.

## Performance (paired A1/B1/B2/A2, 2 warmups + 5 samples, median, HIP events)

Speedup = math ms / tiled ms (>1 = tiled faster). Selected fixtures, full grid
in `bench-attn-r6.json` / `bench-attn-r7.json`:

| Fixture (B=1) | fwd | fwd+bwd | bwd | peak mem tiled/math |
| --- | ---: | ---: | ---: | ---: |
| D64 H16 causal L1024 FP16 | 2.94 | 1.96 | 1.48 | 0.045 |
| D64 H16 causal L4096 FP16 | 2.23 | 1.35 | 1.00 | 0.012 |
| D64 H16 causal L4096 BF16 | 1.96 | 1.18 | 0.87 | 0.012 |
| D64 H16 full L4096 FP16 | 0.59 | 0.44 | 0.37 | 0.012 |
| D128 H16:4 causal L1024 FP16 | 2.56 | 1.42 | 0.98 | 0.064 |
| D128 H16:4 causal L4096 FP16 | 1.72 | 0.80 | 0.52 | 0.017 |
| D128 H16:4 full L4096 FP16 | 0.63 | 0.20 | 0.13 | 0.017 |
| D256 H8:2 causal L1024 FP16 (pure) | 1.55 | 0.88 | 0.71 | 0.116 |
| D256 H8:2 causal L1024 BF16 (hybrid) | 1.54 | 0.71 | 0.72 | 0.718 |
| D128 H16:4 causal L4096 FP32 | 0.71 | 0.37 | 0.26 | 0.019 |

Absolute example: D64 H16 L4096 fwd+bwd extra peak **4152 MiB math vs 48.2 MiB
tiled** (~86×); D128 GQA L4096 4208 vs 72.2 MiB (~58×). Causal forward example:
D64 L4096 48.85 ms math vs 21.88 ms tiled.

Reading: math materializes the quadratic score/probability tensors and (for
causal) computes the full square; tiled keeps O(L·D) state and skips masked
causal tiles — hence the causal forward win and the dramatic memory win.
Tiled's recomputed scalar backward loses to the GEMM-based math backward,
especially full-attention D128 at long L. The BF16 D256 hybrid keeps the math
forward's quadratic temporary, so its memory ratio is only ~0.66–0.80.

**Positioning:** opt-in tiled for (a) long-seq CAUSAL training/inference where
forward speed + memory matter, (b) long-seq FULL attention that would OOM with
math (capacity, not speed). Not a default switch; LoRA-step payoff remains
demoted per D70 (eager core 0.3–0.5% of the step).

## Attempt journey (all retained; harness fixes, not stack defects)

- r1 39/40: release container lacks `transformers` (fixed: owned lora-smoke
  env PYTHONPATH, soak pattern) + higher-order refusal tested at the wrong
  level (`once_differentiable` detaches grads on the create_graph call; the
  RuntimeError surfaces when differentiating them again — test corrected).
- r2 47/47, r3 16/16: qualification green.
- r4/r5: crashed at the first GQA fixture. r5 misdiagnosed a no_grad+GQA
  math-backend boundary; instrumented probes d1–d3 proved the true root cause:
  **the timing helpers called `fn(*xs)` without forwarding `opts`**, dropping
  `is_causal`/`enable_gqa` from every timed call (GQA head-count crash; D64
  "causal" timings actually non-causal on both backends). r4/r5 numbers are
  defect evidence only, used for no claim. Regression test added.
- r6 130/130, r7 38/38: all fixtures re-measured with opts forwarded; no_grad
  forward restored for all fixtures (probe confirms no_grad+GQA works).
- BOUNDS.md addendum records the correction; no numerical bound, drift gate,
  refusal expectation or envelope parameter was ever relaxed.

## Envelope and safety

Every run under the qualified guardian: SH runtime lock, 600 s nonce-acked
boot-guard, 460 s child, `timeout 430` podman backstop, kernel_noise filter,
registry identity, foreign-container refusal, stock-clock gate (pp_dpm_sclk
ceiling 2000 MHz), 900 s orchestrator session. Boot `0d46a6f1` unchanged across
all runs; kernel logs identical (0 benign added); registry unchanged; 0
foreign containers; boards idle pre/post; .105 untouched; host governor/clock/
watchdog/kernel NOT modified. No default/package/release/source change; nothing
installed into any image. Owned synthetic fixtures only; no user application.

## Bounds and remaining scope

D≤256, seq≤4096, FP32/FP16/BF16, eager first-order only; dropout/mask-grad/
higher-order/compile refused (documented). CUDA-graph capture probe is
informational — graph integration/qualification is PLAN2-05. Tile-selection
tuning was out of scope (user chose qualify+document); the causal T=32
heuristic is the existing one. No model-quality/perplexity claim; isolated
attention numbers are not whole-model savings/speed. Larger head dims,
seq>4096, dropout and compile remain separate admissions.

## Usage (opt-in, unchanged)

```python
from torch.nn.attention._gfx1013 import (
    scaled_dot_product_attention as tiled_sdpa,
    register_transformers_attention,
)
out = tiled_sdpa(q, k, v, is_causal=True, enable_gqa=True)  # [B,H,L,D]
# Transformers: register_transformers_attention(); attn_implementation="bc250_tiled"
```

## Reproduction

```bash
python3 build_tools/bc250/attention_plan2_04/run_attention.py --label attn-r2 --script qualify --part 1
python3 build_tools/bc250/attention_plan2_04/run_attention.py --label attn-r3 --script qualify --part 2
python3 build_tools/bc250/attention_plan2_04/run_attention.py --label attn-r6 --script bench --part 1
python3 build_tools/bc250/attention_plan2_04/run_attention.py --label attn-r7 --script bench --part 2
```

23 harness tests (`bc250_attention_plan2_04_test.py`), suite 451, Black clean.

## Evidence

`kernel/artifacts/attention-plan2-04/`: BOUNDS.md (+addendum), qualification
r1–r3, bench r4–r7, guardians + preflights + kernel logs for every run,
orchestrator logs, diagnostic reports d1–d3. Host task dir
`/home/nik/bc250work/attention-plan2-04` retained (tools/configs/outputs/
evidence). Publication: bc250-therock tools+report (no source change), umbrella
curated PLAN2-04.