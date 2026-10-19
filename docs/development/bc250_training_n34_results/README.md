# N3.4 square BF16 projection audit — 2026-10-19

Code `6531025efe4e1d61121df8a29dca0ff94885cd99` reuses the N3.3 guarded controller
with explicit batch/shape/baseline parameters and extra model gates. No native
build or runtime deployment. Installed N3.2 remains runtime code62ab2437c,
rocBLAS8389197c1, native Torchdc8fea0/source2b449c4, other native libraries319a235d6.
The candidate CSV in this repository is **experimental/rejected**, not active.

## Actual layouts and fixed-reference search

The N3.3 trace attributes7.105% of kernel sum to BF16
`[1024,2048] @ [2048,2048]`. Actual synthetic Qwen3.5-2B checkpointed forward/
backward audit finds two layouts, all storage offsets0:

| A stride | B stride | Forward / backward mm calls | rocBLAS transposes | Default / best GPU ms | Candidate / ratio |
| -------- | -------- | --------------------------- | ------------------ | --------------------- | ----------------- |
| [2048,1] | [1,2048] | 42 / 42                     | T,N                | 7.091556 / 5.282978   | -346 / 1.34234x   |
| [2048,1] | [2048,1] | 0 / 42                      | N,N                | 1.870641 / 1.741901   | -533 / 1.07391x   |

Column-major M2048/N1024/K2048, lda/ldb/ldc2048, batch1, BF16 input/output,
FP32 compute, alpha1/beta0, C=D. Each layout enumerates14 admitted existing
solutions, plus default. Full independent CPU-double references pass unchanged
atol0.05/rtol0.015 and relativeL2\<0.004,128-element output guards and immutable
inputs; worst observed relativeL2=0.001702264. Six alternating timing rounds,
three calls per sample, private handle on current Torch stream. Audit inputs
are owned synthetic-model copies; user data/checkpoints/config remain untouched.

The first TN sweep changes timing regime late: default samples3.721–7.292ms,
candidate3.007–5.303ms. Keep the raw paired data; its median ratio is not a
stable clock-controlled throughput claim. No governor/power policy change or
physical clock inference was made. NN samples are much steadier.

## Full-model candidate rejection

One candidate adds only TN solution-346 to the installed N3.2 CSV. The reference
worker **keeps N3.2's existing NT solution-490**, so this does not compare against
the old untuned stack. Both fresh workers use the same installed DSO, no preload,
actual local base weights, new nonzero rank16 adapters and synthetic1024-token
IDs, batch1/accumulation1/BF16+autocast/FP32 adapters/math SDPA/checkpointing.
Each worker checks all372 nonzero clipped gradients for finiteness and its own
first AdamW update/state against independent CPU FP32 replay, then times three
finite warm full steps.

- Reference loss14.294264793395996; candidate14.294599533081055,
  absolute delta0.00033473968505859375. This passes the existing whole-model
  forward-operation loss contract\<0.015. N3.2's backward-only change had the
  stronger exact-loss observation; this forward experiment does not inherit it.
- Worst adapter-gradient relativeL2 **0.0765128583** exceeds the unchanged
  **0.04** model contract. Candidate **rejected**, no tolerance relaxation.
- Each worker's own372 CPU AdamW parameter/state checks pass; max parameter
  abs error1.863e-9. This does not make the reference/candidate gradients equal.
  Worker CPU/GPU RNG hashes are equal.
- Warm medians5.625697s reference /5.581987s candidate, one pair only. Planned
  ABBA stops at the first failed gradient gate; remaining two workers do not run.
  This is not a qualified model speedup or a completed ABBA comparison.

The r2 GPU/host qualification correctly fails, acknowledges its recovery guard,
and retains both gradient files/worker JSON/logs. CPU GEMM correctness does not
bound whole-model gradient sensitivity to changed BF16 reduction order. No
production arithmetic defect or universal invalidity of solution-346 is claimed.

## Follow-up exact-output candidates

r3 compares all14 TN solutions with the original public/default GEMM output on
the captured fixture, retaining CPU references/guards/input preservation.
Four are bitwise equal for this fixture: -343/-338/-345/-350. Other solutions
change624–639 of2097152 output elements. Candidate-346 changes638 elements,
relativeL2 to default5.45847e-5. Exactness here is one input fixture, not all models.

After two seconds sustained alternating warmup per exact candidate, six paired
rounds/eight calls per timing give ratios:

| Exact candidate | Default / candidate GPU ms | Ratio    |
| --------------- | -------------------------- | -------- |
| -343            | 3.613279 / 5.222219        | 0.69190x |
| -338            | 3.644259 / 3.664736        | 0.99441x |
| -345            | 3.625594 / 3.647993        | 0.99386x |
| -350            | 3.650739 / 5.069840        | 0.72009x |

No exact-output candidate demonstrates an isolated gain here. **No square
projection override is deployed.** Native arithmetic sources, prefix metadata,
active preset, normal launchers and user application remain unchanged.

## Preservation, replay and next scope

Four GPU processes: layout/solution/equivalence pass; one model candidate fails
its numerical contract. All four guards ACK600s; dmesg identical across the
whole batch. Final309 protected identities/four FFT AOT DBs/links/runtime.json/
N3.2 CSV+DSO+gfx1013 assets/user trainer/config unchanged. No build, ccache reset,
source/build/release/image/checkpoint deletion, reboot, .105/BIOS/default BLAS/
math/SDMA0/heap-off/fallback change. CPU reference/model threads4. Same r2 boot
`d0532075-88fb-4a2f-974e-f61578d20d9f` / `7.1.5-bc250rocm-r2`.
No active agent job/container/guard; free about5.34GiB. Historical Python CPU
fault remains unresolved/separate from fixed VM-heap/CSR and repaired N3.2 loader.

WS = `/home/nik/bc250work/rocm-10.2-gfx1013`. Commands below require replacing
WS with that absolute path and fresh attempt directories:

```bash
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_training_n34_on_host.py --attempt r1
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_training_n34_on_host.py --attempt r2 --phase extra
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_equivalence_n34_on_host.py --attempt r3
```

The r2 failure is an expected reproducible rejection, not a passing release gate.
Reports, guards, commands and compressed logs/source snapshots are tracked here;
owned fixtures/gradient tensors remain at WS/logs/training-n34-\* and matching
local ignored `kernel/artifacts/training-n34/remote`, copied SHA verified.
Initial r1/r2 sources preserved; no source was changed during a running gate.
Next preservation baseline: WS/logs/training-n34-qualification-r3/final.json.
Helper backup: WS/recovery/training-n34-before; runtime recovery remains
WS/recovery/gemm-n32-before. Current N3.2 preset is still qualified/active.

**Next N3.5:** current-preset1024-token/rank16 accumulation16 gradient/update
qualification and owned checkpoint/fresh-process resume. Audit earlier42-token/
rank4 CLI resume and current N3.2/N3.3 accumulation1 evidence first; qualify the
new geometry rather than repeat completed small tests. Preserve user training
files/checkpoints. Large vocabulary projection tuning remains open and requires
a memory-bounded reference/capture design; it is not hardware-unavailable.
