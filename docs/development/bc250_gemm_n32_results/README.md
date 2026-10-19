# N3.2 exact-layout BF16 GEMM selection — 2026-10-19

TheRock implementation `62ab2437c9fd80e011b23578ffbf9208d85cda64`;
rocBLAS/Tensile source `8389197c134a7337932021ab7137987a585664ae`
(patch0033). Installed only the rebuilt rocBLAS DSO, launcher, exact-problem
preset and prefix metadata. Native Torch remains `dc8fea0`, source `2b449c4`;
other native ROCm libraries retain `319a235d6`. This does not represent a full
ROCm rebuild at the new source revision.

## Actual layout and candidate selection

The [N3.1 profile](../bc250_training_n31_results/README.md) identified logical
BF16 `[1024,6144] @ [6144,2048]`. A dispatch audit plus rocBLAS bench logging
found three distinct layouts, all storage offsets0. Counts are mathematical mm
calls, including checkpoint recomputation, rather than kernel dispatch counts.

| A stride | B stride | Forward / backward calls | rocBLAS transpose / lda / ldb | Paired default / best candidate GPU ms |
| -------- | -------- | ------------------------ | ----------------------------- | -------------------------------------- |
| [6144,1] | [1,6144] | 24 / 24                  | T,N / 6144 / 6144             | 9.680 / 9.616                          |
| [6144,1] | [2048,1] | 0 / 48                   | N,N / 2048 / 6144             | 5.689 / 5.183                          |
| [1,1024] | [2048,1] | 0 / 18                   | N,T / 2048 / 1024             | 173.797 / 4.841                        |

rocBLAS column-major M2048/N1024/K6144, ldc2048, batch1, BF16 input/output,
FP32 compute, alpha1/beta0 and C=D. Enumerated existing admitted solutions
(14/14/8), capped at32; no new arithmetic kernel, emulated matrix instructions,
backend replacement or unmeasured general autotuner. Every candidate passes
full independent CPU-double matmul, fixed atol0.05/rtol0.015 and relative L2
\<0.004, output guards and input preservation. Observed relative L2 about
0.001657–0.001680. Coarse search followed by six alternating rounds, three calls
per timing. Only the third layout is selected: solution **-490** (raw Tensile479).
Its 35.9x isolated-fixture ratio does not describe whole-model throughput.
Other two layouts retain their original selection despite small candidate wins.

## Reproduced cold-loader defect and repair

Two fresh public Torch override attempts with the old DSO failed with an internal
HIPBLAS error, while the same solution worked after enumeration. Tensile debug
logs show solution479 admitted. Eager preloading populated master metadata
without its architecture-specific `codeObjectFilename`; override lookup skipped
normal lazy resolution when the solution index already existed. A diagnostic
that first hydrated metadata passed. This is a loader defect, separate from the
unresolved historical Python CPU segfault and the previously fixed VM-heap/CSR
corruption.

`MasterSolutionLibrary::setOverridesFromFile` now resolves through the lazy
library when a solution is missing **or its code-object filename is empty**.
The `ThreadSafeValue` filename is inspected through `.load()`. An isolated
rebuilt DSO passes the original fresh-process public mm regression without an
enumeration workaround. Installed cold tests repeat that proof and retain both
other layouts, CPU references and guards.

Targeted cached `rocblas` build only: ccache20GiB, OMP4, up to16 jobs,
native gfx1013, builder `fa3e8c254b6a791b541d49d4b0a1000a5d18aaa661b41e89794b82c104770238`,
no mapped GPU/network. Active dist prefix mounted read-only; no global dist,
Torch rebuild, kernel regeneration or cache reset. Candidate DSO SHA256:
`85599e274e3ba5077587f09eed0ba1df611ec19b2f2b595cc3a3724752d2f516`.

Retained failures: first native build called `.empty()` directly on
ThreadSafeValue; corrected r2 uses `.load().empty()`. One r2 controller attempt
failed before launch from Path/string operator precedence; corrected controller
r2b runs the successful native r2 build. A test-only CPU Parameter scalar warning
in model evidence was removed with `.detach()` before the installed rerun.
Original sources and every failed log remain preserved; tolerances unchanged.

## Full-model correctness and bounded timing

Actual local Qwen3.5-2B base weights, owned synthetic nonzero rank16 adapters and
synthetic IDs, batch1/1024 tokens/accumulation1, BF16 base/autocast, FP32 adapters,
math SDPA, activation checkpointing, existing read-only application hooks.
No user training dataset/checkpoint/config edit. Four fresh workers execute
reference/candidate/candidate/reference (AB/BA), all with the same repaired DSO;
only the preset differs. Each worker runs one correctness/update step and three
finite warm complete steps.

| Worker                        | Median warm whole-step wall |
| ----------------------------- | --------------------------- |
| Reference0                    | 8.602978s                   |
| Candidate1                    | 5.653776s                   |
| Candidate2                    | 5.493877s                   |
| Reference3                    | 8.551587s                   |
| Installed automatic candidate | 5.506898s                   |

AB/BA ratios **1.5216x / 1.5566x**. Installed comparison to reference0:1.5622x.
First loss is exactly14.294264793395996 and CPU/GPU RNG hashes match. All372
nonzero clipped adapter gradients checked; worst relative L2 difference
**0.02030649**, below the unchanged **0.04** contract. Reduction order differs,
so gradients are not bitwise equal. Reference repetition is exact. Independently,
each worker's own gradients replay through CPU FP32 AdamW: all372 updated
parameters and step/first/second moments pass, max parameter abs error1.863e-9.
This is not an independent whole-model CPU reference or proof of convergence.

No claim for accumulation16, real labels, DataLoader/evaluation/checkpoint cost,
other shapes/models/dtypes, long-run thermal behavior, sustained throughput or
bitwise continuation when resuming across a tuning setting change.

## Deployment, automatic use and recovery

Exclusive `helpers/update_dev_prefix.py` lock encloses atomic replacement,
five guarded fresh installed gates and preservation checks. New launcher
uses the preset only for the current mutable prefix, after verifying CSV +
DSO + all110 gfx1013 assets. Identity mismatch disables automatic tuning;
manual override wins. `BC250_GEMM_TUNING=0` on the outer command restores the old
selection. Frozen previous prefixes have no preset metadata. See
[usage](../bc250_torch_usage.md#exact-layout-bf16-gemm-tuning-n32-2026-10-19).

Installed gates: cold GEMM3, core7, MIOpen convolution30, model4, opt-out2.
Opt-out public NT fixture returns the old path (~176.15ms) and passes CPU
reference. SDMA0, heap-off, math SDPA and default rocBLAS backend remain.
No reboot: r2 `7.1.5-bc250rocm-r2`, boot
`d0532075-88fb-4a2f-974e-f61578d20d9f`; r3/r2 boot fallbacks, .105 and BIOS untouched.
All GPU attempts acknowledge their600s guards; dmesg unchanged for the entire
batch. Prior193 identities change only active rocBLAS aliases/runtime.json;
frozen previous and user files/four FFT AOT databases unchanged. Final309
protected paths seal installed assets and provenance. Free about6.13GiB.
No running agent job/container/guard at handoff.

WS = `/home/nik/bc250work/rocm-10.2-gfx1013`. Recovery:
`WS/recovery/gemm-n32-before` retains old installed/build DSOs, source/derived
headers, runtime.json, global and helper launchers. The existing gfx1013 kernel
pack is unchanged and hash-pinned; do not pair recovery with a different pack.
Deployment helper rolls back DSO/runtime/launchers on failure and preserves a
rejected preset. For a manual rollback, first take the exclusive lock, verify
recovery/current identities and restore those same files atomically; remove
preset metadata before using the old loader. Preserve original evidence and
frozen previous release. No rollback was needed.

Replayable build/deployment commands:

```bash
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/build_rocblas_gemm_n32_on_host.py --attempt r2
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/deploy_gemm_n32_on_host.py
```

Replace WS with the absolute path above. Helpers intentionally require fresh
log/candidate directories; preserve prior attempts instead of overwriting them.
Exact GPU commands and guards appear in gate-summary.json / command.json.
Raw logs/fixtures/gradients/candidate/recovery remain at WS/logs/gemm-n32-\*,
WS/candidates/gemm-n32-r2, WS/recovery/gemm-n32-before and matching local ignored
`kernel/artifacts/gemm-n32/remote`. Compact tracked reports retain failed and
passed attempts; full large tensors/DSOs stay outside Git.

**Next N3.3:** reprofile the optimized current model with exact profiled versus
unprofiled loss/gradient/update/RNG replay; select the next measured dense-mm
layout bottleneck. Do not assume optimizer or gated-delta now dominates, and do
not repeat tiny eig/optional scientific sweeps without a workload need.
