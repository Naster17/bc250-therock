# N3.3 optimized-model attribution and remaining GEMM audit — 2026-10-19

TheRock code `7838d94bc` reuses the qualified N3.1 collector/analyzer with an
explicit installed-preset assertion and N3.2 preservation baseline. Generic audit
helpers now accept the logical GEMM shape and fixture-log label. No native build,
Torch/library install, kernel change, new preset or application edit. Installed
runtime code remains `62ab2437c`, rocBLAS/Tensile `8389197c1`, native Torch
`dc8fea0` / source `2b449c4`, other native ROCm libraries `319a235d6`.

## Exact replay on the optimized stack

Actual local Qwen3.5-2B base, synthetic nonzero rank16 adapters and input IDs,
1024 tokens/batch1/accumulation1/BF16 base+autocast/FP32 adapters/math SDPA/
activation checkpointing. Existing application hooks read-only; user dataset,
checkpoints, trainer and configuration unchanged. Automatic identity-pinned
N3.2 CSV mounted, no LD_PRELOAD or test-only DSO.

Profiled/unprofiled loss, all372 nonzero clipped adapter gradients, adapter
update, full AdamW state and CPU/GPU RNG match **exactly** within the current
stack. This does not replace N3.2's separately documented non-bitwise comparison
against the old GEMM selection. Loss13.807694435119629;36 convolution/36 gated-
delta calls and one chunked-loss call verify checkpoint recomputation. Three
additional warm complete updates remain finite. Four existing offline analyzer
contract tests, Black/Ruff-F/syntax checks pass.

Warm wall samples5.513378/5.442552/5.435568s, median **5.442552s**; peak allocated
5351892480 bytes. Profile:38296 kernel records, summed **5.403136s**. These are
synthetic accumulation1 measurements, not real dataset throughput, convergence,
a new performance improvement, a thermal sweep or broad correctness proof.

| Launch operation / BF16 logical mm shape   | Share of summed kernel duration |
| ------------------------------------------ | ------------------------------- |
| All `aten::mm`                             | 75.387%                         |
| `[1024,2048] @ [2048,6144]`                | 26.427%                         |
| `[1024,6144] @ [6144,2048]`                | 15.423%                         |
| `[1024,2048] @ [2048,2048]`                | 7.105%                          |
| `[1024,2048] @ [2048,248320]`              | 6.884%                          |
| `[1024,248320] @ [248320,2048]`            | 4.333%                          |
| Frozen causal-convolution launch operation | 7.529%                          |

Backward-submitted kernels sum3.382234s, forward2.016936s; clip+optimizer only
0.003966s. Attribution uses kernel category, launch correlation, CPU submission
stage and External-id operation mapping, excluding duplicate GPU annotations.
Shape counts remain kernel dispatch counts until a separate layout audit.
The previously tuned logical shape has114 dispatches versus132 in N3.1,
consistent with removing18 split-K companion launches. Native mm remains the
largest measured opportunity; optimizer/gated-delta dominance is not established.

## Reverse-projection layouts and existing candidates

Dispatch audit and rocBLAS bench log find **two** layouts, all offsets0:

| A stride | B stride | Forward / backward logical mm calls | rocBLAS transposes / lda / ldb | Default / candidate median GPU ms | Best solution / ratio |
| -------- | -------- | ----------------------------------- | ------------------------------ | --------------------------------- | --------------------- |
| [2048,1] | [1,2048] | 66 / 66                             | T,N / 2048 / 2048              | 11.736940 / 10.745273             | -339 / 1.09229x       |
| [2048,1] | [6144,1] | 0 / 24                              | N,N / 6144 / 2048              | 5.413399 / 4.961911               | -533 / 1.09099x       |

Column-major M6144/N1024/K2048, ldc6144, BF16 input/output, FP32 compute,
alpha1/beta0, batch1, C=D. All14 admitted existing solutions for each layout,
plus default, pass full independent CPU-double references at unchanged
atol0.05/rtol0.015/relativeL2\<0.004,128-element output guards and immutable
inputs. Worst candidate relativeL2:0.001666206. No new arithmetic kernel.
Coarse search followed by six alternating rounds with three calls per timing;
private rocBLAS handles use the actual current Torch stream.

These modest fixture ratios do not demonstrate a full-model benefit. **Neither
candidate is deployed.** The N3.2 preset/default backend/math settings are
unchanged. No new full-model candidate-gradient or convergence claim is made.

## Controller correction and preservation

r1 profile GPU gate passes9 checks. r2 layout2 and solution2 GPU gates both pass.
The first r2 controller then fails because it expected three solution-layout
checks, copied from N3.2; this shape actually has two. Original controller/log/
failed final record retained. Corrected controller derives the expected count
from the audited layouts; r3 verifies existing passed GPU gates without rerun.
r4 additionally preserves inherited user-file hashes rather than replacing them
before comparison. Reuse requires identical original helper/protected-file hashes
(excluding the repaired controller), full passed gate records, acknowledged
guards and unchanged protected identities. No GPU failure or library defect was
reproduced in N3.3; no numerical tolerance change.

All **three** GPU processes exit0, guards ACK600s, dmesg byte-identical across
profile/audit/bench/seals. Final309 protected identities, links/runtime.json,
N3.2 CSV/DSO/110 gfx1013 assets, user trainer/config and four persistent FFT AOT
databases unchanged. No source/build/cache/checkpoint/release/image removal;
no ccache reset or native build. CPU threads4; r2 boot
`d0532075-88fb-4a2f-974e-f61578d20d9f` / `7.1.5-bc250rocm-r2` unchanged.
.105/BIOS/boot fallbacks/SDMA0/heap-off/math/default rocBLAS preserved.
Historical Python CPU fault remains unresolved and separate from fixed
VM-heap/CSR and the repaired N3.2 loader defect. Free about5.92GiB; no active
agent build/test/container/guard at handoff.

WS = `/home/nik/bc250work/rocm-10.2-gfx1013`. Replayable host commands (substitute
WS with that absolute path; fresh attempt directories required):

```bash
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_training_n33_on_host.py --attempt r1 --length 1024
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_training_n33_on_host.py --attempt r2 --phase gemm
python3 WS/helpers/update_dev_prefix.py -- python3 WS/helpers/qualify_training_n33_on_host.py --attempt r4 --phase gemm --reuse-attempt r2
```

Current controller handles two or other positive audited layout counts directly;
replay does not require reproducing the old count failure. Compact results,
commands/guards and compressed logs/source attempts are tracked here. Full raw
trace151619200 bytes and owned CPU fixtures remain in WS/logs/training-n33-\* and
matching local ignored `kernel/artifacts/training-n33/remote`, copied identities
verified against .104. Original staged helper backups remain in
WS/recovery/training-n33-before. Next preservation baseline:
`WS/logs/training-n33-qualification-r4/final.json`; runtime recovery remains
`WS/recovery/gemm-n32-before` (N3.2 still installed).

**Next N3.4:** audit remaining BF16 `[1024,2048] @ [2048,2048]` actual layouts
and admitted scalar solutions, using captured CPU references and alternating
fixture timing. It is the largest remaining untuned moderate-size logical mm
in this trace. Only a demonstrated candidate should proceed to full-model
gradient/AdamW and paired timing gates before any reversible deployment.
