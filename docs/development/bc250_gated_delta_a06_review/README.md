# A06 recheck — 2026-10-17

> **Later correction (2026-10-18):** [A06/A07 qualification](../bc250_a06_a07_results/README.md)
> fixes low-precision model parity using reference-ordered GPU preparation/scan.
> The failure measurements below are retained historical evidence.

**The native preparation remains implemented and passes its recorded operator
references, but broader BF16 LoRA numerical qualification is reopened.** The
stronger nonzero-adapter comparison exceeds the existing per-adapter4% relative
L2 gate. No production Torch/ROCm code was changed in this review. The rejected
composition candidate was not installed. Resolve this A06 issue before A07 or
recommending automatic training integration.

## What the earlier evidence missed

The original actual-model test used freshly initialized LoRA adapters. Its B
matrices were zero, so the A gradients were zero. Comparing372 tensor names did
not exercise372 nonzero gradients. The small random model already used nonzero
B, but that did not establish real2B nonzero-adapter behavior. The old test and
its timings remain valid within that narrower initial-adapter scope; they do not
qualify a resumed trained adapter. Old reports/JSON are preserved.

The new recheck uses synthetic nonzero B (normal std0.001), seed61718, the same
local text-only Qwen3.5-2B base, rank16/alpha32/all decoder linear targets,
checkpointing, math SDPA and the pinned old convolution/loss helper identically
on both paths. No user dataset/checkpoint is opened or changed. Every case has
all372 nonzero finite adapter gradients, exercising both A and B.

| Type / tokens | Worst adapter relative L2 | All-adapter global relative L2 | Existing gate |
| --- | ---: | ---: | --- |
| FP32 /65 |0.0009537% |0.0003809% |Pass (<0.2% worst) |
| BF16 /65 |4.00217% |1.57381% |Fail (<4% worst) |
| BF16 /1024 |5.42540% |0.87004% |Fail (<4% worst) |

The4% gate was not widened. At1024, the worst tensor is
`layers.4.linear_attn.in_proj_a.lora_A`: reference norm0.000497486, difference
norm0.0000269906. Small absolute differences and low total loss error do not
establish optimizer/convergence parity. The unchanged module remains explicit
experimental functionality; these failures are numerical qualification failures,
not GPU faults, OOM or the separate historical Python CPU segfault.

## Why the speedup is modest

For B1/L1024/H16/Dk128/Dv128 BF16, six paired event-instrumented samples after
two warmup rounds give these phase-envelope medians in milliseconds:

| Phase | Reference preparation path | Native preparation path |
| --- | ---: | ---: |
| Forward preparation |1.341 |0.920 |
| Pairwise interactions |0.579 |0.555 |
| Triangular solves |1.653 |1.552 |
| Chunk scan/output |3.971 |4.175 |
| Complete backward |19.836 |19.381 |
| Complete forward/backward |27.384 |26.594 |

Forward preparation is only~4.9% of the reference total, and complete backward
is~72.4%. Fusing preparation cannot remove the remaining solves/scan/gradient
work. The instrumented forward/backward still makes196 `aten.bmm` calls in
both paths; total ATen dispatcher calls1697->1610. These are dispatcher counts,
not kernel counts: views can make no launch and Triton bypasses this count.
Events include host submission and instrumentation overhead; this is a coarse
phase breakdown, not kernel-only timing or a full-model time attribution.
The instrumented forward exactly matches each unmodified implementation. Its
reference backward differs by at most~1.93e-5 relativeL2 due to graph/operation
ordering; native matches exactly. The initial instrumented check incorrectly
required near-bitwise BF16 gradient agreement (8/2097152 entries differed by at
most1.53e-5); it is retained. The final instrumented check uses the already
recorded2% complete-operator BF16 relativeL2 contract and records actual errors.
This did not relax the stronger model4% gate, which still fails.

The nonzero-adapter1024-token repeat has five alternating pairs: median
reference7.7779s/native7.3172s. Individual pair ratios range0.9887–1.0756,
including one slower native sample. Numerical qualification failed, so this is
diagnostic timing, not evidence for deploying an optimization or a promised
training gain. The earlier~3.6% result was one synthetic initial-adapter fixture.
Actual training also includes projections/MLP/full attention/loss and optimizer/
accumulation/data/checkpoint work outside this preparation substage.

## Numerical isolation

An exact ATen preparation control preserves the pinned reference's FP32 cast,
layout, `x*x` normalization, padding, beta and cumsum operation ordering. Using
that control in the unchanged A06 composition gives exactly the stock loss and
all adapter gradients in the65-token diagnostic. Test-only straight-through
substitutions isolate preparation values versus derivatives:

| Preparation values / derivatives | Worst adapter relative L2 | Global relative L2 |
| --- | ---: | ---: |
| Stock ATen / stock ATen |0 |0 |
| Native / native |4.00217% |1.57381% |
| Exact ATen / exact ATen |0 |0 |
| Exact ATen / native |2.14735% |0.64934% |
| Native / exact ATen |3.76388% |1.57510% |

These substitutions are diagnostic only, not deployed autograd implementations.
The first captured layer's prepared Q/K/Kbeta differ by relativeL2~4.6–5.2e-8;
Vbeta is exact; cumulative decay max absolute difference1.53e-5, relativeL2~2.98e-8.
This supports FP32 preparation rounding/derivative-order differences amplified
through BF16 as the issue to investigate. It does not prove a universal root
cause, mathematically wrong gradients, training divergence or safe convergence.
An earlier diagnostic using an algebraically equivalent `square()` ATen control
was not bitwise equivalent in backward; that raw run is retained separately.

A candidate that restores original ATen exp/scan/slice-assignment structure still
fails:4.00217% at65 and5.57270% at1024. It remains an ignored local/remote
experiment; the [candidate patch](rejected-composition-candidate.patch) is evidence
and must not be applied as a qualified fix.

## Independent application edit and protection

During this review, `/home/nik/finetune-n7/replica/bc250_ops.py` changed independently
at2026-10-17 00:08:37 UTC. Its hash changed from `eddfea40...` to
`d058b14a4bae969e5b84a9de8f69706c0f6e3df910409dfc15843f0eeee3d41a`.
It now imports A06 and defaults `BC250_GATED_DELTA` to1. The source of this edit
was not established; it was preserved, not reverted. Our recheck imports the
pinned separate `WS/helpers/qwen35_training_ops.py`, so the compared helper stays
identical on both paths. This independent integration is not qualified by these
failed numerical tests. The previous statement that the ordinary user trainer
necessarily retains the reference path is now outdated.

For a new training process, insert `--env BC250_GATED_DELTA=0` among the inner
`bc250-train` launcher options, before the Python script, to retain the reference
path. A shell environment assignment alone is not inherited by this launcher's
allowlist. No application edit or environment/default change was made by this
review.40/41 prior protected hashes match; only that independently changed app
helper differs. The installed A06 module hash remains
`9ecb6f9014bc139e54cffe0fbb2c80f782035e57a3149e3c9a6b31fe445e8b2b`.
Torchba6cbcd, libraries/system/Triton/kernel/image/boot/fallback and .105 unchanged.
All six review guards ACKed/600s and kernel logs byte-identical53f83f7b; three
retained failures are the initial instrumented check and two failed model gates.
No active agent test/container/guard deadline remains.

## Evidence and next action

[Summaries, per-case samples and worst gradients](review-results.json),
[phase samples/dispatcher counts](phases.json), [guards](gates.json),
[protected-state drift](protection.json). Full per-parameter/raw source hashes and
commands remain `WS/logs/gated-delta-a06-review-*` and ignored local
`kernel/artifacts/gated-delta-a06-review/remote-logs/`. Reproduction tools:
`build_tools/bc250/{review,diagnose}_gated_delta_a06.py`, staged in WS/helpers;
run with the existing exclusive lock and guarded host runner. Formatting and
syntax pass; no new Torch build/codegen or production-source edit was needed.

Next: A06 numerical parity. Use the retained seed61718/nonzero-B fixtures at65
and1024 tokens; isolate reference-ordered Q/K reductions/normalization derivatives
and decay prefix-scan rounding against the exact ATen preparation control.
Require the unchanged4% model gate plus independent operator references before
any default integration. Additional native solve/scan fusion is later performance
work. A07–A09 and N2.7 remain queued; no complete FLA backend is qualified.
