# L01: the authorized Qwen3.5 LoRA application on two BC-250 boards

Completed 2026-11-05. Integration `ee1cc87a4`, export/initial verifier `8bacdd1d2`;
the final verifier also checks initial loss and native pre-clip norm so clipping
cannot conceal a common loss-scaling error.
[Verified metadata](verified-results.json),
[commands and scope](../../../../BC250_LORA_TWO_HOST.md).

The user explicitly authorized `/home/nik/bc250-projects/lora-smoke`, found under
the plural `bc250-projects` directory on .104. Its newer consumer packages are
Transformers 5.22.1, TRL 1.17.1, PEFT 0.21.2, Accelerate 1.15.0, datasets 5.1.0.
The older source-pinned D12/D23 Trainer helpers were not applied to this different
Trainer. Its source already sets Accelerator accumulation to one and counts
shifted causal targets. Three framework source files and both application
revisions are retained with SHA256 inventories in the metadata exports.

Added a separate generated `train_two_host.py`, support module and
`run-distributed`; original script/runner, completed adapters and checkpoints
remain intact. Model, environment and datasets were staged directly to .105;
the 2B model-file SHA256 matched on both boards:
`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`.
No private dataset text or token IDs are published or copied to the workstation.
No `finetune-n7` access or change.

## Training and numerical checks

Actual Qwen3.5-2B, rank128/alpha256/all-linear LoRA, **134,553,600 trainable
FP32 parameters /372 tensors**, FP16 base/autocast/native scaler, eager attention,
non-reentrant gradient checkpointing. Local microbatch2; one host accumulates8,
two hosts4 each, preserving global16. Fresh output paths and base-model weights,
constant LR1e-4/warmup0 in bounded tests. Both use the same first128 eligible
training records, same seed and loader settings; no evaluation/saving in timing
runs. The ready full-training profile preserves the old run's intended schedule
and both datasets, with accumulation4 and a new output path; it was not run fully.

R1 disables LoRA dropout to compare the one-GPU graph with two-node DDP. The first
update's **every post-clip gradient element** is compared in CPU FP64 arithmetic:
whole relative L2 **0.0034373344**, predeclared bound0.03; every parameter's error
norm <=0.05*reference_norm+3e-6*sqrt(elements), worst0.113972 of its bound. Native
pre-clip norms are6.5886802673(single)/6.5882725716(DDP); first losses
2.2005829811/2.2005832195. This is a same-graph GPU reference comparison, not an
independently implemented CPU Qwen reference or universal backward proof.
After four DDP updates, final adapter weights match bit-for-bit between ranks.

R2 restores dropout0.05. Six updates/layout process96 examples; input+label
hash multisets match for every global window, as do global shifted-target,
input-token and padded-token counts. All update gradients are finite and native
optimizer skips are zero in all test phases. Native TRL entropy/accuracy logs
average microbatch statistics and need not match across different grouping.
Different dropout streams/reduction order are not an exact cross-topology replay.

A separate production-mode smoke trains one update on16 records, runs final
evaluation on8 records on both ranks, and saves an adapter. All372 saved tensors/
134,553,600 elements match the live PEFT export **exactly** on readback. Fixed the
original rank-0-only final evaluation branch and added explicit process-group
destruction; the pilot shutdown warning is absent in final-code runs.

The opt-in entry point initializes the native FP16 scaler to128 to avoid spending
a short test warming down the default large scale; automatic growth/overflow
handling remains native. The old run logged10 nonfinite gradient norms in3367
steps; this observation alone does not demonstrate corrupted optimizer updates.

## Matched timings

| Case | Measured updates after excluding first | One GPU | Two GPUs, maximum rank time/window | Throughput ratio |
| --- | ---: | ---: | ---: | ---: |
| R1, dropout0 | 3 | 80.451s | 67.967s | 1.184x |
| R2, dropout0.05 | 5 | 122.892s | 99.861s | **1.231x** |

R2 is **23.1% more examples/s**, or18.7% less update time for these80 measured
examples. All96 input records match including the excluded first window. Timers
enclose synchronized forward/backward/clipping/optimizer work, exclude initial
model/DDP setup, and stop before the audit's reporting collective. The first
update is excluded for allocator/kernel warm-up and R1's numerical dump. These
are sequential bounded runs on the same boards, not a confidence interval or a
full-epoch experiment. CPU allowance4/node, cpuset0–3,10GiB RAM/12GiB total,
512PIDs/256MiB shm/600s recovery guard/460s child; no other workloads stopped.

Native full six-update `train_runtime` was **239.404s single vs248.044s DDP**:
startup makes this short job slower overall. No whole-run time or convergence
claim follows. FP32 LoRA gradient vector alone is538,214,400 bytes/update, not a
physical NIC counter. One long example can keep one rank busy while the other
waits. DDP does not increase per-board model capacity. R2 live allocator peaks:
single 9.167 GiB, DDP .104 9.665 GiB/.105 7.493 GiB; no memory-saving claim.

Recovered GPU allocation-retry warnings occurred in the DDP pilot and both R2
layouts. No GPU phase failed, but the8192 length setting remains memory-sensitive
for long examples; not every8192-token sample was exercised. Reference causal
convolution/gated-delta paths remain in this newer Transformers environment;
optional fast packages and compatibility with older custom hooks are unqualified.

## Evidence, preservation and next action

Eight acknowledged GPU guards passed, no kernel-log/registry changes, no reboot,
native build/install/prefix mutation or package download. Boot IDs remain
`c1199fca-eb06-413f-8d28-34417297ba67` (.104r2) and
`bad537ed-ea08-49e6-ad44-40a9bc1d6914` (.105r3). Shared runtime locks, existing
fallbacks and native board controls retained. Both boards idle at handoff.
Black/critical Ruff+F401/syntax/shell checks and three generator preservation
checks pass; no full upstream CI or repeat of unrelated stack tests.

Workstation `kernel/artifacts/lora-two-host/{104,105}-evidence.tar.gz`:

- .104: 405179 bytes, SHA256 `58557236d72cb6aa6a3781a2c351cb367d9222a40ddd0543b61d27edf43d247d`.
- .105: 282309 bytes, SHA256 `2bbafbb7c270c053480c17eb3618eb14910afb00b9e3508b2611ef2707456902`.

Host originals are `bc250work/evidence/lora-two-host/{104,105}-evidence-r2.tar.gz`.
Initial non-root exports omitted root-only recovery subdirectories; verification
rejected them. Root-only export fixed this; incomplete archives are retained
locally as `*-evidence-incomplete-r1.tar.gz`. Raw gradients and saved adapters
remain on hosts, excluded from metadata archives. The verifier checks hashes,
sources, guard acknowledgments, update/data/dtype/replica contracts and timing
arithmetic; it does not reexecute the GPU math from metadata.

Exact guardian/controller commands, PIDs and logs are in each exported phase's
`running.json`, `final.json`, `gpu.log` and guard records. Guardian PIDs:
.104: 27384/28490/30214/30886/32056;
.105: 20500/22096/23684; all exited 0, no containers/armed guards remain.

**Next application gate L02:** real-application same-world checkpoint distribution
and exact fresh-process optimizer/scaler/scheduler/RNG/cursor replay before relying
on long-job recovery. Old one-host checkpoints are not qualified exact two-host
continuations. The broader stack cursor remains **D24**, BF16 FSDP working
parameters/all-gather buffers with FP32 masters/reductions; L01 does not widen
D23's frozen-parameter or package-version guards. Historical CPU Python/KFD
faults remain distinct from the fixed VM heap, D05 exit and D21 optimizer defects.
