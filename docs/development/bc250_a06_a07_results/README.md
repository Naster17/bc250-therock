# A06 precision correction and A07 head256 attention — 2026-10-18

A06's full gated-delta operation now uses reference-ordered FP32 **GPU** preparation
and scan for FP16/BF16 and CUDA autocast. This fixes the recorded BF16 model parity
gap without widening tolerances. Native non-autocast FP32 preparation and the
explicit standalone native preparation API remain. Standalone native low-precision
preparation is still experimental for deep model integration; the old small
synthetic speedup is not a promise for the corrected default. This is not FLA.

A07 extends explicit `torch.nn.attention._gfx1013.scaled_dot_product_attention`
to head widths1–256, using tile16 above128 to bound per-program working sets.
FP32/FP16 training and inference use scalar tiled kernels. BF16 training at widths
above128 uses math FP32 forward results, native LSE and tiled backward: saved
attention state is linear in sequence length, but forward **temporary** allocations
remain quadratic. Importing it does not change functional SDPA or backend flags.
Math remains the application default.

## Correctness and actual training qualification

- 42 A06 device tests retain independent CPU float64 preparation and token-recurrent
  output/all-input/state-gradient references, with two new low-precision policy
  cases. Four extra autocast cases (FP32 raw inputs, FP16/BF16 target, normalization
  on/off) exactly match reference output and all five gradients.
- 36 A07 tests cover21 CPU-float64 output/QKV-gradient configurations over three
  dtypes, D129/255/256, rectangular causal attention, GQA, broadcast boolean/additive
  masks, fully masked rows, custom streams and strided inputs; four autocast cases,
  two exact checkpoint recomputations, three empty-geometry groups, a1024-token
  actual head geometry, four refusal contracts, three fully tiled inference cases
  and an explicit hybrid-forward dispatch check. Existing tolerances are retained.
- Actual Qwen3.5-2B uses seed61718, synthetic nonzero LoRA B std0.001, rank16/alpha32,
  all372 finite nonzero A/B gradients, math SDPA reference, lengths65/1024 and no
  user data/checkpoint. Both `.training` and checkpointing are asserted. Twelve
  actual head256 calls per run demonstrate six forward and six recomputation calls.

| Actual model case | Corrected A06 worst gradient relativeL2 | A07 worst gradient relativeL2 |
| --- | ---: | ---: |
| FP32 /65 |~9.54e-6 |~8.43e-6 |
| BF16 /65 |0 (all372 exact) |~2.1703% |
| BF16 /1024 |0 (all372 exact) |~1.3072% |

The unchanged model limits are0.2% FP32 and4% BF16 per adapter plus loss difference
<0.015. This is bounded output/gradient evidence, not universal correctness,
long-run convergence, a trained user adapter test, or optimizer parity between
math and tiled attention.

The initial pure-tiled D256 candidate passed32 operator tests but exceeded the
BF16 model gate (~4.04% at65). It was not deployed. The qualified hybrid retains
the reference forward result and native memory-bounded backward. The first model
harness also exposed `.eval()` from `from_pretrained()`: earlier actual-model
fixtures that only enabled checkpointing did not prove its execution. The shared
fixture now returns `.train()`; final runs assert12 calls. Historical operator
results and the old real CLI training/resume evidence remain separate and valid
within their recorded scopes. A06's first ordered-preparation candidate reduced
1024 error to1.76%; preserving the scan graph as well produces the exact BF16
comparisons above. Failed/intermediate attempts and their unchanged dmesg remain.

## Application and checkpoint behavior

The independently added A06 hook in `finetune-n7/replica/bc250_ops.py` is preserved;
it now calls the corrected full GPU operation and reports “precision-safe GPU
gated-delta active.” `BC250_GATED_DELTA=0` remains its explicit reference opt-out.
No warning filter or FLA-package substitution is installed.

The inner trainer and outer `scripts/train_bc250.py` now accept:

```sh
~/finetune-n7/dlvenv/bin/python ~/finetune-n7/scripts/train_bc250.py \
  --attention-backend tiled --output ~/finetune-n7/checkpoints/new-tiled-run
```

Use a new output/run for a changed attention backend. Omitting the flag retains
math, the original profile and original math fingerprint schema; no new default
field is added to old checkpoint configs. Explicit tiled mode records its backend
in the fingerprint and cannot silently resume a math run as if it were identical.
The user's checkpoint500 and all existing datasets/checkpoints are untouched.
Exact fresh/resume checks apply within each newly qualified backend/runtime;
they do not establish bitwise continuation across the old native-preparation bug
or a changed attention backend. Tiled is a memory/performance tradeoff, not a
recommended automatic speed switch.

The reviewed app patch preserves other existing edits and is in
`build_tools/bc250/patches/a06-a07-training-integration.patch`. Full before/after
source copies are retained in ignored local `kernel/artifacts/a06-a07/` and the
host recovery directory. No repository was initialized over the user project.

## Measured performance and allocation limits

Six alternating paired eager forward+autograd.grad calls after two warmups/path,
HIP events and wall envelopes, empty_cache outside timing and peak allocated
minus persistent inputs. Inputs/dY persist equally on both paths; no graphs or
kernel-only/application throughput claim. CPU references are the separate unit
suite; timings also check output/QKV gradients against GPU math with unchanged
relativeL2 bounds.

| Type / L / query:KV heads | Math ms | Tiled/hybrid ms | Math extra peak MiB | Tiled/hybrid extra peak MiB |
| --- | ---: | ---: | ---: | ---: |
| FP32 /256 /4:2 |2.954 |1.496 |8.00 |3.00 |
| BF16 /256 /8:2 |4.636 |5.129 |17.25 |13.75 |
| BF16 /1024 /8:2 |27.416 |32.984 |156.00 |112.01 |
| BF16 /2048 /8:2 |76.374 |85.951 |568.00 |376.02 |
| BF16 /4096 /2:1 |76.917 |91.582 |540.00 |392.01 |

The BF16 policy is slower in these fixtures while reducing peak allocation.
FP32's small measured win is not a general speedup. Whole-model logits, gated-
delta state, projections and optimizer allocations can dominate total memory;
these isolated numbers are not whole-model savings or an OOM guarantee.

## Deployment and evidence

The final exclusive-lock deployment passed eight installed gates:42 A06,
36 A07, four autocast, three actual-model, two CLI backend continuations,
69 A05, seven earlier attention integration and seven core regression groups.
[Deployment](deployment.json) and [independent final verification](installed-verification.json)
record five matching source/app hashes and39 unchanged protected runtime hashes.
Default launcher and explicit flag forwarding also pass [dry-run checks](outer-cli.json).
The user-facing usage file on .104 was backed up before its documentation update.

[32 GPU guards](guards.json) all acknowledged their600s deadlines with unchanged
per-run kernel logs. Four test-process failures are retained: initial eval/checkpoint
harness, pure-tiled BF16 model threshold, obsolete math_ctx assertion, and relocated
chat script. Final gate verifies the current delegated trainer and replica.chat's
math-scoped generation; legacy script/CLI assumptions are not restored into user code.
[First rollback](first-deployment-rollback.json), [moved-chat rollback](modular-chat-rollback.json)
and [r2 refusal](inactive-r2-rollback.json) remain explicit failure evidence.
The r2 attempt refused to arm before GPU execution and is not counted as a GPU test.

At resume .104 was running its persistent r2 fallback. The documented one-shot r3
boot, [base SSH/watchdog/cooling](boot-base.json) and [GPU/VMM gate](boot-gpu.json)
passed before retrying. r2 remains the persistent fallback, r3 one-shot consumed.
Recovery: `WS/recovery/a06-a07-before-r4`; original/r2/r3 attempt backups retained.
Runtime provenance: `WS/SOURCE_A06_A07.json` and current prefix copy.
Raw logs: `WS/logs/a06*`, `WS/logs/a07*`; ignored local
`kernel/artifacts/a06-a07/remote-logs/` retains test logs, including rejected candidates.
Synthetic CLI checkpoints remain in their new host log directories; existing user
checkpoints/data are untouched. No active agent process, container or guard deadline.

Numerical artifacts: [A06](a06.json), [A07](a07.json), [autocast](autocast.json),
[actual model](model.json), [CLI continuation](cli.json), [A05](training.json),
[attention integration](attention-regression.json), [core](core.json),
[paired timings](benchmark.json). [Targeted lint](lint-final.log) passes nine
configured checks through spin/lintrunner `-a`, not full CI/types.
[AST comparison](attention-ast.json) records the forward LSE_ONLY change and
unchanged native backward kernel ASTs; it is not an execution proof.

## Source and remaining scope

Torch `71a0426566bc9b952d9aa9b41f4a9d79ce46e0e8`; TheRock runtime helpers
`e4f4b3854` (full revision in deployment manifest). Python/Triton-only source
updates, no C++ build or standalone codegen; ccache untouched, existing native
Triton caches preserved. No image/kernel/boot/BIOS/SDMA/heap policy change.
Full local Git plus incremental patch0007/bundle onba6cbcd preserve the code;
the older three-file SDPA patch reconstruction gap remains explicitly open.

Dropout, higher-order/forward AD, compile/export/graphs, packed/variable-length
integration, broad concurrency, extreme inputs and long training convergence
remain unqualified. Native scalar kernels do not create MFMA/WMMA hardware.
The historical Python CPU segfault remains unresolved and separate from the
fixed VM-heap/CSR corruption. .105 and frozen/previous runtime cohorts are retained.
After this bounded A07 milestone, the next primary item is A08 dense INT8/INT32;
further native FLA and fully tiled BF16 model-parity work remain future work.
