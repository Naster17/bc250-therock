# D14: native Trainer AMP clipping and rank-local overflow coordination

Completed2026-11-05. Implementation/fixtures625b840d7, exportsb3e3d8a30. New opt-in
BC250ClippedTrainer in tools/distributed-d14 on both hosts. Existing D12
BC250Trainer and D13 FiniteGradientCheck remain unchanged. Native daily/RCCL
Torch a524980/ROCm/D09 launcher/registry/images/profiles/kernels/consumer packages
unchanged. No native build/install/prefix mutation/reboot/user-application access.
[Usage](../../../../BC250_DISTRIBUTED_GUIDE.md#coordinated-trainer-amp-clipping-d14).

## Why coordination must precede clipping

Native Trainer calls Accelerator.clip_grad_norm_ BEFORE on_pre_optimizer_step.
For FP16, native clipping first calls GradScaler.unscale_, which caches each
rank's local overflow decision. A rank-local Inf introduced AFTER DDP reduction
can otherwise make one scaler skip while the other updates. Changing gradients
only at the later public callback cannot repair that already-cached decision.
D13 covered overflow propagated BEFORE final DDP reduction; D14 adds this missing
post-reduction boundary. This is an explicit Python integration policy, not a
new GPU kernel or a global patch of frozen packages.

BC250ClippedTrainer inherits the D12 source/version-pinned normalization adapter
and decorates ONLY its own Accelerator instance's PUBLIC clip_grad_norm_ method.
It delegates actual unscale/L2 clipping to the unchanged native implementation.
Dense FP32 master gradients, one Trainer optimizer, two DDP ranks, positive finite
max_grad_norm and norm_type2. Sparse/other layouts are refused collectively.
No native GradScaler private overflow flags or Trainer loop fork.

Before native unscale, all ranks check complete public scaler-state agreement
and finite positive scale. Any mismatch stops BOTH before the optimizer. If an
FP16 rank has a nonfinite gradient, an all-reduced flag makes EVERY rank mark
one gradient element Inf before native unscale. Both native scalers now observe
overflow themselves, native clipping runs, and native scaler.step skips AdamW
on BOTH. Trainer suppresses StepLR, then native scalers back off in agreement.
BF16 has no scaler; bad input collectively stops before native clipping/update.

After native clipping, invalid gradients/norms from previously finite inputs
collectively stop the job. A public final on_pre_optimizer_step callback is
appended AFTER existing caller callbacks when train starts. If those callbacks
introduce a new nonfinite gradient after unscale, BOTH stop before AdamW rather
than applying a stale finite scaler decision. With a known pre-unscale overflow,
the final gate permits the already-guaranteed native skip. Late failures require
reload from a complete trusted checkpoint; do not continue the dirty scaler/
iterator. User callbacks must not dynamically add/reorder callbacks or mutate
gradients after the final gate; matching collective order remains required.
Finite asymmetric gradient edits are not detected as a numerical consensus API.

## Mathematical and clipping references

Same owned D07 residual MLP1,051,648 parameters and variable-mask MSE/global active
count as D13, actual Trainer5.2.0/Accelerate1.12.0 BF16/FP16. One native gfx1013
GPU/node, FP32 master/reduction.28sequential records/workers0, per-device micro2/
accum2: first three attempts global8, final short global4. No dropout, explicit
RNG probes, strict deterministic fixture setting. Not a pretrained Transformer.

max_grad_norm=.002. Pre-clipping gradients retain independent D07 explicit
low-precision rounding/FP64-matmul oracle checks (BF16 relativeL2<=.02, FP16<=.004).
Observed maxima BF16.00276769444, FP16.000372783621. Independent global loss
references remain within fixed D13 bounds. CPU FP64 sums squares of ALL admitted
unscaled pre-clipping gradients, computes global L2 norm and coefficient
min(1,.002/(norm+1e-6)), then predicts EVERY clipped gradient. Native returned
norm agrees rtol1e-5/atol1e-7; ALL clipped gradients rtol5e-5/atol1e-8. CPU-double
AdamW consumes the independent expected clipped gradients and checks ALL model
parameters and moments under retained D07 bounds. No tolerance changes.

Clipping is ACTIVE in every finite checked window: norms.0186445–.0547800,
coefficients.0365090–.107265. This is not a no-op threshold check. BF16 four
actual updates/LR.00064; FP16 three AdamW/StepLR updates in four Trainer attempts,
LR.0008. Unchanged default scaler65536/growth_interval2000 backs to32768 on
attempt3; final finite short attempt recovers at32768/tracker1.

## Failure injection and native restart

FP16 normal case: rank1-only Inf AFTER final backward/DDP, BEFORE native unscale
at attempt3; pre-clip flags verified [finite,nonfinite]. Both scalers skip,
parameters/AdamW/scheduler remain unchanged, scales agree, next finite update
passes CPU clipping/AdamW references. This closes D13's post-DDP overflow gap.
Three separate expected-refusal cases each reach checkpoint2, then:

- BF16 rank1-only pre-clip Inf: both stop before clipping/update.
- FP16 rank1-only Inf in an ordinary pre-optimizer callback AFTER clipping:
  final gate stops both before stale scaler decision can update weights.
- FP16 rank1 public scaler.update(new_scale=scale/2) AFTER backward: both stop
  on scaler-state disagreement before unscale/update. Intentional scaler mutation
  is not undone; only checkpoint reload is supported afterward.

All six rank refusal observations retain checkpoint2 ALL parameters/AdamW/StepLR,
Trainer global_step and Adam step count2. RNG/data/gradient/scaler transient state
has advanced or been deliberately perturbed; no whole-runtime rollback claim.
Both ranks remain alive; no peer death/crash/network recovery proof.

Native per-host save_on_each_node checkpoint2 including FP16 scaler restores in
new containers/workers with different construction seeds. Initial complete
parameters/AdamW/scheduler/scaler/globalstep/four RNG exactly restored; attempts3/4
then match ALL local losses, gradients, weights, optimizer/scheduler/scaler/RNG/
probe draws/rows/finite flag with zero tolerances and digests:8resumed rank-windows.
Native clipping of Inf can produce NaNs; gradients are represented by exact
NaN/+Inf/-Inf masks plus exact finite values with nonfinite slots canonicalized
to0. Nonfinite classifications are compared, not discarded; no NaN payload-bit
identity claim. All finite state/loss comparisons retain exact zero tolerance.
Fresh CID/UUID/startticks prove process restart. Trusted fixed-world/sequential
workers0 optimizer-boundary format; no arbitrary data/checkpoint migration.

## Evidence and operation

118accepted check groups/14ACK GPU guards/6coherent rank refusals/8exact resumed
rank-windows/42localregressions. First revision all phases pass; Black/critical
Ruff+unused-import/syntax/diff pass; full upstream CI not run. SHA verifier checks
fixture/normalization/finite/clipping/oracle source, per-phase raw reports/kernel
logs/guard nonces/registry, cross-host configs and fresh replay digests.

SH runtime locks,4CPUs/cpuset0–3,10GiB RAM/12GiB memory+swap,512PIDs/256MiB shm,
600s guards/460s children. All agents/phase kernel logs/registries clean. Sequence
controllers10343/10259 exit0; exact commands/PIDs/exits sequence.json. No foreign
process touched; no owned jobs/containers/armed guards remain. Boots dc9e9293(.104r2)/
133b8dc2(.105r3) unchanged; SSH/boot fallback/native controls preserved.
Native all_gather deprecation/barrier warnings and D12 aggregate resume train_loss
caveat remain. Historical Python CPU fault/KFD and invalid-visibility failure are
separate from already fixed VM heap/D05 exit; no native fault or speedup claim.

Both hosts: tools/distributed-d14 and evidence/distributed-d14-r1/
{bf16,fp16}-{seed,resume},{bf16-reject,fp16-late,fp16-scale}-seed. Normal seed
native/reference payload156,059,932B/host retained plus reject/resumed outputs.
Metadata exports omit.pt/.pth/.bin/.pkl/.safetensors, keep source/reports/logs/
registry/kernel/guard diagnostics. Local kernel/artifacts/distributed-d14;
committed [verified summary](verified-results.json).

| Archive | Bytes | SHA256 |
| --- | --- | --- |
|104-evidence-r1.tar.gz|382577|90780de8dff826d6149016bf2db78c58c89d2eb0d7f20c66aa6f6e92ae14ab23|
|105-evidence-r1.tar.gz|343823|bc3b71333f1506f6a87b8ae2375c4cf2cb408cd58146f1fb957c1db3c3bb3667|

Host docs/memory synchronized. No arbitrary model/loss/FP32/clipping threshold/
sparse/multidevice/multioptimizer/FSDP/DeepSpeed/stateful-dynamic-callback/
convergence/capacity/performance/elastic/general resume qualification. Extreme
finite-gradient norm overflow has a collective refusal guard but no separate
GPU injection proof in D14. NextD15: actual owned offline Transformer BF16/FP16
Trainer token-normalized CE +clipping/reference/native restart integration;
retain native/default/recovery settings. Invalid-visibility frontend refusal,
larger capacity and release reconstruction remain separate follow-ups.
