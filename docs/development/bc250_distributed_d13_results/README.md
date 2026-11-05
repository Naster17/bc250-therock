# D13: actual Trainer BF16/FP16 and collective finite refusal

Completed2026-11-05. Fixture/callback70bd075e7, exports62b35356c. Existing opt-in
D12 BC250Trainer file is unchanged; staged alongside the new callback on BOTH
hosts under tools/distributed-d13. Native daily/RCCL Torch a524980/ROCm/D09
launcher/registries/images/profiles/kernels/frozen consumers unchanged. No native
build/install/prefix mutation/reboot or user-application access. English owned
fixtures only. [Usage](../../../../BC250_DISTRIBUTED_GUIDE.md#trainer-bf16fp16-and-finite-gradient-checks-d13).

## Actual mixed-precision path and mathematical references

Actual Transformers5.2.0 Trainer delegates through the source-pinned D12 adapter,
with existing Accelerate1.12.0. Native prepare/DDP/DataLoader/training_step/
backward/no_sync/AdamW/StepLR/checkpoint resume. One gfx1013 GPU per node/world2,
NCCL(AMD RCCL); FP32 master parameters/reduction and observed BF16/FP16 Linear
activation dtypes. The model is the retained D07 residual MLP1,051,648 parameters,
four256/512 blocks/16 tokens. It is an owned torch.nn.Module emitting logits,
not a pretrained Transformer. D12 Bert FP32 qualification remains separate.

Custom fixture loss is masked squared error: sum of active element errors divided
by Trainer's global active count across the entire accumulation window. Variable
mask sizes exercise distributed count normalization. Trainer compensates DDP's
average; BC250Trainer retains ONE accumulation division.28 sequential records,
per-device micro2/accumulation2, global8 first three attempts, final global4 from
one micro/rank. Every input row is checked; gradient traffic zero on nonfinal
micros and4,206,592 FP32 bytes on final micros. Clipping disabled, dropout absent.

Independent D07 CPU oracle uses explicit low-precision rounding of operands,
outputs and backward gradients with FP64 matrix arithmetic. Separate micro/rank
losses scaled with the actual native scaler are accumulated before reference
unscaling/DDP averaging. ALL finite-window gradients and global losses compared:
BF16 relativeL2 limit.02, FP16.004, same retained D07 bounds. Maximum observed
across all parameters/finite windows/ranks: BF16.00276769444, FP16.000372783622.
Loss atol2e-5/rtol.002(BF16),.0005(FP16). No bounds relaxed or harness correction
needed. CPU-double AdamW fed the admitted actual gradients independently checks
ALL parameters and exp_avg/exp_avg_sq against D07 tolerances. This isolates
optimizer arithmetic from accepted low-precision oracle rounding differences.
AdamW lr.001/eps1e-5/decay.01/foreachFalse, StepLR(step2,gamma.8), raw post-step
hook and independent LR/Adam step-state counts. Four BF16 successful updates;
three FP16 successful updates across four Trainer global steps/attempts.

## Nonfinite policies

FP16 uses the unchanged NATIVE GradScaler defaults: initial scale65536,
growth_interval2000. Rank1 injects Inf into the accumulated gradient BEFORE the
third attempt's final DDP reduction. Both ranks therefore receive nonfinite
reduced gradients. Native scaler skips AdamW; parameters, optimizer state and
scheduler are checked unchanged. Trainer does not advance StepLR on skip, but
DOES advance its global_step attempt counter. Both scalers back off to32768/
tracker0, then final short finite attempt recovers at32768/tracker1. End actual
Adam steps3, native Trainer global_step4, scheduler epoch3, LR.0008.
This qualifies nonfinite values propagated through DDP before the reduction;
rank-local errors introduced AFTER reduction and clipping are not covered.

BF16 has no native scaler/automatic nonfinite suppression. New optional
[FiniteGradientCheck](../../../build_tools/bc250/bc250_trainer_finite.py) is a
PUBLIC TrainerCallback: on_pre_optimizer_step examines dense gradients and
all-reduces an invalid flag; if any live rank is invalid, ALL raise
NonFiniteGradientError before the optimizer. It deliberately stops the job;
it does not silently skip a batch or mutate Trainer private overflow flags.
FP16 is explicitly refused by this callback. Recover from a complete checkpoint.

BF16 normal seed/replay exercises callback acceptance. A separate negative
fixture runs two successful updates/saves checkpoint2, injects Inf into ONLY
rank1 AFTER DDP reduction at pre-optimizer time: rank0 local gradients still
finite. The collective flag makes both ranks refuse. Complete parameters/AdamW/
StepLR equal checkpoint2; native Trainer global_step and Adam step count stay2.
RNG/data probes and gradient buffers have advanced and are not claimed unchanged.
Both-ranks-alive test; no peer-loss/hang recovery. FP32 code accepts the callback
but its current GPU qualification is BF16 only; sparse gradients are outside scope.

## Native checkpoint and exact fresh replay

Native Trainer save_on_each_node writes complete safetensors/AdamW/StepLR/Trainer
state/rank-specific RNG and FP16 scaler checkpoint2 on EACH independent host.
Config/source/rank/file sizes/SHA admission before native resume. Entirely new
containers/workers/different constructor seeds call actual
train(resume_from_checkpoint=checkpoint2). Native RNG restoration after skipped
loader batches restores all parameters/optimizer/scheduler/scaler/global_step/
Python/NumPy/Torch CPU/CUDA RNG exactly. Following attempts3/4 then match ALL local
losses, gradients, parameters, optimizer, scheduler, scaler, four RNG streams,
probe draws, input rows and finite flag exactly with zero tolerances and digests.
Eight resumed rank-windows across both precisions/hosts; real CID/UUID/startticks
prove fresh launches. FP16 repeats native skip/backoff/recovery exactly.
No DataLoader worker/shuffle/stochastic graph, arbitrary package/model/loss/
optimizer/partial-micro/elastic/migration/crash/peer-loss claim. Probe draws
exercise four RNG streams; they are not stochastic model layers.

## Evidence and operation

68accepted check groups/10GPU guards/8exact resumed rank-windows/42localtests.
Black/critical Ruff+unused imports/syntax/diff pass; full upstream CI not run.
All phases pass first revision. Native all_gather deprecation/barrier warnings
retained; native aggregate resume train_loss caveat from D12 remains. No native
fault/performance/capacity/convergence claim. Historical Python CPU fault/KFD
uncertainty and invalid-visibility import failure remain distinct from the fixed
VM heap/D05 agent exit defects. D13 does not change HIP visibility parsing.

SH runtime locks;4CPUs/cpuset0–3,10GiB RAM/12GiB memory+swap,512PIDs/256MiB shm,
600s recovery guards/460s child bounds; no foreign process/container touched.
All agents/guards clean/acknowledged, registry/kernel logs unchanged. Sequence
controllers6626/6771 exit0; per-phase commands/PIDs/exits in sequence.json.
Separate BF16 reject8067/8033 exit0 (expected exception caught by owned fixture).
No jobs/containers/armed guards remain. .104r2/.105r3 boots dc9e9293/133b8dc2
unchanged throughout this batch; SSH, fallback kernels and native controls retained.

Both hosts: tools/distributed-d13 and evidence/distributed-d13-r1/
{bf16,fp16}-{seed,resume},bf16-reject-seed. Normal seed checkpoint/reference
payload143,385,444B/host retained, plus reject/resumed native outputs. Metadata
exports omit .pt/.pth/.bin/.pkl/.safetensors data; retain complete raw reports/
source/kernel/guard/registry/log diagnostics. Local kernel/artifacts/distributed-d13
and committed [verified summary](verified-results.json).

| Archive | Bytes | SHA256 |
| --- | --- | --- |
|104-evidence-r1.tar.gz|279479|f7cc87c152d92c700f605ff0d0d35190933d773c513308d1913fd65f5f1dd6b5|
|105-evidence-r1.tar.gz|252034|2d3cf3ca63a0a8eae14bfef5d1b61b5abc90293e47ec5b28d688d72755ff5f2a|

Both host docs/agent memory synchronized. Next D14: actual Trainer AMP gradient
clipping and a coherent rank-local FP16 nonfinite policy, owned bounded references;
keep existing native/default/recovery settings. Trainer AMP Bert/other losses,
larger capacity/elastic operation/release reconstruction remain separate.
