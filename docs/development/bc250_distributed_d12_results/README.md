# D12: actual Trainer DDP, accumulation repair and native restart

Completed 2026-11-05 on .104/.105. Initial514ed8be2; accepted compatibility/source
pins and LR assertion correctionf5edcaac1; exports3f2d1e391, CPU admission-test
correctioncc47efc09. Installed native Torch a524980/ROCm/D09 launcher/images/
registry/profiles/kernels/consumer packages unchanged. No native build/install/
active-prefix update or development reboot. Owned offline fixtures only; no user
application inspection, modification or test.

## Defect and explicit compatibility

The pinned Transformers5.2.0 Trainer normalizes a mean loss by its actual gradient
accumulation window, or uses a token-summed loss normalized by num_items_in_batch.
Its Accelerate1.12.0 instance is also configured with the accumulation count,
and Accelerator.backward divides AGAIN. Stock actual Trainer with two micros/
update produces ALL gradients equal to0.5 times independent CPU-double reference,
while its reported loss matches the undivided global loss. This is a reproduced
Python integration defect, not a GPU arithmetic error. Successful stock diagnostic
checks establish the defect; they do not qualify stock Trainer correctness.

The opt-in [bc250_trainer.py](../../../build_tools/bc250/bc250_trainer.py) supplies
BC250Trainer, subclassing actual Trainer. It delegates the entire native training,
checkpointing and optimizer loop. After native Accelerator creation, it sets
Accelerator.gradient_accumulation_steps=1, while TrainingArguments retains2.
Trainer still sets sync boundaries, performs its own loss normalization, and
handles its actual final window size. No method monkeypatch or package mutation.
Version checks AND exact Trainer/Accelerator file SHA pins refuse changed source
or package pairs, avoiding accidental double-correction after upstream changes.
D12 only qualifies FP32 DDP; source code rejects DeepSpeed/FSDP for this adapter.
Direct Accelerator callers retain their native scaling (D11); do not apply this
Trainer-specific setting to their loops.

BertForMaskedLM accepts **kwargs but computes a local mean CrossEntropyLoss,
ignoring num_items_in_batch. Its signature alone is not a valid token-normalized
loss contract. The equal-active-count fixture explicitly sets
model.accepts_loss_kwargs=False. Variable-mask counts instead use the explicit
masked_token_loss compute_loss_func: sum active token CE divided by Trainer's
global active count for the entire window. average_tokens_across_devices=True
compensates subsequent DDP averaging. This is a MASKED, unshifted LM loss; no
causal/general task semantics are claimed. The helper does not choose a policy
for arbitrary models automatically. [Usage](../../../../BC250_DISTRIBUTED_GUIDE.md#transformers-trainer-on-two-hosts-d12).

## Numerical and state qualification

Both corrected cases use the owned D09 tiny Bert22,561 parameters, vocab97,
hidden32/two layers/four heads, intermediate64, length12, eager attention and
zero dropout. Actual Trainer prepares DDP/DataLoader/optimizer and executes
training_step/backward/no_sync/callbacks/scheduler/save/resume. FP32 masters and
gradient reductions on native gfx1013/RCCL, MULTI_GPU/world2, one GPU/node.
Sequential28 examples, per-device micro2, accumulation2: global8 on first three
updates, final global4 from a SINGLE micro/rank. No duplicate/dropped examples;
all global input row IDs checked. Only final micros reduce90,244 gradient bytes.

At each of FOUR seed updates/case/rank: independently compute the global token
mean CE in CPU double, backpropagate and compare EVERY gradient (atol3e-6,
rtol5e-4), global loss (atol3e-6,rtol5e-5), CPU-double AdamW ALL parameters
(atol4e-6,rtol5e-5) and exp_avg/exp_avg_sq (D07 fixed limits). No numerical
bounds relaxed. AdamW lr.001/eps1e-5/decay.01/foreachFalse, clipping disabled;
StepLR(step2,gamma.8), post-step hook +independent LR formula: four updates,
final LR.00064. The short final window has no extra factor-of-two division.

After step2, native Trainer saves safetensors/optimizer/scheduler/RNG/Trainer
state on EACH independent host filesystem (save_on_each_node=True). It continues
steps3/4 as uninterrupted reference. Fresh containers/processes use different
constructor seeds, then native train(resume_from_checkpoint=checkpoint2) loads
model, AdamW, StepLR, global_step, skips the consumed loader batches and restores
Python/NumPy/Torch CPU/CUDA RNG before the next forward. The fixture asserts
initial restored state EXACT, then losses/ALL gradients/parameters/optimizer/
scheduler/four RNG streams/probe draws/input rows/global_step EXACT and matching
digests for steps3/4:8 resumed rank-windows across both cases/hosts. Explicit RNG
probe draws exercise all four streams; dropout is zero. No stochastic model or
multiworker data augmentation proof follows. Real container IDs, UUIDs/startticks
prove fresh launch. Local config/filesize/SHA admission binds case/source/rank.
No malformed-checkpoint/atomic coordinated load/peer-loss test is added here.

Trainer emits missing-key messages for tied decoder weight/bias omitted by native
safetensors serialization; complete restored model/optimizer and replay equality
pass. Deprecation warning for native all_gather_into_tensor and barrier device
warning remain; not suppressed. Aggregate train_loss printed after resume uses
remaining loss with total global_step denominator; it is not an equal-interval
metric comparable to seed train_loss. Per-window losses, gradients and state are
the qualified comparisons. No aggregate metric semantics repair in this batch.

## Evidence and failures preserved

78 total accepted check groups =6 stock-defect diagnosis +72 corrected integration;
10 GPU guards =2 stock +8 corrected.18 additional pinned-package CPU checks:
version/source replacement refusals, analytic masked-CE loss/gradient and invalid
token-count checks, Torch CUDA uninitialized.42 local regression tests pass.
Black/critical Ruff+unused-import/syntax/diff pass; full upstream CI not run.

Initial r1 corrected mean case passed all four CPU gradient/AdamW checks but failed
a harness LR equality at final step: iterative0.00064 versus power formula's
floating representation. r2 uses abs1e-15 for LR only, independent last_epoch
counter unchanged; no model/loss/gradient bounds changed. r1 source/logs/native
checkpoint remain. r1 stock also preserved; all four historical guards ACK.
CPU-only first invocation incorrectly supplied HIP_VISIBLE_DEVICES=-1: ROCr10.2
rejects this value at import with fatal assertion/abort and exit139; command/logs
retained under cpu-invalid-visibility. Corrected command omits visibility flags
and asserts Torch CUDA remains uninitialized. ROCr's invalid-environment behavior
remains, distinct from historical Python CPU fault/KFD uncertainty and fixed VM
heap/D05 agent exit. CPU test also corrected a stale lazy-module mock by replacing
sys.modules entries; cpu-stale-module-mock diagnostics/source are retained.

GPU phases held common SH locks,4CPUs/cpuset0–3,10GiB RAM/12GiB memory+swap,
512PIDs/256MiB shm,600s guards/460s child deadlines; clean agents and unchanged
registry/kernel logs each phase. r2 coordinators29765/16203 exit0; commands/PIDs/
exits in sequence.json. r1 stock28871/15080 exit0; mean29292/15570 exit1.
CPU corrected controllers4100/4259 exit0 (120s bounds, existing daily SH runner).

GPU qualification boots7344171f/e1cbfa23; both boards rebooted BETWEEN completed
GPU work and CPU/export work, with guards already acknowledged. Current boots
.104dc9e9293-451c-4cd0-8839-7c902c8509ad and
.105133b8dc2-62a1-417b-8acb-fc44aeda6eec; kernels remain r2(.104)/r3(.105).
No development reboot was issued. Final export confirms no containers; no owned
jobs/armed guards remain. Boot fallback/native controls/application data preserved.

Both hosts: tools/distributed-d12 (opt-in helper, fixtures, source-r1),
evidence/distributed-d12-r2/{stock-seed,mean-seed,mean-resume,tokens-seed,tokens-resume,cpu};
r1 and CPU failed diagnostics preserved. Valid seed checkpoint/reference data
3,720,842B/host retained, plus native resumed/historical outputs. Exports omit
.pt/.pth/.bin/.pkl/.safetensors data, retain metadata/source/raw logs. Local
kernel/artifacts/distributed-d12; committed [verified summary](verified-results.json).

| Archive | Bytes | SHA256 |
| --- | --- | --- |
|104-evidence-r2.tar.gz|406142|7ce23877c053426ecd690a764ff1377ae0b839eaec659e3a43580d41db8ccf0d|
|105-evidence-r2.tar.gz|365873|ed7af2bc9f1bf60381749a175456d628b5208f5a801dad77c1c2007ad671497c|

No automatic user-script changes, arbitrary task/model/dataset/loss semantics,
Trainer AMP/FSDP/DeepSpeed/evaluation metrics/convergence/capacity/speedup or
elastic/crash/multiworker/general resume qualification. Host docs/memory include
this report. NextD13: actual Trainer BF16/FP16 integration, bounded mathematical/
nonfinite-update/state references using owned fixtures and this explicit adapter.
