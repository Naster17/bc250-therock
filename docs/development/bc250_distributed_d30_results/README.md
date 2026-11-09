# D30: explicit BF16 PEFT FSDP working parameters

Completed 2026-11-09. Implementation/verifier `e8c6d120c`.
**116 groups/16 acknowledged guards**,28 CPU contract patterns/444 observations.
[Verified results](verified-results.json).
[Usage](../../../../BC250_DISTRIBUTED_GUIDE.md#bf16-peft-working-parameters-with-fp32-masters-d30).
Additive BOTH-host `~/bc250work/tools/distributed-d30`; native providers unchanged.

New `BC250BF16PeftFSDPTrainer` inherits the D28 plain default LoRA/full native
checkpoint/public adapter export/partition/AdamW/root-clip contract. It explicitly
requires BF16 working parameters, FP32 master shards/reductions/buffers/optimizer
state, native BF16 autocast without a scaler, and checkpointing OFF. Original
frozen and trainable parameters must be FP32 before wrapping. The original frozen
helper's AMP refusal is factored into an overridable method; its default refusal
and existing installed helpers remain. Compute-only BF16, low-precision reductions
or retained gradients, FP16 and checkpoint combinations remain refused here.

The class binds precision and three pinned native modeling source hashes in the
full checkpoint. Actual admission requires eager GPT2/dropout0/cacheFalse and no
pre-enabled model checkpointing. Existing plain PEFT source/recipe checks remain.
The inherited collective clip and AdamW gates also check current args/plugin
precision/native autocast/no-scaler and checkpointing state. No package-global
patch, Trainer-loop fork, native rebuild or default policy change.

## Executed numerical scope

Same owned D28 GPT2/eight actual PEFT LoRA Conv1D projections/rank4/alpha8/zeroB,
24,352 unique parameters=21,280 frozen+3,072 trainable. Tied frozen embedding/head,
19 native PEFT FSDP units, original shards12,176 each and adapter/AdamW shards1,536
each. Right padding/prompt masks/28records/workers0/micro2/accum2/global8,8,8,4,
shifted49,57,57,21 and an actual local-zero-target micro are retained.

The shared test harness now admits the explicit frozen-BF16 fixture and constructs
its AMP oracle/gradient lists over ALL trainable names, excluding frozen parameters.
Its existing all-trainable and FP32 paths retain the same math. GPU BF16 loss and
ALL adapter gradients pass CPU BF16 rank-local accumulation plus CPU FP64 at
BF16-rounded working weights, at ORIGINAL D25/D26 BF16 bounds: global/per-parameter
relative norm gates.02/.03 with the original absolute floor; loss.003/.003.
Maximum global gradient relative error is.00760193 versus CPU BF16 accumulation
and.00783804 versus rounded-weight CPU FP64. These are not FP32 bitwise accuracy
or comparisons to unrounded FP32 compute.

Native complete-root clipping passes CPU norm/ALL clipped-entry predictions at
D22's1e-7/1e-5 norm and1e-8/5e-5 entry bounds. CPU-double AdamW consumes predicted
clipped GPU gradients and checks ALL adapter values/moments/counters at unchanged
D21/D22 limits. ALL frozen master values remain EXACT and absent from AdamW state.
Actual hooks observe BF16 working weights/outputs, BF16 parameter allgathers and
FP32 reduction/master/optimizer-facing gradients; no scaler is present.

Matched seed allgathers retain259calls/identical input element sequence to D28,
but input bytes halve623,168→311,584. Replay has111gathers. Reduction input remains
16calls/12,288B per window. This is a parameter-gather input measurement, not total
NIC traffic, whole-memory capacity or a training speedup. No timing claim.

## Checkpoints and refusals

Full native checkpoint2 has six complete DCP model/optimizer files480,247B on both
independent filesystems after the bounded offline union. Original local hashes are
preserved; local scheduler/Trainer/RNG/recipe files stay local. Fresh constructor
seed1773 instead of773/new UUID/start ticks/containers restores ALL model/AdamW/
scheduler/four RNG states exactly. Windows3/4 replay ALL losses/adapter gradients/
clip/updates/state/RNG/rows/counts/digests EXACT within this same BF16 policy.
Native train_loss stays process-local; no cross-precision replay promise.

Ordinary PEFT artifacts at checkpoints2/4/replay4 load ALL parameters/logits EXACT
on matching CPU base, with unchanged four RNG states. These exports require that
base and do not replace full training checkpoints.

Rank1-only missing adapter gradient, injected frozen gradient, post-clip NaN and
post-clip LoRA scaling drift are collectively refused before AdamW, zero updates
and EXACT unchanged parameters/optimizer/scheduler. A PRIVATE checkpoint copy
changes only rank1's saved reduction precision to BF16; BOTH refuse before
optimizer restoration. Original checkpoint bytes remain intact. Model shards may
already have loaded; discard/reload aborted sessions. Retained D28 FP32 PEFT
helper control and CPU tests retaining original frozen AMP refusal pass.

## Evidence and limits

Exact commands/PIDs/CIDs/ACK/kernel logs in evidence/distributed-d30-r1/
frozen-{seed,resume},frozen-{missing,grad,late,peft}-seed,retained-control-seed,
frozen-policy-resume; controls-sequence.json/resumes-sequence.json/controllerlogs.
frozen-policy-seed is a PRIVATE COPY, not an additional test. Source overlays,
checkpoint/reference tensors and public artifacts remain on hosts. Local
kernel/artifacts/distributed-d30 retains source overlay/payload exports/complete
union/metadata archives; tensor/RNG/safetensors payloads are excluded from metadata.
Archive hashes, guards, group counts and errors are in verified-results.json.
.104 metadata681233B SHA0044f8e1acb0380ee4fee0e2a1e0ac2640137f6843dc24b63f1113bdb12ea688;
.105 metadata611707B SHAf2e2b1a9abc07a1d11420aa44924c3e8219d4cb6c019983d8ef4a66aa0e28a09.
Guardian/controller PIDs(.104/.105): seed28859/7650, controls29377/8341,
resumes31579/11167,exports32560/12492, all exit0. No failed GPU phase or relaxed
bound.62 local regressions/Black/critical Ruff+F401/diff pass. No full upstream CI.
Final idle preflight passes/no jobs/containers/armed guards; kernel logs/registries
unchanged. Docs/flow/manifest snapshots synchronized to both host docs entry points.

PLAN-01 EX board reservations+SH runtime locks/4CPUs/cpuset0–3/10GiB RAM/12GiB
RAM+swap/512PIDs/256MiBshm/600s recovery/460s child/15s termination retained.
Boots2d86d523(.104r2)/e902e636(.105r3) unchanged; PLAN-02 MIOpen381eac05/Torch a524/
RCCL/D09/image/packages/defaults/fallbacks retained. No build/download/reboot/
prefix mutation/app inspection/foreign stop. Historical Python CPU fault remains
separate from fixed VM heap/D05 exit/D21 optimizer loading.

No universal PEFT/model accuracy, stochastic/elastic recovery, larger-fit,
convergence, speed, FP16, other adapter variants or BF16+checkpoint qualification.
PLAN-04 remains OPEN. Next D31 (planned): combine explicit BF16 PEFT with native
non-reentrant activation checkpointing; retain refusal until CPU ALL gradients/
AdamW/frozen-preservation/exact-restart proof. Meaningful capacity is PLAN-05.
