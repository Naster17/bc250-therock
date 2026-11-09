# D29: explicit FP32 PEFT activation checkpointing

Completed 2026-11-09; implementation/verifier `8e04fa9b4`.
[Verified results](verified-results.json),
[usage](../../../../BC250_DISTRIBUTED_GUIDE.md#fp32-peft-activation-checkpointing-d29).
Additive BOTH-host `~/bc250work/tools/distributed-d29`; native stack unchanged.

New `BC250CheckpointPeftFSDPTrainer` admits native Trainer checkpointing with
explicit `use_reentrant=False, preserve_rng_state=True`. It inherits D28's plain
single-default LoRA/FP32/partition/AdamW/root-clip/full-checkpoint/export contract.
The ordinary helper's checkpoint refusal is factored into an overridable method;
its default refusal remains, as do D27/D28 and frozen-AMP restrictions. FSDP
plugin activation wrapping, compile, offload and FSDP2 remain refused.

Audit of installed Transformers5.2/PEFT0.18.1/Accelerate1.12 source confirms native
Trainer enables checkpointing before FSDP wrapping; Transformers' GPT2 block
calls its checkpoint function around the module call, including module hooks.
Native enable also installs the frozen embedding output-gradient hook. No manual
embedding hook or replacement Trainer loop is needed. Three additional native
modeling source hashes are checked. The explicit class initially requires eager
GPT2/cache disabled/checkpointing off, validates exact boolean kwargs, and checks
actual enabled module checkpoint functions/options in inherited pre-update guards.
Full checkpoint metadata binds these options/source hashes. Qualified scope is
owned zero-dropout deterministic GPT2; accepting its type is not qualification
of arbitrary configurations, stochastic recovery or other architectures.

## Executed references

Same owned D28 model: 24,352 unique parameters, 21,280 frozen and 3,072 actual
PEFT LoRA A/B entries; tied frozen embedding/head, 19 native PEFT FSDP units.
Original parameter shards12,176 each, trainable/AdamW shards1,536 each. Right
padding/prompt masks/28 records/micro2/accum2/global8,8,8,4 and shifted target
counts49,57,57,21, including the local-zero-target micro, are retained.

ALL adapter gradients/global loss, native root norm/ALL clipped gradients and
CPU-double ALL AdamW values/moments/counters pass D28's unchanged FP32 bounds.
ALL frozen values remain EXACT. Public PEFT exports at checkpoints2/4/replay4
reload ALL parameters/logits exactly without altering four RNG states.

GPU block prehooks observe14 calls/block across seven seed microbatches and6
across three replay microbatches: actual forward plus backward recomputation.
Matching checkpointing-off control observes7/block, with EXACT same training
transcript and final-state digest. Native FP32 allgathers rise259→357 for seed
and111→153 replay. Reductions remain16/window,12,288B input/window; this does
not measure total NIC traffic or establish a training speedup.

Fresh seed1773 instead of773/new UUID/start ticks/containers restores ALL model,
AdamW/scheduler/four RNG state EXACT from complete native checkpoint2. Windows3/4
replay ALL losses/gradients/clip/updates/state/RNG/rows/counts/digests EXACT.
The separate-host offline DCP union contains six files480,247B on each host;
original rank-local files/hashes are preserved. Native train_loss is process-local.

Rank1-only missing adapter gradient, forbidden frozen gradient, post-clip NaN
and post-clip LoRA scaling drift are collectively refused before AdamW, zero
updates and EXACT unchanged parameters/optimizer/scheduler. A PRIVATE checkpoint
copy changes only rank1's saved use_reentrant flag; BOTH refuse before optimizer
restoration. Original files remain intact. Model shards may already load before
refusal; discard the aborted session. Retained D28 ordinary PEFT control passes.

## Bounded allocator comparison and limits

Whole tiny correctness harness, including audits and checkpoint work, peak live
allocation per rank: checkpointed491,008B versus off546,304B (55,296B /10.12%
lower); identical result on both boards. Reserved peak is separately recorded.
This is not isolated activation storage, whole-host memory or a useful large-model
capacity proof. No timing/speed/larger-fit/convergence claim. PLAN-05 still needs
meaningful safe-headroom model sizing and matched one/two-board measurements.

**163 groups/18 acknowledged guards;28 CPU contract patterns/500 observations.**
62 local regressions, Black, critical Ruff+F401 and diff checks pass. No GPU phase
failed. Initial SCP to root-owned evidence was refused; premature launch failed
before guard/GPU setup. Source was then staged through the user's home into the
new namespace, preserving that failed setup log. No native package/provider fix
or full upstream CI claim. Verifier installed after metadata export; its final
source is committed locally and installed on both hosts.

Exact commands/PIDs/CIDs/ACK/logs in evidence/distributed-d29-r1/frozen-{seed,resume},
frozen-{missing,grad,late,peft}-seed,retained-control-seed,frozen-policy-resume,
allocator-control-seed; controls-sequence.json/resumes-sequence.json/controllerlogs.
frozen-policy-seed is a PRIVATE COPY, not another run. Guardian/controller PIDs
(.104/.105): seed22304/30336,controls22858/31101,resumes25019/1388,
allocator25904/2533,exports26385/3193, all final exit0.

Local kernel/artifacts/distributed-d29 retains source audit/overlays/payloads/
complete checkpoint union/metadata archives; tensor/RNG/safetensors payloads stay
in original host evidence, excluded from metadata archives. Archive hashes:

- .104:764130B, SHA256e0f1aa6a1bc129b09d98ad1b99c55f767ded26e13b1cd8a6c78b2e555e137ebe.
- .105:674130B, SHA25692ba18c33be2eee498a8838a775ee03f2255f610483090d459de669ddcd6bdc0.

PLAN-01 reservation+SH runtime locks/4CPUs/cpuset0–3/10GiB RAM/12GiB RAM+swap/
512PIDs/256MiB shm/600s recovery/460s child/15s termination retained. Kernellogs/
registries unchanged; final idle preflight passes/no jobs/containers/armed guards.
Boots2d86d523(.104r2)/e902e636(.105r3) unchanged. PLAN-02 MIOpen381eac05/Torch a524/
RCCL/D09/image/packages/defaults/fallbacks retained. No build/prefix mutation/
reboot/download/app inspection/foreign stop. Historical Python CPU fault stays
separate from fixed VM heap/D05 exit/D21 optimizer loading. Docs synced both hosts.

PLAN-04 remains OPEN. Next D30 (planned): inspect explicit BF16 working-parameter
PEFT frozen-base FSDP admission, checkpointing OFF first, then CPU ALL adapter
gradients/clip/AdamW/frozen preservation and exact restart. Keep existing frozen
AMP refusals until proof; combined BF16 checkpointing and FP16 remain separate.
