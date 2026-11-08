# D27: explicit frozen FP32 FSDP adapter policy

Completed 2026-11-08, implementation/verifier `80080e6a0`.
[Verified results](verified-results.json),
[usage](../../../../BC250_DISTRIBUTED_GUIDE.md#frozen-fp32-fsdp-adapters-d27).

New opt-in `BC250FrozenFSDPTrainer` plus `frozen_fp32_policy(model)` admits a
named mixed frozen/trainable FP32 partition with positive root clipping. The
existing BC250FSDPTrainer STILL refuses frozen parameters. Its original parameter
and gradient checks were factored into methods without weakening the defaults;
new behavior is confined to the explicit subclass. Additive tools/distributed-d27
are installed on BOTH boards; older tools/providers/applications remain preserved.
No native Torch/ROCm kernel defect or performance improvement is claimed.

## Admission, gradients, optimizer and checkpoint contract

The immutable policy contains nonempty disjoint sorted canonical trainable and
frozen name sets. Both ranks agree on policy hash and original parameter shapes.
Every parameter starts FP32; use_orig_params/classic FULL_SHARD/two ranks/no
AMP/offload/compile/activation-checkpointing and pinned Transformers5.2.0 /
Accelerate1.12.0 restrictions remain. Policy/partition drift is refused.

All owned nonempty TRAINABLE original shards need present finite dense FP32
gradients. Frozen optimizer-facing original shards must have grad=None. Native
root-FSDP global L2 clipping still receives the complete root parameter list.
AdamW must contain exactly the trainable Parameters once, with no frozen state.
The actual create_scheduler path installs a public AdamW pre-hook which rechecks
partition, gradients, optimizer membership/state and completed clipping before
an update. It does not patch packages or alter FSDP backward/flattening.

Native checkpoint2 gains bc250-frozen-fp32.json, recording schema/precision,
canonical partition/hash and original shapes. Restoration collectively validates
this metadata BEFORE restoring optimizer/scheduler state. Model shards may
already have loaded before refusal: discard an aborted session. Changing only
rank1's PRIVATE metadata copy makes BOTH ranks refuse with zero AdamW updates
and unrestored optimizer/scheduler. All original checkpoint hashes stay intact.

Public FSDP summon_full_params(with_grads=True) temporarily exposes gradient
views for frozen entries of mixed flat buffers. These are different from
optimizer-facing frozen shards; they are excluded from adapter references and
optimizer membership. R1's probe incorrectly expected None inside that context
and fails on both boards BEFORE any update. Inspection of the patched Torch
_flat_param.py confirms native full-view versus sharded-view behavior. R2 fixes
the probe and passes; R3 adds final immutable-policy/save/restart guards and
reruns all selected phases. No numerical bound was relaxed or kernel patched.
Frozen values remain bitwise equal in both contexts; ordinary sharded-gradient
refusals remain strict. Mixed buffers can still compute/store/reduce base-sized
gradients, so freezing does not establish proportional gradient-memory/traffic
savings.

## Actual two-host reference and refusal results

Owned eager GPT2, tied frozen embedding/head, root+two GPT2Block wraps; native
rank4 adapters around eight Conv1D projections implement base(x)+2*up(down(x)).
Up matrices start zero. ALL21,280 base parameters are frozen; ALL3,072 adapter
parameters train, out of24,352 unique parameters. This is an owned native-module
fixture, not PEFT-package qualification. Each node has12,176 original parameter
elements; trainable/AdamW elements are1,504 and1,568, sum3,072. Mixed FSDP unit
sharding does not give equal adapter shards or adapter-only communication.

D26 owned28examples/right padding/prompt masks/eager/cacheFalse/dropout0/workers0/
sequential/micro2/accum2 gives global8/8/8/final4 and shifted targets49/57/57/21.
An actual rank0 micro has zero local targets and a positive global denominator.
All finite global losses and ALL adapter gradients match CPU FP64 at unchanged
FP32 bounds: loss3e-6/5e-5 and gradients3e-6/5e-4. ALL frozen parameter values
stay EXACTLY equal to the initial CPU snapshot at every inspected step.

Native global norm and clipping at.005 match CPU FP64 actual-gradient predictions
(norm1e-7/1e-5, every clipped entry1e-8/5e-5). CPU double AdamW consumes predicted
clipped adapter gradients; every adapter parameter/moment/counter passes the
retained D21/D22 limits. Full native optimizer state contains exactly adapter
names. Frozen parameters receive no updates or optimizer state. Actual hooks,
FP32 parameter all-gathers and two FP32 reduce-scatter calls/window are observed.
Measured reduce-scatter INPUT is80,640B/window, versus only12,288B for3,072 FP32
adapter elements; flat buffers/padding remain. This is not a total NIC, memory
capacity or speed comparison.

Three rank1-only mutations are collectively refused on BOTH ranks with zero
AdamW updates, unchanged parameter/optimizer/scheduler values and no scheduler
advancement: missing adapter gradient before clipping, forbidden frozen gradient
before clipping, NaN AFTER valid clipping. Aborted gradients/RNG are not rolled
back; arbitrary callback replacement/mutation or rank-loss recovery is unqualified.
A retained ALL-trainable GPT2 FP32 control passes the original helper in this new
namespace. Fourteen causal/frozen CPU contracts plus eight existing policy
contracts exercise partition/AMP/disabled-clip/optimizer/default-refusal guards.

## Fresh exact restart and evidence

Native checkpoint2 DCP model/optimizer union has six files469453B on BOTH
independent filesystems. Existing bounded offline exchange only adds missing
peer payloads; original rank-local Trainer/scheduler/RNG/policy files are kept.
A fresh constructor seed1773 instead of773, new UUID/start ticks/container,
restores ALL frozen/adapter shards, adapter AdamW moments/counters, scheduler and
four RNG states EXACTLY. Windows3/4 reproduce ALL losses, gradients, norms/clip,
parameters/optimizer/scheduler, draws, data/target counts and transcript/final
state digests EXACTLY; tied frozen head stays shared. Native train_loss remains
process-local. No partial-window/stochastic/multiworker/elastic/changed-policy
or larger-model proof.

Final **97 groups/14 acknowledged guards**,22 CPU contract patterns observed294
times,62 local regressions/Black/critical Ruff+F401/diff pass. Retain2 failed R1
probe guards and2 intermediate passed R2 guards; all18 acknowledged, all kernel
logs/D01 registries unchanged. No full upstream CI. Native warnings remain.

Exact commands/PIDs/CIDs/ACKs: evidence/distributed-d27-r3/frozen-{seed,resume},
frozen-{missing,grad,late}-seed, retained-control-seed and frozen-policy-resume;
controls-sequence.json/resumes-sequence.json and controller logs. The extra
frozen-policy-seed folder is a private COPY used for refusal, not another run.
R1 controllers21610/5214 exit1; R2 22748/5962 exit0; R3 seed24866/6696,
controls25338/7199, resumes26837/8794 and exports28532/9786 exit0.
Historical sources/R1–R3/reference tensors/checkpoints remain on hosts. Local
kernel/artifacts/distributed-d27 preserves overlay, exports/complete union and
metadata (tensor/RNG payloads excluded):

- .104:694604B SHA256fec317a80cac76bb74f99b65142e18fa3a605d15825fc15ffac3d631c668527e.
- .105:607645B SHA2563690917d572464734d73bc2cb57f90358218ebc2039d90d060256c4be1283767.

PLAN-01 EX cooperative board reservation+SH runtime locks/4CPUs/cpuset0–3/
10GiB RAM/12GiB RAM+swap/512PIDs/256MiB shm/600s recovery/460s child/15s termination
retained. Final idle preflight/no jobs/containers/armed guards. .104 entryboot
is afcc7be4/r2; .105 enters D27 at570437ed/r3, different from D26's historical
8c396bd1. No agent reboot or boot-policy change occurred in D27. PLAN-02 native
MIOpen381eac05/Torch a524/RCCL/D09/image/package/registry/default/fallbacks remain.
No native build/prefix update/download/app access/foreign stop. Historical Python
CPU faults remain distinct from fixed VM heap/D05 exit/D21 loading.

PLAN-04 remains OPEN. **Next D28 (planned):** inspect the pinned PEFT package's
actual LoRA/FSDP naming/optimizer/save-load paths and qualify an owned actual
PEFT FP32 fixture, retaining current package/default/frozen-AMP guards. Mixed
precision and activation-checkpointing combinations follow their own proof;
real capacity and performance remain PLAN-05/PLAN-10.
