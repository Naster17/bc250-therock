# D31: combined BF16 PEFT FSDP and activation checkpointing

Completed 2026-11-09; implementation `725e495c8`; final verifier maintenance `6c4758c18`.
**163 groups/18 acknowledged guards**,30 CPU contract patterns/536 observations.
[Verified results](verified-results.json).
[Usage](../../../../BC250_DISTRIBUTED_GUIDE.md#combined-bf16-peft-and-activation-checkpointing-d31).
Additive BOTH-host `~/bc250work/tools/distributed-d31`; native providers unchanged.

New `BC250BF16CheckpointPeftFSDPTrainer` inherits D29's native non-reentrant
checkpointing enable/function guards/full checkpoints/export and D28's named
frozen partition/root clipping/trainable-only AdamW. It admits D30's explicit
BF16 working copies/gathers with FP32 master shards/reductions/buffers/optimizer
state, native autocast/no scaler. D30's precision validation/runtime predicate is
factored for reuse, with checkpointing OFF still the default. Existing installed
D29/D30 helpers and their restrictions are preserved.

Both policies are explicit and bound to saved metadata. The new class requires
pinned plain default LoRA/eager GPT2/dropout0/cacheFalse/FP32 original parameters;
model checkpointing initially OFF. Native Trainer enables it before FSDP wrapping,
including the frozen-embedding output-gradient hook. Reentrant/RNG-disabled kwargs,
plugin checkpoint wrapping, low-precision reductions, FP16 and other unqualified
combinations remain refused. No global package patch or Trainer-loop fork.

## Executed references and actual recomputation

Same owned24,352 unique parameters=21,280 frozen+3,072 actual PEFT LoRA weights,
tied frozen embedding/head,19 FSDP units/original shards12,176 each/AdamW1,536 each.
D26 data28/right padding/prompt masks/micro2/accum2/dropout0/cacheFalse/workers0,
global8,8,8,4 and shifted49,57,57,21/local-zero-target micro are retained.

ALL adapter gradient/loss references pass the unchanged D25/D26 BF16 gates versus
CPU BF16 local accumulation and CPU FP64 at BF16-rounded weights. Native root
norm/ALL clipped gradients and CPU-double ALL AdamW parameters/moments/counters
pass unchanged D21/D22 bounds. ALL frozen master values stay EXACT. Actual hooks
observe BF16 working weights/outputs/gathers and FP32 master/reduce/optimizer-facing
gradients, with no scaler. Public adapter exports CP2/4/replay4 load ALL parameters/
logits exactly on the matching CPU base and leave four RNG states unchanged.

GPU GPT2 block prehooks observe14calls/block over seven seed micros and6 over
three replay micros: forward plus real backward recomputation. Matching BF16
checkpointing-off control observes7/block. Both the in-batch control and the
hash-verified published D30 archives have EXACT same seed training transcript and
final-state digest. No tolerance was relaxed to admit the combination.

Native BF16 gathers259→357(seed),111→153(replay) versus checkpointing off; FP32
reduction input remains16calls/12,288B per window. Gather input is not total NIC
traffic. The extra gathers are a recomputation cost, not evidence of a speedup.
The matched tiny full-harness peak live allocation is459,264B checkpointed versus
499,200B off on EACH rank (39,936B/8%lower). This includes audits/checkpoint work,
not isolated activation or whole-host memory. Global gradient relative error
maxima remain.00760193/.00783804 versus CPU BF16/rounded-weight FP64.

## Restart and refusal coverage

Native checkpoint2 DCP union has six model/optimizer files480,247B on each host;
original local hashes and local scheduler/Trainer/RNG/recipe files are preserved.
Fresh constructor seed1773 instead of773/new UUID/start ticks/containers restores
ALL model/AdamW/scheduler/four RNG states EXACT. Windows3/4 replay ALL losses/
gradients/clip/updates/state/RNG/rows/counts/digests EXACT within this policy.
Native train_loss remains process-local; no cross-policy replay promise.

Rank1-only missing adapter gradient, forbidden frozen gradient and post-clip NaN
are collectively refused before AdamW, zero updates/EXACT unchanged parameters,
optimizer and scheduler. The inherited frozen-peft case now deliberately changes
rank1's LIVE checkpoint function preserve_rng_state option after clipping, rather
than LoRA scaling: BOTH refuse before AdamW with the same unchanged-state proof.
A PRIVATE saved checkpoint changes only rank1's use_reentrant flag; BOTH refuse
optimizer restoration. Original checkpoint bytes remain intact. A model may
already load before refusal; discard/reload aborted sessions. Retained D30
checkpointing-off BF16 helper control and default-refusal CPU contracts pass.

## Evidence, resources and limits

Exact commands/PIDs/CIDs/ACK/kernel logs in evidence/distributed-d31-r1/
frozen-{seed,resume},frozen-{missing,grad,late,peft}-seed,retained-control-seed,
frozen-policy-resume,allocator-control-seed; controls-sequence.json and
resumes-sequence.json/controller logs. frozen-policy-seed is a PRIVATE COPY, not
an extra test. Original sources/checkpoints/reference tensors/public artifacts
remain on hosts. Local kernel/artifacts/distributed-d31 retains source overlays,
payload exports/complete union/metadata archives/local-validation-environment.txt.
Metadata excludes tensor/RNG/safetensors payloads; hashes are in verified results.
.104 archive779548B SHAee83847c206e86f03657008e6121c0bd883a247f40021dbb4acbbd75063232b2;
.105 archive689055B SHA2a4190af18c58f9a0a079afc606eaede0a0c6069c45fff97939cdef4a04ed2dd.
Guardian/controller PIDs(.104/.105): seed6265/23444, controls6850/24260,
resumes9014/27004, allocator9961/28238,exports10451/28891, all exit0.
No failed GPU phase;62 local regressions/Black/critical Ruff+F401/diff pass.
Final verifier maintenance6c4758c18 replaces current-source comparisons with
immutable qualified D30/D31 source hashes. Replaying BOTH archived qualifications
produces byte-identical original verified JSON. No GPU rerun or bound change.
Final verifiers installed in D31 namespace AFTER metadata export; committed final
sources are in Git, original verifier variants remain in archives/namespaces.
Final idle preflight passes/no jobs/containers/armed guards; kernel logs/registries
unchanged. Docs/flow/manifest snapshots synced to BOTH host docs entry points.

Initial workstation temporary quota stopped shell heredocs before source changes;
workspace writes worked. Routing tests through /dev/shm then correctly triggered
seven launcher mount-guard errors; an owned workspace TMPDIR passes all62 tests
without weakening guards. No caches/builds/user files deleted. These setup errors
are separate from GPU/native results and historical Python CPU faults.

PLAN-01 EX board reservation+SH runtime locks/4CPUs/cpuset0–3/10GiB RAM/12GiB
RAM+swap/512PIDs/256MiBshm/600s recovery/460s child/15s termination retained.
Boots2d86d523(.104r2)/e902e636(.105r3), PLAN-02 MIOpen381eac05/Torch a524/RCCL/D09/
image/packages/defaults/fallbacks retained. No native build/download/reboot/prefix
mutation/app inspection/foreign stop. Historical Python CPU fault stays separate
from fixed VM heap/D05 exit/D21 loading. No full upstream CI/universal-model/
large-fit/speed/convergence/stochastic/elastic qualification.

This completes PLAN-04's bounded frozen/PEFT/checkpointing memory-feature scope.
FP16 and broader recipes remain separately unqualified. Next PLAN-05 / D32:
inspect retained sizing/timing evidence and implement conservative owned-model
one-board versus two-board DDP/FSDP measurement, matched batch/sequence/precision,
with allocator/host-RAM/swap headroom and bounded warm-update/whole-job timing.
Do not deliberately OOM or claim larger fit from the tiny D31 fixture.
