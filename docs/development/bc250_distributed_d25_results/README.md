# D25: native FP16 FSDP Trainer scaling and exact restart

Completed 2026-11-08. Implementation/verifier `36a215d8c`.
[Verified results](verified-results.json),
[usage](../../../../BC250_DISTRIBUTED_GUIDE.md#trainer-fsdp-fp16-working-parameters-and-scaling-d25).

Explicit `fsdp_fp16_parameter_policy()` extends the source-pinned
BC250FSDPTrainer: FP16 working weights/all-gathers, FP32 master shards,
synchronized gradients/reductions/buffers/AdamW, positive finite root L2 clipping,
enabled native world-group ShardedGradScaler. Trainable incoming FP32 parameters
required. Existing FP32/BF16 policies and old staged helpers retained. Additional
Accelerate optimizer.py/utils/modeling.py source pins protect the audited scaler
selection/step/skip path. No native Torch/ROCm build or prefix/package change.

Per-instance clipping validates gradient format and equal positive scaler state,
calls public Accelerator unscale exactly once, then public root-FSDP clipping
for finite gradients. Native ShardedGradScaler synchronizes overflow at unscale;
known-overflow windows skip root clipping and let native optimizer/scaler logic
skip AdamW/back off. Native Trainer suppresses scheduler advancement on a skipped
update. No private scaler flags, invented skip decision or loop fork. A public
AdamW pre-hook rechecks gradients/scaler immediately before an actual update.
It is installed at scheduler creation, which the pinned Trainer actually calls.
FP16 requires AdamW and positive clipping; other existing restrictions remain.

## Numerical and replay evidence

Same owned eager BertMLM22561/root+two BertLayer units/world2/RCCL/FULL_SHARD/
use_orig_params/dropout0, native gfx1013 per board. Workers0/sequential/micro2/
accum2, right padding and unequal MLM target counts yield global8/8/8/final4.
Native GPU Linear weights/output and training-only parameter gathers are FP16;
post-sync master/gradient shards/reductions FP32. Owned parameter/AdamW elements
11284/11277; FP32 compatibility control11282/11279. Unsynchronized accumulation
may temporarily hold FP16 gradients.

At each finite window CPU FP16 parameter/autocast references replay actual rank
microbatches with scaled FP16 local gradient accumulation, then FP32 averaging/
unscale. CPU FP64 evaluates the global window at FP16-rounded working weights.
Same HF graph/different CPU arithmetic, not an independent Transformer. D15 limits
unchanged: loss atol/rtol.001/.001; gradient relativeL2<=.004 CPU FP16/<=.006 FP64;
every parameter error norm<=bound*reference_norm+3e-6*sqrt(elements). Maximum
observed relative errors **.001115321 / .000713187**. No relaxed bounds.

Native global norm/clipping agrees with CPU FP64 actual-gradient predictions at
D22 limits1e-7/1e-5, every clipped gradient1e-8/5e-5. Cap.05 active, inactive100
coefficient1. CPU double AdamW restores original FP32 masters and consumes
predicted clipped GPU gradients; ALL parameters/moments/counters pass D21/D22
bounds. This isolates optimizer arithmetic from admitted FP16 approximation.

Stress fixture initial128/growth_interval2: four finite updates grow scale
128→128→256→256→512. Separate rank1-owned Inf is inserted AFTER reduction at
window3; raw flags[False,True], native shard-aware unscale makes BOTH ranks skip
AdamW and scheduler, with EXACT unchanged master parameters/optimizer/moments/
scheduler; scale256→128. Window4 is finite, executes successful update3 and scale
remains128/growth_tracker1. Data/RNG and Trainer global_step advance on the skip.
Native production scaler defaults65536/growth2000 remain unchanged; stress
settings are not a performance recommendation or proof of every scale policy.

Native CP2, complete DCP payload on BOTH independent filesystems, original local
Trainer/scheduler/scaler/rank RNG hashes retained. Six payload files864357B for
finite/864365B for overflow fixture. Fresh containers/UUID/startticks/constructor
seed1773/scaler8 restore ALL master shards/AdamW/scheduler/scaler256/four RNG
states EXACT. Both finite and overflow windows3/4 replay ALL gradients, losses,
clip/norms, parameters/moments/counters, scheduler/scaler, RNG draws, data rows and
transcript/final-state digests EXACT. No stochastic/shuffle/multiworker/elastic/
partial-window or changed-policy restore proof. Native aggregate train_loss is
still a current-process statistic, not restored cumulative loss.

Strict FP32 and BF16-working-parameter compatibility controls pass. Rank1 NaN
introduced AFTER clipping is refused on BOTH ranks before AdamW with zero actual
updates and exact unchanged parameters/optimizer/scheduler. Rank1 scaler
mismatch before unscale is also collectively refused with zero updates. Aborted
sessions must be discarded/reloaded; gradients/RNG are not rolled back. Caller
hooks must not mutate state after the gate, reorder collectives or replace/
clear optimizer/scaler internals; arbitrary callback/rank-loss recovery unqualified.

## Failures retained and final status

R1 finite math/optimizer checks pass, but both guards fail a probe's wrong
expected scale256 at fourth finite update; correct native scale512 accepted
with fixed counter formula. R2 late-NaN test catches an adapter defect: the pinned
Trainer bypasses create_optimizer_and_scheduler, leaving the first gate
uninstalled. Isolated fixture AdamW then receives NaN, and the CPU parameter gate
rejects both runs. Fixed installation at actual create_scheduler, unwrapping the
prepared optimizer; successful final windows assert gate identity and gate-call
count equals actual AdamW update count. All R3 phases rerun with the fixed helper,
including successful late-mutation refusal. No failed probe promoted to passed.

Final **128 accepted GPU groups /18 acknowledged guards**,8 CPU contract patterns
observed144times,58 local regressions/Black/critical Ruff+F401/py_compile/diff pass.
Four rejected guards retained, plus10 intermediate R2 passed guards; all32 guards
acknowledged. No full upstream CI. Native FSDP/DCP warnings remain visible.

Both hosts retain tools/distributed-d25, historical-source R1/R2 snapshots,
evidence/distributed-d25-{r1,r2,r3} and all checkpoint/reference payloads. Nine
final phases/node: fp16 seed/resume, overflow seed/resume, inactive, FP32, BF16,
late refusal, scaler refusal. Each final.json records exact command/PID/container/
boot/registry/kernel checks, with rank.json and recovery state/ACK. Final seed
13649/18301, controls14118/18778, finite-resume16256/20940, overflow-resume16657/
21347 all exit0. Ordered control commands/PIDs/exits in controls-sequence.json;
controllers and historical failures retained. All final kernel logs/registries
unchanged, default Docker/Podman idle, no remaining jobs/containers/armed guards.

SH runtime locks,4CPUs/cpuset0–3/10GiB RAM/12GiB RAM+swap/512PIDs/256MiBshm,
600s recovery/460s child retained. Entry bootsafcc7be4(.104r2)/8c396bd1(.105r3)
unchanged. No native build/install/prefix mutation/reboot/download/user-app
inspection/foreign stop; native daily/RCCL Torch a524/D09/ROCm/kernels/fallbacks/
packages/defaults unchanged. Historical Python CPU/KFD faults remain distinct
from fixed VM heap/D05 exit/D21 optimizer-load repair.

Metadata archives exclude tensor/RNG binaries; original payload inventories/hashes
and synthetic owned source retained, no private user data/application exported.
Local kernel/artifacts/distributed-d25 holds source variants, metadata, exports
and completed unions. Archives:

- .104:1024387B, SHA256 `c2f26b9706b7077c901f2b5a03c29f120be6ebf8b59740456806e1d56b93b0f1`.
- .105:873798B, SHA256 `b1a6466e512b331e7fe4eef7e03d9c1734c637770edddbbcc01d0d6772b4df59`.

**Next D26 (planned):** actual Trainer classic-FSDP eager causal-LM/tied-parameter
readiness using owned GPT2, shifted global target normalization, CPU ALL gradients/
clipping/AdamW and exact restart under explicit FP32/BF16/FP16 policies. D16 DDP
coverage is not FSDP proof. Frozen/PEFT, FSDP2, offload, compile, activation
checkpointing, elastic recovery and real larger-model capacity remain separate.
No speed, total-network, peak-memory, general convergence or universal-model claim.
