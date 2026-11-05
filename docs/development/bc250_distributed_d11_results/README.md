# D11: native Accelerate two-host DDP and checkpoint replay

Completed 2026-11-05. Fixture21766d81d, export/verifierf5329413b. The installed
D09 public launcher, native Torch a524980/ROCm/images/registry/profiles/kernels
remain unchanged. No build/install/reboot/user-application access. Accelerate1.12.0
and NumPy2.5.1 are the existing frozen consumers, not new package installations.
[Usage and settings](../../../../BC250_DISTRIBUTED_GUIDE.md#accelerate-on-two-hosts-d11).

## Actual integration and numerical checks

Accelerator discovers MULTI_GPU/world2 with one native gfx1013 GPU/node and NCCL
(AMD RCCL). Actual prepare() wraps the model, AdamW, sequential DataLoader and
StepLR; actual accumulate()/backward() control no_sync, loss division, optimizer,
zero_grad and scheduler gates. Three cases: FP32(no), BF16 and FP16 autocast.
The owned residual MLP has1,051,648 parameters, four256/512 blocks,16 tokens/example.
No user model/application data is accessed.

DataLoader batch2 is split globally into one example/rank, four micros per global8
window. All32 records appear once. A DDP communication hook sees zero gradient
reduction on the first three micros and4,206,592 FP32 gradient bytes on the fourth.
The full mean loss goes to Accelerator.backward without another division by4.
Independent CPU-double loss/all-gradient references qualify FP32; low precision
uses the retained D07 explicit operand/output/gradient rounding oracle with FP64
matrix arithmetic. Two windows/case compare every gradient and CPU-double AdamW
parameters/moments. Fixed D07 limits: FP32 grad atol3e-6/rtol5e-4, BF16 relative
L2<=.02, FP16<=.004. Maximum observed low-precision gradient relative L2 across
both ranks/windows: BF16.00225232, FP16.000435060. No tolerances relaxed.

Explicit split_batches=True, GradientAccumulationPlugin(num_steps4,
adjust_scheduler=False) make StepLR advance once per actual optimizer update.
A raw optimizer post-step hook and independent LR formula check this, including
calls to prepared step/zero_grad/scheduler on every normal micro. StepLR(step2,
gamma.8), AdamW lr.001/eps1e-5/weight_decay.01/foreachFalse. At the third window,
rank1 injects Inf before the final DDP reduction: both ranks observe nonfinite
reduced gradients. Native FP16 scaler skips the update and scheduler; FP32/BF16
explicitly skip both in the fixture after a collective finite check. Accelerator
alone does not automatically guard FP32/BF16 nonfinite steps. Fourth window
recovers: final3 updates/16 micros, LR.0008. FP16 scaler16→16→32→16→16,
final growth tracker1. Strict deterministic algorithms are fixture-only.

## Native save/load and exact fresh-process replay

After8 micros/two updates, BOTH ranks call actual Accelerator.save_state with
ProjectConfiguration(save_on_each_node=True): independent host filesystems need
model/optimizer/scheduler files on each node. Checkpoints contain native safetensors,
optimizer.bin, scheduler.bin, rank-specific random_states pickle, registered
Progress object and FP16 scaler where applicable. Trusted owned native format;
not arbitrary untrusted checkpoint qualification or a new installed checkpoint API.

Seed processes continue two reference windows, including skipped update/recovery.
New containers/processes construct objects from different seeds, validate config,
rank/cursor/file sizes/SHA collectively, then call actual load_state. Four one-rank
bad metadata variants (config/cursor/size/hash) are refused by BOTH live ranks
without state mutation,24 collective refusal observations total. This is a fixture
admission wrapper; native Accelerate's own loading is not atomic or crash-proof.

The sequential workers0 skipped loader is materialized and its first batch
prefetched BEFORE load_state, because iterator initialization consumes CPU RNG.
Native load_state then restores RNG last; its RNG loader can log and suppress
errors, so the fixture explicitly checks exact restored RNG rather than trusting
a successful return. Initial model/AdamW/scaler/scheduler/Python/NumPy/Torch CPU/
CUDA RNG/custom Progress/Accelerator step all match saved state exactly.
Two subsequent windows/case/rank match local loss, all gradients/parameters,
optimizer/scaler/scheduler/RNG/augmentation/indices/cursor exactly (zero tolerance
plus digests):12 resumed rank-windows. Distinct container IDs, worker UUIDs and
start ticks prove fresh launches. Seed/resume final digests match per rank.

## Evidence, operation and limits

78 accepted check groups/12 GPU guards/24 collective refusal observations,
42 local regression tests. Black/critical Ruff+unused imports/syntax/diff pass;
full upstream CI not run. SH runtime locks,4CPUs/cpuset0–3,10GiB RAM/12GiB
memory+swap,512PIDs/256MiB shm,600s recovery guards/460s child deadlines.
All guards acknowledged, agents exit cleanly, kernel logs/registries unchanged.
.104 r2/.105 r3, existing boot IDs7344171f/e1cbfa23. No owned jobs/containers/
armed guards remain. Sequence controllers26401/11021 exit0, per-phase commands/
PIDs/exits in sequence.json. FP32 seed25559/10127 and resume25995/10583 exit0.

Both hosts retain tools/distributed-d11 and evidence/distributed-d11-r1 with
{no,bf16,fp16}-{seed,resume}. Valid native checkpoint/reference data177,094,758B
(.104)/177,094,566B(.105) remain locally. Exports exclude .pt/.bin/.pkl/.safetensors
payloads, retaining metadata, source, reports and raw logs. Local workstation:
kernel/artifacts/distributed-d11; committed [verified summary](verified-results.json).

| Archive | Bytes | SHA256 |
| --- | --- | --- |
|104-evidence-r1.tar.gz|326808|b44001c1a03faa538a284d747de05780372f1a466c2a00e874317887f4c8a2c8|
|105-evidence-r1.tar.gz|292811|a62d43b21ec83c94d174f015467f5f57038e99ee302b4c4dab63fec66c607db8|

Scope: owned deterministic DDP graph, fixed world2, one GPU/node, full four-micro
windows, sequential workers0 loader and trusted optimizer-boundary checkpoint.
No Accelerator FSDP/HF Trainer/multiworker/shuffle/partial-window/elastic/world
migration/peer-loss/crash-during-save/large-capacity/convergence/speedup claim.
Historical Python CPU segfault/KFD uncertainty remains distinct from already
fixed VM-heap corruption and D05 agent lifetime repair. No new native fault found.

Next D12 (planned): actual Transformers Trainer two-host integration, owned offline
fixtures, bounded accumulation/state references; retain native/default/recovery
settings. Host documentation and agent-memory snapshots include this batch.
