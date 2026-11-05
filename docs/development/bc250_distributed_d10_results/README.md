# D10: fresh-process fixed-world checkpoint/restart

Completed2026-11-05 on both hosts. Initial fixture553eebce1; corrected proof and
coordinated restricted-load handling0c0efa810; exports97641e4b6. Installed D09
launcher, native daily/RCCL Torch a524980/ROCm/images/registries/profiles/kernels
unchanged. No native build, package install, prefix mutation, reboot or app access.
This batch qualifies state continuity against an uninterrupted GPU reference;
it adds no new CPU mathematical kernel proof beyond the retained D08/D09 evidence.

## Cases and exact contract

| Case | Existing numerical base | New stochastic behavior |
| --- | --- | --- |
|DDP FP32|D09 tiny offline BertForMaskedLM22,561 parameters, frozen Transformers5.2.0|Hidden/attention dropout0.1, global8/two micros per rank|
|Classic FSDP FP16|D08 residual MLP1,051,648 params, Linear wrapping/use_orig_params, FP32 reduction/master shards|Output dropout0.125, global8/four micros per rank, native ShardedGradScaler|

Both use AdamW lr.001/eps1e-5/weight_decay.01/foreachFalse, StepLR(step_size2,
gamma.8), and strict torch.use_deterministic_algorithms(True) in the test worker.
FP32 DDP scaling is disabled; FSDP starts scale16/growth_interval2. These are
explicit fixture settings, not changed daily defaults or a speed recommendation.

Each rank exercises five RNG streams: Python random, NumPy MT19937, Torch CPU
default, Torch CUDA default (native GPU dropout), and an explicit CPU generator
for a shuffled32-record global cursor. Per-rank augmentation/dropout RNG differs;
the cursor order is shared, with four samples/rank in each global8 window.

Seed workers perform three successful optimizer windows, save an optimizer-boundary
checkpoint (next attempt3, epoch0/offset24), and continue three uninterrupted
reference windows. The middle reference window deliberately injects Inf: before
final DDP reduction on rank1, or into rank1's local FSDP shard after reduction.
Both ranks skip optimizer/scheduler; the data cursor and random draws still
advance. FSDP scaler backoff remains synchronized. Final state has five optimizer
updates after six attempted windows, epoch1/offset16; the reference crosses an
epoch boundary. Independent hook counts/LR formula checks validate update-only
scheduler advancement, not just agreement between two repeated mistakes.

After seed processes/containers exit, new launches construct models and all RNG
streams from deliberately different initial seeds. They restore local parameters,
buffers, AdamW, scaler, StepLR, optimizer update count, cursor/order/generator, and
Python/NumPy/Torch CPU/CUDA RNG, with RNG restored last. Initial restored state
matches exactly. All three following windows then match the uninterrupted local
loss, selected records, augmentation, every local gradient/parameter, optimizer,
scaler, scheduler, RNG and cursor with zero tolerances and matching digests.
FSDP replay crosses scale32→64, injected skip64→32, then finite32/tracker1.
This is12 exact resumed rank-windows across both cases/hosts.

## Admission, storage and fresh-process proof

Each host saves only its own CPU-serialized rank part; no shared filesystem.
Fsync/atomic local file replacement plus a collectively agreed complete rank/
size/SHA manifest. Header binds schema/config/world/next attempt, actual parameter
layouts, package versions and a selected fixture/helper/Torch/agent/worker cohort.
NumPy state is encoded with primitive integers/lists, enabling weights_only=True.
This is an owned fixed-checkpoint format at attempt3, not a generic checkpoint API.

Before application, live ranks coordinate local file/header admission and compare
commit identities. Eleven one-rank failures refuse on **both** ranks: missing
commit, wrong world/config/cohort, duplicate rank, wrong step, wrong digest,
truncated owned part, valid-digest invalid serialization, loaded wrong rank, and
loaded invalid cursor. The latter two operate on loaded owned copies; original
checkpoint parts remain untouched. Restricted-load errors also coordinate before
state application. Model/optimizer/scheduler/scaler/RNG/cursor fingerprints stay
unchanged through all negative checks. Reference files are separately hash-checked.
These are both-ranks-alive cases, not peer death or arbitrary network failure.

Initial r1 DDP seed passed, but resume stopped before state application because
the harness incorrectly required a different PID-namespace inode. That identifier
was equal across two new containers. Both container IDs, worker UUIDs and start
ticks differ; r2 uses those actual birth indicators instead. No numerical tolerance
was relaxed. r1 failure logs/source/checkpoints remain preserved. It was a harness
proof error, not a ROCm/PyTorch arithmetic/heap defect. Failure cleanup now destroys
an initialized process group; r2 exits and agents are clean.

Limits: deterministic owned graphs, one GPU/node, fixed world2/rank/layout/cohort,
optimizer boundaries and synchronous cursor in the main rank process. No DataLoader
worker RNG, partial accumulation, arbitrary HF Trainer checkpoint, model/precision
migration, elastic repartition, crash-during-save or peer-loss recovery proof.
No larger-model fitting, production convergence or speedup claim. Historical
Python CPU segfault/KFD uncertainty remains distinct from fixed VM heap/D05 exit.

## Evidence and operation

48 accepted check groups,44 collective refusal observations,42 local tests,
eight accepted GPU guards plus four acknowledged historical r1 guards. All r2
controllers exit0, registry/kernel logs unchanged, public D09 selected. SH locks,
4CPUs/cpuset0–3,10GiB RAM/12GiB memory+swap,512PIDs/256MiB shm,600s guards/460s
child deadlines; owned container labels prevent cleanup from stopping others.
Black/critical Ruff+unused-import/syntax/diff pass; full upstream CI not run.
.104 r2/.105 r3, boots7344171f/e1cbfa23 unchanged. No owned jobs/containers/guards
remain. r2 coordinators21599/1347 exit0; sequence.json records per-phase commands,
PIDs and exits. r1 controllers20342/31546 seed exit0,20874/32385 resume exit1.

Both hosts: tools/distributed-d10 and evidence/distributed-d10-r2/
{ddp-seed,ddp-resume,fsdp-seed,fsdp-resume}; r1 diagnostics under distributed-d10-r1.
Valid checkpoint/reference payloads33,335,526B(.104)/33,368,766B(.105) remain on
boards, with malformed owned copies/historical checkpoints also preserved.
Exports omit .pt data; source-r1 preserves original fixture source. Workstation
kernel/artifacts/distributed-d10/verified-results.json and raw archives:

| Archive | Bytes | SHA256 |
| --- | --- | --- |
|104-evidence-r2.tar.gz|334281|46e57cbb9b3358d3eb1af197b8dea6c9d58795b88be2950d1eff59fe2876beec|
|105-evidence-r2.tar.gz|300886|2271b7732b3e36ac6763311fa9eb12f4f7a262c933eb45c3d28682c3af3de5c7|

Both host docs/memory synchronized. [Usage](../../../../BC250_DISTRIBUTED_GUIDE.md).
**Next D11 (new planned):** actual Accelerate two-host DDP accumulation and AMP
integration through public bc250-distributed, owned small-model references and
state handling. D09 imported Accelerate but did not qualify its orchestration.
Preserve native/default/recovery settings; larger capacity, other transports,
elastic recovery and release reconstruction remain separate.
