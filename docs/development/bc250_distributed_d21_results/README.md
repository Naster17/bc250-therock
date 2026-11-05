# D21: actual Trainer classic FSDP FP32 training and repaired restart

Completed 2026-11-05. Helper/fixture/payload utility `5189348ba`, initial export/
verifier `39918b800`. [Verified evidence](verified-results.json).
[Usage](../../../../BC250_DISTRIBUTED_GUIDE.md#trainer-classic-fsdp-fp32-d21).

## Added behavior and repaired defect

New opt-in `BC250FSDPTrainer` uses actual Transformers5.2/Accelerate1.12 classic
FSDP1 FULL_SHARD/use_orig_params, FP32 trainable master parameters, no clipping/
AMP/offload/compilation/activation checkpointing. Trainer's native loop, FSDP
wrapping, loss/backward/optimizer scheduling and model checkpointing remain.
As with D12, Trainer owns window normalization; Accelerator accumulation is1
to avoid dividing the already normalized loss a second time. The qualified
plain/DDP helper and its deliberate FSDP refusal remain UNCHANGED.

A fresh native restart demonstrated a real compatibility defect: the pinned
Accelerate sharded optimizer loader passes the fresh EMPTY optimizer state as
a generic DCP load template, then leaves all AdamW moment entries absent. Two
failed attempts per host preserve the assertion and schema diagnosis (actual
state keys[] versus21/22 expected nonempty-parameter entries). No optimizer
step was allowed after the failed exact-restore gate. No tolerance relaxed.

The separate helper repairs ONLY its FSDP optimizer/scheduler load hook using
PyTorch's public `load_sharded_optimizer_state_dict` to allocate from checkpoint
metadata, followed by native `FSDP.optim_state_dict_to_load` and optimizer/
scheduler load_state_dict. No dummy steps/zero-filled-moment workaround/global
package patch/private FSDP mutation. Package versions and five exact HF/
Accelerate source files are pinned and refuse replacement without requalification.
Native installed Torch/ROCm and ordinary package behavior are unchanged.

## Numerical qualification and sharding

Two physical hosts, one gfx1013 GPU each, RCCL Socket/Ring/Simple, actual native
Trainer BertForMaskedLM22561 parameters/vocab97/hidden32/2layers/4heads/inner64/
positions32/eager attention/dropout0/cacheFalse. Root plus each of two BertLayer
modules wrapped (three FSDP units). Tied embedding/decoder remains in root scope.
All trainable weights FP32. Owned28 records, sequence12, right-padding5–12,
unequal masked-label counts. Sequential workers0/per-device micro2/accum2 gives
global8 for the first three updates and final global4. Custom unshifted summed
CE/global active-window count, accepts_loss_kwargsFalse and native world
compensation retain the correct global token denominator. No causal-LM claim.

At each of four uninterrupted updates, a CPU FP64 copy starts from the actual
GPU weights. Same pinned HF graph, independent CPU arithmetic/manual logsumexp
CE. Global loss compared at atol3e-6/rtol5e-5; EVERY full parameter gradient via
public summon_full_params(with_grads=True) at3e-6/5e-4. No new tolerances.

CPU-double AdamW consumes the actual admitted full GPU gradients to isolate
optimizer arithmetic: all parameters4e-6/5e-5, all exp_avg3e-7/5e-4, all
exp_avg_sq3e-9/1e-3, counters exact. Public full optimizer state extraction
resolves actual owned original-param offsets. AdamW lr.001/eps1e-5/decay.01/
foreachFalse, one fixture parameter group, constant scheduler/last_epoch checked.
This is not an independently implemented Transformer or full unrounded FP64
trajectory/convergence proof. Parameters remain on HIP; this is GPU backward.

Parameter and exp_avg elements per rank: **11282(.104),11279(.105)**, summing to
22561. Empty local original-parameter views correctly have no AdamW moment state.
This proves parameter/optimizer sharding, not physical RAM pooling, a larger
model fitting, peak memory savings or faster training. No timing/capacity claim.

## Native checkpoint and exact fresh restart

Native SHARDED_STATE_DICT model+optimizer checkpoint2 and local Trainer state,
scheduler and rank RNG files are retained on each host. Hosts have separate
filesystems: a complete DCP checkpoint requires model/optimizer payload files
from BOTH. New `bc250_fsdp_checkpoint.py` exports/merges/installs only the two
DCP payload directories, with hash manifest/name checks/conflict refusal and
missing-file-only installation. Local Trainer/scheduler/RNG files are untouched.
It is an OFFLINE, trusted, same-window, two-rank utility bounded to256MiB payload,
not streaming, an automatic shared filesystem, run-lineage validation, elastic
checkpoint resharding or a general crash-atomic publication protocol. Callers
must select the same completed checkpoint on both hosts; hashes alone do not
prove that independently chosen inputs belong to the same training run.

Actual payload union has six files/864357 bytes. Every original checkpoint file
retains its native hash after installation. Fresh container IDs/worker UUIDs/
startticks and different constructor seed1773 prove a new process. Initial model
shards/ALL local AdamW moments+counters/scheduler/four RNG states restore EXACTLY.
Subsequent windows3 and4 reproduce ALL local losses, gradient shards, parameter
shards, optimizer, scheduler, RNG/probe draws and data rows EXACTLY; final digests
match uninterrupted training on each rank. Native rank RNG and sampler skip
handling retained. No same-process stand-in for fresh restart; same world/rank/
layout and optimizer-boundary only. No shuffled/multiworker/stochastic graph/
partial-window/elastic/peer-crash/portable-repartition resume claim. Aggregate
native train_loss across resumed sessions retains the earlier D12 caveat.

## Tests, evidence and next action

30 accepted GPU check groups/four accepted guards; four FAILED guards (two
pre-fix restarts per host) retained separately, all acknowledged. Four meaningful
CPU payload utility patterns (actual file union/local-RNG preservation/duplicate+
conflicting input/refuse overwrite) and42 local regressions pass. Black/critical
Ruff+F401/syntax/diff checks pass; full upstream CI not run. Only resume behavior
changed after the already-qualified seed; both corrected replay windows match
that original seed exactly. Prior sources/raw failures/checkpoints remain intact.
Source/helper/native package pins, guard/kernel/registry/agent checks, different
containers, checkpoint hashes and cross-rank config/payload agreement verified.
Archive verification does not reexecute numerical GPU comparisons.

Both host `tools/distributed-d21-r3` hold final code; older tools/distributed-d21
and distributed-d21-r2 preserved. Both `evidence/distributed-d21-r1/{fp32-seed,
fp32-resume,fp32-resume-r2,fp32-resume-r3}` retain commands/PIDs/CIDs/raw logs/
rank.json/final.json/guard state+ACK. Accepted guardians4449/29954(seed),
6613/31909(resume-r3) exit0; failed5328/30719,6003/31345 exit1 before training.
Controller logs `evidence/distributed-d21-{r1-seed,r1-resume,r2-resume,r3-resume}-controller.log`.
Checkpoint2 and4, reference tensors and metadata are PRESERVED on hosts; payload
inventories/hashes are exported, large .pt/.pth/.bin/.distcp omitted from report
archives. Workstation also retains the two payload exports and completed union.

No native build/install/prefix mutation/reboot/download/application inspection/
foreign process stop. Daily/RCCL Torch a524980/ROCm/D09 launcher/consumer packages/
profiles/images/registries unchanged. Shared runtime locks/4 CPUs/cpuset0–3/10GiB
RAM/12GiB memory+swap/512 PIDs/256MiB shm/600s recovery/460s children unchanged.
Boots c1199fca(.104r2, user-restored before batch)/bad537ed(.105r3) unchanged during
work. .104 handoff refreshed at entry, both docs/memory synced at completion.
No jobs/containers/armed guards remain. Native classic-FSDP/ShardedTensor/DCP
API deprecation warnings remain visible; deprecated is not a new hardware fault.
Historical Python CPU fault/KFD/invalid visibility stay separate from fixed VM
heap corruption, fixed D05 exit and this optimizer-load integration defect.

Workstation `kernel/artifacts/distributed-d21` final metadata archives:

- .104:439245 bytes, SHA256 `8b38c968be3530469965f72f95f882b1fed82a1ffa58c6ad0e05bee0500092a2`.
- .105:399839 bytes, SHA256 `f12bde68b9cfa7505882826624e0d4f83ccf63309029bcabfddbe89774701ade`.

**Next D22 (planned):** native global gradient-norm clipping for actual Trainer
FSDP FP32, with CPU all-gradient/norm/clipped-gradient/AdamW and exact fixed-world
restart references. Extend the current max_grad_norm=0 guard only after proof;
AMP/scaler/FSDP2, general model combinations and larger-model fit remain separate.
