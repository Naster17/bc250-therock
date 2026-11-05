# D22: actual Trainer FSDP FP32 global clipping and collective refusal

Completed 2026-11-05. Helper/fixture/controls `2a11969bc`, export/verifier
`790357231`. [Verified evidence](verified-results.json).
[Usage](../../../../BC250_DISTRIBUTED_GUIDE.md#trainer-fsdp-fp32-gradient-clipping-d22).

Extended the separate opt-in `BC250FSDPTrainer` to admit finite nonnegative
max_grad_norm. Zero retains the D21 no-clipping path; positive limits use native
Accelerator→root FSDP.clip_grad_norm_ L2 clipping. No custom distributed norm or
scaling implementation replaces it. Only the complete root parameter list is
admitted. Existing DDP helpers, native packages and ordinary launch defaults are
unchanged; final sources staged BOTH `tools/distributed-d22-r2`.

For positive limits, a per-instance public Accelerator method wrapper checks
all owned nonempty gradients (dense FP32/same device/present/finite), reduces a
bad flag across live ranks, and refuses before native clipping if any rank is
invalid. After native clipping it collectively checks that the returned norm is
finite before returning to Trainer/AdamW. No private FSDP mutation, global package
patch, Trainer-loop fork, dummy updates, silent skip or new clipping math.

A refusal requires abandoning the current session and reloading a completed
checkpoint. Parameter/optimizer/scheduler preservation is tested; transient
gradients/RNG/iterator state are not rolled back. Caller callbacks must not mutate
gradients or clipping settings after these checks; arbitrary late mutation/
collective reordering/peer failure/elastic recovery is outside this qualification.

## Actual forward/backward, norm and optimizer checks

Same owned D21 BertMLM22561/vocab97/hidden32/two layers/four heads/inner64/
positions32, eager/dropout0/cacheFalse/FP32 masters, classic FULL_SHARD with
original params and root+two BertLayer units. One gfx1013 GPU per host/world2/
RCCL Socket/Ring/Simple. Unequal masked-label counts/rightpadding5–12/28 records,
sequential workers0/per-device micro2/accum2; global8/8/8/final4. D21 token loss,
window denominator and AdamW restart repair retained.

At each seed update, public full-gradient snapshots BEFORE native root clipping
are compared against the actual full-global CPU FP64 model/manual CE. D21 loss
bounds3e-6/5e-5 and ALL-parameter gradient3e-6/5e-4 unchanged. CPU FP64 L2 sum of
actual admitted preclip gradients independently predicts native global norm
(atol1e-7/rtol1e-5), coefficient min(1,limit/(norm+1e-6)), and EVERY clipped
full parameter gradient (atol1e-8/rtol5e-5). One counted delegated native root
clip call per completed accumulation window; no per-shard clipping substitute.
CPU-double AdamW uses independently predicted clipped gradients, comparing ALL
parameters4e-6/5e-5, exp_avg3e-7/5e-4, exp_avg_sq3e-9/1e-3, step counters exact.
No tolerances relaxed or failed numerical cases reclassified as passes.

| Update | Native global norm | CPU predicted coefficient, limit.05 |
| --- | --- | --- |
| 1 | 1.4693714380 | .03402813451 |
| 2 | 1.4967209101 | .03340633634 |
| 3 | 1.6263184547 | .03074426846 |
| 4, shorter window | 2.2679159641 | .02204666188 |

An independent one-step positive-limit100 control has coefficient EXACT1;
unchanged gradients/CPU optimizer checks pass. This tests the inactive clipping
branch, not just strongly clipped updates. Source-pinned native routines and
CPU references use the same HF model graph, different arithmetic, not an
independent Transformer implementation. No BF16/FP16/scaler/FSDP2/offload/compile/
activation-checkpointing/frozen-parameter/general architecture qualification.

## Nonfinite and overflow controls

Separate processes inject Inf and NaN into ONE nonempty rank1 LOCAL gradient
shard after the final reduction. Raw nonfinite flags[False,True]; both live ranks
refuse before native clipping or AdamW. A separate finite1e30 rank1 gradient
control has raw nonfinite flags[False,False], but native FP32 global norm becomes
nonfinite: one native root clip call occurs, then BOTH ranks refuse before AdamW.
Native clipping may already alter gradients in that overflow case; no rollback
claim. All three controls prove zero actual AdamW steps and exact unchanged
parameter/optimizer/scheduler state. These are admitted software refusals, not
a GPU memory-fault/reset or VM-heap corruption test.

## Exact fresh-process restart and preserved evidence

Native sharded checkpoint2 is completed on both separate filesystems using the
unchanged D21 offline payload utility: six files/864357 bytes, all original files
retain hashes, local RNG/Trainer/scheduler untouched. New actual container IDs,
worker UUIDs/startticks and constructor seed1773: ALL model/AdamW/scheduler/four
RNG state restores EXACTLY, then windows3/4 reproduce ALL gradients, weights,
optimizer state, norms/coefficients, losses, rows, RNG/probes and final digests
EXACTLY against uninterrupted training. Same fixed world/rank/layout/optimizer
boundary only; not shuffled/multiworker/stochastic/partial-window/elastic proof.
Native aggregate train_loss resume caveatD12 remains.

68 accepted GPU groups/12 acknowledged accepted guards. Two earlier failed guards
are preserved separately: the first Inf harness run could not create its seed
folder under a root-owned parent. No numerical/nonfinite guard was exercised in
that attempt; guardian ownership preparation was corrected and Inf rerun under a
new label. No test/model or native code was weakened to address that setup error.
Two CPU limit-contract patterns observed24 times in accepted workers, plus42
local regressions; Black/critical Ruff+F401/syntax/diff pass, no full upstream CI.
Source/config/guard ACK/kernel/registry/agent, checkpoint hashes, full replay and
cross-rank agreement are verified; archive verifier does not rerun GPU arithmetic.
All native classic-FSDP/DCP/ShardedTensor deprecation notices remain visible.

Seed11245/4083 exit0; controls-r1 coordinators11723/4519 exit1 on folder setup;
corrected controls12595/5309 exit0 with per-phase exact PIDs/commands/exits in
`evidence/distributed-d22-r1/controls-r2-sequence.json`; resume13834/6428 exit0.
Both `tools/distributed-d22` and `tools/distributed-d22-r2` preserved. Both
`evidence/distributed-d22-r1/{fp32-seed,fp32-resume,inactive-seed,inf-refusal,
inf-refusal-r2,nan-refusal,overflow-refusal}` retain source/commands/CIDs/PIDs/
rank.json/final.json/guard/kernel/raw logs. Checkpoints2/4, reference tensors and
native payload exports remain on hosts; metadata archives omit .pt/.pth/.bin/
.distcp but retain inventories/hashes. Workstation keeps payload exports/union.

No native build/install/prefix update/reboot/download/application inspection/
foreign workload stop. Native Torch daily/RCCL a524980/ROCm/D09 launcher/frozen
consumers/images/registries/defaults unchanged. Shared runtime locks/4CPUs/
cpuset0–3/10GiB RAM/12GiB memory+swap/512PIDs/256MiB shm/600s recovery/460s child
unchanged. Boots c1199fca(.104r2)/bad537ed(.105r3) retained. Historical Python CPU
fault/KFD/visibility separate from fixed VM heap/D05 exit/D21 optimizer load.
No jobs/containers/armed guards remain. Both host docs/memory synchronized.

Workstation `kernel/artifacts/distributed-d22` archives:

- .104:526568 bytes, SHA256 `e2863fb13dd4ee32f51a1616cc6de577c6c571ef58109f9fd24893ad3caa36aa`.
- .105:472600 bytes, SHA256 `f99bbe5573367cb21b4d7c5662f7ba9fb51770ed4bfc8efa6389281f6d14e60b`.

**Next D23 (planned):** actual Trainer classic FSDP BF16 readiness with an explicit
parameter/reduction/autocast policy, CPU gradients/global clipping/AdamW and exact
fresh restart. Lift the current AMP refusal only after numerical proof; FP16
scaler/overflow policy and larger-model capacity remain separate priorities.
