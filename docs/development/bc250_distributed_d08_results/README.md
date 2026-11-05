# D08: classic FSDP mixed precision, accumulation and shard-aware skipping

Completed 2026-11-05 on both boards. Numerical fixture/guardian0815ac97b;
export/verifier97e75946a. Reuses D07's owned residual MLP and CPU rounding oracle,
D05 checkpoint primitives and installed bounded launcher/agent exit repair.
Native daily/RCCL Torch a524980, ROCm, images, registry, daily profiles and kernels
unchanged. No backend defect/native patch, build, installation or reboot needed.
Existing application loops remain unchanged; this is an explicit qualified recipe.

## Policy, shards and numerical references

Owned D07 residual MLP width256/hidden512/four blocks/tokens16/1,051,648 parameters,
global batch8/micro1/accum4 per rank. Classic FSDP FULL_SHARD/use_orig_params,
each of eight Linear modules wrapped separately. MixedPrecision param_dtype BF16
or FP16, reduce_dtype FP32, keep_low_precision_grads=False, both root and nested
input casting disabled; outer autocast casts linear inputs while residual/loss
stay FP32. Actual forward weight/intermediate dtypes are asserted. Master shards
and final gradient shards are FP32. This fine wrapping is a correctness fixture,
not a performance recommendation for large models.

Each rank stores525,824 parameter elements and525,824 AdamW exp_avg elements.
Rank-indexed sizes cover every original parameter exactly; CPU optimizer-state
comparisons use those actual shard offsets. Some original parameters have empty
local views, correctly without local optimizer state. Counts prove state sharding,
not whole-board RAM pooling or a larger model fitting.

Before two checked updates, a CPU reference starts from the actual full FP32
weights. Reuses independent FP64 linear/tanh arithmetic with explicit low-dtype
operand/output/backward rounding and FP32 residual/loss. For no_sync, additionally
rounds each rank's accumulated full parameter gradient to param_dtype after each
microbatch, matching the pinned FSDP accumulation path; sync accumulates reduced
FP32 shards. Oracle input-gradient normalization accounts for loss scale/local
batch, and parameter gradients also account for world averaging.

All parameter and input gradient tensors satisfy fixed relative-L2 bounds BF16.02/
FP16.004. Loss atol2e-5, rtol.002/.0005 respectively. No tolerance relaxation.

| Precision | Accumulation mode | Largest parameter-gradient relative L2 error | Largest input-gradient relative L2 error |
| --- | --- | --- | --- |
|bf16|no-sync|0.31388%|0.02906%|
|bf16|sync|0.22560%|0.02914%|
|fp16|no-sync|0.05349%|0.00861%|
|fp16|sync|0.04558%|0.00862%|

Separate FP64 CPU AdamW consumes the actual admitted AMP gradients: every updated
parameter and locally owned moment/counter compared after two updates, using
D07 bounds (parameter4e-6/5e-5; exp_avg3e-7/5e-4; exp_avg_sq3e-9/1e-3; step exact).
This isolates optimizer correctness from admitted mixed-precision approximation;
it is not full-trajectory equivalence to an unrounded FP64 model. No attention,
compiler, arbitrary model, alternative FSDP policy or convergence proof follows.

## Local shard Inf/NaN, scaler and checkpoint

After the final reduce-scatter, inject Inf then NaN into one nonempty **rank1 local
gradient shard**. Rank0's local gradients remain finite: raw flags[True,False].
FP16 uses native ShardedGradScaler, which synchronizes found_inf across ranks;
optimizer post-step hooks prove both skip even though rank0's own gradients are
finite. Local parameter/AdamW state remains exactly unchanged on both. BF16 uses
a disabled scaler plus an explicit collective all-finite guard to skip both.
A rank-local BF16 check alone would not establish that shared decision.

FP16 init16/growth_interval2: finite16→16→32→32; Inf then NaN back off32→16→8,
reset growth tracker; a finite recovery succeeds. Complete scaler state agrees
collectively. One-rank scale and growth-tracker mismatch each refuses on both
ranks before backward, then restores valid state. Full-parameter hashes agree
after each finished window; local AdamW shards are distinct, not duplicate states.

Optimizer-boundary local fsync/atomic parts and complete collective hash/size
manifest save local parameters, AdamW, ShardedGradScaler, config/dtype/mode and
fixed next index6. Six preload malformed-input refusals/rank. Same-worker model/
optimizer/scaler reconstruction with identical rank/world/layout then reproduces
next local loss, all local parameters/AdamW tensors/scaler exactly, crossing8→16
growth boundary. Not fresh-process, partial-window, RNG/sampler/scheduler, elastic,
peer-crash or portable distributed checkpoint qualification.

## Reduction and allocator measurements

Instrument the pinned public reduce_scatter_single call, delegating unchanged to
the original; no custom communication algorithm/hook. Each Linear flat gradient
reduces in FP32: eight calls/4,206,592 input bytes per complete backward.
no_sync performs one pass/window; sync performs four,32 calls/16,826,368 bytes.
All eight windows/worker assert counts/dtypes. Parameter all-gathers and scaler/
control/checkpoint collectives are excluded, so these are neither total traffic
nor NIC wire bytes. no_sync does not eliminate parameter all-gathers. No timing or
speedup is claimed; D05/D06 have different models and are not comparisons here.

After two checked updates, clear gradients/empty the caching allocator, record
baseline and reset peak counters. Measure one complete third training window
through AdamW, including finite/scaler/control work, before full-state hash gathers.
CPU oracle/full-parameter validation/checkpoint work is outside this peak window.
Per-micro live/reserved/gradient-element/dtype snapshots are retained.

| Precision | Mode | Peak live allocator MiB .104 / .105 | Peak reserved MiB .104 / .105 |
| --- | --- | --- | --- |
|bf16|no-sync|75.04102 /75.04102|84 /84|
|bf16|sync|73.60693 /73.60693|84 /82|
|fp16|no-sync|75.04248 /75.04248|86 /84|
|fp16|sync|73.60840 /73.60840|82 /80|

no_sync holds1,051,648 full low-dtype gradient elements during its first three
micros, then525,824 FP32 shard elements after final reduction. Sync holds525,824
FP32 gradient elements after every micro. In this two-rank policy, those gradient
views each total2,103,296 bytes: full low-dtype and half-size FP32 gradients have
equal bytes. Total peak live allocations nevertheless rise1.43408MiB with no_sync
in this fixture. Element counts alone are not a physical memory/capacity estimate;
more ranks/policies/wrapping can have different costs. Native allocator statistics
exclude CPU oracle/runtime memory; raw RSS/meminfo retained, no pooled32GB claim.

## Evidence and next action

Four phases/two hosts/eight workers:100 check groups,48 checkpoint preload
refusals, eight collective scaler-mismatch refusals, eight acknowledged guards.
Clean agents, unchanged kernel logs/registries. SH runtime locks,4CPUs/cpuset0–3,
10GiB RAM/12GiB memory+swap,512PIDs, four Torch CPU threads,600s recovery guards/
460s child deadlines. Black/critical Ruff+unused-import/syntax/diff pass; full
upstream CI not run. Python-only, no ccache/native build. No app inspection/test,
cache/source/release deletion or boot change. .104 r2/.105 r3, boots7344171f/
e1cbfa23 unchanged; recovery/health/native controls retained. Historical Python
CPU segfault/KFD uncertainty stays distinct from fixed VM heap/D05 exit repair.

All controllers exit0, PIDs bf16-no-sync11869/10611, fp16-no-sync12241/11288,
bf16-sync12561/11813, fp16-sync12919/12459. No jobs/containers/armed guards remain.
Both tools/distributed-d08 and evidence/distributed-d08-r1/{bf16,fp16}-{no-sync,sync}-r1
plus controller logs. Four local checkpoint payloads total25,285,292B(.104)/
25,311,948B(.105); preserved on board, omitted from exported metadata/logs.
Workstation kernel/artifacts/distributed-d08/verified-results.json and raw archives:

| Archive | Bytes | SHA256 |
| --- | --- | --- |
|104-evidence-r1.tar.gz|252227|209e51c2b6186f5a7b6cf4be92b9e079850c664a9c4cfaa2f0525057d5875df6|
|105-evidence-r1.tar.gz|230690|c45da6b438bab63adf706b527cebdc734a1cb6ed509e9fcba941b2b3dfe79426|

Both host docs/memory synchronized. [Usage](../../../../BC250_DISTRIBUTED_GUIDE.md).

**Next D09 (new planned batch):** close the recorded distributed consumer-env gap:
add explicit validated env-file/env/mount support by reusing the ordinary Torch
launcher architecture, qualify an owned tiny Transformers training consumer and
private caller/identity/refusal/live rollback gates on both hosts. Keep native
Torch/RCCL pins, D05 exit repair, Ethernet restrictions and ordinary defaults;
no user application/model downloads. Fresh-process checkpoint/failure recovery,
larger-model capacity, other transports and public release audit remain separate.
