# D07: two-host BF16/FP16 autocast, accumulation and scaler state

Completed 2026-11-05. Numerical fixture/guardian `ae52b7306`; verified export
helpers `615caa65e`. Existing installed daily/RCCL native Torch a524980, ROCm,
images, kernels, registries, profiles and D05 launcher/agent exit repair remain
unchanged. No backend defect or native change was needed. This batch qualifies
an explicit training-loop recipe; existing applications are unchanged.

## Fixture and independent references

Owned residual MLP, width256/hidden512/four blocks,16 tokens,1,051,648 parameters;
each block is x +0.125*down(tanh(up(x))). Global batch8, two ranks, microbatch1,
four micros/rank/update. Distinct FP32 inputs per sample. AdamW lr0.001/eps1e-5/
weight_decay0.01/foreachFalse. FP32 master parameters; actual linear intermediates
are asserted BF16 or FP16, residual outputs and mean loss FP32, parameter gradients
and RCCL bucket input FP32. Both sync-every-micro and no_sync-first-three cases
run on both physical hosts, using the public installed distributed launcher code
through the unchanged D05 bounded-resource adapter.

CPU reference explicitly rounds linear operands, output, backward matrix results
and bias gradients to the tested low dtype, using independent FP64 CPU matrix
arithmetic in between. Tanh forward/derivative use FP64 with corresponding low
dtype rounding; residual/loss and gradient accumulation use FP32. The oracle does
not invoke CPU low-precision GEMM or HIP. It models mixed arithmetic/autograd,
not the mathematical derivative of a discontinuous rounding function. It processes
all eight samples and applies the same loss scale/local normalization before
world averaging. Before each of two checked updates it starts at the actual
FP32-rounded model weights. All parameter gradients and each rank's input gradient
are compared; optimizer unscaling does not unscale input gradients automatically.

Fixed relative-L2 bounds per gradient tensor: BF16 0.02, FP16 0.004. Loss bounds
atol2e-5/rtol0.002 or0.0005, respectively. Chosen before execution; no relaxation.

| Precision | Largest parameter-gradient relative L2 error | Largest input-gradient relative L2 error |
| --- | --- | --- |
|bf16|0.22560%|0.02914%|
|fp16|0.04558%|0.00862%|

A separate FP64 CPU AdamW oracle consumes the **actual admitted AMP gradients**.
It compares every updated FP32 parameter and both moments/step counters after
two updates: parameter atol4e-6/rtol5e-5; exp_avg3e-7/5e-4; exp_avg_sq3e-9/1e-3;
step exact. This isolates optimizer correctness from allowed low-precision gradient
approximation. It is not a claim that a full mixed-precision trajectory equals an
unrounded FP64 training trajectory. No long-run convergence, attention, compiler,
arbitrary model or mixed-precision FSDP qualification follows from this MLP.

## Finite/nonfinite decisions and checkpoint

BF16 uses disabled GradScaler plus an explicit finite-gradient check and skips
AdamW when nonfinite. BF16's dtype alone does not provide an automatic skip.
FP16 uses native torch.amp.GradScaler: initial scale16, growth_interval2. Two
finite updates produce16→16→32. An Inf, then a NaN, is injected into one rank's
accumulated parameter gradient immediately before the final synchronized backward.
RCCL propagates it to both ranks; both skip the whole optimizer update. Parameters
and all AdamW state stay exactly unchanged; FP16 scales32→16→8 and resets its
growth tracker. A subsequent finite update succeeds. Optimizer post-step hooks
verify actual update/skip counts, rather than relying on the return value of step.

Finite flags must agree across ranks before an update; divergent flags refuse.
Replica hashes of all parameters, AdamW tensors and public scaler state agree
after each finished window. Before scaled backward, the fixture also compares
complete public scaler states collectively. One-rank scale mismatch and one-rank
growth-tracker mismatch each refuse on both ranks, then restore valid state.
GradScaler is local state: DDP does not automatically synchronize a wrongly restored
scale. Post-reduction rank-local gradient mutation and general fault recovery are
not qualified by the injected-before-reduction cases here.

Optimizer-boundary local checkpoints fsync/atomic rename and use the D05 complete
collective rank/hash/size manifest, extended config includes dtype/mode. Save
parameters, optimizer, scaler and fixed next fixture index5 after scaler.update.
Six malformed inputs/rank refuse before loading. Reconstruct model/optimizer/
scaler objects in the same worker, reload the same world/rank/config, and compare
next local loss, every parameter, every AdamW tensor and scaler state **exactly**.
FP16 replay crosses a growth boundary8→16, including growth-tracker restoration.
This is not fresh-process, partial-window, stochastic sampler/RNG/scheduler,
elastic repartition, interrupted-peer or general distributed-checkpoint proof.

## Communication, resources and preservation

Every synchronized backward reduces4,206,592 gradient input bytes/rank. no_sync
uses one complete pass/update; sync-every-micro uses four,16,826,368 bytes. Initial
one-bucket construction may rebuild to two; byte contracts hold for all seven
window executions/worker, including nonfinite skips and replay. Actual input
bytes are counted through Torch's standard averaging hook. These are not NIC
wire-byte measurements; initialization/control/checkpoint collectives excluded.
Autocast does not compress these FP32 gradients. No timing or speedup is claimed;
this is a different model from D05/D06 and is not a performance comparison.

Four phases/two hosts, eight workers:92 check groups,48 checkpoint preload
refusals, eight collective scaler-mismatch refusals, eight ACKed GPU guards.
All agents exit cleanly; kernel logs and registry unchanged within/across phases.
SH common runtime locks,4CPUs/cpuset0–3,10GiB RAM/12GiB memory+swap,512PIDs,
600s recovery guards/460s child deadlines; four Torch CPU threads. Black, critical
Ruff plus unused-import checks, syntax and diff pass; full upstream CI not run.
No native build/ccache step, active-prefix mutation, cache reset, installation,
reboot or user-application inspection/test. .104 r2/.105 r3 and observed bootIDs
7344171f/e1cbfa23 unchanged from D06. Boot recovery, immutable releases and host
health/controls preserved. Historical Python CPU segfault/KFD uncertainty remains
separate from the fixed VM heap and D05 agent-exit repair.

## Evidence and next action

Both hosts retain tools/distributed-d07 and evidence/distributed-d07-r1/
{bf16,fp16}-{no-sync,sync}-r1 with final/rank JSON, GPU/controller logs, kernel
logs, guard ACK and owned optimizer/scaler checkpoints. All four controllers/host
exit0; known PIDs FP16-no-sync9453/8276, BF16-sync9783/8438, FP16-sync10112/8607.
Initial BF16-no-sync controller PID was not captured; its final/ACK/clean-container
evidence is present. No jobs/containers/armed recovery guards remain. Four owned
checkpoint payloads/host total50,551,948 bytes; exports omit .pt payloads.
Workstation kernel/artifacts/distributed-d07/verified-results.json and raw archives:

| Archive | Bytes | SHA256 |
| --- | --- | --- |
|104-evidence-r1.tar.gz|239789|2cc7285d29a9a43deeac5ee8e830cf7f696ebfc22b00421969bd9aa29b090627|
|105-evidence-r1.tar.gz|217630|f21d02b249a4ff53962000c3e6642c1f0cc0414a9c3801b65382702b51085c67|

Both host docs/memory are synchronized. See the
[usage guide](../../../../BC250_DISTRIBUTED_GUIDE.md).

**Next D08 (new planned batch):** classic FSDP mixed-precision accumulation with
independent gradient/optimizer references, actual local shard/peak-memory counts,
shard-aware nonfinite decisions (inspect/use ShardedGradScaler) and same-world
checkpoint replay. Keep no_sync full-gradient memory cost explicit; preserve FP32
defaults and admit each BF16/FP16 path separately. No large-model capacity or
speedup claim until directly measured. Other transports/release audits stay open.
