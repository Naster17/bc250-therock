# D06: DDP accumulation, communication counts and optimizer semantics

Completed 2026-11-05 on both physical BC-250 hosts. Numerical fixture and guardian
`bd3726047`; fail-fast sequencing and verified exports `539951ced`. Reuses D05's
owned Block/config/checkpoint primitives and installed bounded launcher. Native
Torch daily/RCCL source a524980, ROCm, images, registry and daily selections remain
unchanged. The D05 source-derived agent exit repair remains active in both explicit
`bc250-distributed` commands. No native build, installation, reboot or app access.

## Workload and acceptance

Same four-block width512/heads8/sequence64 causal math-attention transformer as
D05: 12,609,536 FP32 parameters, AdamW lr0.001/eps1e-5/weight_decay0.01/foreachFalse.
No dropout, AMP, compilation, scheduler, user model or download. Add distinct
per-example input/target offsets, then FP32-round the inputs for both backends.
The independent FP64 CPU oracle processes groups of four, rather than the GPU
microbatch layout. Two successive steps compare global mean loss, every gradient,
every updated parameter, both AdamW moments and scalar step counters.

Fixed bounds: loss atol3e-6/rtol5e-5, gradient3e-6/5e-4, parameter4e-6/5e-5,
exp_avg3e-7/5e-4, exp_avg_sq3e-9/1e-3; step counters exact. No tolerance relaxation.
The tested configurations are:

| Global examples / optimizer step | GPU microbatch | Accumulation on one host | Accumulation per two-host rank |
| --- | --- | --- | --- |
|8|1|8|4|
|32|2|16|8|

One-host baselines run independently on both boards. Distributed cases either
synchronize every microbatch (`sync`) or only the final microbatch (`no-sync`).
The latter encloses both forward and backward in DDP.no_sync for preceding
microbatches. Each mean loss is divided by local accumulation count; DDP averages
between ranks. Zero gradients and optimizer updates occur once per global batch.
All ranks execute equal workloads. This does not qualify uneven batches or
masked-token weighting, and larger global batches are separate training settings.

The communication hook counts bucket invocations/input bytes and delegates to
Torch's existing allreduce_hook, including its world-size normalization. Every
microbatch is checked: zero hook calls on no_sync branches, complete parameter
bytes on synchronized branches. These are summed gradient-buffer input bytes per
rank, **not NIC packet counters or total physical wire bytes**. Loss checks,
barriers, timing-control reductions, initialization and checkpoint collectives are
outside this counter. No compression or altered collective algorithm is used.

## Timing and memory

Each worker checks two reference steps, warms two additional steps, then times
five optimizer steps. Each timing includes forward/backward, accumulation, AdamW,
Python hook/counter bookkeeping and GPU synchronization. Distributed samples use
the maximum rank time. CPU oracle, checkpoint, rendezvous, agent exit and timing
control collectives are excluded. Repeat orders: single→sync→no-sync, then
no-sync→sync→single, for each batch configuration. Ring/Simple Socket over the
existing1000Mb/s Ethernet links; P2P/IB/SDMA/symmetric remain off.

| Global batch | One host .104, two order medians ms | One host .105 ms | Two-host sync ms | Two-host no_sync ms | sync/no_sync ratio |
| --- | --- | --- | --- | --- | --- |
|8|63.50 /62.67|63.43 /64.86|1738.56 /1738.02|462.36 /462.00|3.760 /3.762|
|32|123.35 /126.22|128.16 /127.80|3474.24 /3476.18|497.33 /496.08|6.986 /7.007|

The improvement is relative to synchronizing every microbatch. Both distributed
strategies are still slower than one host for this model. D06 uses different data
and microbatch layouts from D05; its single-host numbers are not a D05 regression
or an end-to-end application speed comparison. These measurements establish no
universal ratio, production convergence or larger-model fitting claim.

At measured steady state, each complete reduction uses three buckets totaling
50,438,144 input bytes/rank. The global8 case changes12 calls/201,752,576 bytes
to3/50,438,144; global32 changes24/403,505,152 to3/50,438,144. First backwards
and checkpoint reconstruction can use a different initial bucket count; all
microbatch byte contracts hold throughout all11 optimizer-update executions
per worker, not just timed steps.

| Global batch | Mode | Peak live allocator MiB/rank | Peak reserved MiB/rank |
| --- | --- | --- | --- |
|8|single|275.43|292|
|8|sync|270.43|306|
|8|no_sync|321.53|342|
|32|single|288.43|320|
|32|sync|282.43|314|
|32|no_sync|333.53|350|

no_sync raises peak live allocations by51.10MiB versus sync in these fixtures,
even with gradient_as_bucket_view=True. Traffic savings are not a memory-saving
claim. These are allocator statistics, not physical whole-board usage. Worker
RSS and host meminfo are retained; CPU/GPU also share the board's physical RAM.
FSDP no_sync/state-capacity behavior remains separate and unqualified here.

## Checkpoint, lifecycle and preservation

Reuses D05's local fsync/atomic part and collectively completed hash/size manifest;
no shared filesystem. Configuration includes global and microbatch sizes; mode,
world and rank must match on reconstruction. Six malformed cases are rejected
before loading: missing commit, wrong world/config, incomplete peer set, wrong
local digest and truncated owned copy. After reconstruction, the next local loss,
all parameters and every AdamW tensor equal uninterrupted execution exactly.

This is an optimizer-boundary fixture with a fixed next data index. Partial
accumulation windows, RNG/data sampler/scheduler state, elastic world changes,
peer failures and generic distributed checkpoint APIs are not qualified.

12 phases/host,24 workers total: eight single and sixteen distributed workers.
192 worker check groups,144 pre-load refusals,24 matching guard ACKs. Both
reference updates cover every parameter and AdamW state; all accepted GPU phases
have clean agent exits, unchanged kernel logs and unchanged distributed registry.
Black, critical Ruff, syntax and diff checks pass; full upstream CI not run.
Harmless broadcast_buffers deprecation and current-context barrier warnings remain
visible in fixture logs; they are not hidden or treated as backend failures.

Common SH runtime lock,4 CPUs/cpuset0–3,10GiB RAM/12GiB memory+swap,512 PIDs,
four Torch CPU threads,600s recovery guard and460s child deadline per phase.
No prefix mutation, native build or ccache work is needed for this Python-only
qualification. .104 kernel7.1.5-bc250rocm-r2 and .1057.2.10-bc250rocm-r3 remain;
current observed boots7344171f/e1cbfa23 are newer than D05, already present at the
first D06 phase. This batch performs no reboot. Boot/SSH fallbacks, caches,
releases and user applications remain intact. finetune-n7 is untouched. Historical
Python CPU segfault/KFD uncertainty is separate from the fixed VM heap and repaired
agent exit lifetime; D06 establishes no fix for those historical issues.

## Evidence and next action

Both boards: `/home/nik/bc250work/tools/distributed-d06/` and
`evidence/distributed-d06-r1/<case>-<mode>-<a|b>/`, plus per-phase controller logs.
Sequence journals contain exact commands/PIDs/exit codes. Coordinators4856(.104)
and3882(.105) exit0; all12 guards/host passed, no owned jobs/containers/armed
recovery guards remain. Each host retains12 complete owned checkpoint files,
1,816,412,148 bytes total; exports deliberately omit .pt payloads. Small initial
single/sync phases precede the journal beginning small-no-sync-a.

Workstation: `kernel/artifacts/distributed-d06/verified-results.json` and raw
archives below. Independent verifier checks archived source digests, worker/final
identity, kernel-log bytes, guard nonces, cross-rank configuration/timing and
communication reduction ratios. It does not infer physical network counters.

| Archive | Bytes | SHA256 |
| --- | --- | --- |
|104-evidence-r1.tar.gz|647863|e68c3c32a77a14dbc9e77fa68ada38f3a3897026699c36d300ee61a013bb3d64|
|105-evidence-r1.tar.gz|578803|08fda612dd67472bd22fa4ab8c03854548331e4d742c4f540130ac9024b76448|

The [distributed guide](../../../../BC250_DISTRIBUTED_GUIDE.md) includes the
explicit accumulation pattern. Existing application loops are not changed or
automatically accelerated. Native installed Torch already exposes no_sync.

**Next D07 (new planned batch):** bounded two-host DDP mixed-precision accumulation:
BF16 autocast and FP16+GradScaler, dtype-appropriate independent references,
synchronized finite/nonfinite update decisions and same-world optimizer/scaler
checkpoint replay. Preserve FP32 defaults and qualify each path separately;
FSDP capacity/memory, other transports and public release reconstruction remain
separate open work. No AMP success or performance is claimed before those tests.
