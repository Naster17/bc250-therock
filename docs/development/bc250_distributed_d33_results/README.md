# D33: fresh training-only memory controls

Completed 2026-11-09. Implementation/verifier `199d3f2ab`.
[Verified results](verified-results.json), [larger-shape plan](larger-shape-plan.json),
[usage](../../../../BC250_DISTRIBUTED_GUIDE.md#fresh-training-only-measurements-d33).
Additive BOTH-host tools/distributed-d33; native providers unchanged.

D32 mixed CPU qualification/audit history into the process later used for memory
measurement. D33 reuses that completed CPU qualification through small immutable
proof records extracted from the published, hash-verified D32 archives. It does
not repeat the same CPU test. The guardian and worker refuse changed proof bytes,
training-source bytes or registry identity before model allocation. Native Torch
version, Transformers5.2 and the three audited modeling source hashes are checked.

The fresh worker reuses byte-identical D32 build/data/loss/backward/AdamW functions.
It retains no CPU base duplicate and executes no CPU FP64 oracle, full optimizer
export or full-parameter audit; those entry points are actively blocked. It moves
the initialized FP32 model to GPU in place, performs the same initial GPU update,
two warmups and five measured updates, then exits. Separate containers/new UUIDs/
start ticks establish fresh workers. Proofs bind the correct host/mode and original
CPU loss/ALL gradients/clip/ALL AdamW checks; they do not admit changed model shapes.

## Result and attribution limits

Same actual owned eager GPT2:45,871,104 all-trainable FP32 parameters, tied head,
six blocks/width768/12heads/vocab4096/sequence256/global8/micro1/dropout0/cacheFalse.
Single runs eight micros; DDP/FSDP run four each with native no_sync until final.
AdamW/root clipping/data/update order and resource bounds match D32. Initial and
five measured global losses match the qualified D32 trajectories within1e-5.
ALL parameter/AdamW ownership counts match; no new CPU ALL-gradient execution is
claimed for the fresh workers. Those checks are inherited from source-bound D32.

| Mode | Warm update ms across both orders/hosts | Peak live GPU MiB/rank | Fresh measured cgroup maximum MiB/rank | D32 cgroup maximum MiB/rank |
| --- | --- | --- | --- | --- |
| One board |247.897–249.549|940.725|1080.95–1081.41|2100–2332|
| Two-host DDP |1680.839–1681.057|1114.484|1870.41–1872.30|2792–3048|
| Two-host FSDP |6879.746–6882.628|767.607–769.358|1642.55–1646.23|3260–3448|

The fresh measurement corrects the physical-memory baseline; it is NOT a native
ROCm/Torch memory optimization or a training speedup. Previous CPU oracle/audit
work and retained base construction affected the old process history. These
controls remove that combined history, without identifying each contributor's
causal share. One board remains fastest. FSDP reduces master/AdamW ownership to
22,935,552 elements/rank and uses less cgroup memory than DDP here, but still more
than a single worker. No larger-model fit follows from this fixture.

New memory.stat/RSS and GPU snapshots cover before-model, after-wrap, initial
update, owned-cache release, warmup and measured-end stages. In .104 order a,
post-measurement anon/file-shmem are approximately959/104MiB single,1315/530MiB
DDP,1314/304MiB FSDP; socket peaks are around4.3MiB distributed. The counters are
categories/overlapping subsets, not independent memory pools to add to GPU memory.
Their names alone do not identify which library owns the pages.

DDP shmem grows during this short warm-up: measured endpoints458.11→514.11→
530.11→530.11→530.11MiB on BOTH hosts and in BOTH orders. Its final three endpoints
plateau in this bounded run; no ongoing leak, exact allocation owner or long-run
stability claim. FSDP's shmem remains near303.52MiB after its first GPU update.
This explains why two warmups did not establish DDP memory steady state; the
reported maxima include the observed later plateau rather than only the first
sample. Timing medians remain consistent with D32.

Whole-worker cgroup peaks are at most1081.42MiB single,1874.44DDP,1647.80FSDP.
Sampled host available memory never falls below10.625GiB, host/cgroup swap usage
and OOM counters stay zero. CPU/GPU share physical RAM; do not sum allocator,
cgroup and RSS counters. The200ms sampler and stage snapshots are not a complete
physical allocation trace. Existing device-free≥4GiB/host-available≥10GiB admission
and PLAN-01 resource/recovery limits are preserved.

Complete guarded phases take roughly17–18s single,30–31s DDP,70–71s FSDP. Removing
CPU validation from those phases explains their reduction versus D32; it does not
make the GPU updates faster. As in D32, single workers use two-node orchestration
but initialize no training process group and perform independent one-board work.

## Larger-model planning and next action

The new CPU-only planner emits exact plain tied-GPT2 parameter counts and FP32
state LOWER BOUNDS for159,807,488 /352,957,440 /1,024,471,040 parameter candidates.
For example,159.8M requires at least2.381GiB for single FP32 masters+gradients+
AdamW,1.191GiB/rank for fully sharded states, or1.488GiB/rank with unsharded
no_sync gradients. The1.024B candidate's corresponding floors are15.266/7.633/
9.541GiB. These exclude activation, gather overlap, reservations, initialization,
runtime and communication costs. **Every larger candidate is GPU_admitted=False.**
No automatic linear extrapolation, OOM search or larger fitting proof is performed.

Next PLAN-05 / D34: start with159.8M only. Implement memory-bounded initialization
and shape-specific CPU/GPU correctness qualification, then a fresh one-board/FSDP
measurement at a capacity-oriented global2/micro1/sequence128, with positions256.
Use separate qualification/measurement phases and bounded CPU optimizer checking
so a large oracle/moment export does not set the memory limit. One local micro per
FSDP rank avoids unsharded no_sync accumulation. Keep larger candidates refused
until their own initialization/activation/collective envelope is qualified.

## Evidence and preservation

**84 fresh-worker groups/12 ACKed guards**,68 local tests including altered-proof/
source and unsafe-admission refusals; Black/critical Ruff+F401/diff pass. No failed
GPU phase or relaxed bounds. D32's72 prior CPU/reference groups are reused, not
counted as fresh executions. No full upstream CI or universal model accuracy proof.

Exact commands/PIDs/CIDs/UUIDs/ACK/kernel logs: evidence/distributed-d33-r1/
{single,ddp,fsdp}-{a,b}, sequence.json/controller logs. Guardians104:
11524,12442,12911,13571,14232,14687;105:16770,18210,18815,19775,20733,21332.
Sequence controllers12441/18209,exports15427/22461 exit0. Local artifacts retain
proof-generation inputs, overlays, pilot JSON and complete metadata archives:

- .104:720475B SHAf3559f7bcdfee1dad7ded60d2100c4601cdeec4fb3f933e1f58a5b02bea00ee0.
- .105:690668B SHAedc434002dec47f2c7feb05189976695ecee66ee4775f00577b62f533463509e.

PLAN-01 EX reservations+SH runtime locks/4CPUs/cpuset0–3/10GiB RAM/12GiB RAM+swap/
512PIDs/256MiBshm/600s recovery/460s child retained. Kernel logs/registries unchanged,
final idle/no jobs/containers/armed guards. Boots2d86d523(.104r2)/e902e636(.105r3)
unchanged; native PLAN-02 MIOpen381eac05/Torch a524/RCCL/D09/image/packages/defaults/
fallbacks preserved. No native build/download/reboot/prefix mutation/user app
inspection/foreign stop. Docs/flow/manifest snapshots synced BOTH hosts. Historical
Python CPU fault stays distinct from fixed VM heap/D05 exit/D21 loading.
PLAN-05 remains open; speed and larger-than-single fit are not established.
