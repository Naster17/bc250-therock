# D05: two-host training cost, state sharding and agent exit repair

Completed 2026-11-04 within the bounds below. Training tools `9ab6aec7f`, exit
bridge/tools `41bafbdd0` plus final durable-payload/export maintenance. Patched
Torch source `8d8b18df558cb478727ba2179f2e01dfce20ea40` fixes agent lifetime;
installed daily/RCCL native cohorts remain based on `a524980`, byte-identical.
No ROCm/Torch native build, kernel change, reboot or user-application access.

## Fixed workload and measurements

Owned eager FP32 transformer: four pre-norm blocks, width512, eight attention
heads, explicit causal math attention, GELU MLP expansion4, sequence64, global
batch8, 12,609,536 parameters. AdamW lr0.001/eps1e-5/weight_decay0.01/foreachFalse;
no dropout, scheduler, AMP, compilation or model downloads. Single-host workers
process all eight examples; two-host DDP/FSDP workers process four each. Both
single-host baselines run independently, concurrently, one on each board. Torchrun
provides orchestration but the single baseline does not initialize a process group.

Each fresh worker checks two CPU-double global-batch gradient/AdamW updates,
warms two further steps, then measures five optimizer steps. Timings include GPU
synchronization and exclude CPU reference work, checkpoint I/O, rendezvous and
exit. Distributed samples use the maximum rank time; barriers/control reductions
are outside the measured windows. Order is single→DDP→FSDP, then FSDP→DDP→single.
Both links negotiate1000Mb/s, Ring/Simple Socket, no P2P/IB/SDMA/symmetric features.

The installed launcher code is used through a test-only resource adapter:
four CPUs/cpuset0–3,10GiB memory/12GiB memory+swap,512 PIDs; four Torch CPU threads.
Each GPU phase holds the common SH runtime lock and a600s per-host recovery guard.
No active profile/native environment changes are needed to run the experiment.

| Mode | Median step time in the two orders | Peak live GPU allocator, per rank | Peak reserved allocator, per rank |
| --- | --- | --- | --- |
| One host .104 | 27.22 /23.55 ms | 287.35 MiB | 320 MiB |
| One host .105 | 23.75 /25.83 ms | 287.35 MiB | 320 MiB |
| Two-host DDP | 438.41 /438.25 ms | 299.44 MiB | 308 MiB |
| Two-host classic FSDP | 651.70 /651.55 ms | .104:212.78/.105:211.78 MiB | .104:360/.105:386 MiB |

This fixture is substantially slower on two hosts. As a diagnosis, full-buffer
FP32 AllReduce at1/8/32MiB passes exact constant-value references; .104 medians
are9.21/71.87/286.30ms, .1059.21/71.94/286.47ms (three timed calls after one warmup).
The training and communication results are consistent with Ethernet traffic
being the dominant cost here. These are fixture measurements, not a universal
DDP/FSDP speed ratio or model-size limit.

FSDP actually stores6,304,768 parameter elements/rank versus12,609,536 for DDP.
AdamW exp_avg covers exactly the local parameter elements. Optimizer tensors are
about50.44MB/rank for FSDP versus100.88MB for DDP (small scalar-step differences).
Live allocated memory falls, but reserved memory rises: no larger-model fitting
claim follows. Allocator statistics are neither physical whole-board memory nor a
pooled32GB device. Raw worker RSS/status and host meminfo are retained; the CPU
oracle and runtime also consume the board's shared memory.

## Numerical and checkpoint acceptance

Fixed gradient bounds atol3e-6/rtol5e-4, parameter bounds atol4e-6/rtol5e-5, against
FP64 CPU computation from identical FP32-rounded initial weights. No tolerance
relaxation. Every owned parameter gradient and update is compared. After timing,
local parameter/AdamW shards are fsync-written and atomically renamed; a complete
rank-indexed hash/size manifest is collectively gathered and atomically committed
on each host. No shared filesystem is assumed.

Each worker rejects six malformed inputs before loading: missing commit, wrong
world, wrong config, empty peer set, wrong local digest and truncated owned copy.
Reconstructed same-world/rank models then match the next uninterrupted loss,
all local parameters and AdamW tensors exactly. The next-step index/data are
fixture-defined; scheduler, sampler, stochastic RNG, interrupted-peer recovery,
elastic repartition and portable distributed checkpoint APIs remain unqualified.

Across nine phases/two ranks:140 worker check groups,108 checkpoint refusals,
18 ACKed GPU guards and unchanged kernel logs/registry within each phase. Owned
checkpoint payloads stay on the boards; exports include their manifests and byte
counts but intentionally omit `.pt` data. Four single workers plus fourteen
DDP/FSDP workers are covered; counts overlap and are not API coverage percentages.

## Agent exit race found and repaired

Original DDP-a on .105 logs TCPStore EOF/DistNetworkError after successful GPU
training; torchrun catches it and returns0. Original guards accepted numerical
work but did not inspect agent logs. This is recorded as a lifecycle defect, not
silently promoted to clean launch qualification.

Source inspection shows a symmetric terminal barrier can let the rank-zero
TCPStore owner close before a peer sends its final wait. The repair reuses the
existing synchronize/get_all rank-aware completion protocol, which keeps rank zero
alive until peer reads complete. A CPU-only two-process real TCPStore test adds a
late peer wait: old code reproduces the error, repaired code passes two repetitions.
Two existing exit-flag/timeout unit cases also pass. The first CPU harness attempt
failed directory ownership before the test; corrected r2 passes, failed log retained.
The unit timeout case deliberately logs its mocked exception; it is not an active
launcher failure.

A versioned source-derived bridge applies only the patched `_exit_barrier` method
at torchrun startup, checking the installed API SHA and repair-source SHA first.
Expected old API SHA:
`625fa4ccf0fa2beaec4dcee1ad995a9d2b4fceba52546cf60bc943cb3f328afc`.
The full licensed patched source is bundled; errors are not suppressed. This
avoids mutating frozen Torch payloads. Direct torchrun outside bc250-distributed
still uses the old installed Python API until a separately qualified Torch update.

Candidate DDP/FSDP pass32 worker groups/four guards with strict clean-agent checks;
only then both public commands activate `/opt/bc250-stack/tools/distributed-d05`.
Live EX-locked rollback/reactivation is exercised, including durable install
payload reuse. Post-activation DDP adds16 groups/two guards with clean exit.
These48 groups are included in the140 above. A real UID1000/doas/caller-context
CPU two-host Gloo test verifies cwd, spaced argv, sum and a false Torch CUDA-initialization flag on
both hosts, with clean ordinary command exits. The raw field HIP_initialized
records that Torch flag, not a whole-driver/device-memory measurement.29 local launcher/lock/deployment
unit tests, Black/critical Ruff/syntax/diff checks pass. Full upstream CI not run.

Registry runtime pins, daily selections, native Torch/RCCL/ROCm files, loopback
profile, old images/releases, caches and r2/r3 boot fallbacks remain. Common EX
locks protect versioned tool registration/rollback; launches hold SH locks.
Recovery/installation tools and payload live under
`/home/nik/bc250work/tools/distributed-d05/`; records are in
`evidence/distributed-d05-tool-install-r1`. See the
[guide](../../../../BC250_DISTRIBUTED_GUIDE.md) for commands and limits.

Historical Python CPU segfault/KFD uncertainty remains separate from the fixed
VM-heap corruption. The TCPStore repair establishes no fix for those defects.

## Evidence and next action

Both hosts retain `bc250work/evidence/distributed-d05-r1/<phase>/`, source tools,
owned checkpoints and controller logs. CPU repro/unit evidence is under .104
`evidence/distributed-d05-exit-cpu-r2/tests`; ordinary caller reports under both
`evidence/distributed-d05-user-cpu`. No jobs/containers/armed guards remain.

Controller PIDs, all exit0: singles-a32062/8330, DDP-a32630/8726,
FSDP-a1520/9588, FSDP-b2054/9955, DDP-b3035/10588, singles-b3880/11139,
DDP-fixed5967/12323, FSDP-fixed6627/12776, installed-DDP7354/13266.
CPU r1PID4494exit1 (ownership), corrected r2PID4819exit0.

Workstation `kernel/artifacts/distributed-d05/` retains all raw metadata/logs,
verified aggregate results, the pinned install payload and both r1/r2 archives:

| Final r2 archive | Bytes | SHA256 |
| --- | --- | --- |
|104-evidence-r2.tar.gz|660837|8a546b712339bffc724370eb64c1348f4eea57329824f092250f9fad958ebcdf|
|105-evidence-r2.tar.gz|495817|62c1e81c258700b6651791b38c4b1d0d5248a344e38ac39ff95c3549fe4b854b|

D06 next: matched-global-batch DDP gradient accumulation/no_sync, independent
CPU-double gradient/AdamW and checkpoint references, synchronization counts and
one-versus-two-host timings. Amortize communication without changing intended
optimizer semantics. FSDP capacity/reservation, other transports/dtypes/models,
production convergence and broad release reconstruction remain separate work.
