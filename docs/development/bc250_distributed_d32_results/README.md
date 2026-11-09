# D32: representative one-board versus two-board training cost and memory

Completed2026-11-09. Implementation/verifier `6af433cb8`.
[Verified measurements](verified-results.json),
[usage](../../../../BC250_DISTRIBUTED_GUIDE.md#owned-capacity-and-timing-measurements-d32).
New additive BOTH-host tools/distributed-d32; native providers unchanged.

Owned actual eager GPT2: six blocks/width768/12heads/vocab4096/sequence256,
**45,871,104 all-trainable FP32 parameters**, tied embedding/head, zero dropout,
cacheFalse. Synthetic deterministic IDs/targets, global batch8/micro1,2040 shifted
CE targets. Independent one-board workers process eight micros; DDP/FSDP process
four each. DDP and FSDP use native no_sync for all but the final micro. Fixed
native AdamW lr.0001/eps1e-5/weight_decay.01/foreachFalse, root L2 clip.5.
No PEFT/AMP/checkpointing/compile/model download/application access in this fixture.

D05/D06 already covered the smaller custom12.6M transformer and accumulation;
D32 instead exercises a3.64x larger actual HF causal model with4x D05's global
batch×sequence positions, plus changed width/layer count. Those ratios describe
parameters and inputs, not measured memory or architecture-independent costs. PLAN-05 remains open: this model fits
one board, so it cannot establish a larger-than-one-board capacity benefit.

## Measured result

Five warmed update samples per fresh worker, in forward order single→DDP→FSDP
then reverse FSDP→DDP→single. Distributed samples report the maximum rank time.
Each timing includes forward/backward, accumulation, clipping, AdamW and GPU
synchronization; CPU reference/audit, rendezvous/exit, checkpoint I/O and timing
control collectives are outside the update windows. No checkpoint I/O is tested.

| Mode | Median update ms in orders a/b | Peak live GPU allocator MiB/rank | Peak reserved MiB/rank | Measured-phase cgroup maximum MiB |
| --- | --- | --- | --- | --- |
| Single .104 |247.963 /248.057|940.725|1028|2100–2168|
| Single .105 |249.278 /249.128|940.725|1028|2242–2332|
| Two-host DDP |1681.584 /1682.027|1114.484|1224|2792–3048|
| Two-host classic FSDP |6885.565 /6885.373|768.366 /770.856|1108 /1112|3260–3448|

For this fixture, one board is fastest. DDP is about6.8x slower and FSDP about27.8x
slower than .104 single; this is not a universal ratio. Both physical eth0 links
report1000Mb/s. Actual steady collective input/rank/update: DDP183,484,416B;
FSDP gathers707,186,688B plus reductions183,484,416B. These are instrumented input
bytes, not total NIC wire traffic. DDP delegates to Torch's native averaging hook;
FSDP delegates unchanged native all-gather/reduce-scatter. No transport changes.
The observed communication volume and timings are consistent with Ethernet being
the dominant distributed cost; this is an inference, not a traced time breakdown.

FSDP owns22,935,552 master and AdamW-moment elements/rank, exactly half of single/
DDP45,871,104. AdamW tensor bytes366,969,136 single/DDP versus183,484,636 on .104 and183,484,528
on .105 for FSDP (small step-scalar ownership differences). Its live GPU allocator peak falls
about18% versus single, while reserved memory and measured cgroup memory rise.
Thus the smaller GPU allocator counter is NOT proof of lower physical board RAM
or a useful larger-model fit. CPU allocator/runtime/collective contributions are
not separately attributed here; D33 must isolate fresh training-only processes
before using these measurements to select a larger model.

Sampled host MemAvailable during measured training stays at least9.27GiB across
all workers; minimum including CPU validation is8.635GiB. All sampled host and
cgroup swap usage is zero; OOM/oom_kill counters remain zero. Whole-validation
cgroup peak is at most4745.49MiB. CPU/GPU share physical RAM: do not add device,
process and cgroup counters as independent memory pools. Sampling is200ms;
kernel memory.peak captures an aggregate cgroup high-water mark separately.

The seven-update training segment (two warmups+five measured, including control/
bookkeeping between windows) takes about1.85–1.98s single,11.87–12.89s DDP,
48.19–48.49s FSDP. Complete guarded phases take33–34s,53–55s,93–94s respectively.
Those phase times include imports, CPU qualification, agent orchestration and exit;
they are not production training-only times. Both single baselines use the existing
two-node torchrun orchestration but initialize no training process group and do
no cross-board model computation. Their GPU workloads are independent one-board
baselines, run concurrently on separate boards.

## Correctness, admission and evidence

Every fresh worker checks one full global CPU FP64 loss/backward, ALL76 named
parameter gradients/45,871,104 elements at original3e-6/5e-4 bounds, and loss at
3e-6/5e-5. CPU norm and ALL clipped gradients pass D22's1e-7/1e-5 and1e-8/5e-5
limits. CPU-double AdamW consumes the predicted/checked clipped GPU gradients and
checks ALL parameters/moments/counters at original D21/D22 bounds. The first
loss8.477344513 matches CPU8.477344611. Five later finite global losses agree
across both orders/modes/hosts within1e-5; they are not a convergence test or
bitwise cross-topology replay. Previous native checkpoint qualifications remain.

The fixed sizing/admission helper refuses before model allocation unless host
MemAvailable≥10GiB and device free≥4GiB. It records parameter/state estimates with
activation/bucket/temporary/runtime exclusions rather than promoting a formula
to physical fit.64 local tests include insufficient-host, insufficient-device and
unknown-mode refusals. No deliberate OOM, native build or cache deletion.

**72 groups/12 acknowledged guards**,64localtests/Black/criticalRuff+F401/diff pass.
No failed GPU phase, no relaxed bounds and no full upstream CI claim. Exact
commands/PIDs/CIDs/guard nonces/ACK/kernel logs in evidence/distributed-d32-r1/
{single,ddp,fsdp}-{a,b}, sequence.json/controller logs. Sequence controllers
26567/25318 and exports29863/30120 exit0. Original metadata/source evidence
retained on hosts; kernel/artifacts/distributed-d32 contains overlays/raw pilot
JSON/metadata exports. Archive identities:

- .104:689853B SHA15f13d943bc8541654fc9c8c504a6374616dd86d9183b59f80c1d7b719c61337.
- .105:657137B SHA5f3b45e1fc26481c871c68f453e14b530c708ec885e24576753b5de76075224b.

PLAN-01 EX reservations+SH runtime locks/4CPUs/cpuset0–3/10GiB RAM/12GiB RAM+swap/
512PIDs/256MiBshm/600s recovery/460s child/15s termination retained. Kernel logs/
registries unchanged/final idle/no owned jobs/containers/armed guards. Boots
2d86d523(.104r2)/e902e636(.105r3) unchanged; PLAN-02 MIOpen381eac05/Torch a524/RCCL/
D09/image/packages/defaults/fallbacks retained. No prefix mutation/reboot/download/
foreign stop/user app inspection. Docs/flow/manifest snapshots synced BOTH hosts.
Historical Python CPU fault remains distinct from fixed VM heap/D05 exit/D21 load.

Next PLAN-05 / D33: separate source-bound CPU qualification from fresh training-only
workers, repeat the representative memory control without preceding CPU oracle/
audit work, then use that measured physical-memory envelope for larger
FSDP initialization/admission planning. Keep large-fit and speed claims withheld;
choose safe bounds before allocating a larger model.
