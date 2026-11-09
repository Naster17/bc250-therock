# D34: bounded 159.8M GPT2 qualification and fresh training

This owned opt-in fixture extends PLAN-05 to 159,807,488 all-trainable FP32
parameters. It does not change native Torch, ROCm, runtime defaults or user
applications. Both boards use additive `~/bc250work/tools/distributed-d34`.

The fixed eager GPT2 has width1024, 12 blocks, 16 heads, vocabulary8192,
positions256, sequence128, tied input/output weights, dropout0 and cache disabled.
Single and two-host classic FSDP match global batch2/micro1. Each FSDP rank has
one local micro, so it does not retain unsharded `no_sync` gradients. Native
AdamW uses lr1e-4/epsilon1e-5/weight-decay.01/foreachFalse and root clip.5.
There is no AMP, checkpointing, PEFT, compilation or model download in this batch.

## Qualification and memory bounds

Separate qualification workers on each host and in each mode compare the global
loss and ALL148 gradients with CPU FP64 at unchanged D32 tolerances. They check
identical initialized parameters/tied weights, an independent CPU root norm,
ALL clipped gradients and ALL AdamW updated values. A CPU AdamW optimizer exists
for only one parameter at a time; the largest pair of FP64 moment arrays totals
16,777,216 elements (128MiB). This is a scratch-state bound, not total peak RAM.
Local GPU AdamW moments/counters are checked against actual clipped local
shards. No full CPU optimizer or FSDP optimizer-state export is retained.

All four qualification workers observe GPU loss9.19343376159668 versus
CPU FP64 loss9.193433560988957; maximum absolute gradient error is
1.8551279880318905e-7. Root norm is about7.061389 and clip coefficient.0708075861.
Single owns159,807,488 master/AdamW elements; FSDP owns79,903,744 on EACH rank.
These are first-update checks for this shape, not arbitrary-model convergence,
checkpoint/restart or other-precision qualification.

Fresh measurement workers bind the qualification JSON digest, exact math/helper
sources, shape, mode, rank and Torch identity. Guardian additionally checks the
qualified successful result, ACK and runtime registry under reservation+SH lock.
Native Transformers5.2 and three modeling hashes are checked. Measurement workers
forbid full-parameter audits/full optimizer exports and never execute the oracle.
They perform the qualified first update, two warmups and five measured updates;
mode order is single→FSDP then FSDP→single, with fresh CIDs/PIDs/UUIDs. Initial
losses and the later cross-mode trajectories are checked separately. Timings omit
control collectives; complete guarded-phase times include imports/orchestration.

## Replay

Coordinate each command/port on BOTH idle hosts as root. Qualification directories
are intentionally immutable; use a new evidence series for a changed fixture.
The qualified series is `evidence/distributed-d34-r1`.

```bash
doas python3 ~/bc250work/tools/distributed-d34/guard_distributed_d34_on_host.py \
  --mode single --phase qualify --label qualify-single --port 31210
# After successful qualification on BOTH:
doas python3 ~/bc250work/tools/distributed-d34/run_distributed_d34_on_host.py
```

The sequence runs qualification-fsdp, single-a, fsdp-a, fsdp-b, single-b and stops
on a failure. It leaves unrelated processes/containers untouched. It retains
PLAN-01 EX cooperative reservation, runtime SH lock, four CPUs/cpuset0–3,
10GiB cgroup RAM/12GiB combined RAM+swap/512PIDs/256MiBshm, 600s recovery guard
and460s child deadline. Admission requires host available≥10GiB and device
free≥6GiB before model allocation. The floor estimate excludes activations,
gathers, allocator reservations, runtime and communication. No deliberate OOM.

```bash
python3 therock/build_tools/bc250/verify_distributed_d34_exports.py \
  --host104 kernel/artifacts/distributed-d34/104-evidence-r1.tar.gz \
  --host105 kernel/artifacts/distributed-d34/105-evidence-r1.tar.gz \
  --output therock/docs/development/bc250_distributed_d34_results/verified-results.json
```

The verifier checks immutable source identities, qualification digests, ALL-state
ownership, fresh processes, trajectories, both-rank timing, guard ACKs, kernel
logs, registry preservation and memory/swap/OOM observations. Original sources,
results and logs remain on the hosts. Local metadata archives omit tensors/user data.

## Results and limitations (2026-11-09)

Implementation `58e91cb5d`. [Verified metadata](verified-results.json):
64 executed qualification/measurement groups, 12 ACKed guards, 71 local tests,
Black/critical Ruff+F401/diff pass. Three private metadata-copy source/proof/
measurement corruptions refuse as required; original archives remain unchanged.
No failed GPU phase or relaxed numerical bound; no full upstream CI claim.

| Matched mode | Median update, both orders/hosts | Peak live GPU / rank | Peak reserved GPU / rank | Sampled measured cgroup max / rank |
| --- | --- | --- | --- | --- |
| Single | 150.288–151.092ms | 2737.507MiB | 2748MiB | 1082.047–1082.340MiB |
| Two-host FSDP | 8088.982–8103.416ms | 1493.502–1495.365MiB | 1956–2064MiB | 1644.629–1647.766MiB |

FSDP lowers live GPU allocation about45.4%, yet updates are about54x slower here.
Each rank's measured native collective INPUTS are621,924,352B all-gather plus
639,229,952B reduce-scatter per update, not wire-byte totals. Both eth0 links are
1000Mb/s. Complete guarded measurement phases are17.5–18.2s single/81.6–82.1s
FSDP; they include imports and two-node agent coordination, even for independent
single workers. Qualifying CPU phases take37–38s single/60–61s FSDP separately.
Do not compare D33/D34 timings as a speedup: shapes, batch and token counts differ.

All later losses match across modes/hosts/orders within1e-5. Qualification sampled
cgroup maxima are4314–4329MiB single/4927–4968MiB FSDP; kernel whole-cgroup peak
is at most4991.34MiB. Minimum host available during qualification is5.811GiB;
during fresh measured training,10.093GiB. Host/cgroup swap and OOM counters stay0.
Sampling is200ms and may miss transients; memory.peak records the cgroup peak.

BC-250 shares physical CPU/GPU RAM. GPU live/reserved, cgroup, RSS and whole-host
available counters overlap and cover different allocations; never sum them as
independent pools. Single live GPU allocation exceeds its cgroup reading. Host
available during measured training is10.107–10.234GiB single and10.093–10.337GiB
FSDP: these runs do NOT establish a clear whole-host capacity improvement, despite
lower live GPU allocation. Both modes safely fit this model. A model unable to
fit one board but safely fitting two remains unqualified. PLAN-05 remains OPEN.
352.96M/1.024B remain planning-only. No claim of universal performance, convergence,
long-run stability, new restart coverage, arbitrary FSDP layouts or native changes.

Evidence on BOTH hosts: `~/bc250work/evidence/distributed-d34-r1/`, controller
logs and `sequence.json`; original archives listed in verified-results.json.
Guardian PIDs104:27806,28632,29278,29666,30412,31143;
105:12390,13628,14498,14981,16058,17125. Sequence28630/13627 and exports31744/17992
all exit0. Sources, registries, exact commands, CIDs, worker UUIDs/start ticks,
guard nonces and kernel logs are retained in the archives. Local artifacts are
`kernel/artifacts/distributed-d34/`; original host evidence is preserved.

Final idle preflight passes BOTH; no owned jobs/containers/armed guards remain.
Kernel logs/registries unchanged. Boots2d86d523(.104/r2) and e902e636(.105/r3)
unchanged. Native PLAN-02 MIOpen381eac05/Torch a524/RCCL/D09/image/packages/defaults/
fallbacks unchanged. No native build, reboot, download, prefix mutation, app
inspection or foreign stop. The initial non-root staging attempt was refused by
host directory permissions; root staging succeeded before any GPU phase. Its
short prestage error log remains preserved. Historical Python CPU fault is still
distinct from the fixed VM-heap/D05 exit/D21 load defects.

Next: D35: implement staged, source-bound CPU gradient reference artifacts and storage/memory admission for the352,957,440-parameter candidate, so the full CPU oracle need not coexist with GPU training state. Qualify that reference pipeline before any352.96M GPU admission, then shape-specific single/FSDP checks and fresh measurements. Keep1.024B blocked; no extrapolated fit or deliberate OOM.
