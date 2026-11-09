# D37: exact CPU gradient streaming with bounded scratch

Implementation `2610b2b76`, additive `~/bc250work/tools/distributed-d37` on BOTH
hosts. This is CPU-reference infrastructure for larger-model qualification.
GPU execution, ordinary training, native libraries and defaults are unchanged.
The fixed352,957,440-parameter FP64 CPU GPT2 uses D35's identical FP32-initialized
weights, global2/sequence128/positions256, dropout0/cacheFalse and one full-global
backward. CPU activation checkpointing is disabled. CUDA remains uninitialized;
there is no training process group or GPU tensor in either worker.

## Implementation and exact checks

A fresh retained-gradient control computes the complete CPU backward and checks
all gradients against the retained D35 raw FP64 files. The streaming variant
registers native `register_post_accumulate_grad_hook` on each unique parameter.
Each completed leaf gradient is checked against its original reference, written
to one owned raw scratch file, read back exactly, then removed; `p.grad=None`
releases that parameter's gradient before the next callback. Hooks never step an
optimizer. This full-global-backward recipe does not qualify multi-micro gradient
accumulation, arbitrary shared-parameter graphs or production training hooks.

All148 named parameters (including the tied embedding/head) complete exactly once.
The hook rejects reentrancy, duplicate/unknown completion and incomplete coverage.
The maximum visible model `.grad` elements at a callback is12,582,912; after
backward no parameter gradients remain. This does not bound internal autograd or
reference-read buffers: those are included in measured RSS/cgroup peaks. Scratch
is at most100,663,296B (96MiB), only one file, with5GiB free-disk reserve. No full
3.9446GiB reference copy, full gradient spool or global cache flush is created.
Original references are read-only and reverified at export.

Both variants check every initial parameter against the FP32 reference promoted
to FP64, with exact FP64 equality before and after backward. ALL gradients match
FP64 bytes/SHA256 exactly, and CPU loss is exactly9.364227904214559 across both
variants/hosts and D35. Hook handles are removed, successful owned scratch is
removed, and original weights/references remain unchanged. Old oracle source
identity and the additional D37 source identity are checked separately.

## Results (2026-11-09)

[Verified metadata](verified-results.json):24 final CPU check groups/four ACKed
guards,83 local tests,Black/critical Ruff+F401/diff pass. Four new local tests cover
once-only completion, unknown/duplicate/missing callbacks and disk refusal.
Three private metadata-copy source/hook-count/gradient-hash corruptions refuse,
including synchronized rank/final/handoff edits. No failed final CPU phase or
relaxed numerical bound; no full upstream CI claim. No fresh GPU tests are needed
for this CPU-only harness change; D35/D36 GPU qualifications remain unchanged.

| Host | Peak cgroup, retained→stream | Peak process RSS, retained→stream | Host available minimum, retained→stream |
| --- | --- | --- | --- |
| .104 | 7556.621→5218.113MiB | 7563.840→5158.750MiB | 7.033→9.212GiB |
| .105 | 7481.609→5202.461MiB | 7492.652→5145.844MiB | 6.984→9.109GiB |

The streaming path lowers measured cgroup peak by30.5–30.9% in this fixture.
Retained gradient elements fall352,957,440→0 after backward. Streaming cgroup
usage after backward is4818–4833MiB; the actual process still retains its full
FP64 parameter set plus runtime/allocator buffers. Counters overlap; do not sum
RSS/cgroup/host available. Kernel cgroup peak and VmHWM are recorded separately
from200ms samples. Swap and OOM counters remain0 on BOTH hosts.

Streaming trades memory for I/O/check overhead: complete guarded phases are
about76s versus57s retained; worker durations57–63s versus40–44s include reference
validation, forward/backward, per-gradient checks/I/O and parameter rechecks.
There is no CPU/GPU training speedup or timing-significance claim from one paired
order. Model size, graph, precision and runtime remain fixed for the comparison.
1.024B CPU/GPU fit, other graph/hook patterns and long-run stability are unqualified.

An initial retained-control draft (`distributed-d37-r1`) passed gradients but
compared initial weights after an FP32 cast. Before streaming, direct FP64 weight
equality was added and BOTH corrected controls/streams rerun in final r2. Initial
results and source backup `tools/distributed-d37-preliminary-r1` are preserved,
not counted among the24 final groups; local overlay-r1 preserves the first source.
This was a harness-validation tightening, not a demonstrated Torch numerical bug.

## Replay and evidence

Keep completed runs immutable. On a clean target with the qualified D35 reference,
coordinate each command/port on BOTH idle hosts (the installed D09 launcher still
coordinates two independent CPU workers):

```bash
doas python3 ~/bc250work/tools/distributed-d37/guard_distributed_d37_on_host.py \
  --variant retained --label retained-control --port 31511
# After successful controls on BOTH:
doas python3 ~/bc250work/tools/distributed-d37/guard_distributed_d37_on_host.py \
  --variant stream --label stream-gradient --port 31512
```

The final series is `~/bc250work/evidence/distributed-d37-r2`. Admission requires
host available≥12GiB,96MiB scratch+5GiB disk. PLAN-01 EX cooperative reservation+
SH runtime lock/idle Podman/Docker/GPU ownership preflight are retained, along with
4CPUs/cpuset0–3/10GiB RAM/12GiB combined RAM+swap/512PIDs/256MiBshm/600s recovery/
460s child. Guard ACK follows owned-container exit. No foreign stop, download,
reboot, native build or prefix mutation. User applications are uninspected.

```bash
python3 therock/build_tools/bc250/verify_distributed_d37_exports.py \
  --host104 kernel/artifacts/distributed-d37/104-evidence-r2.tar.gz \
  --host105 kernel/artifacts/distributed-d37/105-evidence-r2.tar.gz \
  --baseline104 kernel/artifacts/distributed-d35/104-evidence-r1.tar.gz \
  --baseline105 kernel/artifacts/distributed-d35/105-evidence-r1.tar.gz \
  --baseline-results therock/docs/development/bc250_distributed_d35_results/verified-results.json \
  --output therock/docs/development/bc250_distributed_d37_results/verified-results.json
```

The verifier replays published D35 provenance, binds original manifest/worker and
all source bytes, checks exact values/hook counts/retained state/scratch bound,
fresh process/container identities, ACKs/kernel logs/registry and memory/swap/OOM.
Raw original tensors stay on hosts; exports contain metadata/source/logs only.

Guardians104:32310/630,105:10185/11364; exports1769/13133 all exit0. Preliminary
r1 guardians31542/9018 also exited safely; their original evidence remains.
Local artifacts: `kernel/artifacts/distributed-d37/`. Final idle preflight passes
BOTH; no owned jobs/containers/armed guards. Registry/kernel logs unchanged;
boots2d86d523(.104/r2),e902e636(.105/r3) unchanged. Native PLAN-02 MIOpen381eac05/
Torch a524/RCCL/D09/image/defaults/fallbacks preserved. Historical Python CPU fault
remains distinct from fixed VM heap/D05 exit/D21 optimizer restoration.

Next D38: qualify native non-reentrant CPU activation checkpointing combined with
this streaming path, first against the existing352.96M reference. Require exact
loss/initial values/ALL FP64 gradients, one hook per parameter, actual recomputation
and measured CPU memory. Preserve one-tensor scratch/original refs. Then resolve
1.024B oracle/storage and clipping-spool admission separately; D37 memory savings
do not admit the larger shape. PLAN-05 remains OPEN; six important packages remain.
