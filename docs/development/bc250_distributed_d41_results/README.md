# D41: larger CPU oracle admitted and published

Completed 2026-11-09 on **.104 only**. Producer implementation `3afa04930`,
independent finite-file audit `f75cb361e`. PLAN-05 remains open.

A native **1,024,471,040-parameter FP64 CPU GPT2 oracle** completed forward and
backward with checkpointing and per-gradient streaming. All244 gradients are
finite, round-trip bit-exact to raw storage and covered exactly once; no gradients
remain after backward. Peak CPU-container memory is **9.05GiB**, host available
memory stays above5.36GiB, swap/OOM0. A fresh352.96M control exactly matches the
qualified D35 loss and ALL initial/gradient hashes. This publishes a CPU reference;
**1.024B GPU fit/training is still unqualified**.

## Shape, provenance and stored files

Width2048/20 GPT2 blocks/32 heads/vocab8192/positions256, sequence128, full global
batch2/micro1, eager attention/dropout0/cache off, tied embedding/head. The same
qualified D35 deterministic construction, shifted CE and data are reused through
an explicit shape selector; original sources are unchanged. FP32 initialization
is promoted to FP64. Package identities/checkpoint source match the installed
native Torch/Transformers. The oracle manifest binds the original ten source
files plus six checkpoint/stream/shape/producer helpers separately from future GPU
code. No optimizer, process group or elastic agent executes in the CPU producer.

`cpu_d41_bounded_launch.py` uses the installed D09 ordinary Torch launcher and its
identity-checking bootstrap, removes both GPU device mappings, and runs one CPU
process. All parameters/gradients are CPU FP64; CUDA remains uninitialized. The
control uses existing D35 refs and one temporary96MiB roundtrip file at a time,
removed after checks. Large mode retains each raw output instead of a temporary
full in-memory gradient set. Initial FP32 and gradient FP64 tensors round-trip
exactly; initial values are checked again in FP64 after backward.

The new488 raw files occupy **12,293,652,480B =11.449GiB**, plus the source-bound
manifest at:

```text
/home/nik/bc250work/evidence/distributed-d41-r1/large-oracle/reference/
```

Manifest SHA: `99b36b92073fddf2e68da39525392b202992ac8263693e8b2247866165077efb`.
CPU FP64 loss: **9.416439180620552**. Keep this reference directory; do not treat it
as cleanup debris or copy its full payload to105/workstation. Original D35 refs on
both hosts are unchanged. Final exports omit all raw payloads.

Native nonreentrant checkpointing uses exact `use_reentrant=False` and
`preserve_rng_state=True`; runtime partial/module checks pass,21 capable modules,
20 block entries after forward/40 after backward. Nonreentrant early-stop can stop
recomputation before a full second forward. All244 hooks run once, maximum visible
gradient16,777,216 elements, largest roundtrip128MiB, zero retained grads. Dropout0
CPU RNG bytes are unchanged; no stochastic/general graph claim.

## Admission and safety

Large CPU budget before allocation: FP64 parameters7.633GiB +1.25GiB runtime/
launcher allowance +0.5GiB checkpoint/engine allowance +three largest128MiB tensor
buffers =9.758GiB, below the10GiB soft threshold. This is an explicit planning
allowance, not a scaling-based fit claim. Actual execution now qualifies this one
shape. Require at least12GiB host available, planned host reserve3GiB, full output
payload+5GiB disk reserve+64MiB metadata allowance. Worker checks headroom at phase
and per-tensor boundaries. Guardian samples100ms and stops only its labeled
container if memory reaches10GiB, host available falls below3GiB, or disk below5GiB.
No headroom stop was needed. Hard cgroup limit11GiB retains1GiB beyond the soft
threshold; combined RAM+swap limit12GiB. This explicit CPU-only limit differs from
the earlier10GiB fixture cap and leaves the native stack unchanged.

Four CPU/cpuset0–3,512PIDs/256MiB shared memory,600s recovery guard/460s child bound,
PLAN-01 EX reservation+SH runtime lock and ACK only after container exit remain.
Owned-file fsync/fadvise only; no global cache flush. Disk after the producer is
5.424GiB free, above reserve. Future builds must recheck this tight budget.

| Shape | CPU loss | Cgroup peak MiB | Process VmHWM MiB | Min host available GiB | Max roundtrip MiB | Worker / guarded seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 9.364227904214559 | 3748.035 | 4046.621 | 10.674 | 96 | 62.8 / 71.7 |
| large | 9.416439180620552 | 9268.129 | 9498.918 | 5.365 | 128 | 172.9 / 181.8 |

Counters overlap; do not add RSS/cgroup/host observations. Different model shapes
and the new agent-free CPU launcher make this no matched training-performance
comparison. CPU timings include model construction, backward/recompute, publication,
roundtrips and full file hashing. No GPU speed/memory/capacity, long-run convergence
or universal correctness claim follows.

## Final qualification

16 CPU check groups/two acknowledged guards;100 local tests, Black/critical
Ruff+F401/diff pass. New admission tests exercise disk/RAM boundaries and soft stops;
new finite-file tests distinguish IEEE finite boundaries from Inf/NaN for both
widths/signs and reject misalignment. Three private metadata archives with changed
source/hook count/admission policy refuse. D35 provenance is replayed offline.

The final r2 exporter independently rescans **all1,024,471,040 initial FP32 and
all1,024,471,040 gradient FP64 values**, checks finite exponent patterns and SHA,
in bounded1MiB file chunks without loading Torch or recreating the model. This
strengthens initial-value finiteness validation beyond the producer's finite-loss/
gradient and exact-weight assertions. Original D35 and new raw refs are rehashed.
The earlier r1 metadata export is retained as preliminary evidence before this
additional scan; the CPU jobs were not rerun or modified. Final r2 is authoritative.
No failed CPU model phase/headroom abort/relaxed bound/full upstream CI claim.

Registry/kernel logs unchanged, .104 boot2d86d523-c909-415b-86fa-a75be9460dde/r2,
no new native install/build/download/reboot/prefix mutation or user application work.
.105 received documentation only, no oracle/build/runtime change. Both remain idle.
Historical Python CPU fault remains open and distinct from fixed VM-heap/D05/D21;
these clean CPU exits do not establish a universal repair.

## Evidence and replay

Tools104: `~/bc250work/tools/distributed-d41`.
Evidence104: `~/bc250work/evidence/distributed-d41-r1/{control,large-oracle}`.
Logs: `evidence/distributed-d41-{control,large-oracle,export,export-r2}-controller.log`.
Guardians22608/23355, initial exporter24339 and final exporter25094 exited0. Exact
commands/CID/UUID/ACK/cgroup/headroom samples/source SHA remain in final records.
Local metadata/source transfers: `kernel/artifacts/distributed-d41`.

Run only on free104 as root; existing phase directories deliberately refuse reuse:

```sh
python3 ~/bc250work/tools/distributed-d41/guard_distributed_d41_on_host.py --shape control --label control
# After successful control/ACK, only if current admission still passes:
python3 ~/bc250work/tools/distributed-d41/guard_distributed_d41_on_host.py --shape large --label large-oracle
```

Large mode requires the same-source successful control. Export once per revision
with `export_distributed_d41_on_host.py`; it produces `distributed-d41-evidence-r2.tar.gz`.
Local replay:

```sh
python3 therock/build_tools/bc250/verify_distributed_d41_exports.py \
 --host104 kernel/artifacts/distributed-d41/104-evidence-r2.tar.gz \
 --baseline104 kernel/artifacts/distributed-d35/104-evidence-r1.tar.gz \
 --baseline-results therock/docs/development/bc250_distributed_d35_results/verified-results.json \
 --output therock/docs/development/bc250_distributed_d41_results/verified-results.json
```

[Verified results](verified-results.json), [negative controls](negative-controls.json).

## Next action

D42: admit and qualify a source-bound1.024B two-host per-unit FSDP GPU audit using the D41 oracle on104 and bounded remote reads on105. Review host/GPU/scratch budgets and the verification/startup protocol before launch: the two remote12.294GB verification passes alone exceed the old120s PG timeout on1Gb/s Ethernet. Add an explicit readiness barrier or reviewed bounded timeouts/guards; do not blindly reuse D40 timing. Keep original loss/raw/norm/clip/AdamW gates,128MiB reader cap, old refs/fallbacks/defaults. No deliberate single-host OOM; the all-resident single FP32 AdamW state floor exceeds RAM. GPU capacity remains unqualified until the actual guarded audit passes; training-only measurement is separate.

The old352M two remote verification passes had a smaller startup budget. For the
new12.294GB payload, two passes alone need at least~197s at ideal1Gb/s, already over
120s before hashing/import overhead. Resolve that coordination before a GPU test.
Six important work packages remain; PLAN-05 is not yet a GPU capacity proof.
