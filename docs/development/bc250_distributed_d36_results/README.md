# D36: bounded per-unit FSDP qualification

Implementation `22447a4a9` adds opt-in `tools/distributed-d36` on BOTH hosts.
The fixed352.96M FP32 GPT2, native backward/clipping/AdamW recipe and existing
source-bound D35 CPU oracle are preserved. No native Torch/ROCm/default/user
application change, CPU oracle rerun or new multi-GiB reference copy is required.

## Implementation

`Auditor` enumerates the public FSDP wrappers, normalizes their module names and
assigns every CPU-reference parameter name to its deepest owning wrapper. This
qualified GPT2 has13 units: the root (tied embeddings/positions/final norm) plus
12 blocks. Duplicate/missing/empty units refuse; prefix matching uses a component
boundary, so block1 does not capture block10. Three new local tests cover name
ownership, refusal cases and148-name coverage. The actual GPU worker checks the
expected wrapper set and all parameter shapes/coverage in every audit pass.

The unit path uses public `FSDP.summon_full_params(..., recurse=False,
writeback=False, with_grads=True)`. A process-local guard refuses whole-model
summons, including an accidental call through the old helper. Matrices outside
the active unit must remain one-dimensional optimizer-facing shards. Each
raw-gradient/clipped-gradient/updated-parameter pass visits ALL148 parameters
once in13 contexts, versus one whole-model context in the control. Only one unit
is materialized at a time. Native FSDP communication/reduction/clipping and AdamW
are delegated unchanged; no private handle layout or training loop fork is used.

Both paths retain D35's CPU FP64 global loss/ALL raw gradients, independent CPU
norm/ALL clipped entries, streamed CPU-double ALL AdamW updated values and ALL
local moment/counter gates. Exact per-parameter raw/clipped-gradient and updated-
value digests, plus local AdamW moment/counter digests, are recorded for paired
comparison. The temporary actual-gradient spool is owned by each new phase,
removed only after successful clipping checks; original D35 references stay intact.
CPU moment scratch remains bounded per parameter; no full CPU optimizer/export.

CPU gradient squares are summed in the original CPU manifest order, consistently
for both new paths. D35's FSDP named-parameter order differs: report-only CPU norm
moves8.457222465649355→8.457222465649352 and coefficient changes2.08e-17. The
GPU/CPU qualification gates are unchanged. Historical metadata comparison accounts
for double rounding in those two derived scalars; all raw-gradient hashes and
other common D35 reference fields stay exact. Paired D36 scalars/digests must be exact.

## Replay and evidence

The declared series is `~/bc250work/evidence/distributed-d36-r1`; preserve completed
runs. On a clean replay target with the original qualified D35 references present,
coordinate each command/port on BOTH idle boards:

```bash
doas python3 ~/bc250work/tools/distributed-d36/guard_distributed_d36_on_host.py \
  --audit whole --label whole-control --port 31410
# After successful controls on BOTH:
doas python3 ~/bc250work/tools/distributed-d36/guard_distributed_d36_on_host.py \
  --audit unit --label unit-audit --port 31411
```

Guardians retain PLAN-01 EX cooperative reservation/SH runtime lock, idle workload
preflight,4CPUs/cpuset0–3/10GiB cgroup RAM/12GiB combined RAM+swap/512PIDs/256MiBshm,
600s recovery/460s child deadline and ACK only after the owned container exits.
No foreign stop. GPU admission stays D35 host≥11GiB/device free≥9GiB. CPU raw-file
hashes/sizes/source identity are checked before allocation. Temporary spool plus
5GiB free disk is required; no original data is overwritten or removed.

Worker bytes reuse immutable D35 math sources and bind three additional D36 audit
sources. The old CPU source table is verified separately from the new audit table;
a changed CPU fixture cannot inherit the proof. Package modeling pins and native
D09 worker/agent identities remain checked. Original references are rehashed at
final export. Archives contain metadata/source/logs, not tensor payloads.

```bash
python3 therock/build_tools/bc250/verify_distributed_d36_exports.py \
  --host104 kernel/artifacts/distributed-d36/104-evidence-r1.tar.gz \
  --host105 kernel/artifacts/distributed-d36/105-evidence-r1.tar.gz \
  --baseline104 kernel/artifacts/distributed-d35/104-evidence-r1.tar.gz \
  --baseline105 kernel/artifacts/distributed-d35/105-evidence-r1.tar.gz \
  --baseline-results therock/docs/development/bc250_distributed_d35_results/verified-results.json \
  --output therock/docs/development/bc250_distributed_d36_results/verified-results.json
```

The verifier replays the hash-verified published D35 provenance, checks the exact
CPU manifest/worker identity, fresh CIDs/UUIDs, all new source bytes, complete audit
scope, unchanged numerical gates, paired digests, guard ACKs, registry/kernel logs,
OOM/swap/headroom and measured qualification allocation. Prior CPU executions are
reused, not counted as new tests. Whole-control and unit-audit are separate fresh
processes. Peak GPU statistics reset immediately before the first backward;
these are qualification/audit peaks, excluding earlier model initialization.
There is no training-throughput benchmark or universal model/capacity claim here.

## Qualified results (2026-11-09)

[Verified metadata](verified-results.json):24 executed GPU audit groups/four
ACKed guards,79 local tests,Black/critical Ruff+F401/diff pass. Three private
archive corruption cases (source bytes, synchronized scope coverage, synchronized
paired-gradient digest) refuse. The latter two update rank/final/handoff together
to exercise scope/pair checks beyond simple JSON-copy consistency. Original
archives remain intact. No failed GPU phase or relaxed GPU/CPU numerical gate;
no full upstream CI claim. Original CPU executions are reused, not recounted.

| Host | Whole audit peak live GPU | Unit audit peak live GPU | Host available minimum, whole→unit |
| --- | --- | --- | --- |
| .104 | 5456.828MiB | 2980.129MiB | 5.260→7.838GiB |
| .105 | 5450.629MiB | 2973.930MiB | 4.920→7.246GiB |

Peak live qualification allocation is about45.4% lower. Maximum materialized
parameter scope falls352,957,440→28,331,520 elements. Every pass covers all148
names once, root+12 blocks, three whole contexts versus39 unit contexts. Raw and
clipped gradients/updated parameters are byte-exact between paths and both hosts;
local AdamW moments/counters are byte-exact between paths on each rank. ALL CPU
loss/gradient/norm/clip/updated-value/local-moment gates pass unchanged. Each rank
still owns176,478,720 master/moment elements. Native training state is unchanged.

Reserved audit peaks6264→3734MiB(.104),6484→4136MiB(.105); whole-process cgroup
peaks remain about2501–2506MiB. Host/cgroup swap and OOM counters stay0; minimum
host available across all phases4.920GiB. Host counters/GPU/RSS/cgroup overlap:
do not sum them. The paired headroom gain applies to qualification; it is not
ordinary training memory saving, a training-speed claim, a larger-model fit proof
or long-run stability. Model initialization precedes the GPU peak reset. Complete
guarded phases144.3–144.6s whole/134.4s unit include imports/reference I/O/digests,
not production throughput. Only one paired order was needed for this correctness/
materialization scope; no timing significance claim.

Evidence BOTH: `evidence/distributed-d36-r1/{whole-control,unit-audit}` with exact
commands/logs/CIDs/UUIDs/start ticks and guard ACKs. Guardian PIDs104:20709/21890;
105:22408/24273; exports23300/26549 all exit0. Local artifacts are
`kernel/artifacts/distributed-d36/`. Original3.9446GiB D35 references remain on
BOTH hosts and are reverified at final export; no new full reference copy.

Final idle preflight passes BOTH, no owned jobs/containers/armed guards remain.
Registry/kernel logs unchanged; boots2d86d523(.104/r2),e902e636(.105/r3) unchanged.
Native PLAN-02 MIOpen381eac05/Torch a524/RCCL/D09/image/defaults/fallbacks retained.
No build/reboot/download/prefix mutation/foreign stop/application inspection.
Historical Python CPU fault remains distinct from fixed VM heap/D05 exit/D21 load.

PLAN-05 remains OPEN. D36 removes one audit-materialization prerequisite only;
1.024B execution remains blocked by CPU-oracle and storage constraints in the
[D35 capacity plan](../bc250_distributed_d35_results/next-capacity-plan.json).
Next CPU streaming should validate against existing reference files with at most
one parameter of disposable scratch, rather than allocating another full set on
.105. Its hook count, exact FP64 gradients and actual memory reduction need their
own proof before larger CPU/GPU admission. Six important packages remain.

Next: D37: implement CPU post-accumulate gradient streaming on the existing352.96M fixture. Compare ALL gradient bytes/loss/initial values against the retained D35 FP64 reference, verify one completed hook per named parameter and no retained full gradient set, measure CPU/RSS/cgroup/host peaks. Use only one parameter of owned scratch at a time; do not duplicate the3.9446GiB reference set. Qualify CPU checkpointing separately only if needed, then resolve1.024B CPU-oracle/storage admission. Keep1.024B CPU/GPU blocked; preserve original references and avoid deliberate OOM.
