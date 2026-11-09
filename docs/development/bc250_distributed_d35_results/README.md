# D35: staged CPU references and bounded352.96M qualification

Implementation `02107159f` adds an owned, fixed352,957,440-parameter FP32 eager
GPT2 fixture under `~/bc250work/tools/distributed-d35` on BOTH hosts. Native
Torch/ROCm/default launchers and user applications are unchanged. The model has
width1536/12blocks/24heads/vocab8192/positions256/sequence128/tied weights,
dropout0/cacheFalse, global2/micro1, native AdamW1e-4/eps1e-5/wd.01/foreachFalse
and root clip.5. No AMP, PEFT, checkpointing, compilation or downloads.

## What changed

D34's co-resident full CPU oracle is not used for this larger GPU qualification.
A separate **CPU-only** producer builds the identical deterministic FP32 model,
converts it in place to FP64 and computes the full-global loss/ALL148 gradients.
It writes FP32 initial values plus FP64 gradients as raw little-endian files;
no pickle, checkpoint loaders or global cache flushing. SHA256, exact size,
shape/name/source identity and full parameter coverage are bound in a manifest.
Every tensor is read back and compared bit-for-bit before publication. Both hosts'
complete reference manifests match exactly. CUDA remains uninitialized in the
producer; its process exits before GPU qualification starts.

The payload is **4,235,489,280B (3.9446GiB) per host** in
`evidence/distributed-d35-r1/cpu-reference/reference/`. CPU production reserves the
payload plus5GiB free disk; GPU qualification reserves a temporary FP32 actual-
gradient spool (1,411,829,760B) plus5GiB. Existing files refuse overwrite. Partial
or failed evidence is preserved. Only newly-created successful audit-spool files
are deleted after ALL clipping checks; original references/evidence are retained.

The guardian verifies the successful/ACKed source-bound CPU stage and every raw
file before launching the GPU child. The child repeats manifest/source/size/hash
checks before CUDA model allocation. CPU reference data is read one parameter at
a time. The actual raw GPU gradients are temporarily spooled to avoid holding a
second full CPU gradient set during clipping checks. CPU AdamW checks stream one
parameter at a time, including the first-step weight-decay/epsilon behavior;
ALL GPU moment states/counters are checked on optimizer-facing local shards.
No full CPU optimizer or FSDP optimizer-state export is retained. The maximum
per-parameter FP64 moment pair is192MiB, not a total process-memory bound.

CPU FP64 global loss is9.364227904214559. This is a deterministic shape-specific
reference, not a general model/precision certificate. GPU admission requires
host available≥11GiB/device free≥9GiB; CPU producer requires host available≥12GiB.
The fixed state floors exclude activations, allocator reservations, gathers,
runtime and communication. Shared physical CPU/GPU RAM counters overlap.

## Execution and replay

The declared series is `~/bc250work/evidence/distributed-d35-r1`; it must not
already exist when replaying on a clean test target. Do not overwrite completed
runs. Existing targets retain their qualified data for inspection. Coordinate the
same commands/ports on BOTH idle hosts, using rootful Podman through the installed
D09 distributed launcher:

```bash
doas python3 ~/bc250work/tools/distributed-d35/guard_distributed_d35_on_host.py \
  --mode single --phase reference --label cpu-reference --port 31310
# After successful CPU stages on BOTH hosts:
doas python3 ~/bc250work/tools/distributed-d35/run_distributed_d35_on_host.py
```

Sequence: qualify-single, qualify-fsdp, single-a, fsdp-a, fsdp-b, single-b.
It stops on failure. Every phase retains PLAN-01 EX cooperative reservation+SH
runtime lock, idle Podman/Docker/GPU preflight, 4CPUs/cpuset0–3,10GiB cgroup RAM/
12GiB combined RAM+swap,512PIDs,256MiBshm,600s recovery and460s child deadline.
Recovery ACK occurs only after the owned container exits. Foreign workloads are
never stopped. CPU-only stages use launcher orchestration but no training PG.
Each FSDP rank uses one micro; no unsharded `no_sync` gradient accumulation.

Fresh measurement workers require both the CPU-artifact manifest and the
successful mode/rank-specific GPU qualification digest. They forbid full-state
exports/full-parameter audits, perform initial update+two warmups+five timed
updates in both mode orders, and record actual native collective INPUT bytes,
allocator/RSS/cgroup/host available/swap and source/process/container identities.
CPU oracle execution is absent from every GPU worker. Timing excludes control
collectives; complete guarded phase durations include imports and launch agents.

```bash
python3 therock/build_tools/bc250/verify_distributed_d35_exports.py \
  --host104 kernel/artifacts/distributed-d35/104-evidence-r1.tar.gz \
  --host105 kernel/artifacts/distributed-d35/105-evidence-r1.tar.gz \
  --output therock/docs/development/bc250_distributed_d35_results/verified-results.json
```

Exports contain metadata, exact sources, raw-file digests, original logs and guard
ACKs. Multi-GiB raw references remain on their original hosts; exports do not
contain those tensor payloads. Final host export reverifies all original raw files.
The offline verifier checks the recorded manifest identities/coverage and both
hosts' identical manifests; replaying the original blob hashes themselves requires
access to the host payloads or rerunning the owned CPU producer in a new series.

## Results (2026-11-09)

[Verified results](verified-results.json):72 executed groups/14 ACKed guards,
76 local tests/Black/critical Ruff+F401/diff pass. Five new file-format tests cover
exact roundtrip/no-overwrite, corruption/truncation, path/symlink refusal, storage
reserve and manifest integrity/coverage. Three additional private archive source/
manifest/measurement corruptions refuse; original evidence is unchanged. No failed
GPU phase or relaxed mathematical gate; no full upstream CI claim.

Both modes/hosts pass CPU FP64 loss/ALL148 gradients, exact initialized/tied
weights, independent CPU norm/ALL clipped entries, CPU-double ALL AdamW updated
values and ALL local moment/counter checks at unchanged D32 tolerances. GPU
loss9.364227294921875 versus CPU9.364227904214559; maximum absolute gradient
error2.1940768522681064e-7. Native norm8.4572229385 versus CPU8.4572224656;
clip coefficient.0591210581. Single owns352,957,440 master/moment elements;
FSDP owns176,478,720 EACH rank. The148 raw actual-gradient digests are retained
in qualification metadata, while their successful temporary spool is removed.

| Matched mode | Median update across hosts/orders | Peak live GPU / rank | Peak reserved GPU / rank | Measured cgroup max / rank |
| --- | --- | --- | --- | --- |
| Single | 265.535–266.563ms | 5863.493MiB | 5912MiB | 1082.641–1083.191MiB |
| Two-host FSDP | 17968.643–17973.422ms | 3183.965–3189.670MiB | 4238–4254MiB | 1650.941–1651.590MiB |

FSDP uses about45.6% less live GPU allocation but updates are about68x slower.
Native collective INPUTS/rank/update are1,385,871,360B all-gather plus
1,411,829,760B reduce-scatter, not total NIC wire bytes. Both eth0 links1000Mb/s.
All initial and five later measured global losses agree across modes/hosts/orders
within1e-5. This is not convergence, large checkpoint/restart or arbitrary-model
qualification. Do not compare D34/D35 latency as a native performance improvement:
model size differs, and native code did not change.

Matched host-available minima are6.997–7.129GiB single versus7.940–8.081GiB FSDP;
within each host/order, FSDP provides0.927–0.956GiB additional observed headroom.
This is a bounded whole-host observation, not proof of a model that only fits two
boards. Both modes fit352.96M. GPU/RSS/cgroup/host counters overlap and must not be
summed. In particular, cgroup alone omits parts of shared GPU-memory accounting.
The CPU producer peaks7426–7550MiB cgroup; GPU qualification peaks2127–2128MiB
single/2503–2504MiB FSDP. Whole-worker measured-process cgroup peaks reach2129MiB
including initialization. Host available stays≥4.932GiB across all phases and
≥6.997GiB during measured training. Host/cgroup swap and OOM counters remain0.
Sampling200ms may miss transients; kernel cgroup peaks are recorded separately.

Complete guarded phases: CPU producer63–65s; GPU qualification72s single/131–132s
FSDP; fresh measurement28–30s single/170–171s FSDP. These include reference I/O,
imports and two-node agent orchestration, even for independent single workers;
they are not production training-only durations. Guards/resources/locks retained.

Original evidence: BOTH hosts `evidence/distributed-d35-r1/` and controller logs,
with exact commands/CIDs/UUIDs/start ticks/guard nonces and kernel logs. Guardian
PIDs104:11869,12781,13478,14492,14971,16185,17385;
105:8161,9572,10590,12173,12813,14743,16676. Sequence12780/9570 and exports18108/
17748 all exit0. Local artifacts: `kernel/artifacts/distributed-d35/`. Raw CPU
references remain on hosts and are verified again at export; metadata archives
omit those payloads. Sources/history/builds/caches/checkpoints/apps remain intact.

Final idle preflight passes BOTH; no owned jobs/containers/armed guards remain.
Kernel logs/registry/default prefix unchanged. Boots2d86d523(.104/r2) and e902e636
(.105/r3) unchanged; native PLAN-02 MIOpen381eac05/Torch a524/RCCL/D09/image/
packages/fallbacks retained. No build, reboot, download, prefix mutation, foreign
stop or application inspection. Historical Python CPU fault remains separate
from fixed VM-heap corruption/D05 exit/D21 optimizer restoration.

## Next capacity boundary

[CPU-only plan](next-capacity-plan.json) records the1,024,471,040-parameter
candidate and observed MemTotal/free disk. Its all-resident FP32 parameter+
gradient+AdamW floor is15.266GiB, above either board's physical RAM; two-rank floor
is7.633GiB/rank before excluded overheads. The unchanged full CPU FP64 oracle also
needs at least15.266GiB, and the unchanged reference payload is11.449GiB/host.
`.105` cannot satisfy that storage requirement plus5GiB reserve, or the unchanged
3.816GiB actual-gradient spool plus reserve. No1.024B CPU/GPU allocation was made.
These are lower bounds and refusal prerequisites, never GPU admission proof.

A whole-model full-parameter+gradient summon adds a theoretical3.816GiB per rank
above sharded state at1.024B, versus about192MiB for the largest single block.
The next audit implementation must be qualified on the existing352.96M references
before using this planning bound. CPU streaming/storage must be resolved separately.
PLAN-05 stays OPEN; no universal speed, precision, long-run stability or new
large-model restart claim. Six important work packages remain.

Next: D36: replace whole-model FSDP qualification summons with bounded per-FSDP-unit ALL-gradient/AdamW audits, first qualified against the existing352.96M source-bound CPU references at unchanged numerical gates. Then resolve streamed CPU-oracle and storage admission for1.024B. Keep1.024B CPU/GPU execution blocked until its own envelopes pass; no deliberate OOM, deletion of retained references or inference of larger-than-single fit.
