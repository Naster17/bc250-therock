# D42: source-bound 1.024B two-host GPU audit

Qualified on 2026-11-09; independently exported/replayed on 2026-11-10.
Implementation: `48f58f74c`, unused-cache repair `f9f9ab563`, bounded CPU AdamW
reference `4582251e3`. Final candidate: `tools/distributed-d42`, series
`distributed-d42-r5` on both hosts. PLAN-05 remains open.

A complete FP32 FSDP forward, backward, gradient clipping and native GPU AdamW
update passed on two BC-250 boards for **1,024,471,040 parameters**. All244
parameters, raw/clipped gradients, updated values and both local optimizer moments
and step counters were checked. Both ranks have identical raw/clipped/updated
hashes. Each rank owns512,235,520 FP32 parameter elements and the corresponding
AdamW state. This qualifies one deterministic update of the owned GPT2 fixture;
training-only step time, repeated updates, checkpoint restart at this size and
other models/precision policies are separate work.

## Numerical coverage and provenance

D41 configuration:20 blocks/width2048/32 heads/vocab8192/positions256, sequence128,
global batch2/micro1, eager attention/dropout0/cache off/tied weights. D35's model,
shifted loss and optimizer construction remain intact. The immutable D41 CPU FP64
oracle manifest is `99b36b92073fddf2e68da39525392b202992ac8263693e8b2247866165077efb`.
Its488 raw files remain only on104;105 retains only the manifest and reads bounded
blobs. No raw-gradient spool, full CPU model or full-model CPU optimizer is created
in a GPU worker. CPU reference checks are explicit; model forward/backward and the
training optimizer execute on the native gfx1013 GPUs.

All original tolerances are retained (absolute/relative): loss3e-6/5e-5,
raw gradients3e-6/5e-4, norm1e-7/1e-5, clipped gradients1e-8/5e-5,
updated values4e-6/5e-5, first moments3e-7/5e-4, second moments3e-9/1e-3;
step counters equal1. Clip references use FP64 CPU gradients multiplied by the
coefficient derived from the checked actual-gradient norm. Maximum observed raw
error3.21960e-7; clipping error1.54533e-8 passes the combined absolute/relative bound.
GPU loss9.416439056396484 vs CPU9.416439180620552.

The fresh352.96M control preserves all qualified D39/D40 reference/state/scalar
hashes, except the intentional CPU moment-scratch accounting change. Per-unit
summons cover21 units/63 scopes at1.024B, without a whole-model summon. Every
updated parameter value is checked in1,048,576-element CPU AdamW chunks:1139
chunks per rank, no sampling. The largest FP64 moment pair is16MiB. The initial
FP32 tensor still occupies up to64MiB and other audits have separate scratch;
16MiB is not a total CPU-memory claim. A synthetic2M+13-element, three-chunk check
matches whole-vector CPU AdamW weights, moments and step **bit-exactly**, on both
hosts/control/large. Elementwise optimizer settings and first-step scope are fixed.

## Repairs driven by actual memory stops

The r3 large attempt completed backward but hit the3GiB host reserve during the
raw-gradient audit. r4 releases only unused GPU cache after wrapping/before
backward and at audit phase boundaries: about4GiB after wrapping and2GiB after
backward at this shape. Recorded live allocations are unchanged across each trim.
The r4 large attempt then passed all raw/clipped gradients and executed native
AdamW, but the full-parameter CPU updated-weight oracle crossed the same host
reserve. r5 bounds that CPU oracle in chunks and trims unused cache before local
moment checks. **Neither reserve nor numerical tolerances was lowered.**

Both failed large attempts stopped only the owned containers, acknowledged both
guards and left kernel logs unchanged, without OOM/reboot. Their metadata and
source backups are preserved in the final archives. r1 also preserved two safe
pre-GPU refusals (per-host registry identity and the original600s guard cap).
r2/r3 controls and r4 control/recheck evidence remain available.

| Shape / host | Live GPU audit peak MiB | Reserved GPU audit peak MiB | Cgroup peak MiB | Min host available GiB (worker) | Worker / guarded seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 352.96M /104 | 2991.670 | 3186 | 2255.129 | 8.784 | 208.6 /258.9 |
| 352.96M /105 | 2973.930 | 3202 | 2241.508 | 8.589 | 208.9 /221.7 |
| 1.024B /104 | 8265.713 | 8514 | 4715.746 | 3.488 | 603.6 /723.0 |
| 1.024B /105 | 8265.291 | 8578 | 4715.992 | 3.283 | 603.7 /616.8 |

Guardian100ms samples observe minima3.484/3.249GiB for large104/105. Swap/OOM0.
Host/cgroup/device accounting overlaps on this unified-memory board: do not sum
these columns. Timings include construction, readiness, network and extensive
CPU/GPU reference checks, not ordinary update throughput. No speedup follows.
The single-board all-resident FP32 parameter/gradient/two-moment floor is
16,391,536,640B, above either host's physical RAM. We did not provoke single-host
OOM; this does not rule out alternative AMP/offload recipes.

## Startup, transport and recovery

105 performs a guardian verification before launcher startup and a worker
verification before GPU initialization. Their aggregate24.587GB needs at least197s
on ideal1Gb/s Ethernet. The elastic agent rendezvous already waits for guardian
verification; the second pass alone needs at least98s plus overhead and can race
a120s GPU process-group timeout. A separate CPU TCPStore readiness barrier600s
now confirms both source/manifest identities with CUDA still uninitialized, before
native GPU PG initialization. **The native GPU timeout remains120s.** Final CPU
teardown waits for client completion before destroying the store.

The temporary104 UID1000 HTTP source on port29842 serves only manifest/whitelisted
regular files to105:1MiB chunks,256MiB address-space limit,64 descriptors, no writes
or listings. At1.024B it served1953 requests,49,174,609,920 raw payload bytes plus
one manifest; each raw file exactly four times, no failures/rejections. Worker105
verifies488 and reads976 blobs, maximum retained blob128MiB, zero diskcache or
local fallback. The source stops before guard acknowledgment.

PLAN-01 exclusive board reservations/shared runtime locks remain. Four CPUs0–3,
10GiB hard memory/12GiB combined,512PIDs/256MiB shm; own-container stop at9GiB
cgroup, host available below3GiB, disk below5GiB. Child1400s/isolated recovery1800s.
The original kernel recovery helper's600s cap remains unchanged: the test-local
copy differs only in cap/documentation/error (AST compared), with max-ACK,
expiry-dry-run and above-max refusals tested. This is no new arbitrary GPU-hang
recovery qualification. No prefix installation, reboot or native-default change
was performed for this task.

The r4 first control passed math but105's strict log gate refused
`clocksource: Watchdog remote CPU 3 read timed out`. It remains **open**. Both hosts
use unchanged TSC. Qualified kernel source0957f1aa4 uses `pr_info_once` for this
remote-read timeout and retries without marking TSC unstable. Quiet later logs
cannot prove no recurrence or a repaired cause. The r4 recheck and all r5 logs are
unchanged. This observation is separate from the historical Python CPU fault and
the already fixed VM-heap, D05 agent-exit and D21 optimizer-restore defects.

Both r5 controls/large phases completed on the recorded original boot IDs, with
four acknowledged guards and clean agent shutdown.104 subsequently rebooted at
2026-11-10 02:21 local, after its large final at23:55:41 the previous day; cause not
established here. Export records its new boot ID separately and rehashes the
original references. Verifier pins **test/ACK boots**, independently of the later
export boot. Both post-export ownership preflights pass with no active containers.
Native kernels remain104r2/105r3; native Torch/ROCm/MIOpen/D09 profiles, fallbacks,
production controls and applications are unchanged.

## Replay and next action

[Verified results](verified-results.json):24 check groups/four guards,
108 local tests, Black/critical Ruff+F401/diff pass. Predecessor archives
D35/D36/D39/D40/D41 are replayed before D42. Four private mutated archives
(source, pre-GPU readiness, local-state count, coherently altered test/ACK boot)
are refused; see [negative controls](negative-controls.json). No complete CI claim.

```bash
python3 therock/build_tools/bc250/verify_distributed_d42_exports.py \
  --host104 kernel/artifacts/distributed-d42/104-evidence-r5.tar.gz \
  --host105 kernel/artifacts/distributed-d42/105-evidence-r5.tar.gz \
  --output /tmp/d42-replayed.json
```

Metadata archives preserve earlier attempts/sources; no raw payloads. SHA/size
are recorded in verified-results.json. Host evidence is under
`~/bc250work/evidence/distributed-d42-r5/`; tools under
`~/bc250work/tools/distributed-d42/`. Existing labels must not be reused.
Guarded replay requires free boards, current identity and full admission; start104
before105, with matching unused ports and fresh labels, using
`guard_distributed_d42_on_host.py --shape large --label <fresh-label> --port <port>
--ready-port <ready-port>` as root on each host. Never run workers without guards.

**Next D43:** a fresh training-only1.024B FP32 FSDP capacity gate, bound to this
oracle proof but without full reference reads/audits in timed workers. Admit memory,
run a short finite multi-update test, measure complete-job/warm-step/host/device
memory and communication, and retain reserve/ownership/recovery checks. Treat any
new capacity failure as evidence rather than lowering reserves. PLAN-05 stays open
until practical training capacity/value is measured; ordinary helpers retain their
existing qualified policies.
