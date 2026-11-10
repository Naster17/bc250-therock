# D43: fresh two-host 1.024B training capacity

Completed 2026-11-10. Initial implementation `8a590d02f`, export/replay
`7c0dc6234`, forward-cache repair `4f684e984`, backward-cache repair `31aa4415d`. Final candidate is the optional
`tools/distributed-d43` on both hosts, series `distributed-d43-r3`.

**Eight consecutive native FP32 FSDP updates at 1,024,471,040 parameters pass on
two BC-250 boards**, without raw CPU-reference transport or full-state gathers/CPU audits in
timed updates. This completes PLAN-05's bounded owned-fixture capacity scope.
It establishes capacity for this recipe, not faster training or general support
for arbitrary billion-parameter models. Five important work packages remain.

## Fixed recipe and independent prerequisite

The unchanged D35 training loop uses the D41/D42 GPT2 dimensions:20 blocks,
width2048/32 heads/vocab8192/positions256, sequence128, globalbatch2/micro1,
eager attention/dropout0/cache off/tied embedding/head. FP32 parameters,
gradients and AdamW moments, lr1e-4/eps1e-5/weightdecay.01/foreachFalse, root
clip.5; no AMP, checkpointing, offload or compilation. Steps use the original
shifted/global target-normalized CE and deterministic data indices0–7.

D42's complete FP64 CPU reference comparison is replayed before admitting these
results. `qualification_d43.json` pins both shapes/ranks' actual D42 workers and
their archive identities. Original model/math/optimizer sources and package pins
must match before GPU initialization. First global loss must match the qualified
GPU loss within1e-5. A fresh process, native NCCL/RCCL group and optimizer are used;
no live model/optimizer is inherited from qualification.

The worker loads metadata only: raw artifact read/write/verification, CPU oracle,
full optimizer export and full-parameter summons are explicitly forbidden. No
HTTP reference source is started. The eight updates comprise one initial loss
check, two warmups and five timed updates. All eight clipping norms are finite.
After training, ALL local FP32 parameters, gradients and both moments are scanned
for finiteness on GPU in at most1M-element chunks; every native AdamW counter is8.
Each rank has512,235,520 local parameter elements and corresponding moments.
This final scan is outside update timing; it does not copy full state to CPU.
Only step0 has D42's independent mathematical oracle, not a new eight-step CPU
oracle or a convergence certificate.

## Actual failure and cache repair

The r1 control passed eight updates on both hosts (median17.9963s). The first
1.024B update completed, but105 crossed the unchanged3GiB host reserve during
its second forward:3,214,176,256B available, cgroup1,937,588,224B. Its guardian
stopped only the owned container.104's peer was then explicitly stopped; its
RCCL timeout after peer cancellation is preserved as a consequence of that stop,
not evidence of a separate RCCL mathematical or historical Python/VM-heap defect.
Both recovery guards acknowledged, logs unchanged, no OOM or reboot.

The r2 repair additionally releases unused GPU cache before each FSDP unit's
forward and after the root forward, using public module hooks. The existing
before/after-backward trims remain. Every trim synchronizes and records allocated/
reserved bytes, asserts live allocation unchanged and reserved memory nonincreasing.
The forward-only r2 policy had21 units/192 trims at1.024B and13 units/128 at352.96M. No tensor, wrapping policy, prefetch setting, collective or optimizer
arithmetic changes. These synchronization/reallocation costs are included in
step timings. This is a fixed explicit memory policy, not a universal cache-speed
optimization or an ordinary launcher default.

The first r2 control passed all numerical/finite/exit checks, and both hosts'
initial loss, all eight norms, five global losses and state counts exactly match
r1.105's overall phase was refused because unrelated Docker veth activity changed
kernel logs. That failed exclusivity gate is preserved without filtering messages.
The unmodified r2 `control-recheck` passed strict checks. R2 large then completed
its second forward but crossed the host reserve during backward on105 at
3,216,723,968B available (cgroup1,735,995,392B). Again only owned containers were
stopped; both guards acknowledged and logs remained unchanged. This is why one
audited update did not establish useful multiupdate capacity.

R3 additionally trims unused cache after each non-root FSDP unit backward via
public full-backward hooks returning None. All gradients are preserved; no
optimizer or collective arithmetic changes. Across eight updates there are
160 backward-unit callbacks/352 total trims at1.024B, or96/224 at352.96M. The
final r3 control still matches ALL eight r1 norms, five losses and state counts
exactly on both ranks. R3 control and large training are the final qualification.

## Measured results

| Shape / host | Median update s | Measured live GPU peak MiB | Reserved peak MiB | Cgroup peak MiB | Min sampled host available GiB | Worker / guarded s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 352,957,440 /104 | 18.347 | 3205.011 | 3378 | 2136.121 | 8.955 | 153.0 /166.5 |
| 1,024,471,040 /104 | 53.365 | 8587.883 | 8808 | 4731.906 | 3.620 | 441.6 /455.6 |
| 352,957,440 /105 | 18.347 | 3183.235 | 3374 | 2141.059 | 8.661 | 152.7 /166.1 |
| 1,024,471,040 /105 | 53.365 | 8585.430 | 8884 | 4729.129 | 3.282 | 441.4 /455.2 |

All five timed global losses and maximum-rank step times match across ranks.
Container swap/OOM0; host swap occupancy stays0 on104 and a pre-existing
2,756,608B (2.63MiB) on105, identical from the first sample through the last in
both shapes. Replay initially refused its absolute-zero host-swap assumption;
the final checker explicitly requires unchanged sampled host occupancy and zero
container swap. Tests and a fourth private archive mutation reject host occupancy
growth. Paging-I/O counters were not collected, so constant occupancy does not
prove no swap I/O. Reserves and runtime stop thresholds are unchanged. All owned containers exit and four final guards acknowledge. GPU,
cgroup and host counters overlap on this unified-memory board; do not add them.
`worker_seconds` ends before native PG teardown/final finite scan; guardian time
includes those and launcher startup. No full CPU reference reads occur in either.

The complete all-resident single-board FP32 p/g/m/v floor is16,391,536,640B,
above either host's actual physical RAM. Two sharded ranks safely execute this
one recipe; no deliberately failing single-host OOM experiment was run. AMP,
checkpointing or offload may admit different single-board recipes, and other
models/sequence lengths change memory requirements. Earlier352.96M comparisons
already show Ethernet FSDP is much slower than single-board training when the
same model fits; this does not claim training speedup.

Collective instrumentation records native FP32 all-gather and reduce-scatter
**input bytes**, not Ethernet wire bytes or isolated communication completion time.
The1Gb/s link remains a major cost; compute/communication/reallocation overlap
cannot be separated from these counters alone. Metric scalar all-reduces and inter-step barriers are outside the timer; the
gradient-norm reduction remains inside each training update. No throughput extrapolation or long-run
stability/checkpoint-restart claim follows from eight updates.

## Safety, replay and remaining work

PLAN-01 EX cooperative board reservations/SH runtime locks, four CPUs0–3,
10GiB hard RAM/12GiB combined,512PIDs/256MiB shm,100ms owned-stop thresholds
(host below3GiB/cgroup at9GiB/disk below5GiB),1400s child/isolated1800s recovery
remain. Original kernel helper600s and native GPU PG120s caps are unchanged.
No active prefix installation, rebuild, reboot, foreign stop or application work.
Native Torch/ROCm/MIOpen/D09 profiles, safe defaults and boot fallbacks remain.
105's earlier clocksource remote-read timeout remains open; `pr_info_once` means
quiet logs cannot prove absence of recurrence. Historical Python CPU fault stays
separate from fixed VM-heap corruption, D05 agent exit and D21 optimizer restore.

[Verified results](verified-results.json):24 check groups/four final guards,
116 local tests, Black/critical Ruff+F401/diff pass, four private source/finite-
coverage/update-counter/host-swap-growth archive mutations refused ([controls](negative-controls.json)).
Predecessor replay includes D42 and its D35/D36/D39/D40/D41 prerequisites. No full
upstream CI claim. Final r4 metadata exports include the corrected baseline-aware reader; original
r3 exports remain preserved. No GPU rerun was needed for metadata-reader fixes.
The initial reader also corrected a guardian qualification-field name mismatch.
Metadata archives include earlier failed runs and sources,
exclude raw reference payloads; hashes and sizes are in verified-results.json.

```bash
python3 therock/build_tools/bc250/verify_distributed_d43_exports.py \
  --host104 kernel/artifacts/distributed-d43/104-evidence-r4.tar.gz \
  --host105 kernel/artifacts/distributed-d43/105-evidence-r4.tar.gz \
  --output /tmp/d43-replayed.json
```

Host records are under `~/bc250work/evidence/distributed-d43-r3/` with final labels
`control` and `large-training`. Tools are under
`~/bc250work/tools/distributed-d43/`; existing labels refuse overwrite. Future
GPU replay needs free boards and renewed admission: launch the guardian on104
then105 with the same fresh label/port and `--shape control` or `--shape large`.
Workers must not be launched directly without the recovery/reservation procedure.

**Next PLAN-06:** inspect retained Python CPU fault, KFD/invalid-visibility and
clocksource evidence; select a bounded reproduction and concrete front-end
validation/diagnostics repair before a predeclared mixed-workload stability pass.
Do not reopen completed kernel work or claim unknown historical faults repaired
from quiet tests. Release reconstruction/qualification and modern environment/
model admission remain the following packages.

After successful tests/exports and documentation sync,105's final handoff preflight
found a new unrelated Docker workload. It was left untouched;104 was idle. This
does not change the earlier qualified phases, but the next GPU task must check
ownership again rather than assuming105 is free. All owned guards are acknowledged.
