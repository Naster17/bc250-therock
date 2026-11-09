# D39: spool-free CPU-reference clipping audit

Completed 2026-11-09, implementation `a82aca0ba`. PLAN-05 remains open.

The optional per-unit GPU qualification now reads the original D35 FP64 CPU
reference gradient for each clipped-gradient comparison. It needs **zero raw
actual-gradient spool bytes**, versus **1,411,829,760 bytes (1.315 GiB) per host**
in the legacy path at352,957,440 parameters. The original CPU references and
legacy D36 control are preserved. Ordinary Torch/ROCm training is unchanged.

## What was qualified

Same owned FP32 GPT2 fixture, tied embeddings,12 blocks/width1536, sequence128,
global batch2/micro1, eager attention/dropout0/cache off. Two physical gfx1013
ranks use native FSDP/use_orig_params, native clipping0.5 and AdamW. No CPU oracle
executes in the GPU worker. Original D35 files are rehashed before allocation and
at final export; their source identities remain independent of the new audit.

All148 initial tensors, raw gradients, norm, clipped gradients, updated values,
local AdamW moments and counters pass the original gates. The clipping comparison
is now **CPU FP64 reference gradient × checked coefficient** against the native
clipped GPU gradient, at unchanged **atol1e-8/rtol5e-5**. The coefficient still uses
the independently CPU-summed norm of the actual GPU raw gradients, already checked
against the CPU oracle; this is not a claim of a separate CPU-reference-only norm.
Maximum absolute clipped-gradient difference is1.300937118860028e-8 on both hosts;
the combined absolute/relative bound passes for every entry. No bound was relaxed.

All raw/clipped/updated tensor hashes and per-rank local AdamW hashes are **exact**
against both fresh legacy control and published D36 unit-audit. Both routes retain
39 public nonrecursive summon contexts across three complete passes, max28,331,520
materialized parameter elements; whole-model summons remain forbidden. CPU moment
scratch remains bounded to one parameter (maximum192MiB for its two FP64 moments).
The compatibility key `actual_raw_spool_sha256` holds the same raw tensor hashes
computed directly in memory in reference mode; no raw spool file is created.
`actual_raw_spool_removed=True` records final absence, not a created/deleted file.

## Results and limits

24 GPU check groups/four acknowledged guards;86 existing local tests, Black,
critical Ruff+F401 and diff checks pass. Three private altered archives are refused:
changed audit source, relaxed clipping tolerance and changed clipped-gradient hash.
Independent replay revalidates the immutable D35 and D36 archive provenance.
No failed GPU phase, swap, OOM, new kernel log or registry change. Watchdog guards
are acknowledged only after the owned containers exit. Both boards finish idle.

| Host | Audit | Peak live GPU MiB | Peak reserved GPU MiB | Whole cgroup peak MiB | Min host available GiB | Guarded seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| .104 | legacy-control | 2980.129 | 3734 | 2507.645 | 7.802 | 132.7 |
| .104 | reference-clip | 2980.129 | 3734 | 2502.086 | 7.705 | 134.6 |
| .105 | legacy-control | 2973.930 | 4136 | 2502.293 | 7.290 | 133.9 |
| .105 | reference-clip | 2973.930 | 4136 | 2503.176 | 7.270 | 135.9 |

GPU audit peaks reset immediately before backward and exclude initial construction.
Counters overlap; do not add them. Whole-guard timings include imports, I/O,
validation and hashes; no training-speed improvement or timing significance is
claimed. GPU live peaks are unchanged. This improvement removes a qualification
**storage** dependency, not ordinary training memory or throughput.

Both phases use PLAN-01 EX board reservations/SH runtime locks, pinned D09 launcher,
4 CPU/cpuset0–3,10GiB RAM/12GiB combined RAM+swap,512 PIDs,256MiB shared memory,
600s recovery guards/460s child bounds. New reference mode requires5GiB free reserve
with no spool payload; legacy still admits its full spool plus5GiB. Native profiles,
Torch a524980/MIOpen381eac05/RCCL/image/defaults, boot IDs and fallbacks are unchanged.
No native build, install, reboot, package download or user application work occurred.
Historical Python CPU fault remains distinct from fixed VM-heap/D05/D21 defects.

## Evidence and replay

Host tools: `~/bc250work/tools/distributed-d39`.
Evidence: `~/bc250work/evidence/distributed-d39-r1/{legacy-control,reference-clip}`.
Controllers: `evidence/distributed-d39-{legacy-control,reference-clip,export}-controller.log`.
Guardians10410967/11784 and10521718/22538; exporters10412572/10523340 all exited0.
Exact native command, CID, UUID, source hashes, guard ACK and logs are in each
final record. Original3.9446GiB D35 references remain on both hosts; archives omit
all raw tensor payloads. Local owned evidence: `kernel/artifacts/distributed-d39`.

Launch one matching guardian on each host (root, only on free boards):

```sh
python3 ~/bc250work/tools/distributed-d39/guard_distributed_d39_on_host.py --audit legacy --label legacy-control --port 29739
# After both finish/ACK:
python3 ~/bc250work/tools/distributed-d39/guard_distributed_d39_on_host.py --audit reference --label reference-clip --port 29740
```

Existing output directories deliberately refuse reuse. Export once with
`export_distributed_d39_on_host.py`. Local verification:

```sh
python3 therock/build_tools/bc250/verify_distributed_d39_exports.py \
 --host104 kernel/artifacts/distributed-d39/104-evidence-r1.tar.gz \
 --host105 kernel/artifacts/distributed-d39/105-evidence-r1.tar.gz \
 --baseline104 kernel/artifacts/distributed-d35/104-evidence-r1.tar.gz \
 --baseline105 kernel/artifacts/distributed-d35/105-evidence-r1.tar.gz \
 --baseline-results therock/docs/development/bc250_distributed_d35_results/verified-results.json \
 --legacy104 kernel/artifacts/distributed-d36/104-evidence-r1.tar.gz \
 --legacy105 kernel/artifacts/distributed-d36/105-evidence-r1.tar.gz \
 --legacy-results therock/docs/development/bc250_distributed_d36_results/verified-results.json \
 --output therock/docs/development/bc250_distributed_d39_results/verified-results.json
```

Archive identities and measurements: [verified results](verified-results.json).
Refusals: [negative controls](negative-controls.json).

## Next action

D40: qualify bounded read-only per-tensor reference transport from .104 to .105 using the existing352.96M D35 oracle, source/manifest/length/SHA checks, bounded buffers and explicit truncation/corruption/refusal controls. Preserve original refs and local fallback; do not duplicate the full reference set. Then separately admit a1.024B CPU checkpoint+streaming oracle on .104 and a GPU audit only after shape-specific RAM/storage/transport budgets pass. No deliberate OOM or inferred fit.

1.024B CPU/GPU remains unqualified. The new route removes the hypothetical3.816GiB
raw audit spool, but its11.449GiB full CPU-reference payload+5GiB reserve still does
not fit .105. CPU-oracle RAM and eventual GPU capacity need their own admission.
Six important packages remain; this is one prerequisite within PLAN-05.
