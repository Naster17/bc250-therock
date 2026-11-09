# D40: bounded reference streaming between hosts

Completed 2026-11-09, implementation `470c4046b`. PLAN-05 remains open.

The optional GPU audit can read reference tensors from .104 without creating a
full reference copy on .105. The new remote route passes **24 GPU groups/four
acknowledged guards** at352,957,440 parameters. Raw/clipped gradients, updated
weights and local AdamW states are **bit-exact** versus fresh local control and
published D39/D36. All CPU FP64 correctness bounds are unchanged. Ordinary native
training/defaults are unchanged; this removes a qualification storage dependency.

## Architecture and bounds

`serve_reference_d40.py` starts only inside .104's guarded phase, after the source
manifest and all296 D35 blobs have been reverified. It runs as UID1000, with a
256MiB address-space limit/64 file descriptors, one synchronous request handler
and1MiB file-read chunks. It binds192.168.1.104:29840 and accepts only .105's IP.
Only the immutable manifest and its explicitly listed blobs are served; no directory
listing or write method. File opens use O_NOFOLLOW and regular-file/size checks.
The service is plain HTTP on the private board LAN, not an authenticated/encrypted
public file service. It stops before recovery acknowledgment and leaves no daemon.
The parent owns the existing EX board reservation and SH runtime lock throughout.

`RemoteStore` requires the original manifest SHA and length before receiving it.
Every blob requires exact HTTP status, length, digest header and independently
computed SHA over received bytes. Redirects, encodings, unknown items and roots
refuse.20s socket/60s per-blob bounds remain inside the phase's overall timeout.
Remote verification uses1MiB scratch; tensor reads allocate one manifest-sized
bytearray (96MiB maximum in this fixture,128MiB hard transport cap). Calls do not
retain a second blob or a disk cache. The descriptor directory contains only the
68,123-byte manifest. The reader has no silent local fallback. Existing qualified
D35 local references on both hosts are preserved, not deleted to demonstrate this.
The explicit local-control route remains available.

The .104 GPU worker keeps local reference reads; .105 uses remote references for
preflight, preallocation validation and the entire per-parameter GPU audit. Source
SHA identities separately bind the original CPU oracle and new reader/service.
Each source blob is served exactly four times: .105 guardian verification, worker
verification, and two audit reads. Source counters show1,185 complete requests,
16,941,957,120 raw payload bytes plus68,123 manifest bytes, zero failed/rejected
requests. These are application payloads, not Ethernet wire-byte measurements.
No full reference set or actual-gradient spool is created by the remote path.

## Numerical qualification

Same owned FP32 tied GPT2 as D35/D39:12 blocks/width1536/24 heads, sequence128,
positions256/vocab8192, global batch2/micro1, eager/dropout0/cache off. Native
FSDP/use_orig_params and AdamW/clipping0.5 remain unchanged. ALL148 initial/raw/
clipped/updated tensors, local moments and counters pass the original checks;
clipping stays atol1e-8/rtol5e-5. The coefficient retains D39's CPU-summed checked
actual-gradient norm. Loss remains9.364227294921875 versus CPU9.364227904214559;
max raw-gradient error2.1940768522681064e-7 and clipped error1.300937118860028e-8.
All native hashes exactly match both hosts/modes and D39/D36, local moments perrank.
No CPU model/oracle executes in GPU workers;39 per-unit summon contexts remain,
max materialized unit28,331,520 elements, whole-model summon forbidden.

93 local tests pass, including seven new methods exercising actual socket traffic:
truncated/changed source and responses, excessive lengths, redirects, bad digest
headers, encodings, symlinks, unknown/traversal paths, peer/method rejection, wrong
manifest pins and out-of-scope reads. One preliminary test inspected server counters
before its handler finished; draining the server before inspection fixes the test.
Three private altered archives (source, fallback flag, retained-buffer size) refuse.
Black/critical Ruff+F401/diff checks pass; immutable D35/D36/D39 evidence replayed.
No failed GPU phase or relaxed bound/full upstream CI claim.

## Measurements and limits

| Host | Route | Peak live GPU MiB | Peak reserved GPU MiB | Whole GPU-container cgroup peak MiB | Min host available GiB | Guarded seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| .104 | local-control | 2980.129 | 3734 | 2503.555 | 7.779 | 135.3 |
| .104 | remote-reference | 2980.129 | 3734 | 2504.148 | 7.827 | 257.3 |
| .105 | local-control | 2973.930 | 4136 | 2504.457 | 7.333 | 136.4 |
| .105 | remote-reference | 2973.930 | 4136 | 2515.375 | 7.361 | 220.2 |

The source service runs outside the GPU container with its own256MiB address-space
cap; whole-host available memory includes it. Memory counters overlap, do not sum.
GPU audit peaks reset before backward, excluding earlier model construction.
Remote worker duration~207s versus~121.5s local includes reads/hashing/validation;
this is a storage-for-network-I/O tradeoff, not a training-speed improvement. The
.105 guarded phase excludes its earlier remote preflight, while .104 waits for it;
do not treat their phase durations as identical brackets or infer network throughput.
No long-run, larger-model fit, generic transport reliability or production speed claim.

Swap/OOM0, kernel logs/registry/boot IDs unchanged. PLAN-01 reservations/runtime
locks,4CPU/cpuset0–3,10GiB RAM/12GiB combined,512 PIDs/256MiB shared memory and
600s guard/460s child limits retained;5GiB free disk reserve retained. Native
Torch a524980/MIOpen381eac05/RCCL/D09/image/packages/defaults/fallbacks unchanged.
No native build/install/reboot/download/prefix mutation or user application work.
Historical Python CPU fault remains distinct from fixed VM-heap/D05/D21 defects.

## Evidence and commands

Host tools: `~/bc250work/tools/distributed-d40`.
Evidence: `~/bc250work/evidence/distributed-d40-r1/{local-control,remote-reference}`.
Logs: `evidence/distributed-d40-{local-control,remote-reference,export}-controller.log`.
Guardians10416356/17139 and10526759/27099; source10417198; exporters10418491 and
10527467 all exited0. Exact native commands/CIDs/UUIDs/source SHA/ACK/logs remain
in each final record. Both boards finish idle, no containers/source service/armed
guards. Original raw refs remain; metadata exports omit raw tensor payloads.
Local evidence: `kernel/artifacts/distributed-d40`.

Run each phase's matching guardian on BOTH free hosts as root, .104 first:

```sh
python3 ~/bc250work/tools/distributed-d40/guard_distributed_d40_on_host.py --transport local --label local-control --port 29741
# After both complete/ACK:
python3 ~/bc250work/tools/distributed-d40/guard_distributed_d40_on_host.py --transport remote --label remote-reference --port 29742
```

Existing output directories refuse reuse; exporter runs once. `remote` means .104
provides the bounded source and .105 reads it; no manual persistent service needed.
The service and fixture remain pinned to352.96M; a larger shape must get separate
source/config/RAM/storage/transport admission. Offline verification:

```sh
python3 therock/build_tools/bc250/verify_distributed_d40_exports.py \
 --host104 kernel/artifacts/distributed-d40/104-evidence-r1.tar.gz \
 --host105 kernel/artifacts/distributed-d40/105-evidence-r1.tar.gz \
 --baseline104 kernel/artifacts/distributed-d35/104-evidence-r1.tar.gz \
 --baseline105 kernel/artifacts/distributed-d35/105-evidence-r1.tar.gz \
 --baseline-results therock/docs/development/bc250_distributed_d35_results/verified-results.json \
 --legacy104 kernel/artifacts/distributed-d36/104-evidence-r1.tar.gz \
 --legacy105 kernel/artifacts/distributed-d36/105-evidence-r1.tar.gz \
 --legacy-results therock/docs/development/bc250_distributed_d36_results/verified-results.json \
 --previous104 kernel/artifacts/distributed-d39/104-evidence-r1.tar.gz \
 --previous105 kernel/artifacts/distributed-d39/105-evidence-r1.tar.gz \
 --previous-results therock/docs/development/bc250_distributed_d39_results/verified-results.json \
 --output therock/docs/development/bc250_distributed_d40_results/verified-results.json
```

[Verified counters/measurements/archive SHA](verified-results.json),
[altered-archive refusals](negative-controls.json).

## Next action

D41: implement and admit a CPU-only FP64 checkpointed/streaming oracle producer for the planned1,024,471,040-parameter GPT2 on .104. Validate config/parameter count/source provenance before allocation, enforce a conservative RAM budget and11.449GiB reference payload+5GiB disk reserve, and abort owned work if headroom fails. Check full initial/loss/ALL-gradient coverage, finite values, raw write/read hashes, checkpoint recomputation and zero retained gradients. Preserve D35 refs; no GPU allocation or larger-fit claim until oracle, storage and shape-specific GPU/transport admission pass separately.

1.024B remains CPU/GPU unqualified. PLAN-05 stays open; six important packages
remain. Transport at352.96M does not prove the larger CPU oracle or GPU fit.
