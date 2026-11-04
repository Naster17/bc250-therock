# D01–D04: two-host RCCL/DDP and bounded FSDP rollout

Completed 2026-11-04 for the stated fixtures. Initial export/staging tools:
`1f533d8f1`; installed launcher, qualification and recovery tools: `5f45c242a`.
Torch source remains `a5249802372882dedccb91d3fd4d73be9fcdf2bd`; existing distinct
daily/RCCL native payloads were copied without rebuilding. ROCm library source
`96e5a7dfc43` includes isolated B07 work; installed native base remains `8389197`.
B07.12 private precision/plugin artifact was not silently installed. B07.13 is
still deferred by the user's distributed-training priority.

Usage: [BC250_DISTRIBUTED_GUIDE.md](../../../../BC250_DISTRIBUTED_GUIDE.md).
Deployment/recovery: [HOST105_UPDATE.md](../../../../HOST105_UPDATE.md).

## Executed results

One GPU rank on each physical host, both eth0 links at 1,000 Mb/s:

| Gate | References and result |
| --- | --- |
| D01 Gloo | GPU sums FP32/FP64/FP16/BF16, small/1 MiB payloads, three FP32 global-batch CPU-double DDP gradient/SGD steps: eight checks/rank, 16 total. Initial native Torch revisions differed. |
| D02 matched RCCL | Same 16 checks across real Socket transport, Ring/Simple on eth0. Logs show network sends/receives and two-rank communicator initialization. |
| D03 installed command | Same 16 checks through `bc250-distributed` and torchrun, using env:// rendezvous. |
| D04 sharding/resume | Nested classic FSDP/use_orig_params, AdamW, 1,608 total parameter elements/804 per rank, AdamW exp_avg 804 per rank. Three CPU-double gradient/parameter reference updates, including same-world/rank checkpoint resume: seven checks/rank, 14 total. |
| .105 target daily | Existing owned suite: 656 checks/nine guarded phases (core7, CPU distributed31, sparse253, FFT245, vision51, audio40, compiled dense14/CNN14, profiler1). |
| .105 selected ordinary commands | bc250-torch four exact native forward/backward checks, bc250-train five including frozen consumer imports: nine total. |

DDP fixed gradient/parameter bounds: atol2e-6/rtol2e-5. D04 gradients use those
bounds; parameters atol4e-6/rtol4e-5. Ordinary dyadic matrix forward/input gradient/
weight gradient compare exactly to CPU double. No tolerance relaxation. Passed
GPU gates retain unchanged kernel logs. All 21 created guard instances are
acknowledged (five .104, sixteen .105), including the failed rendezvous and
ordinary-consumer harness attempts. No active owned jobs/containers/armed guards.
29 local launcher/lock/deployment tests, Black, critical Ruff and syntax pass.
Workload counts overlap; do not sum them into universal support coverage.

## Deployment and preservation

All 24,664 inventoried ROCm file/link entries matched with zero delta. Keep that
identical core and kernels; .105 needed matching current Torch/RCCL/consumer image.
Source image and imported .105 image:
`1bb4ac0ed3766d9c70431c1d5119e03cd4d10e710f4c04cd54254b5b87cd45d5`.
Frozen source export under .104 shared runtime lock:

| Artifact | Bytes | SHA256 |
| --- | --- | --- |
| userspace.tar.gz | 1490746029 | fb5b7f0672b07bdb913e323fcdcf2a2f3ee2fb034f578366b78c324ce07b607d |
| image.tar | 1208947200 | 6e82caf42cd72b0eac0b72894396c3780886ca598d26f1cc72eda505d7a9d5c7 |

.105 selects `/opt/bc250-stack/releases/10.2-gfx1013-20261104-d01` after the full
656-check target gate and EX-locked atomic promotion. Its ROCm link explicitly
depends on retained M105.1 `/opt/bc250-stack/releases/10.2-gfx1013-20261023-host105`.
Old image506f3b26, base manifest and all five operational boot-health tools are
verified preserved. Independent new writable caches; old caches unchanged.
Only the redundant new .105 image transfer file was removed after verified load;
source/workstation exports and all releases/images remain. The later large disk
free-space increase has no established cause and is not claimed as our cleanup.

Both hosts have root-owned versioned tools plus the explicit distributed registry.
.104 provider `/opt/bc250-stack/dev/rccl-single-b053`; .105 candidate's
`rccl-profile`. Existing ordinary commands and loopback rccl-single registration
remain. Distributed command uses Socket/Ring/Simple only, P2P/IB/SDMA/symmetric off.
It holds the host's common SH lock; installs/metadata/promotion used EX locks.
Promotion recovery is `distributed-d01-recovery/activation.json`; the reviewed
daily-selection rollback helper is available but live D01 rollback was not tested.
Optional distributed registration is independent of daily selection rollback.

No kernel or boot-policy changes/reboots: .104 r2 boot `d10514df-543d-433b-a352-0e3ce79bc7a8`;
.105 r3 boot `92b47338-3011-46a9-99a0-a94a5c8f146c`, retained r2 fallback.
No user application inspection/testing, source/build/cache deletion or native
rebuild. Historical Python CPU segfault/KFD uncertainty remains separate from the
fixed VM-heap defect; this batch establishes no root-cause fix for those issues.

## Harness failures retained

D03r1 used an independent TCPStore port inside torchrun's environment; both ranks
timed out. D03r2 refused still-exiting workers before GPU/guard creation. After
owned workers exited, D03r3 used env:// under TORCHELASTIC_RUN_ID and passed.
The first ordinary daily harness incorrectly expected Transformers in plain
bc250-torch; GPU math passed before that import failed. Plain Torch intentionally
lacks the frozen consumers supplied by bc250-train; corrected r2 passes all nine.
No application/backend arithmetic fix followed from these harness errors.

## Evidence and exact next action

Both hosts `/home/nik/bc250work/` retain:
`distributed-d01-20261104`, `distributed-d02-rccl-r1`, `distributed-d03-cli-r1/r2/r3`,
`distributed-d04-fsdp-r1`, matching controller logs and `distributed-d04-handoff.json`.
.105 additionally retains `distributed-d01-target-qualification-r1`,
`distributed-d01-ordinary` and `distributed-d01-ordinary-r2`, and recovery metadata.
D04 owned checkpoint files on both ranks are retained.

Successful controller PIDs (all exited0): export22485, stage22363;
D01 ranks21432/9019; D02 ranks894/23562; D03r3 ranks5499/29248;
D04 ranks6820/30882; .105 target323. Controller log stems identify each phase;
ordinary-r2 log is `distributed-d01-ordinary-r2-controller.log`. No job is running.

Workstation `kernel/artifacts/distributed-d01/` holds frozen archives, inventory,
initial evidence and final raw archive snapshots verified after transfer:

| Final archive | Bytes | SHA256 |
| --- | --- | --- |
| 104-final-evidence-r2.tar.gz | 352782 | b68218d22d139822a32f872d75440d0e7dd1c35c185a894147718e0fb807dc26 |
| 105-final-evidence-r2.tar.gz | 790849 | 2b856fdf4ae2d1c2a0acefb070eb11b6155b56bf5ea0785ed6fb47f16b650c67 |

D05 next: representative owned DDP/FSDP training with matched global batch,
one-versus-two-node timing and actual peak-memory evidence; checkpoint robustness
and model-block wrapping before larger-model claims. Current FSDP proof is tiny,
FP32/classic API, same-world/rank resume. No practical maximum model capacity,
production convergence, elastic recovery, other RCCL transports/protocols,
FSDP2/ZeRO/HF Trainer integration or measured training speedup is qualified.
