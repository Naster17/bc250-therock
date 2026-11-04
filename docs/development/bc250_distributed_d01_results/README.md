# D01: two-board distributed training workstream (in progress)

User priority override2026-11-04: .105 updates and cross-host training experiments
are authorized. Preserve the M105.1 base/image/boot health, r3/r2 fallback and user
applications. B07.13 remains pending, not completed or silently deployed.

The initial Gloo baseline runs one GPU rank per physical host over eth0. Both
1Gb/s links carry GPU-origin FP32/64/16/BF16 AllReduce sums, small and1MiB payload
checks, then three DDP gradient/SGD updates compared to a CPU-double reference
for the combined32-example batch. Eight checks per rank pass (16 total). Both
600s guards ACK and kernel logs remain unchanged. Kernels are .104 r2 and .105
r3. Existing Torch native revisions differ; both report2.13.0a0+gfx1013. This is
host-staged Gloo transport, not RCCL qualification, speed scaling or sharded memory.

Read-only source/target inventories match all24,664 ROCm entries, with zero content
or link delta. Retain this identical core. Source .104 image1bb4ac0e and current
daily/RCCL Torch variants need deployment on .105 (old image506f3b26). The source
export freezes these plus RCCL/consumers under the existing SH runtime lock.

The prepared candidate stager uses the .105 control EX lock documented in
HOST105_UPDATE.md. It stages a fresh release, sharing the retained immutable
M105.1 ROCm base as an explicit dependency; it never changes current. Image import
is verified before retiring only the redundant new transfer copy; source and
workstation image archives remain. Existing releases/images/caches stay intact.
Staging and RCCL scripts are prepared but not yet target-executed.

The next transport test will use the matching native RCCL375-object build,
Ring/Simple/socket eth0 only, P2P/IB/SDMA/symmetric features off. The existing
registered rccl-single loopback profile stays unchanged; command-only experimental
settings and independent watchdogs isolate this qualification.

Ordered direction:

1. D01: freeze/transfer/stage matched .105 userspace and verify target references.
2. D02: two-host RCCL collectives and global-batch DDP/optimizer references,
   failure/timeout cleanup and isolated communication timing; promote only after
   target tests/rollback are ready.
3. D03: durable ordinary launch/profile/torchrun instructions, checkpoint/resume
   and representative one-vs-two-host training timings; qualify useful workloads.
4. D04: model/optimizer sharding (FSDP/ZeRO applicability), independent gradients,
   actual per-rank memory evidence and checkpoint recovery. DDP replicates weights;
   two16GB boards do not become a single32GB CUDA device. Ethernet traffic can
   outweigh compute savings; no speed or maximum model-size promise yet.

Initial evidence on both boards:
`/home/nik/bc250work/distributed-d01-20261104/`, controller
`/home/nik/bc250work/distributed-d01-controller.log`, inventories
`/home/nik/bc250work/distributed-d01-rocm-inventory.json`. Local ignored artifacts
are under `kernel/artifacts/distributed-d01/` and `tmp/distributed-d01/`.
Source export .104 PID22485, log
`/home/nik/bc250work/distributed-d01-export-controller.log`, output
`/home/nik/bc250work/distributed-d01-export/`. Consult flow.md for current job state
before resuming; do not restart or overwrite the export/candidate directories.
