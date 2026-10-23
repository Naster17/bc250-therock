# B03.7: FP8 admission guards and bounded RCCL feature profile

An independent native gfx1013 Torch candidate now rejects unsupported numeric
FP8 reductions while preserving FP8 byte copies. It passes76 new guard/readiness
checks plus the retained1,365 checks (**1,441 across22 gates**). The candidate is
isolated; daily Torch/ROCm and defaults remain unchanged.
[Verified results and identities](verification.json).

Tooling revision:`819bb624f`.

## Source fixes and actual evidence

Torch `f807099fbb874d61eacfb573b7b1095f9514992c` rejects Float8 in the low-level
AllReduce/Reduce/ReduceScatter wrappers before communicator creation. Their
existing uint8 storage mapping supports byte copies, not numeric FP8 reduction
semantics. This low-level correction applies to NCCL/RCCL wrappers generally;
only native gfx1013 was runtime-qualified here. Broadcast/AllGather retain their
mapping. A device-generic upstream TestCase covers four FP8 formats; targeted
spin/lintrunner static-format checks pass. Full upstream CI was not run.

ProcessGroupNCCL's existing reduction policy now examines the input tensor's
actual HIP device and refuses all FP8 formats on the filtered gfx1013 profile.
Other GPU devices retain existing version-based admission. No architecture is
spoofed, and no claim of permanent hardware impossibility is made.

The first build succeeds but qualification-r1 fails after33 checks:AllReduce
refuses e4m3fn, yet process-group Reduce accepts it. Torch
`14cfef2d8c5e8c2c40e3b24587233227ea8dadc3` adds the missing checks to Reduce and
its internal out-of-place helper. Rebuild-r2 succeeds; qualification-r2 passes.
The failed source/native payloads and logs are preserved. This is an observed
admission gap, not a multi-GPU wrong-result reproduction. [Complete patch](fp8-guards-complete.patch).

## Qualified subset and exclusions

| Path | Evidence and boundary |
| --- | --- |
| Low-level FP8 numeric reductions |15 refusals:five formats × three wrappers; no arithmetic submitted |
| Process-group FP8 reductions |20 refusals:five formats × AllReduce/Reduce/tensor ReduceScatter/coalesced AllReduce; internal out-of-place check also reviewed, not directly executed |
| FP8 byte copies |32 exact comparisons:four formats ×17/257 × Broadcast/AllGather × low-level/process-group; offset guards/unchanged inputs, raw uint8 patterns including all256 encodings; no numeric FP8 claim |
| Multicast allocator | Existing `supports_tensor_alloc=False` and allocator refusal both checked; multicast remains unavailable through this ROCm path |
| Generic registration | Default MemPool register, existing and later allocation, GPU PreMulSum/CPU references, deregister; later allocation need not create a new segment; no transport/zero-copy/speed proof |
| Symmetric windows | Explicit `NCCL_WIN_ENABLE=0` returns success with a null handle:registration is skipped, not functional symmetric support |
| Suspend/resume | Memory stats/state0→1→0 and subsequent FP32 GPU arithmetic pass; all tracked byte counters are zero, so payload relocation/reclamation is unqualified |
| Symmetric device API | Source macro excludes ROCm; full symmetric Tensor/rendezvous/device API remains unqualified |
| Lifecycle/device/DSO | Native gfx1013/version and actual isolated loaded DSO verified |

The76 checks include36 expected refusals. Nontrivial registration/post-resume
arithmetic uses independent CPU values. Retained B03.6 contracts178, B03.5
RCCL64/GPU Gloo41 and the18-gate1082 suite pass with unchanged tolerances,
including analytical BF16/FP32 DDP, sparse/FFT/vision/audio/image/video/files,
compiled models, profiler and core checks. RCCL uses one rank/physical GPU;
Gloo's two ranks share it and host-stage communication. No multi-device,
Ring/Tree/LL transport, P2P/XGMI/GDA/SDMA, scaling, speed, broad FP8/BF16 accuracy,
higher gradients, graphs, concurrent registration, shrink or real offload-payload
claim follows from these tests. [Source audit/profile](applicability.json).

## Reproduce and preserve

On .104, `WS=/home/nik/bc250work/rocm-10.2-gfx1013`. Run each helper through
`WS/helpers/update_dev_prefix.py -- ...` using the existing exclusive lock:

1. Stage exact source metadata and four source/test overlays in `helpers/b037-*`.
   `build_rccl_b037_on_host.py` copies B03.5 into independent
   `WS/pytorch-rccl-b037`; build uses pip only. No original tree/reset/hardlink edit.
2. `stage_rccl_b037_on_host.py` creates release-r1/baseline and backs up/installs
   the shared qualifier. Qualification `--attempt r1` records the missed Reduce
   guard. Stage the corrected metadata/source as recorded in `source-r2.json`.
3. `rebuild_rccl_b037_on_host.py` preserves old source/native libraries and resumes
   the same isolated build. Stage `--attempt r2 --build-label rccl-b037-build-r2`.
4. `qualify_rccl_b037_on_host.py --attempt r2` runs the22 guarded gates.
5. `export_rccl_b037_on_host.py` exports accepted and failed evidence once.

Build4 CPUs/four Ninja jobs/one linker,10GiB RAM/12GiB memory+swap, shared20GiB
ccache/no statistics reset; compile containers have no GPU/network. Cumulative
cache statistics are not a timing claim. B03.5/B03.4 and active sources, builds,
releases/caches remain intact. Within the new B03.7 build, before-fix native/source
payloads live in `WS/recovery/rccl-b037-reduce-before`; release-r1 links reference
the reused build, not immutable before-fix binaries. Shared helper/metadata backup:
`WS/recovery/rccl-b037-before`. Final baseline:
`WS/logs/rccl-b037-qualification-r2/final.json`.

All356 protected hashes, four AOT DBs and links match. Accepted gates/batch dmesg
match;23 guards including the failed run acknowledge. Local ignored artifacts:
`kernel/artifacts/rccl-b037`, with native payloads/recovery/raw logs. The
1,083,504,640B archive matches SHA256963a1d6416ae1aaca62565253250b1e29812def7bc22cbd83688f9db8c225ff8;
291 raw logs total8,996,665B. Compiled follow-up delta is exactly six guard lines.
Daily
Torch2b449c4/NCCL-off/image506f3b26/runtimeJSON32fd4559/native HIPa1162308/systems
1f9609/ROCm/math/SDMA0/heapoff and isolated RCCL38fd6/DSOca790dc7 are unchanged.
Same r2 boot2db1c7c2; no reboot/.105/BIOS/fallback/user application inspection,
modification/test or cleanup. Unknown historical Python CPU faults remain
unresolved/distinct from fixed VM-heap/CSR/B04.2 hostcall queue defects. Final
reports supersede historical running-job checkpoints. No own jobs/root SSH remain.

**Next B03.8:** package the bounded opt-in single-device RCCL profile, including
read-only DSO mounting/loading and explicit window/CUMEM/P2P/IB settings. Current
extra library mounting belongs to the test harness, not ordinary launcher
integration. Validate the packaged launcher with owned output/backward checks;
retain the daily NCCL-off/Gloo runtime and rollback. The profile excludes numeric
FP8, multicast/symmetric device execution, real offload payloads and multi-device
qualification; do not advertise the filtered375-object RCCL as a full backend.
