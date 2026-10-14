# BC-250 native ROCm development handoff

Updated 2026-10-14. Read this first when continuing this port.

## Purpose and working rules

Port ROCm 10.2 and PyTorch to the AMD BC-250's native **gfx1013** GPU, covering
what the hardware can execute correctly. Use scalar/vector implementations where
matrix instruction hardware is absent. Do not spoof another GPU architecture.
All work, communication and documentation are in English.

Implement locally in this repository, transfer explicit changed files to `.104`,
then build and qualify on its GPU. User authorized development, installation and
kernel experiments on `.104`; preserve SSH recovery and boot fallback. `.105`
is the unchanged ROCm 7.2 comparison machine: do not upgrade it yet.
Keep previous releases, images, sources, build objects, caches and checkpoints.
Do not reset submodules with `fetch_sources.py` over port changes. Do not touch
the unrelated untracked local `src/` directory. Group commits and update this
file once at the end of a work batch; avoid repeated documentation and long tests.

## Hardware and validated runtime profile

- Hosts: `nik@192.168.1.104` (development), `nik@192.168.1.105` (reference).
- `.104`: unlocked 8 CPU cores / 16 threads, 40 CU / 20 WGP / 80 SIMD, wave32,
  64 KiB LDS. HIP reports 20 multiprocessors, not 40 CU.
- Native target `gfx1013:xnack-`, KFD device 100103. No MFMA/WMMA hardware.
- GPU reports 15,938,355,200 bytes, approximately 14.84 GiB. GPU and CPU share
  approximately 16 GB physical RAM; these are not independent memory pools.
- Kernel `7.1.5-bc250rocm-r2`; fallback workspace
  `/home/nik/bc250work/bc250-rocm-lab` retained. OpenRC watchdog active, actual
  GPU hang recovery is not qualified.
- Keep **`HSA_ENABLE_SDMA=0`**. Experimental gfx13 VM heap stays off; ordinary
  VMM and asynchronous pools remain available. Media engine rings unavailable
  under current kernel. Counter/debug/SDMA paths need separate qualification.
- SSH keys work. Elevated operations use interactive doas; never put credentials
  in source files, manifests, logs or scripts.

## Active development stack and recovery

The launcher uses the mutable Priority 2 development stack. The directory retains
its original `priority1` name; its runtime profile is `gfx1013-priority2-dev`:

| Item                        | Current value                                                      |
| --------------------------- | ------------------------------------------------------------------ |
| Active symlink              | `/opt/bc250-stack/current` → `/opt/bc250-stack/dev/priority1`      |
| Workspace (WS)              | `/home/nik/bc250work/rocm-10.2-gfx1013`                            |
| ROCm                        | `WS/build/dist/rocm`                                               |
| Torch editable installation | `WS/pytorch-work`                                                  |
| User cache                  | `WS/cache/dev-user`, uid/gid 1000                                  |
| Image tag                   | `localhost/bc250-stack:10.2-gfx1013-priority1-candidate`           |
| Image ID                    | `5df0a58392de13916819f6d84342b0338ae2c2bbb2f3c21a1f3ad38be2724a59` |
| Build image                 | `localhost/bc250-rocm-build:10.2-ccache`                           |
| Recovery link               | `/opt/bc250-stack/before-dev-current`                              |
| Recovery release            | `10.2-gfx1013-20261013-4b000d542`                                  |

The initial `10.2-gfx1013-20261012-a640c9296` release is also retained.
`bc250-torch` was tested after activation as nik: native GPU, compiled
forward/backward, checkpoint save/load and uid-1000 ownership passed.
`/home/nik/bc250work/priority1-dev-launcher.json` and
`priority1-dev-checkpoint.pt` preserve this evidence.
`bc250-train` mounts the retained Python consumer bundle from the previous
release, using its training.env; earlier training/LoRA qualification remains
available, but no additional long training qualification was run this batch.

## Source identity and reproduction

Local branch: `users/Naster17/rocm-10.2-gfx1013`. Use `git log -1` for the final
TheRock commit containing this handoff (a file cannot contain its own commit ID).

| Repository     | Current source revision                    |
| -------------- | ------------------------------------------ |
| rocm-libraries | `01566e3144d854944c7bb1f6b045e965ddb5aae9` |
| PyTorch        | `4f13bf86476252e3d57b17d1c72f8512814400e9` |
| rocm-systems   | `dbdb7b8d1c56316d873a9b90e548e5a72853496d` |
| Triton         | `ee65ff3b85f99d52e678b252a99c8425fa46e329` |

Apply library patches in `patches/amd-mainline/rocm-libraries` in order through
0017, and Torch patches in `patches/bc250-pytorch` through 0002.
`.104` source trees are exported, gitless copies: `WS/source`,
`WS/pytorch-source`, `WS/pytorch-work`. Their GIT_REVISION files and
`WS/SOURCE_PRIORITY2.json` record the latest provenance. SOURCE_PRIORITY1.json
retains its earlier snapshot identity. Earlier source archives and
SOURCE_CANDIDATE.json retain their original baseline meaning. The generated
Torch version git label can retain its original export revision; use the
source manifest for the patched revision. Triton wheel SHA256:
`2d65d6fefe359116521640e3cc2257bd03b4b24f6877fc8e28946ee9c57544d8`.

## Completed Priority 1 batch

These are bounded GPU qualification scopes, not full library support claims.

| Done | Area                        | GPU evidence and limits                                                                                                                                                                                                                                                     |
| ---- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [x]  | hipBLASLt scalar consumers  | 403 existing + 465 new API cases = 868. C++ Gemm/GroupedGemm, host group lists, descriptors/lifetimes/streams/query APIs, f32/f16/bf16 and added activation/AUX epilogues. FP32 accumulation. No device group arguments, matrix kernels, FP8/INT8/FP64 or arbitrary tuning. |
| [x]  | CK Tile generic operations  | 247 numerical cases: top-k 60, LayerNorm 18, scalar GEMM 160, Pool3D 9; plus 400 GEMM argument rejections. Installed public ScalarGemmKernel header. Explicit API, not an automatic replacement for matrix kernels.                                                         |
| [x]  | hipDNN consumers            | 100 graph cases: hipBLASLt provider 48, HIP provider 18, MIOpen 3D forward/dx/dw 18, convolution+bias+activation fallback 16. Six additional host parameter tests.                                                                                                          |
| [x]  | Torch/Triton/Inductor       | 64 checks: 52 eager/compiled indexing, normalization, training and attention checks; 12 explicit hipBLASLt forward/backward checks with 36 observed successful native library calls. CPU references, dynamic shapes, real generated GPU kernels.                            |
| [x]  | Dev launcher and provenance | Mutable stack activated; nik GPU checkpoint/Inductor smoke passed. Recovery stack preserved.                                                                                                                                                                                |

Final kernel log matched the pre-test baseline with no new messages. No build or
GPU qualification job remained running at handoff. Local verification: 92
meaningful tests, plus source syntax/formatting checks.
Reports are under `docs/development/bc250_scalar_library_results/priority1-*`
and `WS/logs`; exact case scopes are encoded in the checked-in probes.

Previously qualified runtime/compiler, rocBLAS/hipBLAS, solver/FFT/RNG/sparse,
rocPRIM/hipCUB, RPP and training subsets remain available. Previous scalar
hipTensor (76 cases) and hipSPARSELt (162 cases) are retained. Consult
`docs/development/bc250_support_matrix.md` and
`docs/development/bc250_scalar_library_expansion.md` for those narrower scopes.

## Build and test procedure / important findings

- Use ccache (20 GiB limit), typically 16 build jobs, 6 for providers/hipDNN,
  12 for Torch with one link job. Keep objects; do not expunge to iterate.
- In the existing build container, `helpers/build_priority1.sh /workspace/source /workspace/build` configures opt-in gfx13 scalar backends and the hipBLASLt
  provider, builds components and stages dist. `BC250_STAGE_ONLY=1` omits the
  final global dist rebuild. Container mounts must match `/workspace` paths.
- **Stop GPU tests/user Torch during updates of the mutable dist prefix.**
  Build/stage first, then run the bounded GPU validators. An earlier concurrent
  rebuild/test crash has no established root cause; avoid this unsafe overlap.
- **Explicitly regenerate derived HIP sources before every Torch source rebuild:**
  `python tools/amd_build/build_amd.py --out-of-place-only`, then
  `python -m pip install -e . -v --no-build-isolation`. Existing CMake caches
  otherwise leave HIPHooks stale. Both Torch build helpers implement this.
- Use `THEROCK_GFX1013_SCALAR_BACKENDS=ON`; options remain off for other profiles.
- The hipBLASLt provider option is `THEROCK_ENABLE_HIPBLASLTPROVIDER=ON`.
  Native Torch hook checks an actual linked zero-workspace f32 heuristic.
  Default Torch BLAS stays hipBLAS/rocBLAS; explicit hipBLASLt selection works.
- SWISH follows the public fixed-beta-1 contract; activation arg0 is not beta.
- MIOpen legacy bias computes alpha*y + alpha*b + beta\*y: use beta=0.
  Legacy BF16 and FP16 NHWC pointwise behavior required the opt-in private HIP
  bias/activation shader, with strided layouts and rounding after each stage.
  Mixed low-precision output / FP32 virtual fusion intermediates stay rejected.
- Public CK headers and provider TOML/HIP files require explicit superbuild
  tracking, now included. A noop component build does not prove headers staged.
- LD_PRELOAD hipBLASLt observation is test-only; handle RTLD_LOCAL via explicit
  library lookup. Never load the observer in the normal launcher.
- hipSPARSELt scalar compression/pack format is separate from hardware sparse
  matrix formats; retain its guards. Do not imply hardware acceleration.

## Completed Priority 2 batch

The user's P2 scope is tensor algebra, sparse Lt, math libraries, primitives,
performance and candidate packaging. Debug/telemetry/communication bring-up was
explicitly deferred in this batch; `.105` remains unchanged.

| Done | Area                     | Final evidence                                                                                                                                                                                                                                                                                                                          |
| ---- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [x]  | hipTensor                | 166 native GPU cases. Nineteen additional real unary operations, strided dot and three contracted modes; common single-contracted-mode loop avoids repeated mixed-radix division. Seven supported storage/compute combinations retained; no complex or unilateral-reduction claim.                                                      |
| [x]  | hipSPARSELt              | 234 native GPU cases. ReLU, ABS, LeakyReLU, GELU, Sigmoid and parameterized Tanh; clipped ReLU, activation parameter set/get and plan snapshot independence. FP32/FP16/BF16 scalar execution retained; no integer or hardware sparse speedup claim.                                                                                     |
| [x]  | Math libraries           | 149 GPU checks: BLAS 18, FFT 30, solvers 32, Torch sparse 25, direct hipSPARSE triangular solves 16, Torch RNG 6, direct rocRAND 22. CPU references, solver residuals/condition-10,000 systems, gradients, odd/strided/batched FFTs, COO/CSR, six pseudo and four Sobol generators.                                                     |
| [x]  | rocPRIM/hipCUB/rocThrust | 138 int32 GPU reference cases: 42/48/48. Stable signed-key sorts, selection, scans, reductions, wave32/block algorithms and default/nonblocking streams, Thrust par/par_nosync. Guards and counts checked. Installed headers required no new port fix.                                                                                  |
| [x]  | Performance              | 71 bounded CPU-checked timing cases. hipBLASLt register-tiled GEMM verified by 403 existing + 561 expanded API cases = 964. Measured large GEMM approximately 2.1–2.4x faster than the prior scalar Lt implementation; small-shape timings roughly flat/slightly worse. Default Torch BLAS unchanged.                                   |
| [x]  | Candidate                | Exact source-tree reconstruction from 17 library patches, three runtime-system patches and two Torch patches. Independent ROCm/Torch copy with 191 shared-object/kernel-pack hashes; nik GPU checkpoint plus 20 sparse/compiled-forward/backward/FFT/allocation cycles passed. Separate candidate, not a newly activated daily release. |

Reports: `docs/development/bc250_scalar_library_results/priority2-*.json` and
`WS/logs/priority2-*`. Independent package:
`/opt/bc250-stack/candidates/priority2-20261014`; its runtime manifest records
payload hashes. `bc250-torch` remains on the mutable dev prefix with the new work.
Previous releases, images, caches, checkpoints and source snapshots are retained.

Performance findings are workload-specific. Plain Inductor reductions often
measured slower than eager. Explicit HIP graph replay of the compiled fused
SiLU/square/sum at 1,048,576 elements measured about 4x faster than eager; replay
requires fixed captured storage/shapes. Sum/dot graph replay improved the tested
large cases about 1.35–1.4x. No automatic graph or BLAS policy was changed.
For known SPD systems, measured Cholesky factorization/solve was cheaper than a
general solve, particularly with a reused factor; this does not justify silently
assuming arbitrary matrices are SPD. Benchmark source contains the exact cases.

The archived segfault is a Python CPU execution fault at address `0x3b3e6`,
without an adjacent reported GPU reset/fault. The logs contain no useful stack
or identified operator, so the cause remains **unconfirmed**. Independent-copy
and stable-prefix sparse checks passed; they do not establish the cause.
The installed launcher now holds a shared `.runtime.lock` during developer
workloads. Use the host `helpers/update_dev_prefix.py -- COMMAND...` wrapper for
**every mutable-prefix update**; it takes the exclusive lock. Arbitrary writes
that bypass this wrapper remain unprotected. Lock exclusion, exception cleanup
and symlink rejection passed local regression tests.

Use component `hipBLASLt+dist hipTensor+dist hipSPARSELt+dist` targets for these
changes. A global `dist` unnecessarily started rebuilding MIOpen and was stopped
(exit 137); objects were preserved. Targeted staging passed (exit 0), all updated
native kpacks were flattened, and GPU gates ran after staging. No full clean
rebuild was needed. Future global builds can resume the preserved MIOpen objects.

Local final verification: 27 launcher/lock/config/deployment/performance tests,
source syntax/format hooks, nested diff checks and exact patch reconstruction.

## Remaining work, ordered

- [ ] Further measured tuning, especially small/skinny GEMM, scalar tensor/sparse
  throughput, reduction launch policies and representative long training steps.
- [ ] Wider numerical/type/layout/API coverage: floating/custom and segmented
  primitives, additional solver/FFT/sparse operations, RPP and generic tensors.
- [ ] Feasible hipTensor complex/unilateral reductions and further plan APIs;
  hipSPARSELt additional types and API combinations. Keep hardware-instruction
  limitations and backend-specific compression explicit.
- [ ] hipDNN mixed intermediate types, more fusion graphs and provider coverage;
  wider hipBLASLt grouped APIs and epilogues where feasible.
- [ ] Longer training/checkpoint/resume and sustained shared-memory/stability
  checks. The current developer profile is mutable and only bounded-qualified.
- [ ] Diagnose the historical crash if it recurs with a stable prefix and capture
  a real stack; current successful gates are not proof of its root cause.
- [ ] Distributed/RCCL qualification later; preserve `.105` for comparison until
  `.104` is stable and user authorizes its migration.
- [ ] Protected driver/telemetry/debug/counter investigations; SDMA, reset and
  experimental VM heap require isolated qualification with SSH/boot recovery.
- [ ] Media engines only if kernel/firmware exposes working hardware rings.

Do not call this 100% ROCm support. Hardware matrix instructions are unavailable;
unsupported paths must remain explicit rather than architecture aliases.
