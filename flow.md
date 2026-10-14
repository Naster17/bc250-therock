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

The launcher now uses a mutable development stack, not the frozen daily release:

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

| Repository     | Final Priority 1 revision                  |
| -------------- | ------------------------------------------ |
| rocm-libraries | `f12e43e3f9be393dca54de17d7b34a29daffc57b` |
| PyTorch        | `4f13bf86476252e3d57b17d1c72f8512814400e9` |
| rocm-systems   | `dbdb7b8d1c56316d873a9b90e548e5a72853496d` |
| Triton         | `ee65ff3b85f99d52e678b252a99c8425fa46e329` |

Apply library patches in `patches/amd-mainline/rocm-libraries` in order through
0016, and Torch patches in `patches/bc250-pytorch` through 0002.
`.104` source trees are exported, gitless copies: `WS/source`,
`WS/pytorch-source`, `WS/pytorch-work`. Their GIT_REVISION files and
`WS/SOURCE_PRIORITY1.json` record provenance. Earlier source archives and
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

## Remaining work, ordered

- [ ] Performance: compare actual hipBLASLt versus rocBLAS dispatch and CPU;
  representative Torch training/attention, GEMM shapes and memory pressure.
  Tune native scalar kernels only after correctness checks. No new speedup claim.
- [ ] Broader existing-library numerical/shape/layout/type coverage, including
  solver, sparse, FFT, RPP, CK Tile and hipTensor. Expand unsupported public API
  subsets only where hardware allows; maintain honest operation-level tables.
- [ ] hipDNN mixed intermediate types, more fusion graphs and provider coverage;
  wider hipBLASLt epilogues and grouped APIs where feasible.
- [ ] Longer training/checkpoint/resume and sustained memory/stability checks,
  especially under shared CPU/GPU memory pressure. Current dev profile is mutable.
- [ ] Distributed/RCCL qualification later; preserve `.105` for comparison until
  `.104` is stable and user authorizes its migration.
- [ ] Protected driver/telemetry/debug/counter investigations; SDMA, reset and
  experimental VM heap require isolated qualification with SSH/boot recovery.
- [ ] Media engines only if kernel/firmware exposes working hardware rings.

Do not call this 100% ROCm support. Hardware matrix instructions are unavailable;
unsupported paths must remain explicit rather than architecture aliases.
