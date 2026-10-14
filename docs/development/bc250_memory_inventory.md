# BC-250 durable implementation and hardware inventory

Audited 2026-10-14 through **code milestone `ce0025df5`**. This is the detailed
project memory; [flow.md](../../flow.md) is the short resume cursor and ordered
plan. Read the relevant section here before investigating a supposedly missing
feature. Update affected entries after substantive work, preserving evidence.

## Evidence, precedence and audit coverage

Current host manifests, exact source snapshots, committed patches and executed
reports outrank prose from older milestones. The audit covered 143 BC-250 branch
commits after base `7a803357a6e0de10c780a683aa7492da39579938`, all 17 library,
three system, two Torch and four Triton patches, component histories, retained
validation/benchmark reports, `.104` manifests/kernel/sysfs and a read-only
`.105` legacy-source inventory. Twenty project threads were present in the local
Codex state index. Private transcripts are recovery sources, not support proof.
The unrelated untracked `src/` was preserved and excluded from modification.

Distinguish **implemented**, **built**, **executed/numerically qualified**,
**unqualified**, and **hardware unavailable**. Counts describe particular scopes;
counts from different milestones overlap and must not become a coverage percentage.
No claim below establishes every public operation, shape, type or concurrency mode.

The durable entry point is the root AGENTS.md rule plus flow.md; keep the detailed
inventory on demand instead of injecting its entire history into every turn.
OpenAI documents project instructions and workspace files as continuity tools:
[AGENTS.md discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md),
[workspace memory and compaction](https://developers.openai.com/cookbook/examples/agents_sdk/building_reliable_agents_memory_compaction).
Files persist independently of conversational summaries; no hidden permanent
model memory or guaranteed perfect recall is claimed. A versioned documentation
copy under `WS/agent-memory/latest` is separate from qualified source snapshots.

This audit changed documentation only. A later documentation commit does not
invalidate the installed code revision or require a ROCm/Torch rebuild.

## Hardware: measured board versus background descriptions

Community reverse-engineering documentation describes BC-250 as an ASRock mining
APU board with Zen 2 CPU cores, shared GDDR6, DisplayPort, Ethernet and USB. Its
stock core counts and marketing architecture names describe a different
configuration from our unlocked board. No manufacturer datasheet was recovered;
use measurements and native compiler capabilities for implementation decisions.
See the [community hardware documentation](https://github.com/H6-Technologies/BC-250).

The compiler execution model is **gfx10.1.3**, not a substitute gfx1010/gfx1030
identity. LLVM lists gfx1013 as an APU and exposes wave64 as a target option;
our native qualification uses wave32/XNACK−. Target options alone do not prove
wave64 or fault-migration execution. See [LLVM's processor table](https://llvm.org/docs/AMDGPUUsage.html#processors).

| Property                    | Measured `.104` value / durable evidence                                                                                                                        |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU                         | AMD BC-250, family 23/model 71, eight cores/16 threads, SMT active; lscpu and `/sys/devices/system/cpu/online`                                                  |
| CPU instruction/cache facts | AVX/AVX2/FMA; 256 KiB L1d, 256 KiB L1i, 4 MiB L2, 8 MiB L3; one NUMA node. Do not assume AVX512.                                                                |
| PCI identity                | `0000:01:00.0`, vendor/device `1002:13fe`, subsystem `1022:0000`; DRM device uevent/sysfs                                                                       |
| VBIOS string                | `113-AMDRBN-003`; not a complete BIOS image provenance claim                                                                                                    |
| Native GPU                  | KFD target 100103; HIP `gfx1013:xnack-`                                                                                                                         |
| Topology                    | 80 SIMDs, two SIMDs/CU → 40 CU, arranged as 20 WGP; HIP multiprocessor count 20 is not a lost-CU bug                                                            |
| Execution                   | Qualified wave32, 64 KiB LDS; KFD max 20 waves/SIMD; historical HIP L2 report 4 MiB                                                                             |
| Compute queues              | KFD 24 CP queues; DRM compute ring field is a mask, not a queue count                                                                                           |
| SDMA enumeration            | Two engines, eight queues/engine; zero XGMI SDMA engines and zero P2P links. Enumeration is not user-SDMA qualification.                                        |
| Memory                      | MemTotal 15,579,344 KiB ≈14.858 GiB; GTT 15,938,355,200 bytes =15,200 MiB ≈14.844 GiB; VRAM aperture 512 MiB; KFD local_mem_size 0                              |
| Memory implication          | CPU/GPU share physical board memory. GTT and system RAM are overlapping views, not separate ~16 GB budgets. OS/cache/CPU allocations consume the same capacity. |
| PCI link sysfs              | Current/max 16 GT/s, width 16. This is an advertised link property, not measured shared-memory bandwidth.                                                       |
| Firmware fields             | KFD fw_version 144, sdma_fw_version 44, num_xcc 1. Firmware-file presence does not independently prove which image loaded.                                      |

Hardware-relevant sources include `compiler/amd-llvm/llvm/lib/Target/AMDGPU/GCNProcessors.td`,
`AMDGPU.td`, ROCr `core/runtime/isa.cpp`, the native HIP probes and
`WS/logs/baseline-torch.log`. `WS` means
`/home/nik/bc250work/rocm-10.2-gfx1013` throughout this file.

| Capability                          | Current conclusion                                                                                                                                                               |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scalar FP32/FP64, FP16/BF16 storage | Native numerical GPU paths qualified for documented operations; availability is not a throughput guarantee                                                                       |
| Low-precision atomics               | Existing HIP CAS fallback passed contention, returned-value and neighbor-guard tests; no newly implemented/native FP16/BF16 atomic instruction claim                             |
| Packed arithmetic versus dot        | Some scalar/packed-half arithmetic is usable; unsupported packed dot lowering is replaced by widened scalar operations. Do not equate all packed arithmetic with matrix support. |
| MFMA/WMMA                           | No usable native path. Never whitelist another architecture or enable matrix-only engines to bypass this. Scalar implementations of the mathematics remain possible.             |
| Wave64 / XNACK+                     | Compiler/runtime registration exists; our board execution/fault migration is unqualified                                                                                         |
| SDMA                                | Keep `HSA_ENABLE_SDMA=0`; ROCr also intentionally defaults gfx1013 user SDMA off. Explicit override is an experiment, not a supported default.                                   |
| VMM / async pools                   | APIs retained. Only the experimental VM-heap default is disabled on gfx1013.                                                                                                     |
| GPU reset                           | Current boot and live parameter `amdgpu.gpu_recovery=0`; reset behavior unqualified                                                                                              |
| Media                               | DRM VCN decode/encode, JPEG, UVD and VCE queries return −EINVAL with zero exposed rings. Current driver exposure is not proof silicon lacks every media block.                   |
| Profiling                           | API/kernel/copy tracing works. Hardware counter tables, PC sampling, ATT and SPM are unqualified.                                                                                |
| Debugger                            | Native architecture/ELF registration patched; consumers and attach/step/wave suites not qualified                                                                                |

## Kernel, unlock, cooling and recovery

Active kernel: `7.1.5-bc250rocm-r2 #2 SMP PREEMPT_DYNAMIC`, built 2026-09-16.
The running kernel CONFIG was not available through `/proc/config.gz`. A later
workspace-organization task recovered its original build-tree configuration from
.105 at `bc250stack/kernel/configs/r2.config`; provenance records this distinction.
Do not infer it from unrelated 7.0.10 headers.

Live boot files:
`/home/nik/bc250work/bc250-rocm-lab/{vmlinuz,initramfs}-7.1.5-bc250rocm-r2`.
GRUB persistent default is `bc250-rocm-r2-kernel`, timeout three seconds.
`WS/recovery/` retains kernel/initramfs, grub.cfg, grubenv, grub and cmdline.txt.
Additional lts6.18.52/n17/stable7.0.10 boot files are present, but their presence
alone is not a verified fallback boot. Use one-shot GRUB selection for a new
kernel, preserving the known working persistent default until SSH gates pass.

Relevant current boot arguments (see `/proc/cmdline` for exact root UUID):

```text
amdgpu.sg_display=0 mitigations=off amdgpu.gttsize=15200
ttm.pages_limit=3891200 ttm.page_pool_size=3891200
```

The exact TTM arguments are `ttm.pages_limit=3891200` and
`ttm.page_pool_size=3891200`; additional arguments are
`amdgpu.gpu_recovery=0`, `amdgpu.bc250_cc_write_mode=3`,
`amdgpu.bc250_flush_pasid_kiq=0`, `amdgpu.bc250_flush_by_runlist=3`, and
`bc250.vulkan_v33_r1=1`. Preserve this known profile when investigating changes.

SP5100 TCO watchdog: OpenRC service active, `/dev/watchdog0`, 10-second
keepalives/120-second timeout. Configuration lives in `/etc/conf.d/watchdog`.
Activity is not proof of recovery from a CPU-responsive GPU hang. Deliberate
traps/reset/SDMA experiments need independent SSH/boot recovery planning.

CPU unlock service `/etc/init.d/bc250-8core-unlock` calls
`/usr/local/bin/bc250-8core-unlock boot`. The script describes SMU mailbox 0x98,
mask register 0x0115A870, stock mask 0x77→0xff and warm-reset persistence;
these descriptions are not newly executed register experiments. The 40-CU
patch is retained at
`/home/nik/bc250work/bc250-gfx1013-fix/patches/kernel/40cu-bc250-unlock.patch`,
with older work under `/home/nik/bc250-40cu-unlock`. Imported performance claims
in that patch are not measurements from this audit.

Important operational facts:

- No cpufreq policy or scaling_governor exists on `.104`; do not report a
  configured CPU performance governor.
- GPU power policy is auto. `pp_dpm_sclk` and hwmon freq1_input disagree; clock
  telemetry remains untrusted. Advertised maxima are not attained-clock proof.
- The SMU OC service requests CPU 3500 MHz, scale −19, max_temperature 90;
  these are settings, not verified operating frequency/voltage.
- `/usr/local/bin/bc250-fan-curve` reads CPU k10temp/Tctl despite comments saying
  GPU temperature. NCT6686 pwm2/CPU_FAN1 controls the fan. The existing curve
  uses PWM90 below60°C,175 below70°C,250 below80°C,255 above. Do not assume
  hwmon numbering survives reboot. AMDGPU hwmon has no PWM attributes here.
- Retain cyan_skillfish2 CE/ME/MEC/MEC2/PFP/RLC/SDMA firmware files under
  `/lib/firmware/amdgpu`; file presence is not verified runtime load identity.
- Retain `/home/nik/bc250work/{bc250-rocm-lab,bc250-gfx1013-fix,bc250-vulkan-v33,strata-port}`,
  `/home/nik/{bc250-40cu-unlock,bc250_smu_oc,rocm-systems}`. Older Vulkan7.0.10
  module/initramfs artifacts are distinct from the live r2 kernel.

## Installed stack and precise provenance

| Item                   | Current code / location                                                                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Branch                 | `users/Naster17/rocm-10.2-gfx1013`                                                                                                                           |
| Qualified TheRock code | `3dae3539c3ffe46bb10cb83176cfdfebe1fd2dd7` (N1.2); newer docs commits need no runtime rebuild                                                                       |
| Libraries              | `c6ac34f0ba3c4de11a0aa7e21fc9ec19adc6ef9e`                                                                                                                   |
| Systems                | `dbdb7b8d1c56316d873a9b90e548e5a72853496d`                                                                                                                   |
| LLVM                   | Upstream `7667f52dd687899256a3285ab0bce3c53bc73394`; no custom LLVM architecture patch                                                                       |
| Torch                  | `4f13bf86476252e3d57b17d1c72f8512814400e9`; release/2.13 baseline `c779020750d7dc99a39ea8c104ef8df1fc9d0755`                                                 |
| Triton                 | `ee65ff3b85f99d52e678b252a99c8425fa46e329`; baseline `66b3f9ff575eb8db6adec658bbf0badd9a8d2e25`                                                              |
| Distribution/version   | TheRock 10.2.0; HIP component reports 7.17.0; Torch 2.13.0a0+gfx1013, qualified Triton 3.8 wheel. Different component version strings are not a failed port. |
| Active link            | `/opt/bc250-stack/current` → `/opt/bc250-stack/dev/priority1`; directory name retained, runtime profile `gfx1013-priority2-n11-n12-dev`                              |
| ROCm / Torch           | `WS/build/dist/rocm`, editable `WS/pytorch-work`                                                                                                             |
| Runtime image ID       | `5df0a58392de13916819f6d84342b0338ae2c2bbb2f3c21a1f3ad38be2724a59`                                                                                           |
| Runtime tag            | `localhost/bc250-stack:10.2-gfx1013-priority1-candidate`; tag name predates P2, immutable ID above is authoritative                                          |
| Build image            | `localhost/bc250-rocm-build:10.2-ccache`; Ubuntu24.04/glibc inside Alpine host                                                                               |
| Triton wheel SHA256    | `2d65d6fefe359116521640e3cc2257bd03b4b24f6877fc8e28946ee9c57544d8`                                                                                           |
| Candidate              | Independent `/opt/bc250-stack/candidates/priority2-20261014`, bounded-qualified, not activated as a daily release                                            |
| Recovery link          | `/opt/bc250-stack/before-dev-current` → retained `10.2-gfx1013-20261013-4b000d542` release                                                                   |
| Initial release        | Retained `10.2-gfx1013-20261012-a640c9296`; `/opt/bc250-stack/previous` records prior deployment                                                             |

`.104` exported source trees are gitless. Inspect `WS/source/GIT_REVISION`,
`WS/source/rocm-libraries/GIT_REVISION`, and both Torch GIT_REVISION files.
`SOURCE_PRIORITY2.json` records 51 transferred file hashes and code provenance;
workspace-root/dev/candidate copies exist. `SOURCE_PRIORITY1.json` and
`SOURCE_CANDIDATE.json` retain their older meanings. Source archives are baselines,
not magically updated by patching extracted trees. Generated torch.version's
Git label may retain the original export revision; use source manifests instead.

Reconstruction: 19 patches in `patches/amd-mainline/rocm-libraries`, three in
`patches/amd-mainline/rocm-systems`, two in `patches/bc250-pytorch`, four in
`patches/bc250-triton`. Apply in order. The P2 reconstruction helper verified
exact library/system/Torch trees using private indexes; it did not recreate the
entire compiler or claim a clean-room full rebuild. Nested committed branches
and `.smrev` metadata preserve the port independently from root submodule pins.
Do not reset sources or casually change pinned submodules. N1.1 adds
`SOURCE_N11.json` (7 verified changed files, qualified root/library revisions,
active hipTensor hash) on the P2 snapshot; P2 manifests/candidate stay unchanged.
N1.1 library reconstruction matched all18 ordered patches exactly. N1.2 adds
`SOURCE_N12.json` (8 verified changed files) on N1.1/P2 and exact19-patch
library reconstruction; retained earlier manifests keep their original meaning.

## Library patch index

| Patch | Durable change / commit anchor                                                         |
| ----- | -------------------------------------------------------------------------------------- |
| 0001  | Native rocBLAS/Tensile target, capability/library predicates: `55e9abea68b`            |
| 0002  | MIOpen HIP-direct convolution applicability: `b6b37dcdf0d`                             |
| 0003  | Honest hipBLASLt HOST_API_ONLY build: `9ca9c61165f`                                    |
| 0004  | rocRAND native host/device dispatch: `039b79bcb43`                                     |
| 0005  | CK scalar DL/generic CK Tile architecture integration: `3f6e5c01b3c`                   |
| 0006  | Tensile CMake native target validation: `f4302e457a9`                                  |
| 0007  | Omit device-management discovery for host-only Lt: `2efc3717efe`                       |
| 0008  | hipTensor generic-only native profile: `a9e578151e5`                                   |
| 0009  | Execute generic arithmetic cases rather than silently skip: `b35368dd8f5`              |
| 0010  | RPP audio FFTS object dependency: `0a912025cc0`                                        |
| 0011  | Isolate generated headers/read-only sources and generic CK requirements: `e0eb4c93e7a` |
| 0012  | rocPRIM native dispatch in standalone and two hipCCL copies: `a8fc909de24`             |
| 0013  | RPP spectrogram destination row stride: `b9c8ad41e5b`                                  |
| 0014  | Native scalar Lt/tensor/sparse backends and generic CK operations: `a6656565253`       |
| 0015  | Wider types/epilogues/pruning, generic operators, decibel fixes: `7bc253117ee`         |
| 0016  | Lt C++/grouped consumers, CK Tile, hipDNN fusion/provider integration: `f12e43e3f9b`   |
| 0017  | Register-tiled Lt GEMM, tensor unary expansion, sparse activations: `01566e3144d`      |
| 0018  | hipTensor unilateral scalar contraction modes: `681d2f8a25d` |
| 0019  | hipTensor native C32F/C64F scalar contractions and identity/NEG/CONJ: `c6ac34f0ba3` |

System patches: build-time ROCr hotswap default; native ROCdbgapi architecture
registration (`fa95eec5a2`); gfx1013 safe pool default (`dbdb7b8d1c`). The first
patch has a synthetic format-patch ID; its applied tree, not that ID, is evidence.
Torch patches: NHWC average-pool gradient padding (`b36877506d3`), qualified
native Lt backend selection (`4f13bf8647`). Triton patches: widened low-precision
dot (`c815e192`), signed/rounding tests (`63915ab7`), BF16 multiplication fallback
(`270bfb9d`), RDNA1 buffer flags (`ee65ff3b`).

## Component coverage: implemented versus existing native backend

| Component                                   | What exists and was executed                                                                                                                                                                                 | Boundary / where to continue                                                                                                                                |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLVM/Clang/device libs/COMGR/HIPRTC         | Existing upstream gfx1013 support; native compile/runtime compile, builtin probes, Triton instruction-selection checks and GPU kernels                                                                       | Not a newly written compiler target; wider builtin/debug/offload suites remain                                                                              |
| ROCr/HIP/CLR                                | Native allocation/copy/stream/churn tests; safe pool default; VMM and async APIs retained                                                                                                                    | SDMA/reset/VM heap/fault migration are separate controlled experiments                                                                                      |
| Low-precision atomic APIs                   | Existing HIP CAS fallback qualified by `hip_atomic_low_precision.cpp`, contention/returned values/neighbor guards                                                                                            | Do not patch working atomic headers or claim native low-precision atomic instructions                                                                       |
| rocBLAS/Tensile + hipBLAS                   | Native identity and scalar capability/predicate patches, 570 fallback kernels; large SGEMM and four Torch precisions; P2 BLAS18                                                                              | No complete tuned database/BLAS suite. hipBLAS uses this backend, not custom new kernels.                                                                   |
| hipBLASLt                                   | 964 C/C++ API cases: f32/f16/bf16 with f32 accumulation, layouts/transposes/batches, host/device scalars, bias/activation/AUX/DGELU, host grouped queries/events/lifetimes; register tile                    | No FP8/INT8/FP64, device user groups, arbitrary tuning or matrix engine. Default Torch BLAS remains rocBLAS.                                                |
| hipTensor generic                           | Historical154 rank-two and23 higher-rank reduction/permutation/binary/trinary checks in generic-only profile                                                                                                 | Wider rank/layout/operators remain; these counts are separate from contractions                                                                             |
| hipTensor contraction | N1.1 real273 API cases (253 numerical +20 descriptor); N1.2 complex450 (426 numerical +24 descriptor) and273 real regressions. Same-type C32F/C64F, identity/NEG/CONJ, public complex scalars, shared/unilateral sums, padded/permuted layouts, zero/imaginary scalars, span/alias/guard bytes, in-place conjugated C/D,16-mode union with unit axes, plan lifetime and3algorithms×3workspace preferences/type with JIT NONE | Next N1.3: plan enum/attribute-size validation and honest cache/JIT contracts. Mixed/widened complex and other complex unary ops unsupported; small-fixture timing is not general performance. |
| hipSPARSELt                                 | 234 cases: f32/f16/bf16, strip prune/check/compress/matmul, low-precision maximum-L1 TILE, sparse A/B, layouts/transposes/padded batches, descriptor/plan/Search; six activation families                    | Separate backend compression. No FP32 TILE, integer/FP8, bias/vector scaling, split-K/multi-stream/broadcast guarantees or measured hardware sparse speedup |
| CK scalar DL                                | CPU-checked FP32/FP16/signedINT8 GEMM examples; native architecture applicability changes                                                                                                                    | Admitted DL convolution/grouped classes are not automatically executed/qualified                                                                            |
| CK Tile                                     | Prior19 generic runs; P1 numerical247: top-k60, LayerNorm18, scalarGEMM160, Pool3D9;400 argument rejections; installed public ScalarGemmKernel                                                               | Explicit generic API, not automatic matrix-kernel replacement; more ops/types/layouts remain                                                                |
| MIOpen                                      | CPU-reference f32/f16/bf16 convolution forward/dx/dw, groups1/2; native HIP-direct solvers; 3D/provider tests                                                                                                | No full neural operator/solver selection/tuning DB qualification; CK integration stays off in current profile                                               |
| hipDNN providers                            | Historical pass-by-value5 +HIP9 +MIOpen9, scalar-expansion14; P1 Lt48 +HIP18 +3D18 +fusion16; six host parameter tests                                                                                       | Normalization/training/backward/provider/fusion subsets only. Mixed FP32 virtual/low-precision output rejected; no automatic Torch provider routing claim   |
| RPP/FFTS                                    | Native image layouts16 plus audio26; decibel zero/reference/reduction fixes, spectrogram row-stride and FFTS build/link fixes                                                                                | Broader image/audio/volume operations, types and layouts remain                                                                                             |
| rocPRIM/hipCUB/rocThrust                    | Historical reductions/scans and208 block-reduction tests; P2 int32 cases42/48/48 with guards/streams, signed stable sorts, selection, scans, blocks/wave32 and Thrust policies                               | Floating/custom/segmented algorithms and broader tuning remain. hipCCL is header packaging, not RCCL.                                                       |
| rocFFT/hipFFT                               | Existing native backend, persistent 16-worker HIPRTC AOT cache; prime DFT andP2 FFT30                                                                                                                        | No custom rocFFT component architecture patch; more plans/types/layouts remain                                                                              |
| rocSOLVER/hipSOLVER                         | Existing native/BLAS backend; Cholesky andP2 solver32 with residuals, gradients and condition10,000                                                                                                          | Wider difficult/complex/blocked cases and policy/performance remain                                                                                         |
| rocSPARSE/hipSPARSE                         | Existing backend, Torch COO/CSR25 anddirect triangular16 inP2                                                                                                                                                | More formats, analyses/solves/API contracts remain; old corruption fix was CLR, not sparse kernel rewriting                                                 |
| rocALUTION                                  | CG40 ordinary cases andfive fresh-process200 passes after pool-default fix                                                                                                                                   | Larger systems, other solvers/preconditioners/concurrency remain                                                                                            |
| rocRAND/hipRAND                             | Native dispatch patch; historical44 basic+6 dispatch; P2 direct22 acrosssix pseudo/four Sobol variants andTorch RNG6                                                                                         | More generator/distribution/type combinations remain                                                                                                        |
| OpenCL                                      | Native CL1.2 kernels, LDS/atomics/images/events/profiling checked                                                                                                                                            | OpenCL2 completeness and other feature combinations remain                                                                                                  |
| OpenMP/Flang                                | Mandatory GPU SAXPY/reduction/task dependency checks; bundled Flang compiled                                                                                                                                 | Fortran device execution and hipThreads remain unqualified                                                                                                  |
| HIPIFY                                      | hipify-perl CUDA translation → native compilation → exact GPU check                                                                                                                                          | hipify-clang requires parser/SDK input qualification                                                                                                        |
| hipFile                                     | Five synchronous/asynchronous compatibility I/O roundtrips                                                                                                                                                   | Not a direct-storage/DMA fast-path claim; registration/concurrency/ordering remain                                                                          |
| rocprofiler/ROCTX                           | Native marker/API/kernel/copy tracing with 3628 APIs,263 dispatches,814 copies, CPU workload correlations                                                                                                    | Counter mapping/advanced profiling/debug consumers separate                                                                                                 |
| AMD SMI / RDC                               | AMD SMI built without EPYC ESMI; disposable generated-header source copy                                                                                                                                     | Read-only telemetry remains incomplete/untrusted; RDC not built/qualified                                                                                   |
| RCCL/rocSHMEM/MPI                           | Future applicability work, not built in current profile                                                                                                                                                      | Local readiness/software transports first; no XGMI/GDA/P2P/SDMA claims; two-board work deferred                                                             |
| rocDecode/rocJPEG                           | Current driver does not expose media rings                                                                                                                                                                   | Qualify feasible host decoding/HIP interoperability separately; do not invent an upstream JPEG software fallback                                            |
| rocWMMA / matrix-only engines               | Not applicable as native MFMA/WMMA execution                                                                                                                                                                 | Distinct scalar emulation/math paths could be work; never enable them by architecture spoofing                                                              |
| rocRoller/emulation/other optional projects | No blanket qualification                                                                                                                                                                                     | Audit pinned API/instruction applicability before enablement; CPU fallback must be identified                                                               |

Code roots for actual port backends:

- `rocm-libraries/projects/hipblaslt/library/src/amd_detail/gfx1013_scalar.{cpp,hpp}`
- `rocm-libraries/projects/hiptensor/library/src/contraction/contraction_scalar.{cpp,hpp}`
- `rocm-libraries/projects/hipsparselt/library/src/hcc_detail/gfx1013_scalar.hip`
- `rocm-libraries/projects/composablekernel/include/ck_tile/`
- `rocm-libraries/dnn-providers/miopen-provider/engines/plans/`
- `rocm-libraries/projects/rpp/src/modules/hip/`
- `rocm-systems/projects/clr/rocclr/device/rocm/rocdevice.cpp`
- `sources/bc250-pytorch/aten/src/ATen/{native/cuda/AveragePool2d.cu,cuda/detail/CUDAHooks.cpp}`
- `sources/bc250-triton/third_party/amd/` (verify repository path before edits)

Supported scalar contraction type combinations (A/B share the input type; C/D
share the output type): f16→f16, bf16→bf16, f32→f32, f16→f32 and bf16→f32 use
f32 computation; f32→f64 and f64→f64 use f64 computation. Reduced-precision
accumulation, f16→f64 and narrowed outputs are not supported in this profile.
The contraction source currently accepts 23 real unary operators (19 added in P2).
Low-precision Lt inputs may likewise use f32 outputs; sparse Lt uses its documented
backend-specific FP32 1:2 / FP16-BF16 2:4 compression, not a portable hardware format.

## Torch, training, files and resource utilization

[Training evidence](bc250_training_memory.md), root `d744f0952`, proves real
SmolLM2-135M full FP32 fine-tuning and FP16/BF16 AMP LoRA for a bounded example.
Model revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, weights SHA256
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`;
134,515,008 base parameters. LoRA rank 8 / alpha 16 / q_proj+v_proj gives 460,800
trainable parameters. Batch 1 / 32 tokens / one text, mathematical SDPA and AdamW:
four updates plus one fresh-container resume update. This is not a convergence
study or proof of larger-model/full-model-AMP/QLoRA support.

Full FP32 loss fell 4.647392→2.845694→2.507597, peak~2.92GB. LoRA loss fell
~4.647392→4.0175→3.803, peak~704MB; frozen base remained unchanged, activation
checkpointing used, FP16 scaler skipped no updates. Resume preserved
model/adapter, optimizer, scheduler, scaler and CPU/GPU RNG; logits and exact RNG
probes were checked. CPU/GPU logit relative L2 was5.56e−6; representative gradient
relative L2 was8.36e−5–4.68e−4. Near-zero elementwise reference failures were
retained and replaced with justified aggregate/max-error checks, not ignored.

Memory/concurrency gates passed 10 GiB GPU pressure + 768 MiB CPU storage, fragmented
reuse, four streams, eight basic graph replays and two GPU processes in 123.99 s.
This respects shared memory; near-OOM/hang recovery was not established.
No CPU/RAM cgroup cap is imposed by the normal launcher; its default 1 GiB shared
memory mount is separately configurable. GPU use still competes with host RAM.

Retain `WS/{training-model,training-env-stable,training-validation}` and
`WS/logs/training-memory/`. Qualified consumer pins are Transformers 5.2.0,
PEFT 0.18.1, Accelerate 1.12.0. `bc250-train` mounts the frozen consumer bundle
through `current/python-consumers`; it does not replace our custom Torch.
Do not redownload or recreate these because conversation context was compacted.

P1 Torch evidence:64 checks, including 52 eager/compiled dynamic-shape indexing,
normalization, training andmath SDPA cases, plus 12 explicit Lt cases with 36
observed actual successful library calls. LD_PRELOAD tracing is test-only and
must resolve RTLD_LOCAL dependencies; never enable it in the ordinary launcher.
Additional prior eager 56/compiler 18/operator 55 checks remain historical scopes.

Launcher `build_tools/bc250/bc250_torch.py` handles Python/script/-c/-m/stdin/shell,
retains host HOME/cwd/absolute paths, mounts script parents, supports explicit
mount modes and private environment handoff across doas. It runs rootful Podman
as uid/gid 1000 plus device groups. Arbitrary data outside automatic mounts needs
an explicit mount; do not claim an unrestricted host filesystem. Thirteen IO
scenarios include persistent packages, HTTP downloads, DataLoader spawn,
interruption handlers and checkpoint resume. Ctrl-C/termination checkpoints
require the user's application to implement a handler.

Current P2 launcher source added a shared runtime lock; cooperating host updates
must use `helpers/update_dev_prefix.py -- COMMAND...` with the exclusive lock.
This prevents overlap of launcher workloads and prefix replacement, not arbitrary
manual writes. `/usr/local/bin/bc250-torch.before-priority2` and recovery launcher
copies are retained. Metadata `/opt/bc250-stack/launcher.json` predates the P2
lock change; installed-script hash plus committed source and P2 launcher report
are newer evidence. Do not overwrite historical metadata to fake provenance.

N1.1 .104 gate: `WS/logs/n11-hiptensor-r3/result.json`, root report
`bc250_scalar_library_results/n11-hiptensor.json`. Old library rejected the new
case; targeted12-job/20GiB-ccache build17.1s and GPU gate2.7s passed under the
exclusive runtime lock. Kernel logs identical (1009 lines). Thirty-repeat HIP
stream-event timings across seven types: A-only12.6–13.5µs, mixed27.7–29.3µs;
includes submission gaps, no previous implementation/speedup comparison.
Independent prior library: `WS/recovery/n11-hiptensor-before.so.0.1`.
Two initial container failures (`WS/logs/n11-hiptensor{,-r2}`) happened before
compilation: cgroups were unmounted after a host reboot, and runc rejected
NoCgroups. Existing cgroups service started in unified mode and enabled in boot;
no kernel change or reboot, SSH/fallback intact. Do not use disabled cgroups
with the installed runc. N1.1 did not rebuild Torch or alter .105/frozen candidates.

N1.2 .104 gate: `WS/logs/n12-hiptensor-r2/result.json`, root reports
`bc250_scalar_library_results/n12-{hiptensor,reconstruction}.json`. C32F/C64F
scalar path uses HIP complex addition/multiplication, shared geometry/guards;
public host scalar ABI and FP64 capability check retained.450 complex cases
(426 std::complex<double> references,24 descriptor rejections), then273 real
regressions passed. All27 identity/NEG/CONJ A/B/C combinations executed on seven
layouts/two types; mixed/unilateral sums, guards, signed-zero/NaN and imaginary
scalars, null zero-beta C, in-place conjugated C/D,16-mode union with unit axes,
destroyed source descriptors/preferences, nine algorithm/workspace pairs/type.
First17.1s build passed; gate stopped on test-only false alias expectation for
adjacent scalar C/D tensors (`WS/logs/n12-hiptensor`, preserved attempt1 report).
Prior active library restored before releasing lock. Fixed assertion checks
multi-element spans; force source timestamps on retry to regenerate staging
rather than retain the restored old dist hardlink. Cached restage4.9s,
complex2.3s/real2.8s gates passed; kernel logs identical1009 lines.12jobs/20GiB
ccache, targeted hipTensor+dist/exclusive lock. Thirty-repeat tiny stream-event
timings: GEMM8.7–9.9µs, mixed21.5–23.2µs including submission gaps; no speedup
or large-performance inference. Independent prior library:
`WS/recovery/n12-hiptensor-before.so.0.1`. No new host/kernel/Torch changes.

## Defects and traps already investigated: do not rediscover them

| Issue                           | Finding / resolution                                                                                                                         | Evidence to preserve                                                                                                                           |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Intermittent CG/CSR zero output | Proven live CSR analysis metadata overwrite associated with experimental VM heap; compiled default now uses the established pool backend     | Root `47925cbd5`, systems patch0003, `rocalution-step-trace.log`, allocation/raw-metadata/first-kernel probes, 40 plus fresh-process200 checks |
| Experimental VM heap            | Default bypass is a mitigation, not a repair. VMM/async pools retained; explicit opt-in remains experimental.                                | `rocdevice.cpp`, root `bd6e4c8a4`, release gates                                                                                               |
| Later Python CPU segfault       | Execution fault at0x3b3e6, no identified stack/operator; cause remains unknown. Stable-prefix sparse gates and independent candidate passed. | Older operator-candidate/kernel logs, P2 candidate/launcher20 cycles each; do not merge with the proven CSR defect                             |
| Triton incorrect buffer reads   | Missing RDNA1 descriptor flag branch; globals passed while buffer path failed. Fixed.                                                        | Triton patch0004/root `9fdae0b5d`; both memory paths validated                                                                                 |
| Triton low-precision lowering   | Unsupported gfx13 packed dot/BF16 multiply replaced by widened scalar arithmetic; signed limits and rounding-tie/NaN tests retained          | Four Triton patches and MLIR/LLVM/assembly/GPU gates                                                                                           |
| NHWC avgpool backward           | Padding correction in Torch source                                                                                                           | Torch patch0001, operator coverage                                                                                                             |
| MIOpen bias/fusion              | Legacy bias beta1 doubles output; beta0 is correct. BF16 and FP16 NHWC pointwise required private scalar HIP fallback; output routing fixed. | Library patch0016, provider CPU references; mixed-intermediate rejection                                                                       |
| Lt SWISH                        | Public beta is fixed1; activation arg0 is not Swish beta                                                                                     | 465 then561 extended consumer cases; hipDNN Lt48                                                                                               |
| RPP audio                       | Spectrogram destination stride and decibel reduction/reference/zero handling fixed; FFTS dependencies/linking fixed                          | Patches0010/0013/0015 and audio26                                                                                                              |
| Source generation               | Torch CMake cache can leave HIPHooks stale even after pip; explicitly hipify before editable rebuild                                         | `build_torch.sh`, `rebuild_torch_pooling.sh`                                                                                                   |
| Installed headers               | CK public headers and provider TOML/.hip require explicit superbuild tracking; noop build alone does not prove staging                       | `math-libs/CMakeLists.txt`, `ml-libs/CMakeLists.txt`                                                                                           |
| Kernel packs                    | hipSPARSELt owns a separate sparselt artifact; extracted pack collisions fixed                                                               | `math-libs/BLAS/CMakeLists.txt`, descriptors; do not merge it back into generic sparse                                                         |
| Reference/test mistakes         | CPU.double() can alias float64 inputs; gradients need detached clones. Some upstream mixed-inference CPU fixtures were limited.              | Corrected P2 math references and meaningful training fixtures, not all-skipped success                                                         |
| Build resources                 | Global dist restarted unrelated MIOpen and was stopped137; targeted component+dist passed0                                                   | P2 stage logs; preserve objects and resume global builds only when needed                                                                      |
| Old/new timing                  | Seed-only fixtures differed; replaced with identical serialized bytes. Clock reports conflict.                                               | Matched reports; excluded raw runs are not comparison proof                                                                                    |

## Performance already measured, with the correct interpretation

Root `24ea38d84`, [performance report](bc250_performance.md), and
`bc250_performance_results/` record nine matched workloads on `.104` 10.2/Torch 2.13
versus `.105` 7.2.4/Torch 2.9.1. GEMM/copy mostly differed ~1–6%; copy~0.2%.
Small CNN results were highly variable on `.105` (59.9% IQR); an apparent 3.4x ratio
is not a stable improvement. Different boards, thermal states and untrusted
clock telemetry prevent attribution of small differences to ROCm alone.

SmolLM2 batch 1/32-token timings were ~131.56ms full FP32 update,~111.72ms FP32 LoRA
without checkpointing,~260ms AMP LoRA with checkpointing and~139–141ms without.
These are training-step processed tokens, not generation tokens/second. The
smallest model may benefit from FP32 because overhead outweighs low-precision compute.

P2 is a separate same-board scalar-kernel comparison: 71 CPU-checked timing cases,
964 Lt API checks, medium/large Lt GEMM~2.1–2.4x faster than prior Lt. Small shapes were
roughly flat/slightly worse. Default Torch BLAS did not change; this is not a 2.4x
whole-Torch or old/new board speedup. Plain compiled reductions often lost to eager.
Explicit HIP graph replay for compiled SiLU/square/sum at 1,048,576 elements measured
~4x over eager; tested large sum/dot~1.35–1.4x. Captured storage/shapes must remain fixed.
For known SPD matrices, Cholesky factorization/reuse was cheaper than general LU solve;
never silently assume all matrices are SPD. The benchmark source encodes exact cases.

## Evidence registry and reproducible workflow

| Scope                             | Durable location                                                                                                                       |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Initial/expanded native gates     | `WS/logs/`, `bc250_port.md`, historical support matrix/roadmap                                                                         |
| Pool corruption/regressions       | `rocalution_diagnostic.cpp`, `sparse_analysis_trace.cpp`, `WS/logs/*rocalution*`, allocator/raw-metadata logs                          |
| Training/memory                   | `bc250_training_memory.md`, `WS/logs/training-memory/`, retained model/adapter artifacts                                               |
| Matched old/new/model performance | `bc250_performance_results/*.json`; `/home/nik/bc250work/rocm-performance-20261013` on both hosts                                      |
| P1/P2 numerical evidence          | `bc250_scalar_library_results/priority1-*.json` and `priority2-*.json`; host logs                                                      |
| Independent frozen release        | 600-second soak40,339 cycles; prior candidate39,657; review46,378 file hashes. These do not qualify current mutable P2 long stability. |
| P2 candidate                      | `candidates/priority2-20261014/runtime.json`,191 shared-object/kernel-pack hashes; validation/candidate.json                           |
| P2 launcher                       | `/home/nik/bc250work/priority2-launcher.{json,pt}`,20 cycles/40 sparse matmuls, compiled gradients and checkpoint                      |
| Kernel comparison                 | P2 report records82 new bridge/veth messages, no new fault/reset/OOM; logs are not byte-identical                                      |

Build without mapped GPU devices; run bounded GPU gates serially after staging.
Use existing objects/20 GiB ccache:16 normal jobs, hipTensor 12, hipDNN providers 6,
rocALUTION 8, Torch 12/link 1, Flang 4/LLVM link 1. rocFFT HIPRTC AOT cache has its own
persistent 16-worker pool, not ccache. Evaluate setup_ccache.py in each builder shell.
Native bootstrap compiler remains under `/usr/lib/llvm-18`; masquerade aliases preserve
old LLVM command lines, not architecture identity.

Exact opt-in: `THEROCK_GFX1013_SCALAR_BACKENDS=ON` with exclusive
`THEROCK_AMDGPU_FAMILIES=gfx1013`. A leftover cache variable named
`THEROCK_ENABLE_GFX1013_SCALAR_BACKENDS=OFF` is not this option. Top-level group
flags can be OFF while dependencies/selected child projects are activated; inspect
actual targets/artifacts, not just group flags. HIPBLASLt provider is enabled;
RCCL/rocSHMEM/MPI/debug/media are off in this profile.

Prefer component `hipBLASLt+dist hipTensor+dist hipSPARSELt+dist` targets to global
dist for local changes. `.104` source exports include root-owned directories: transfer
explicit changed files via tar/checksums, not rsync directory timestamps over them.
Update docs/provenance separately; do not relabel older archive baselines.

Torch rebuild must use explicit HIP generation:

```bash
python tools/amd_build/build_amd.py --out-of-place-only
python -m pip install -e . -v --no-build-isolation
```

Triton lowering changes require `make` before MLIR/LLVM tests. Keep qualified wheel
unless source changed. Exact patch reconstruction helper is
`build_tools/bc250/verify_priority2_reconstruction.py`; it needs local Git objects,
not gitless host exports. Do not rerun all GPU gates just to edit memory/docs.

## Earlier agents, legacy work and archive recovery

BC-250 milestones:

| Anchor                          | Work already done                                                 |
| ------------------------------- | ----------------------------------------------------------------- |
| 7a5c2ca0f                       | Native gfx1013 target/bootstrap                                   |
| a640c9296                       | Initial independent ROCm/Torch/Triton deployment                  |
| 47925cbd5 /4b000d542            | Safe allocator default; expanded candidate/release gates          |
| b2707998a /761ec5803 /f1895f712 | Persistent host paths/files/read-only mounts/shell                |
| d744f0952                       | Real SmolLM2/AMP LoRA/resume/memory concurrency                   |
| 24ea38d84                       | Matched old/new and training performance                          |
| c8dc436d3                       | Frozen snapshot review/pause policy, subsequently revoked by user |
| fdde5561c                       | Scalar backends/operators expansion                               |
| 71f9f0ee4                       | P1 consumers/CK Tile/DNN/Torch/dev activation                     |
| ce0025df5                       | P2 algebra/primitives/math/performance/candidate/runtime-lock     |

Separate branch `users/nik/alpine-musl-port`, tip `36c62955f`, has 22 commits after
the common base. It is **not an ancestor of the active BC-250 branch**. It contains
native-musl toolchain/runtime/COMGR-stack/portable FFT/SMI/OpenCL work, qualified
on Ryzen 8645HS/Radeon 760M **gfx1103**, not BC-250. Examples: `a05baf1ae`,
`7aacbbd89`, `ef108c4ff`. Recover its historical doc with
`git show 36c62955f:docs/development/alpine_port.md`. This work is reusable research,
not current BC-250 native-musl qualification; don't switch branches or integrate it
silently. The current port runs glibc userspace in Podman on Alpine.

Local project thread index: `/home/nik/.codex/state_5.sqlite`, read-only lookup by
cwd `/home/nik/Documents/Vibe/therock`. Raw rollouts live under
`/home/nik/.codex/sessions/YYYY/MM/DD/`. Retrieve only relevant project transcripts if
source/evidence cannot answer a question; do not copy titles/prompts/credentials into
committed memory. Root BC-250 thread ID `01a12a83-7fad-70d1-ab81-57e2e3cf6253`;
rollout filenames may change across compaction/resume, so find by ID rather than hardcode
one current filename. Prior roles include build_guidance, triton_review,
remaining_runtime_audit, remaining_libraries_audit, hipblaslt_scalar,
hiptensor_scalar, hipsparselt_scalar, p1_hipblaslt/p1_ck_tile/p1_torch_compiled,
p2_communication/p2_tensor_sparse/p2_primitives. The P2 communication agent was
redirected to math coverage; its name is not proof RCCL was ported.

`.105` remains the old reference; read-only audit found Alpine 3.24.2/current r2 kernel,
legacy sources under `/home/nik/bc250work/rocm-libraries-gfx1013`, branch
`gfx1013-cyanskillfish`, revision `b578bcd410d0641337ce0f34be055b88560f5c24`.
Other retained legacy work: `/home/nik/bc250-stack`, `pytorch-build-gfx1013`,
`research-handoff`, `bc250-rocm-lab`, `bc250-gfx1013-fix`. Do not copy old patches
blindly or upgrade/prune/reconfigure the host. Its private cgroup2 mount namespace
benchmark workaround was ephemeral, not persistent migration.

## Historical documents and how to avoid stale tasks

`bc250_port.md`, `bc250_release_20261013.md`, `bc250_development_handoff.md`,
`bc250_support_matrix.md`, `bc250_scalar_library_expansion.md`, the old roadmap,
training memory and performance docs are valuable milestone evidence. They may
say the frozen release is active, Lt GEMM unsupported, tensor contractions absent, sparse Lt
excluded, CK Tile untested, Torch unpatched or performance not started. **Those statements
are superseded by the current inventory for its specific scopes.** Do not delete snapshots
or mark every old box done; map completed work to evidence and select an actual remaining gap.

Before repeating any task: identify the component/operation, search this inventory,
patch subjects and validation registry, compare current code/profile, then state why
new work/test is necessary (new change, uncovered case, genuine failure or invalid old
reference). An old unchecked box, context compaction, stale PID, mutable tag name or
component version number mismatch is not a reason to restart the port.

## Unified workspace organization (after the memory audit)

Current local workspace: `/home/nik/Documents/Vibe/bc250stack`. The existing
TheRock checkout moved intact to `therock/`; `/home/nik/Documents/Vibe/therock`
is a compatibility symlink. The separate fresh PyTorch checkout moved intact to
`references/pytorch-upstream`, with its original `Vibe/pytorch` path preserved.
Canonical `torch/` and `triton/` aliases point to the qualified patched sources
inside TheRock, so there is no duplicated or silently upgraded port checkout.

Kernel source was found on .105 at
`/home/nik/bc250work/alpine-rocm-7.1/sources/linux-7.1-bc250-test` and copied
read-only into `kernel/source`. Five source/configuration anchors were verified.
The kernel binary SHA256 is
`792f904835f27574b680d2fd5cfd7e42b348fb3a3d9b39d8e8c87348873c3b0d`, identical
on both hosts. Configuration, legacy scripts and the existing bc250-linux
packaging repository are preserved separately. This is an already-patched source
snapshot, not a clean upstream checkout; generated build products/private signing
keys/initramfs archives are excluded. No remote installation or reboot occurred.

Workspace README/AGENTS and workspace-manifest.json explain component identities.
The workspace repository tracks coordination files; source changes stay in their
component repositories. Board support copies retain driver and CU-unlock history.
New session history may use the new workspace cwd; the earlier Codex thread-index
lookup by the old therock cwd remains useful for archived sessions.
