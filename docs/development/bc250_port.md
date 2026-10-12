# BC-250 ROCm 10.2 port

This is an experimental native `gfx1013` port. Registering the target or
building a component does not establish hardware support; execution results
are recorded separately.

## Source and host policy

- Base TheRock revision: `7a803357a6e0de10c780a683aa7492da39579938`.
- Local branch: `users/Naster17/rocm-10.2-gfx1013`.
- All source edits and commits happen locally. Transfer committed source
  snapshots to `nik@192.168.1.104` for builds and hardware tests.
- Use an Ubuntu 24.04 glibc Podman build image without installed ROCm.
- Preserve `.105` as the old-stack comparison host. Defer llama.cpp and
  multi-host distributed validation.
- PyTorch baseline: ROCm `release/2.13`, commit
  `c779020750d7dc99a39ea8c104ef8df1fc9d0755`.

## Hardware and recovery

On 2026-10-11, `.104` reports 8 cores/16 threads, 80 GPU SIMDs (40 CUs),
`gfx_target_version=100103`, 14.9 GiB system memory, and kernel
`7.1.5-bc250rocm-r2`. GPU memory shares the physical GDDR6 with CPU memory.

The existing GRUB default boots kernel and initramfs files from
`/home/nik/bc250work/bc250-rocm-lab`. Never remove these as old userspace
artifacts. Copies of those files, GRUB configuration, GRUB environment, and
the kernel command line are retained in the new workspace's `recovery/`.
Keep the current working kernel as the persistent default; use GRUB one-shot
selection for any replacement kernel until SSH and hardware gates pass.
The host exposes an SP5100 TCO hardware watchdog. Use
`build_tools/bc250/configure_watchdog.sh` as root to enable Alpine's existing
OpenRC service with 10-second keepalives and a 120-second timeout. Its prior
configuration is backed up alongside `/etc/conf.d/watchdog`. Verify watchdog
reset behavior separately before relying on it for unattended tests. A hard
GPU hang may still require physical power cycling if the chipset timer cannot
recover it.

## Build

Build the image from the source root using
`podman build -f build_tools/bc250/Containerfile -t bc250-rocm-build:10.2 .`.
Run `build_tools/bc250/build_core.sh` inside it with the transferred source
and a separate build directory. Do not map GPU devices into build containers.
Build the ccache variant with
`podman build -f build_tools/bc250/Containerfile.ccache -t bc250-rocm-build:10.2-ccache .`.
The host runner mounts a persistent NVMe cache and defaults to 16 compilation
jobs, four Flang compilation jobs, and one LLVM linker job. Override these with
`BC250_BUILD_JOBS` and `BC250_FLANG_JOBS`. The cache defaults to 20 GB, configurable
with `BC250_CCACHE_MAXSIZE`.

The bootstrap image uses ccache masquerade aliases for Ubuntu Clang while
retaining the real compiler under `/usr/lib/llvm-18`. The opt-in
`THEROCK_BOOTSTRAP_CCACHE_MASQUERADE` flag suppresses duplicate bootstrap
launchers, preserving existing LLVM Ninja command lines. Subsequent runtime
and HIP builds use standard CMake ccache launchers. Compiler fingerprints use
TheRock's compiler and shared-library hashing script.

The target belongs to `igpu-all`, `gfx101X-all`, and `gfx101X-igpu`, and is
directly selectable as `gfx1013`. CK's scalar DL paths are enabled with XDL and
WMMA disabled. hipBLASLt supplies an explicit host/API-only profile while GEMM
uses native rocBLAS. hipSPARSELt is omitted when its device-target filter is
empty. Keep MIOpen's CK integration disabled during initial validation.

`build_math.sh` extends the core build with the native math libraries, MIOpen,
and scalar CK. The topology also enables the profiler SDK and its legacy ROCTX
library, which PyTorch's marker bindings require even with Kineto disabled.
AMD SMI uses a disposable writable source copy for its generated headers.
rocFFT's ahead-of-time cache compiler uses 16 workers, independently configurable
with `BC250_FFT_JOBS`. Its cache in the build tree preserves completed kernels
when resuming; it uses HIPRTC directly rather than ccache. Its producer pool
is capped by the CPU affinity count, so larger limits do not create more than
16 workers on this host.

`continue_torch_on_host.sh` waits for the math build, runs CPU-reference GEMM,
Cholesky, prime-length FFT and RNG checks, and verifies the host-only hipBLASLt
contract. It then builds the pinned PyTorch source with
`pip install -e . -v --no-build-isolation` in a separate tool environment.
Its successful container is saved as `bc250-pytorch-runtime:2.13-gfx1013`.
Torch defaults to 12 compile jobs and one linker job; use `BC250_TORCH_JOBS`
to adjust the heavier template build independently from ROCm's 16 jobs.
The runtime image requires the persistent ROCm and Torch working-directory
mounts; the editable installation is not a standalone wheel.
The math profile includes libhipcxx, required by ROCm 8+ consumers. Torch uses
the controlled CMake 3.31.12 executable because some bundled projects retain
older minimum-version declarations. A CMake toolchain hook runs upstream HIP
generation inside the pip build before CMake discovers generated sources and
sets native GPU targets before importing HIP packages in the GPU-less builder.

Torch execution gates include native target identity, markers, FP16/BF16
indexed accumulation and embedding gradients, four GEMM precisions, sparse
CSR multiplication, solvers, FFT, convolution gradients, mathematical SDPA,
and a CPU-compared 30-step training run. Flash attention, memory-efficient
attention, CK matrix backends, and distributed execution are disabled in the
initial profile.

## Validation status

The old 7.2 runtime passed a CPU-compared 64x64 FP32 torch GEMM on the
8-core host, with SDMA disabled. This establishes a reference only.
The fresh ROCm 10.2 core passed the native HIP hardware gate on `.104` with
SDMA disabled: 256 allocation/dispatch cycles, LDS reduction, global atomics,
and CPU-checked copies from 4 KiB through 16 MiB. The reported architecture is
`gfx1013:xnack-`, with wave32. PyTorch is still building; its execution gates
have not yet run.
The packed FP16/BF16 atomic gate also passed with unmodified HIP headers,
including contended packed updates, neighboring scalar lanes, untouched
sentinels, and returned old values. LLVM's target-aware `__has_builtin` selects
the existing CAS fallback on `gfx1013`; no atomic header patch is required.
rocRAND passed six dispatch/configuration tests and all 44 basic tests on the
new runtime. rocBLAS built 570 native fallback kernels and passed the upstream
CPU-compared 1023×1024×1025 SGEMM example. The host-only hipBLASLt descriptor,
heuristic, extension enumeration and unsupported-execution gate passed.
On 2026-10-12, the complete math build and the CPU-reference math gate passed:
rocBLAS/hipBLAS GEMM, rocSOLVER Cholesky, hipFFT prime-length DFT, and rocRAND
reproducibility/distribution checks. This is initial operation coverage, not
full upstream test-suite coverage.

`validate_ml_on_host.sh` passed CK's upstream scalar DL FP32, FP16 and signed
INT8 GEMM examples with CPU verification. MIOpen's driver passed FP32, FP16
and BF16 convolution for groups 1 and 2, including forward, input gradients,
and weight gradients. Set `MIOPEN_DRIVER_USE_GPU_REFERENCE=0` explicitly:
the pinned driver's default reference is the GPU. Its initial tests use
fresh per-container caches; no gfx1013 tuning database is supplied yet.

Triton baseline `66b3f9ff575eb8db6adec658bbf0badd9a8d2e25` builds with the
upstream pinned LLVM 23 package. The patches under `patches/bc250-triton/`
replace unsupported gfx1013 packed FP16/BF16/INT8 dot lowering with widened
scalar operations, and BF16 elementwise multiplication with the existing
FP32 conversion fallback. Other targets retain their packed paths.
`validate_triton_lowering.sh` rebuilds through `make` and checks MLIR and LLVM
instruction selection against native gfx1013 and gfx1100. Preserve generated
fixture functions through instruction selection so LLVM cannot discard them.
The first six MLIR dot checks passed; the extended BF16 multiplication check
reproduced an LLVM instruction-selection failure before its fix. The updated
native rebuild and all eight extended MLIR/LLVM/assembly checks passed on
2026-10-12. Triton GPU execution and Inductor execution remain pending until
Torch finishes.

`validate_primitives_on_host.sh` passed 227 upstream GPU tests: eight
rocPRIM reductions, ten rocPRIM scans, hipCUB's device version test, and
all 208 hipCUB block-reduction cases. Kernel logs gained no new messages
during the ML and primitive hardware gates.

`validate_compilation_on_host.sh` requires successful eager Torch validation
and a successful Triton build. It runs the compiler gate, the three upstream
native gfx1013 dot tests, fused elementwise kernels, and a CPU-compared
layernorm/softmax/matmul forward-and-backward graph through Inductor.
`finish_validation_on_host.sh` waits for successful eager Torch, ML,
primitive and Triton build gates before continuing compiler validation.

The BC-250 math build disables AMD SMI's EPYC-only ESMI library. Its default
configuration downloads sources into the source tree, which is intentionally
read-only; the BC-250 CPU does not expose the EPYC management interface.

The passive DRM probe on `.104` reports one graphics engine, one compute engine,
and two SDMA engines. VCN decode, VCN encode, JPEG, UVD, and VCE counts return
`-EINVAL` with zero available rings. The current kernel therefore exposes no
hardware media decode backend; firmware file presence alone does not establish
one.
