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

## Validation status

The old 7.2 runtime passed a CPU-compared 64x64 FP32 torch GEMM on the
8-core host, with SDMA disabled. This establishes a reference only.
No ROCm 10.2 GPU component has been validated yet.

The passive DRM probe on `.104` reports one graphics engine, one compute engine,
and two SDMA engines. VCN decode, VCN encode, JPEG, UVD, and VCE counts return
`-EINVAL` with zero available rings. The current kernel therefore exposes no
hardware media decode backend; firmware file presence alone does not establish
one.
