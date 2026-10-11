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
A hard GPU hang may prevent both SSH and a software watchdog from recovering
the machine; physical power cycling remains necessary in that case.

## Build

Build the image from the source root using
`podman build -f build_tools/bc250/Containerfile -t bc250-rocm-build:10.2 .`.
Run `build_tools/bc250/build_core.sh` inside it with the transferred source
and a separate build directory. Do not map GPU devices into build containers.
`CMAKE_BUILD_PARALLEL_LEVEL=4` also bounds nested builds; the compiler has a
single linker job and a single Flang compilation job.

The target belongs to `igpu-all`, `gfx101X-all`, and `gfx101X-igpu`, and is
directly selectable as `gfx1013`. The initial excluded projects match the
existing gfx101X profile. They remain audit targets, with hardware-compatible
generic paths considered separately.

## Validation status

The old 7.2 runtime passed a CPU-compared 64x64 FP32 torch GEMM on the
8-core host, with SDMA disabled. This establishes a reference only.
No ROCm 10.2 GPU component has been validated yet.
