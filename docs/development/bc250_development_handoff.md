# BC-250 development handoff: pause after the 2026-10-13 release

> **Historical snapshot notice:** this file preserves a previous milestone.
> Read [flow.md](../../flow.md) and the
> [current memory inventory](bc250_memory_inventory.md) before acting on its
> active-release claims or unchecked tasks. Newer qualified scopes supersede
> older unsupported/pending rows; retain the original evidence.

Resume manually around **2026-10-18**. Preserve all existing work. This is an
operational handoff, not an instruction to schedule autonomous work.

**Update:** the user explicitly resumed library development on 2026-10-13 before
that planned return. The frozen active/previous releases below remain unchanged.
The new developer candidate is documented in
[bc250_scalar_library_expansion.md](bc250_scalar_library_expansion.md), including
native scalar hipBLASLt/hipTensor/hipSPARSELt, generic operation coverage, source
commits, kernel-pack separation and the Torch pooling fix. Read that milestone
before assuming the developer sources still match the frozen baseline.

## Starting state

- Local source: `/home/nik/Documents/Vibe/therock`.
- Branch: `users/Naster17/rocm-10.2-gfx1013`.
- User-owned untracked `src/` must remain untouched.
- `.104` workspace: `/home/nik/bc250work/rocm-10.2-gfx1013`.
- Active core: `/opt/bc250-stack/releases/10.2-gfx1013-20261013-4b000d542`.
- Previous core: `/opt/bc250-stack/releases/10.2-gfx1013-20261012-a640c9296`.
- Immutable active image: `f460edb6e72ae084041a2a86129db6b48d90faf541b17d137fb98d319364e4d4`.
- Existing kernel: `7.1.5-bc250rocm-r2`, 8 cores/16 threads, 40 CU/20 WGP.
- `.105` retains its ROCm 7.2.4/Torch 2.9.1 reference installation.
- Daily-use commands: `bc250-torch` and `bc250-train`.

The frozen training consumers are under `current/python-consumers`. Keep the
original `training-env-stable` developer environment too. Review manifest and
runtime hashes live under `current/review-20261013`.

## Implementation provenance

- Core build source: TheRock `4b000d54287ee047adbe6febdcd6b2ea459dac85`.
- Libraries: `b9c8ad41e5b2c67c00bc28dff68b0c6f0df72ac5`, 13 ordered patches.
- Systems: `dbdb7b8d1c56316d873a9b90e548e5a72853496d`, three ordered patches.
- Compiler baseline: `7667f52dd687899256a3285ab0bce3c53bc73394`.
- Torch: `c779020750d7dc99a39ea8c104ef8df1fc9d0755`, editable build, no source patch.
- Triton: `ee65ff3b85f99d52e678b252a99c8425fa46e329`, four native backend patches.
- Performance milestone: root `24ea38d84`.
- Training/memory milestone: root `d744f0952`.

The root submodule pins and ordered patch directories are the reconstruction
mechanism; nested repositories also retain their local committed branches.
`fetch_sources.py` resets/reapplies sources and must not be used during the pause.
Before using it later, inspect and preserve all existing changes.

The gfx1013-specific CLR change disables `DEBUG_HIP_MEM_POOL_VMHEAP` only when
its flag is left at default. VMM APIs and async pools remain available. Do not
remove this default based on one passing rerun: earlier rocALUTION failures were
traced to live CSR analysis metadata being overwritten by the VM heap.

## First actions on return

1. Read [bc250_release_20261013.md](bc250_release_20261013.md) and collect any real
   workload failures the user encountered during everyday use.
1. Read Git status/logs and the installed release/launcher manifests before editing.
1. Keep development builds in the existing workspace, separate from the active
   runtime and frozen consumers. Update transferred files by checksum so unchanged
   timestamps do not cause unnecessary LLVM rebuilds.
1. Reuse the persistent 20 GiB ccache: 16 ordinary build jobs, rocALUTION 8,
   hipTensor 12, HIPDNN graph tests 6, LLVM linker 1.
1. Group related source changes into milestone commits. Do not commit after
   every small action. Do not push or publish without explicit instructions.

## Next useful work

- Investigate variable small-convolution behavior on the old reference stack
  without modifying its installed software or migration state.
- Repair/query trustworthy GPU clock telemetry before attributing small cross-board
  differences to ROCm alone.
- Expand useful performance coverage: larger training shapes, BF16-resident base
  models, inference, FFT/sparse/reductions and compilation overhead.
- Continue operation coverage and additional scalar backends according to
  [bc250_full_support_roadmap.md](bc250_full_support_roadmap.md).

The current 135M-model benchmark showed FP32 LoRA without activation checkpointing
fastest for batch 1/32 tokens. That finding is shape-specific; keep correctness
checks when changing precision, memory policy or kernels. Raw matched performance
results are committed under `docs/development/bc250_performance_results`.

Real model artifacts are retained under `.104`'s `training-validation/` and
`training-model/`. Performance fixtures/results are under
`/home/nik/bc250work/rocm-performance-20261013` on each host. The reference host
needed a private cgroup2 mount namespace to run its existing image after reboot;
its persistent mount configuration was left unchanged.

## Recovery rules

Retain `/home/nik/bc250work/bc250-rocm-lab`: GRUB uses its working kernel/initramfs.
Do not prune images, delete caches/checkpoints, expunge builds or remove previous
runtimes. SSH-only fault/reset experiments require independent recovery planning;
watchdog keepalives do not prove recovery from a CPU-responsive GPU hang.

PyTorch rebuilds/code generation must continue through the prescribed
`pip install -e . -v --no-build-isolation` workflow. Triton native/compiler source
checks must run `make` first. Do not rewrite the port around architecture identity
spoofing or enable unsupported matrix instructions.
