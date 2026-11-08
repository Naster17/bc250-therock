# BC-250 native gfx1013 ROCm roadmap

> **Historical snapshot notice:** this file preserves a previous milestone.
> Read [flow.md](../../flow.md) and the
> [current memory inventory](bc250_memory_inventory.md) before acting on its
> active-release claims or unchecked tasks. Newer qualified scopes supersede
> older unsupported/pending rows; retain the original evidence. The current
> priority checklist is [BC250_REMAINING_PLAN.md](../../../BC250_REMAINING_PLAN.md).

Last reviewed: **2026-10-13**. This is a manual progress snapshot, not a live
dashboard. Host logs and committed source revisions are the authoritative
evidence when the state changes.

## Goal and scope

Build a reliable, efficient, native ROCm stack for the AMD BC-250: compiler,
runtime, hardware-feasible libraries, PyTorch, debugging/profiling tools,
packaging, recovery and performance tuning.

The target is `gfx1013`, without pretending to be another GPU. A library that
currently requires matrix instructions may still support the mathematical
operation through a correctly implemented scalar/SIMD backend.

Scope does not require unsupported physical instructions to exist. Hardware
acceleration, software fallback and unavailable features must be reported
accurately. CPU fallback must be identified explicitly.

- [x] Establish native ROCm + Torch execution on `.104`.
- [x] Extend beyond the initial Torch dependencies to additional ROCm components.
- [ ] Complete all hardware-feasible implementation work described below.
- [ ] Complete broad numerical and stability qualification.
- [ ] Tune performance with reproducible measurements.
- [ ] Deliver a reproducible, recoverable expanded release.

## How to interpret the checkboxes

- `[x]` means the **specific action or test stated on that line** is complete.
- `[ ]` means work remains, validation is pending, or applicability still needs
  investigation.
- A completed registration patch is not a completed library port.
- A completed build is not proof of GPU execution or numerical correctness.
- A passing operation does not establish all shapes, precisions, layouts,
  concurrency modes or algorithms in the library.
- A short passing run does not establish long-term reliability.
- Do not compute a completion percentage from checkbox counts: the remaining
  tasks differ greatly in effort and some depend on hardware investigations.

The operation-level evidence is summarized in
[bc250_support_matrix.md](bc250_support_matrix.md). Build, deployment and recovery
details are in [bc250_port.md](bc250_port.md).

## 1. Current milestone: qualify and release the expanded stack

This is the work immediately ahead of us. The installed working release and the
developer distribution are separate.

- [x] Build the initial usable ROCm/Torch/Triton stack.
- [x] Install the initial versioned release with independent runtime copies.
- [x] Build the expanded profile with OpenCL, RPP audio, rocALUTION, hipFile,
  hipTensor and hipDNN providers.
- [x] Fix the HIP spectrogram output-row stride exposed by the CPU reference.
- [x] Execute RPP host/HIP audio tests successfully after the stride fix.
- [x] Execute 154 hipTensor generic CPU-reference cases successfully.
- [x] Execute five HIP-provider runtime-scalar regressions successfully.
- [x] Execute nine CPU-reference HIP-provider graph cases successfully.
- [x] Execute nine CPU-reference MIOpen-provider graph cases successfully.
- [x] Trace the native HIP workload and validate the profiler records.
- [x] Find the VM-heap corruption responsible for intermittent rocALUTION failures.
- [x] Demonstrate that the established pool backend passes the CG diagnostics.
- [x] Add a gfx1013 default that disables the experimental VM heap while retaining
  async pools and virtual-memory APIs.
- [x] Preserve an explicit VM-heap opt-in for future controlled investigations.
- [x] Rebuild with the compiled allocator default.
- [x] Pass all 40 CPU/GPU CG diagnostic cases with that compiled default and no
  allocator environment override.
- [x] Repeat the ordinary GPU CG gates across fresh processes and allocator reuse.
- [x] Finish the entire expanded validation sequence after the allocator change.
- [x] Queue automatic candidate eager Torch, compiled Torch/Triton and stability
  checks after the expanded library gates pass.
- [x] Pass candidate eager Torch validation against the new ROCm distribution.
- [x] Pass candidate compiled Torch/Triton validation.
- [x] Pass the queued 10-minute CPU-checked candidate stability run.
- [x] Review kernel messages and resource growth across those gates.
- [x] Reconcile source snapshot metadata, actual transferred files and helper
  revisions before creating a release manifest.
- [x] Extend deployment requirements to include the expanded-component evidence.
- [x] Copy the candidate into a new versioned release and validate that copy.
- [x] Switch the active userspace symlink only after release-copy gates pass.
- [x] Retain and verify rollback to the previous working release.
- [x] Verify `bc250-torch` against the newly activated release as `nik`.

## 2. Hardware inventory and applicability

The current host reports 8 CPU cores/16 threads, 80 GPU SIMDs/40 CUs, wave32 and
64 KiB LDS. HIP reports 20 multiprocessors on this RDNA arrangement; do not confuse
that count with the physical 40-CU inventory. CPU and GPU share physical memory.

- [x] Confirm native KFD target encoding `100103` and HIP `gfx1013:xnack-`.
- [x] Confirm the unlocked CPU configuration and current kernel.
- [x] Record PCI/device identity, topology, queue counts and memory characteristics.
- [x] Record two reported SDMA engines, without treating that as working user SDMA.
- [x] Record no reported XGMI SDMA engines or P2P links on the current host.
- [x] Probe native scalar FP32/FP64 and packed-half arithmetic support.
- [x] Keep MFMA, WMMA and unsupported packed-dot paths disabled for this target.
- [x] Record unavailable media-ring queries in the installed kernel.
- [ ] Maintain a per-feature capability inventory that separates silicon, firmware,
  kernel, runtime, compiler and library limitations.
- [ ] Investigate unknown capabilities individually instead of copying another
  architecture's capabilities wholesale.
- [ ] Verify practical memory limits under simultaneous CPU/GPU workloads.
- [ ] Measure clock behavior, temperatures, throttling and sustained power limits.
- [ ] Investigate whether unavailable media engines can be exposed safely by a
  supported firmware/kernel configuration.
- [ ] Record genuinely unavailable features with evidence and an explicit boundary.

## 3. Kernel, firmware and SSH recovery

The current working kernel is `7.1.5-bc250rocm-r2`. Boot files remain under
`/home/nik/bc250work/bc250-rocm-lab`; that directory is not obsolete userspace.

- [x] Preserve the working GRUB default and kernel/initramfs files.
- [x] Save boot/recovery configuration and copies in the new workspace.
- [x] Enable existing SP5100 watchdog keepalives and confirm active state.
- [ ] Verify actual watchdog reset behavior independently.
- [ ] Establish a recovery procedure for a GPU hang with a still-responsive CPU.
- [ ] Qualify SSH reconnection after controlled reboot and recovery events.
- [ ] Use one-shot boot selection for replacement kernels until qualification passes.
- [ ] Keep a known-good persistent boot entry through kernel/firmware experiments.
- [ ] Audit gfx1013 KFD/DRM topology, trap handling and firmware selection.
- [ ] Investigate SDMA through guarded correctness and stress experiments.
- [ ] Qualify reset/recovery before deliberately faulting or halting GPU queues.
- [ ] Record kernel/firmware combinations, regressions and required patches.
- [ ] Package a reproducible kernel configuration and installation procedure if a
  new kernel becomes part of the stable stack.

## 4. Source control and reproducible builds

- [x] Work on `users/Naster17/rocm-10.2-gfx1013`.
- [x] Make source edits and commits locally before transferring them to `.104`.
- [x] Preserve `.105` as the old-stack comparison host.
- [x] Use pinned upstream source baselines and ordered component patch series.
- [x] Keep nested vendor pins reproducible rather than relying on unpublished
  nested-submodule commits.
- [x] Build ROCm in a glibc container on the Alpine host.
- [x] Keep GPU devices out of ordinary build containers.
- [x] Configure persistent ccache with a 20 GB limit.
- [x] Use 16 ordinary compilation jobs and a single heavy LLVM linker job.
- [x] Use separate limits for memory-heavy components: rocALUTION 8, hipTensor 12,
  hipDNN graph tests 6 and Flang compilation 4.
- [x] Reduce graph-test parallelism after observing RAM exhaustion and swap thrashing.
- [x] Replay the library patch series against the pinned baseline and compare the
  resulting modified files with the committed tree.
- [ ] Repeat clean patch-application checks when patch series change.
- [ ] Export complete source manifests for each qualified candidate.
- [x] Preserve unchanged-file timestamps when updating snapshots to avoid needless
  configure/code-generation rebuilds.
- [ ] Verify each candidate from a clean source reconstruction, not only an
  accumulated development directory.
- [ ] Record compiler versions, build flags, images, source revisions and hashes.
- [ ] Add suitable regression coverage for architecture defaults and exclusions.
- [ ] Keep patches limited to gfx1013 or separately justified general bug fixes.
- [ ] Check supported comparison architectures before considering upstream submission.
- [ ] Prepare upstreamable patches and documentation when the implementation is ready.
- [ ] Publish or push changes only when explicitly requested.

Useful baselines: TheRock `7a803357a6e0de10c780a683aa7492da39579938`,
rocm-libraries `3d8f334b9a72ad99a0ecd87552613970433e0500`, rocm-systems
`d5c528019d0fcfbbc32d96b53ac2c61df6d0e67d`, LLVM
`7667f52dd687899256a3285ab0bce3c53bc73394`, Torch
`c779020750d7dc99a39ea8c104ef8df1fc9d0755`, and Triton
`66b3f9ff575eb8db6adec658bbf0badd9a8d2e25`.

## 5. LLVM, device libraries, COMGR, HIPRTC and HIPIFY

- [x] Build HIPIFY tools.
- [x] Translate CUDA runtime APIs and a kernel with hipify-perl, compile the output
  for gfx1013 and verify execution against exact CPU results.
- [ ] Qualify hipify-clang translation with the required CUDA SDK/parser inputs.
- [x] Compile native gfx1013 HIP code and execute it on the board.
- [x] Build and use native device-library bitcode.
- [x] Use COMGR/HIPRTC through actual runtime compilation consumers.
- [x] Probe instruction capabilities instead of assuming all RDNA-family features.
- [x] Verify relevant native lowering and assembly through the Triton compiler gate.
- [ ] Audit target registration and feature assumptions throughout the pinned compiler.
- [ ] Expand intrinsic tests for integer, floating-point, subgroup and memory operations.
- [ ] Check atomics and address-space conversions across supported types and widths.
- [ ] Test debug information, code-object compatibility and symbol resolution.
- [ ] Qualify relocatable device code, device linking and representative library users.
- [ ] Test optimization-sensitive cases and avoid unsupported-instruction emission.
- [ ] Audit HIPRTC architecture selection, diagnostics and compilation caches.
- [ ] Qualify additional hardware-feasible ISA features with execution evidence.
- [ ] Document unsupported instructions and tested software alternatives.

## 6. ROCr, HIP and memory management

- [x] Execute native allocation/copy/stream/LDS/integer-atomic smoke tests.
- [x] Verify packed FP16/BF16 CAS atomics, neighboring lanes and returned old values.
- [x] Use the existing target-aware HIP atomic fallback without an unnecessary patch.
- [x] Keep SDMA disabled in the validated profile.
- [x] Identify live CSR metadata corruption in the VM-heap allocator path.
- [x] Use the established pool backend by default on gfx1013.
- [x] Retain virtual-memory APIs and asynchronous pool allocation availability.
- [ ] Verify explicit/default pool attributes, reuse, release thresholds and trimming.
- [x] Verify pinned copies, four streams, events and CPU results in the bounded Torch gate.
- [ ] Expand allocation/free ordering and explicit pool coverage across streams/events.
- [ ] Test pool import/export and IPC where supported.
- [ ] Qualify virtual address reserve/map/unmap/access/import/export operations.
- [ ] Investigate and fix the VM-heap corruption before restoring it as the default.
- [ ] Build a focused regression for live pool allocations across module loading and
  subsequent kernel launches.
- [ ] Qualify graph allocation/free nodes and repeated graph execution.
- [ ] Test managed/pinned memory, asynchronous transfers and host/device visibility.
- [x] Verify 10 GiB GPU allocations, fragmentation/reuse and complete pressure-buffer release.
- [ ] Test near-capacity allocation failures and recovery.
- [x] Verify two simultaneous independent Torch GPU processes.
- [ ] Broaden multi-process and multi-thread runtime use and IPC.
- [ ] Qualify images/textures, events, callbacks and supported interoperability paths.
- [ ] Run broader upstream HIP/ROCr tests with meaningful pass/skip accounting.

## 7. rocBLAS and hipBLAS

- [x] Add native target support to the relevant Tensile architecture paths.
- [x] Use actual compiler capabilities when selecting scalar fallback kernels.
- [x] Build 570 native fallback kernels in the initial profile.
- [x] Pass the upstream CPU-compared 1023×1024×1025 SGEMM example.
- [x] Verify representative rocBLAS/hipBLAS GEMM and Torch FP32/FP64/FP16/BF16 GEMM.
- [ ] Expand Level 1/2/3 BLAS operation coverage.
- [ ] Verify complex arithmetic, transposes, conjugation and non-contiguous layouts.
- [ ] Verify batched/strided-batched operations and irregular leading dimensions.
- [ ] Test scalar pointer modes, small dimensions, empty dimensions and numerical extremes.
- [ ] Investigate additional mixed-precision and integer paths that can be implemented.
- [ ] Produce native tuning logic for representative BC-250 workloads.
- [ ] Optimize scalar/SIMD GEMM without introducing unsupported dot/matrix instructions.
- [ ] Measure dispatch overhead and small-operation performance.
- [ ] Verify deterministic and concurrent operation modes where provided.

## 8. hipBLASLt and hipSPARSELt

Native scalar device paths are now qualified in the development candidate; the
frozen daily-use release still has its original host-only/excluded profiles. See
[the scalar expansion evidence](bc250_scalar_library_expansion.md) for exact limits.

- [x] Provide an explicit host/API-only hipBLASLt profile.
- [x] Test descriptor APIs, heuristic enumeration and honest unsupported execution.
- [x] Exclude unsupported hipSPARSELt device builds from the current profile.
- [x] Design a native scalar/SIMD hipBLASLt backend with correct layouts and semantics.
- [x] Compile/load the scalar kernels for native gfx1013 without matrix instructions.
- [x] Supply a real scalar algorithm, valid heuristics and zero-workspace handling.
- [x] Implement and verify bias, ReLU and GELU epilogues.
- [ ] Extend auxiliary/backward and other applicable epilogues.
- [x] Qualify FP32/FP16/BF16 input combinations, row/column layouts, transposes,
  padded/broadcast batches and zero-scalar corner cases: 403 GPU cases, including bias/ReLU/GELU epilogues.
- [ ] Extend supported precision/layout/epilogue combinations beyond that scalar subset.
- [ ] Integrate and qualify the hipBLASLt hipDNN provider only after device GEMM works.
- [x] Qualify hipSPARSELt strip pruning/checking/compression independently of matmul.
- [x] Implement a documented scalar compression format with faithful metadata decoding.
- [x] Test sparse matmul against dense CPU references: 162 strip/TILE dtype/layout/transpose/batch
  combinations, decoded metadata, guards and zero-scalar/Search regressions.
- [x] Add genuine maximum-L1 FP16/BF16 TILE pruning with independent mask references.
- [ ] Extend remaining sparse API/epilogue/type combinations.
- [ ] Tune both libraries after correctness is established.
- [x] Keep native matrix-instruction capability reports false on this hardware.

## 9. rocSOLVER and hipSOLVER

- [x] Pass initial CPU-reference Cholesky and Torch solver checks.
- [x] Run the user's Torch solve/LU checks successfully.
- [ ] Expand LU, QR, Cholesky and related factorization coverage.
- [ ] Qualify eigenvalue/SVD and other applicable routines.
- [ ] Test batching, workspaces, complex types and irregular dimensions.
- [ ] Test singular, ill-conditioned and failure-reporting cases.
- [ ] Verify numerical residuals and reconstruction, not just finite outputs.
- [ ] Tune representative solver workloads and measure memory requirements.

## 10. rocSPARSE, hipSPARSE and rocALUTION

- [x] Pass Torch sparse CSR multiplication checks.
- [x] Build the sparse libraries and rocALUTION in the expanded profile.
- [x] Diagnose CG with CPU/GPU SpMV, norms and solution-error comparisons.
- [x] Compare transferred CSR arrays and initialized vectors with exact host inputs.
- [x] Execute direct legacy CSR tests with/without/repeated analysis and unit/non-unit
  alpha, signed/constant inputs, beta values and allocation guards.
- [x] Trace the live metadata overwrite rather than relaxing numerical tolerances.
- [x] Pass 40 CPU/GPU CG cases after the allocator-default fix.
- [x] Pass 200 ordinary CG diagnostic cases across five fresh processes on the
  installed expanded release.
- [ ] Repeat upstream CG tests with larger and more varied systems.
- [ ] Qualify more rocALUTION solvers and preconditioners.
- [ ] Expand sparse-format coverage: CSR/COO and other hardware-feasible formats.
- [ ] Qualify SpMV/SpMM, conversions, sorting and applicable sparse solves.
- [ ] Test index widths, empty rows, long rows, transposes and complex types.
- [ ] Test concurrent execution and analysis-cache reuse.
- [ ] Benchmark realistic sparse matrices and document algorithm choices.
- [ ] Treat allocator-related historical failures as regressions requiring coverage.

## 11. rocRAND and hipRAND

- [x] Add native architecture detection and appropriate generic parameters.
- [x] Pass six rocRAND dispatch/configuration tests and 44 basic tests.
- [x] Verify reproducibility/distribution checks and Torch RNG smoke tests.
- [ ] Cover all relevant generators, distributions and supported precisions.
- [ ] Verify seeds, offsets, subsequences and independent concurrent streams.
- [ ] Test very small/large requests and alignment/tail handling.
- [ ] Verify device API consumers as well as host APIs.
- [ ] Tune generation throughput without weakening statistical checks.

## 12. rocFFT and hipFFT

- [x] Build native FFT libraries and the ahead-of-time cache.
- [x] Pass a prime-length transform against a CPU DFT reference.
- [x] Pass Torch FFT checks and the user's two-dimensional roundtrip test.
- [ ] Expand rank, size, real/complex, direction and precision coverage.
- [ ] Test irregular strides, distances, batching and in-place/out-of-place operation.
- [ ] Test cache hits/misses, runtime compilation and cold-start behavior.
- [ ] Verify normalization and difficult numerical inputs.
- [ ] Tune common signal-processing and ML transforms for the board.

## 13. rocPRIM, hipCUB, rocThrust, libhipcxx and hipCCL

hipCCL here is the primitive/container-library packaging project, not RCCL.

- [x] Pass eight rocPRIM reductions, ten scans, a hipCUB version test and 208
  hipCUB block-reduction cases in the initial gate.
- [x] Add native host/device architecture classification to rocPRIM.
- [x] Apply corresponding detection changes to the two bundled hipCCL copies.
- [x] Verify native architecture dispatch in all three copies on the GPU.
- [x] Build libhipcxx and its initial consumers.
- [ ] Qualify the remaining reductions, scans, sorting, selection and other primitives.
- [ ] Verify subgroup widths, shared-memory limits and architecture tuning choices.
- [ ] Test hipCUB device/warp/block algorithms beyond initial reductions.
- [ ] Exercise rocThrust algorithms, iterators, allocators and stream policies.
- [ ] Test libhipcxx device containers, synchronization and applicable utilities.
- [ ] Integrate/qualify hipCCL packaging without conflicting duplicate headers or ABI.
- [ ] Keep native architecture identification separate from unmeasured tuning claims.

## 14. composable_kernel and CK Tile

- [x] Enable scalar DL paths for gfx1013 with XDL/WMMA disabled.
- [x] Pass upstream FP32, FP16 and signed INT8 scalar GEMM examples against CPU results.
- [x] Use existing generic CK support for hipTensor operations.
- [x] Qualify selected CK Tile reduction/permutation/transpose/elementwise examples:
  12 CPU-reference executions, including odd/padded strides and tails.
- [x] Qualify FP16 CK Tile 3D max pooling and output indices against CPU results.
- [x] Qualify six FP16/BF16/FP32 RMSNorm runs with odd/padded widths.
- [ ] Qualify top-k and additional normalization/pooling/other operations.
- [ ] Extend scalar DL GEMM coverage to layouts, tails, batching and supported types.
- [ ] Audit architecture checks and wave32 assumptions throughout generic kernels.
- [ ] Integrate useful generic CK paths into consumers after independent validation.
- [ ] Tune vectorization, LDS use, occupancy and memory access for BC-250.
- [ ] Keep instruction-dependent XDL/WMMA paths excluded.

## 15. MIOpen

- [x] Add native architecture recognition and relevant direct-solver guards.
- [x] Pass CPU-reference FP32/FP16/BF16 convolution for groups 1 and 2.
- [x] Verify forward, input gradients and weight gradients.
- [x] Keep unsupported matrix/CK-dependent solver paths out of the current profile.
- [ ] Expand convolution ranks, dilation, stride, padding, groups and layouts.
- [ ] Qualify normalization, activation, pooling and other applicable operators.
- [ ] Qualify supported fusion combinations individually.
- [ ] Supply a measured gfx1013 tuning database and solver-selection policy.
- [ ] Test deterministic behavior and cache/thread/process concurrency.
- [ ] Verify CPU references explicitly; the driver's default GPU reference is insufficient.
- [ ] Avoid enabling existing RDNA fusion exclusions until an appropriate backend exists.

## 16. hipTensor

- [x] Add native device registration and scalar FP64 capability recognition.
- [x] Introduce an opt-in generic-only profile without contraction device instances.
- [x] Fix generic test capability checks so actual gfx1013 tests execute.
- [x] Generate headers outside the read-only source tree.
- [x] Request only the CK packages required by the generic profile.
- [x] Pass reduction, permutation, binary and trinary operation gates.
- [x] Pass extended FP16/BF16/FP32/FP64 reductions with sum/product/min/max and
  nonzero beta: 154 total executed cases across the current gates.
- [x] Add 23 executed rank-three/four generic reduction/permutation/binary/trinary cases.
- [ ] Extend remaining ranks, modes, layouts, strides, operators and dtype combinations.
- [ ] Qualify memory/workspace handling and unsupported-operation reporting.
- [x] Implement a scalar contraction backend for the supported real tensor subset.
- [x] Verify scalar contraction values, plan ownership, fixed algorithm and zero
  workspace: 76 numerical/descriptor cases, including FP16/BF16→FP32 and FP32→FP64.
- [ ] Extend contraction types, unary operations, mode/rank coverage and plan caching.
- [ ] Tune generic operations and any new contraction implementation.

## 17. hipDNN and providers

- [x] Build hipDNN, integration tests, samples and the two applicable providers.
- [x] Enable generic HIP MLOPS while disabling instruction-specific engines.
- [x] Enable the legacy C++ graph fixtures needed by the chosen validation gate.
- [x] Pass five runtime-scalar tests in the HIP kernel provider.
- [x] Pass nine CPU-reference normalization/resampling graph cases.
- [x] Pass nine CPU-reference MIOpen convolution graph cases across three precisions
  and forward/data-gradient/weight-gradient directions.
- [x] Finish requalification after the compiled allocator default changes.
- [ ] Expand forward/backward normalization, training and layout coverage.
- [x] Add 14 CPU-reference LayerNorm/RMSNorm backward and MIOpen batchnorm training/backward
  graph cases, including NCHW/NHWC and FP32/FP16/mixed normalization.
- [ ] Expand supported provider graphs and bundle-based integration coverage.
- [ ] Verify unsupported graphs, plans, caching, dynamic loading and serialization.
- [ ] Qualify provider interaction, discovery and installed-prefix relocation.
- [ ] Add hipBLASLt provider coverage once its native device backend exists.
- [ ] Evaluate generic attention/MoE routes independently of matrix-only engines.

## 18. PyTorch, Triton and Inductor

- [x] Build the pinned Torch source through the required editable pip workflow.
- [x] Build patched native Triton and install a wheel with backend metadata.
- [x] Implement scalar FP16/BF16/INT8 dot fallbacks without unsupported instructions.
- [x] Fix BF16 arithmetic lowering and the RDNA1 buffer-descriptor flags.
- [x] Pass native lowering/assembly checks and relevant GPU tests.
- [x] Pass 56 eager Torch checks on the installed initial release.
- [x] Pass compiled elementwise and forward/backward graph checks; verify masked
  global/buffer memory, offset views, sentinels and BF16 rounding cases.
- [x] Pass the user's Torch matrix twice after fixing the launcher.
- [x] Pass the initial 120-second installed-runtime soak: 8,074 checked cycles.
- [x] Pass the queued eager/compiled gates on the new ROCm candidate.
- [x] Pass the queued 10-minute candidate soak.
- [x] Fine-tune the pretrained 135M model with FP32, finite gradients and decreasing loss.
- [x] Train rank-eight PEFT LoRA with FP16/BF16 autocast and activation checkpointing.
- [x] Resume model/adapter, optimizer, scheduler, scaler and RNG state in fresh containers.
- [x] Compare full-model CPU/GPU logits, loss and four representative parameter gradients.
- [ ] Run longer training and mixed-operation stress tests.
- [ ] Expand upstream Torch/Triton coverage for shapes, layouts and supported dtypes.
- [x] Qualify bounded 10 GiB GPU + 768 MiB CPU pressure and fragmented allocation reuse.
- [x] Qualify four streams/events, eight basic HIP graph replays and two concurrent processes.
- [ ] Expand memory pressure, graph allocation/free nodes, OOM and process/IPC coverage.
- [ ] Test eager/compiled agreement and convergence on representative real models.
- [ ] Tune scalar dot, memory access, reductions and autotuning for the board.
- [ ] Evaluate useful storage/conversion support for additional low-precision types
  without claiming unavailable hardware matrix acceleration.
- [ ] Rebuild Torch when source/header/ABI changes require it, using the prescribed
  pip workflow and meaningful regression gates.

## 19. OpenMP, Fortran, hipThreads and OpenCL

- [x] Build compiler/OpenMP support and execute mandatory native GPU offload.
- [x] Verify odd-length SAXPY, reduction and task dependencies against CPU values.
- [x] Build a private OpenCL ICD/runtime configuration.
- [x] Verify CL1.2 kernels, LDS reduction, atomics, event timestamps and RGBA8 images.
- [ ] Qualify Fortran GPU offload examples with Flang.
- [ ] Qualify more OpenMP mapping, synchronization and runtime behavior.
- [ ] Build/test hipThreads where the native device bitcode permits it.
- [ ] Test applicable OpenCL 2.0 features individually; advertising 2.0 is not full coverage.
- [ ] Test OpenCL/HIP concurrency and applicable interoperability.
- [ ] Qualify installed ICD discovery and packaging without modifying unrelated ICDs.

## 20. RPP image, audio and volume processing

- [x] Build native RPP image processing.
- [x] Verify brightness and flips in NHWC/NCHW with CPU comparisons and padding guards.
- [x] Integrate bundled static FFTS, its package metadata and license artifacts.
- [x] Fix Clang/SSE3/static-link integration issues in the bundled FFTS path.
- [x] Enable RPP audio and fix the discovered HIP spectrogram row-stride error.
- [x] Verify host and HIP pre-emphasis for three border modes.
- [x] Verify host and HIP spectrogram in NTF/NFT against independent CPU DFT values.
- [x] Fix and qualify 1D/2D decibels, non-unit/automatic references, zero inputs and padding.
- [x] Add 16 independent CPU-reference blend/contrast/exposure/offset-crop cases
  across NHWC/NCHW source/destination layouts, padded rows and allocation guards.
- [ ] Expand image augmentations, ROI handling, types and layouts.
- [ ] Qualify normalization/statistical and applicable volumetric operations.
- [ ] Expand audio transforms, window/FFT sizes, padding and batch lengths.
- [ ] Qualify resampling, decibels, downmixing and mel/filter operations as applicable.
- [ ] Extend guards and numerical cases for vectorized tails.
- [ ] Tune throughput and fusion after correctness is established.

## 21. Storage: hipFile

- [x] Build hipFile and its examples.
- [x] Pass five verified compatibility-backend roundtrips, including asynchronous,
  nonblocking-stream, registered multistream and batch cases.
- [ ] Test wider file sizes, offsets, alignment, partial I/O and failure handling.
- [ ] Test concurrent access, stream ordering and registration lifetimes.
- [ ] Investigate whether a direct-storage backend is supported by the board,
  driver, filesystem and NVMe configuration.
- [ ] Qualify any fast path independently; do not label compatibility I/O as direct DMA.

## 22. Communication: RCCL, rocSHMEM and multi-host use

- [x] Preserve `.105` during development on `.104`.
- [ ] Build RCCL with native gfx1013 device compilation.
- [ ] Qualify single-device APIs/readiness without claiming full collective coverage.
- [ ] Qualify local multi-process behavior where useful and supported.
- [ ] Build rocSHMEM with generic wave32 paths and suitable initial backend settings.
- [ ] Override hardware-specific GDA/SDMA assumptions when they are not qualified.
- [ ] Integrate the supported MPI setup and test bootstrap/runtime loading.
- [ ] Qualify software/network collectives, correctness, ordering and failure behavior.
- [ ] Qualify distributed Torch operations only after the local stack is stable.
- [ ] Use `.105` for reproducible old/new comparisons before migrating it.
- [ ] Migrate `.105` only after the `.104` stack and rollback procedures are stable.
- [ ] Qualify two-host RCCL/rocSHMEM/Torch behavior after both hosts are ready.
- [ ] Measure actual networking and shared-memory performance.
- [ ] Keep XGMI/P2P/GDA acceleration claims gated by actual topology and execution.

## 23. Debugging, tracing and counters

- [x] Build the profiler SDK and required marker libraries.
- [x] Verify native tracing: 3,628 HIP API records, 263 dispatches and 814 copies.
- [x] Validate timestamps, dispatch names/counts and correlations with a CPU-verified workload.
- [x] Patch native gfx1013 ELF/architecture registration in ROCdbgapi.
- [ ] Qualify the debugger consumers and required debug-agent/ROCgdb suites.
- [ ] Qualify attach/detach, supported breakpoints, stepping and wave inspection.
- [ ] Establish recovery protection before deliberate traps or fault tests.
- [ ] Extend tracing coverage to applicable allocation, event, graph, storage and
  communication domains.
- [ ] Validate architecture-specific counter register mappings and topology.
- [ ] Add native counter configuration only after physical measurements agree.
- [ ] Verify derived metrics, repeated sessions and profiling overhead.
- [ ] Investigate PC sampling, ATT and SPM applicability separately.
- [ ] Record unsupported advanced-profiling modes honestly when hardware/kernel
  capabilities do not provide them.

## 24. AMD SMI, RDC, media and other optional components

- [x] Build AMD SMI without the BC-250-inapplicable EPYC ESMI dependency.
- [x] Record the current kernel's unavailable VCN/JPEG/UVD/VCE ring queries.
- [ ] Qualify read-only AMD SMI telemetry and its unsupported-field reporting.
- [ ] Build/test applicable RDC monitoring and health functions.
- [ ] Keep reset/clock/power-management controls separate from telemetry qualification.
- [ ] Qualify rocDecode host decoding and HIP interoperability.
- [ ] Investigate hardware decode/encode availability independently of the host decoder.
- [ ] Investigate rocJPEG hardware availability; the current upstream backend does
  not establish a working software fallback on this host.
- [ ] Implement a clearly identified software media backend only if useful and in scope.
- [ ] Audit optional rocRoller and similar generators for legal scalar/SIMD kernels.
- [ ] Assess rocWMMA-style APIs honestly: native WMMA hardware is unavailable;
  any useful software emulation would be a separate implementation.
- [ ] Evaluate other optional components against the pinned source inventory and
  tested capabilities instead of enabling them by name alone.

## 25. Performance and resource efficiency

- [x] Retain build/cache work across interrupted builds.
- [x] Use measured memory consumption to set heavy-component job limits.
- [x] Establish shared-fixture GPU benchmarks with synchronized warm measurements.
- [ ] Extend first-invocation measurements to controlled cold-cache/startup cases.
- [x] Compare nine GPU workloads against `.105` with identical tensor bytes and precision policies.
- [ ] Record compiler, kernel, firmware, clock, temperature and memory conditions.
- [x] Measure FP32/FP16/BF16 GEMM, convolution forward/backward and device copies.
- [ ] Extend measurements to FFT, sparse operations and reductions.
- [x] Measure SmolLM2-135M full training and LoRA step throughput on `.104`.
- [x] Measure LoRA precision and activation-checkpointing tradeoffs for the small workload.
- [ ] Extend model shapes/inference measurements and compilation overhead coverage.
- [ ] Separate correctness improvements from performance improvements.
- [ ] Tune kernels and databases with numerical checks retained.
- [ ] Investigate launch overhead, memory bandwidth, LDS and occupancy bottlenecks.
- [ ] Check that an optimization does not regress irregular shapes or other operations.
- [ ] Keep repeatable before/after results and reject unexplained benchmark outliers.

## 26. Packaging, launcher, maintenance and deployment

- [x] Use versioned releases and an active userspace symlink.
- [x] Run Torch as the host user's UID with required GPU groups.
- [x] Keep release runtime files separate from mutable developer builds.
- [x] Freeze the training consumers separately and provide `bc250-train`.
- [x] Verify standalone GPU/Inductor/PEFT execution without development mounts.
- [x] Record runtime checksums, review evidence and a preservation handoff for the pause.
- [x] Fix absolute script paths and scripts outside the launch directory.
- [x] Verify the launcher as `nik`, from home and from `/tmp`.
- [x] Preserve Python `-c`, `-m`, option and stdin argument handling.
- [x] Provide interactive shell mode in the launcher.
- [x] Qualify interactive shell behavior and representative file/import workflows.
- [x] Document additional data mounts when arbitrary host paths are needed beyond
  the current-directory/script-directory mappings.
- [x] Preserve host absolute paths, caller cwd, home and host identity in the container.
- [x] Support explicit read-write/read-only host file and directory mounts.
- [x] Verify checkpoint model/optimizer/RNG save and resume across separate containers.
- [x] Verify host edits, sibling/module imports, stdin, exit codes and file ownership.
- [x] Verify HTTP downloads persist in the host home cache across invocations.
- [x] Verify private environment handoff through doas as nik.
- [x] Verify spawn DataLoader workers, pinned memory and more than 64 MiB of shared memory.
- [x] Verify persistent extra Python packages in a mounted target directory.
- [x] Verify torch.compile through the updated launcher.
- [x] Verify application file-save handlers receive SIGINT and root-issued SIGTERM.
- [x] Document training file workflows in [bc250_torch_usage.md](bc250_torch_usage.md).
- [x] Record component profiles and limitations in release manifests.
- [x] Require fresh expanded, Torch and stability evidence before activation.
- [ ] Validate installed-prefix relocation and dynamic library resolution.
- [ ] Package licenses, headers, tools, plugins, tests and relevant cache assets.
- [x] Verify rollback, launcher compatibility and cache ownership after each release.
- [ ] Add a concise diagnostic/reporting command for installed releases.
- [ ] Document known unsupported features and how experimental options are enabled.
- [ ] Establish an update/rebase procedure with reproducible regression gates.
- [ ] Keep SSH recovery instructions and source/build instructions usable by another
  person without relying on this conversation.

## 27. Release completion criteria

A component can be called qualified only for its recorded scope. The eventual
expanded release needs all of these conditions.

- [ ] Sources reconstruct from recorded baselines and patch series.
- [ ] The selected components build successfully for native gfx1013.
- [ ] Supported operations execute with independent expected results or CPU references.
- [ ] Required tests actually execute; zero-test/all-skipped success is rejected.
- [ ] Unsupported physical instructions are not emitted or falsely advertised.
- [ ] Runtime allocator and concurrency regressions have dedicated coverage.
- [ ] Stable Torch eager/compiled behavior is verified on the release copy.
- [ ] Longer stress, training and memory-pressure tests pass.
- [ ] Kernel logs and resource growth show no unexplained regressions.
- [ ] Performance is measured and relevant regressions are explained or fixed.
- [ ] The support/limitation matrix and manifest match the installed artifacts.
- [ ] Rollback and SSH recovery remain available.

## 28. Recommended order after the current milestone

- [x] **P0:** finish compiled allocator-default qualification, candidate Torch/soak
  gates and safe deployment of the expanded release.
- [ ] **P1:** expand numerical/concurrency coverage and tune the libraries already
  serving real Torch workloads.
- [ ] **P1:** qualify CK Tile and additional generic hipTensor/hipDNN operations.
- [ ] **P2:** implement scalar Lt/tensor-contraction backends, with operation-level
  tests before integration or tuning.
- [ ] **P2:** qualify debugger, telemetry and performance counters under safe recovery.
- [ ] **P2:** bring up communication libraries locally, then migrate/qualify the
  second host only after the local stack is stable.
- [ ] **P3:** investigate SDMA, VM-heap repair, advanced profiling, media engines and
  other optional capabilities using separate guarded experiments.
- [ ] Maintain a living applicability inventory so every feasible component ends
  in a tested implementation or a documented reason it remains unavailable.

## Evidence and work locations

- Local source: `/home/nik/Documents/Vibe/therock`.
- Host workspace: `/home/nik/bc250work/rocm-10.2-gfx1013` on `.104`.
- Initial fallback release: `/opt/bc250-stack/releases/10.2-gfx1013-20261012-a640c9296`.
- Candidate build result: `logs/expanded-build.exit`.
- Expanded gate result/details: `logs/expanded-validation.exit` and
  `logs/expanded-validation.log`.
- CG diagnostics: `logs/rocalution-diagnostic.log` and allocator investigation logs.
- Candidate Torch result: `logs/candidate-torch-validation.exit`.
- Candidate eager/compiled/soak evidence: `logs/candidate-runtime/`.
- Committed implementation patches: `patches/amd-mainline/` and
  `patches/bc250-triton/`.
- Build and validation helpers: `build_tools/bc250/`.

An in-progress result file can contain a sentinel rather than a final outcome.
Check the runner state and detailed logs before treating an old or intermediate
status as a new pass/failure. The compiled-default expanded sequence and
candidate Torch/compile gates passed. The candidate ten-minute soak completed
39,657 CPU-checked cycles in 600.005 seconds, with unchanged kernel logs.
The independent release copy passed eager/compiler gates and 40,339 checked
cycles in 600.001 seconds, with unchanged kernel logs. The active expanded
release is `/opt/bc250-stack/releases/10.2-gfx1013-20261013-4b000d542`.
Five fresh release processes passed all 200 CG diagnostic cases.

Rollback verification switched to the previous release, passed native GEMM,
restored the expanded release and passed native GEMM again. The user
`bc250-torch ~/bc250work/torch-matrix.py` invocation passed as `nik`.

The short real-model training/memory milestone passed nine gates in 123.99 seconds
on 2026-10-13, with unchanged kernel logs. Performance was not started. See
[bc250_training_memory.md](bc250_training_memory.md) for scope, commands, losses,
memory figures, artifacts and remaining qualifications.

Initial Performance measurements are now complete: nine matched-input GPU
comparisons plus six SmolLM2 training profiles. See
[bc250_performance.md](bc250_performance.md) for raw data, variability, hardware
confounds and remaining work. The earlier stop-before-Performance entry above
is the historical boundary of the previous milestone.

The current daily-use release is reviewed and retained for the development pause.
See [bc250_release_20261013.md](bc250_release_20261013.md) and
[bc250_development_handoff.md](bc250_development_handoff.md). All source, build,
cache, image, checkpoint and prior-release data remains in place.
