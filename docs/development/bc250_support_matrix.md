# BC-250 gfx1013 support matrix

This records operation-level evidence for the native ROCm 10.2 port on `.104`.
It is not an upstream support declaration or a claim that every operation in
a listed library works. `.105` remains the unchanged ROCm 7.2 reference host.

The installed PyTorch release and the expanded developer build are separate.
New components are not activated in the installed release until their gates
pass and the expanded distribution passes the existing Torch gates again.
See [bc250_port.md](bc250_port.md) for source revisions and recovery policy.

## Executed successfully

Evidence below was collected on 2026-10-12. Tests use native `gfx1013:xnack-`
without architecture overrides. SDMA is disabled in the validated profile.

| Component                                   | Executed coverage                                                                                                                   | Limits of this evidence                                                                   |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| LLVM/Clang, device libraries, COMGR, HIPRTC | Native HIP compilation, runtime kernel compilation, Triton instruction selection and GPU execution                                  | Not every builtin, language feature, or optimization has been tested                      |
| ROCr/KFD and HIP                            | Allocation churn, copies, streams, LDS, integer atomics, native identity; packed FP16/BF16 CAS atomics with neighboring-lane guards | Current kernel and firmware only; SDMA, reset and debug traps need separate qualification |
| rocBLAS and hipBLAS                         | CPU-compared GEMM, including Torch FP32/FP64/FP16/BF16; upstream 1023×1024×1025 SGEMM                                               | Generic scalar kernels; no native tuning database or complete BLAS suite                  |
| rocSOLVER and hipSOLVER                     | Cholesky and Torch solver checks                                                                                                    | Wider factorizations and difficult numerical cases remain                                 |
| rocFFT and hipFFT                           | Prime-length CPU-compared transform and Torch FFT checks                                                                            | Wider ranks, real/complex combinations and strides remain                                 |
| rocRAND and hipRAND                         | 44 rocRAND basic tests, six dispatch/configuration cases, CPU-checked distribution/reproducibility checks                           | Complete generator/type coverage remains                                                  |
| rocSPARSE and hipSPARSE                     | Torch sparse CSR multiplication                                                                                                     | Direct library test coverage across formats and operations remains                        |
| rocPRIM and hipCUB                          | Eight device reductions, ten scans, device-version test, 208 block reductions                                                       | Remaining primitives and native runtime architecture/tuning selection need auditing       |
| composable_kernel                           | Upstream scalar DL FP32, FP16 and signed INT8 GEMM examples with CPU verification                                                   | CK Tile generic operations need their own gates; matrix-instruction engines stay disabled |
| MIOpen                                      | CPU-reference FP32/FP16/BF16 convolution for groups 1 and 2, forward and both gradients                                             | Other operators, layouts, solver coverage and tuning remain                               |
| PyTorch                                     | 56 eager checks, sparse operations, convolution gradients, mathematical SDPA and training                                           | Broad upstream suites, long qualification and `.105` performance comparisons remain       |
| Triton and Inductor                         | Native dot fallbacks, exact BF16 rounding, masked global/buffer memory, fused kernels and compiled forward/backward graph           | No hardware matrix-instruction path; wider shape/layout coverage remains                  |
| LLVM OpenMP                                 | Mandatory GPU offload, odd-length SAXPY, reduction and task dependencies with CPU comparison                                        | Fortran offload and wider runtime cases remain                                            |
| OpenCL                                      | CL1.2 kernels on the native device, LDS reduction, atomics, event profiling and RGBA8 image sampling                                | Device advertises OpenCL 2.0; this does not establish all 2.0 features                    |
| RPP image operations                        | CPU-compared brightness and flips in NHWC/NCHW, padded rows and allocation guards                                                   | Other image, audio and volumetric operations remain                                       |
| hipFile                                     | Five verified host/device file roundtrips, asynchronous/registered/multistream/batch cases                                          | Compatibility backend only; no direct-storage fast-path claim                             |

The installed release also passed 8,074 CPU-checked allocation/GEMM/copy/compiled
kernel cycles over 120 seconds. This is an initial soak, not long-term stability
qualification. No new kernel messages appeared during that release gate.

## Expanded work and remaining qualification

| Component or feature                     | Current state                                                                    | Next implementation or hardware gate                                                                                                   |
| ---------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| RPP audio / bundled FFTS                 | Audio build integration added; static FFTS stages successfully                   | Independent CPU formulas for pre-emphasis and spectrogram, including host FFT and HIP paths                                            |
| hipTensor generic operations             | Native device registration and opt-in generic-only profile added                 | Build and run reduction, permutation, binary and trinary operations; verify no skipped tests                                           |
| hipTensor contraction                    | Generic profile contains no device contraction instances                         | A scalar backend is feasible work; implement and verify it before claiming contractions                                                |
| rocALUTION                               | Built; four small GPU CG cases passed a serial rerun after intermittent failures | Freeze the distribution; repeat CPU/GPU SpMV, norm and CG comparisons before declaring support                                         |
| hipDNN and MIOpen provider               | Built in the initial expanded profile                                            | CPU-reference forward/data-gradient/weight-gradient convolution graphs in three precisions                                             |
| hipDNN HIP kernel provider               | Built with generic HIP MLOPS; ASM SDPA, Flash2 and ingestor disabled             | Runtime scalar regression cases and CPU-reference normalization/resampling graphs                                                      |
| hipBLASLt                                | Explicit host/API-only profile tested; GEMM reports unsupported                  | Implement native scalar TensileLite kernels and valid metadata/heuristics; no whitelist-only port                                      |
| hipSPARSELt                              | Excluded from gfx1013 device build                                               | Generic prune/compress may be enabled separately; sparse matmul needs a scalar implementation with correct metadata decoding           |
| CK Tile                                  | Generic core recognizes gfx1013/wave32; examples not yet qualified               | Reduction, permutation, transpose, elementwise, normalization and pooling with CPU references                                          |
| rocThrust / libhipcxx / hipCCL packaging | Headers and initial consumers build                                              | Audit architecture detection and run algorithms; avoid duplicate incompatible primitive headers. hipCCL is not a communication library |
| RCCL                                     | Not yet built/qualified for this profile                                         | Build native single-device readiness and test APIs; preserve `.105` until two-host validation is authorized for the new stack          |
| rocSHMEM                                 | Not yet built/qualified                                                          | Begin with generic wave32 paths and GDA/SDMA disabled; MPI and multi-process operation need separate gates                             |
| ROCprofiler SDK                          | Built; tracing not yet executed                                                  | Trace the CPU-verified HIP workload and verify timestamps, names, counts and correlations                                              |
| Hardware performance counters            | gfx1013 counter configuration missing                                            | Validate register mappings, topology and counter values before adding native tables                                                    |
| PC sampling / ATT / SPM                  | Not established                                                                  | Query runtime/kernel capabilities and architecture requirements individually; compiler acceptance is insufficient                      |
| ROCdbgapi, rocr-debug-agent and ROCgdb   | Native gfx1013 debugger class/ELF registration patched                           | Build consumers and run prescribed debugger suites with qualified SSH recovery protection                                              |
| AMD SMI / RDC                            | AMD SMI builds without EPYC-only ESMI; RDC not qualified                         | Read-only telemetry tests first; GPU management and reset are separate operations                                                      |
| hipThreads / Fortran offload             | Not qualified                                                                    | Build generic device support and execute native compiler/runtime examples                                                              |
| rocDecode                                | Current kernel exposes no VCN decode ring                                        | Qualify upstream host decoder and HIP interoperability; investigate media firmware/kernel availability separately                      |
| rocJPEG                                  | Current kernel exposes no JPEG ring; upstream software fallback absent           | Establish actual hardware capability first, or implement a distinct software backend                                                   |
| SDMA                                     | KFD reports two engines, but ROCr intentionally disables gfx1013 user SDMA       | Guarded firmware/kernel/runtime experiments and correctness/stress tests                                                               |
| GPU reset and fault recovery             | Not qualified; watchdog keepalives active, reset behavior untested               | Verify recovery independently before tests that deliberately fault or halt GPU queues                                                  |
| hipIFY, packaging and tooling            | Initial expanded build includes HIPIFY                                           | Translation/tool smoke tests and consistent source provenance for the expanded release                                                 |

## Hardware constraints

The host reports 80 SIMDs, 40 CUs, wave32, 64 KiB LDS and eight CPU cores with
16 threads. KFD exposes 24 compute queues, two SDMA engines and no XGMI SDMA
engines or P2P links. CPU and GPU share the board's physical memory.

Compiler capability probes establish scalar FP32/FP64 and packed half arithmetic,
but no usable MFMA, WMMA or packed dot path for this target. Absence of matrix
instructions constrains the backend implementation and performance; it does not
make the mathematical operation impossible. Generic HIP/scalar implementations
remain candidates for libraries currently coupled to matrix engines.

The passive DRM probe reports graphics/compute engines but unavailable VCN,
JPEG, UVD and VCE ring queries in the installed kernel. This describes current
driver exposure, not proof that the silicon physically lacks every media block.

## Validation rules

- Preserve the active userspace release and working boot files while developing.
- Compile without mapped GPU devices; run hardware gates serially against a
  finished distribution, not shared libraries being replaced by a build.
- Require CPU references or independent expected values, actual executed test
  counts, and allocation guards where appropriate. An all-skipped exit code zero
  does not establish support.
- Keep architecture gates for unsupported instructions. Add a real alternative
  backend rather than changing the reported GPU identity.
- Record the exact profile, source revisions, logs and failed cases. A passing
  rerun does not erase an intermittent correctness failure.
- Qualify tracing separately from counters. The [rocprofv3 command-line
  reference](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.14.0/quick-reference/rocprofv3-cli-options.html)
  documents separate tracing domains and output controls.
- Extend deployment gates before activating an expanded versioned release, then
  rerun the existing eager/compiled Torch and stability checks on that copy.
