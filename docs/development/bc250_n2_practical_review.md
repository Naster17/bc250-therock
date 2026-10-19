# Practical N2 gap review — 2026-10-19

N2.1–N2.18 are completed within their individual recorded scopes. N2 remains an
open scientific/library coverage workstream. The practical review finds no
recorded missing mandatory library or unresolved numerical defect blocking the
current Qwen3.5 BF16 LoRA path. This is a workload decision, not universal Torch
qualification or a statement that no hidden defect exists.

The current native Torch is `dc8fea0`, source/test revision `2b449c4`, ROCm
libraries `319a235d6` with source-documentation `c29021b40e7`. Existing guards,
CPU-reference evidence and protected artifact hashes remain authoritative.
A01–A03 address the demonstrated MIOpen timing/workspace, legacy reductions,
recurrent gradients and database/probe problems; the original audit failures
are historical. The historical Python CPU fault remains unresolved and separate
from fixed VM-heap/CSR corruption.

| Remaining area                                                  | Practical disposition                                                  | Next trigger / boundary                                                                                                                                                        |
| --------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Mandatory loaders / native BLAS / current model operators       | Retain qualified stack; no replacement/rebuild required by this review | Fresh current-model N3 forward/backward/optimizer replay exercises the installed route; retain protected identities and loader errors                                          |
| BF16 FFT input, B01                                             | Optional numerical API extension                                       | Explicit FP32 promotion already uses supported FFT; define result/gradient dtype contract before adding an API. No native complex-BF16 claim                                   |
| General sparse factorization/solve, B02; triangular autograd    | Real missing capabilities, separate from qualified triangular forward  | Select an actual sparse workload, residual/singular/info and gradient contract; dense fallback is not a scalable sparse solver                                                 |
| Dense difficult solvers; nonsymmetric eig                       | Retain host-LAPACK production default and bounded native prototype     | Larger/ill-conditioned cases and native performance remain open; more tiny eig optimization is not a LoRA prerequisite                                                         |
| FFT plans, RNG, scans/sorts and custom primitives               | Broad bounded coverage now recorded by N2.14–17                        | Add operation/type/layout/concurrency cases when a consumer needs them or a defect reproduces; no exhaustive-combination gate                                                  |
| Scalar CK convolution backward                                  | Hardware-feasible candidate for future direct-library coverage         | Audit existing backward-data/weight implementations first; MIOpen backward is already available to Torch within recorded bounds. A proposed N2.19 was not assigned or executed |
| MIOpen modern CK plugin, B07; other CK Tile/provider operations | Optional scalar integration work                                       | N2.18 direct CK forward does not enable a plugin or change Torch routing. No XDL/WMMA admission                                                                                |
| RPP image/audio/volume                                          | Optional workload coverage                                             | Not used by this text-only LoRA path; qualify actual image/audio consumer operations when selected                                                                             |
| torchao / bitsandbytes / low-bit recipe, B05                    | N3 feasibility work, not a required dependency                         | A08 INT8 GEMM is a primitive, not QLoRA. Audit a concrete packing/dequantization/training recipe before integration                                                            |
| torchvision/torchaudio/ONNX, B04/B06                            | Optional packages                                                      | Add pinned compatible extensions/export only for an actual consumer; RPP is not automatic torchvision routing                                                                  |
| Distributed/RCCL, B03                                           | N5 workstream                                                          | Local/software readiness first; .105 remains read-only, no unqualified P2P/XGMI claims                                                                                         |
| Physical counters/telemetry / advanced profiling                | N4 workstream                                                          | A09 API/kernel tracing and CPU-first initialization repair already available; trustworthy counters are a separate problem                                                      |
| Historical Python CPU segfault                                  | Open stability investigation                                           | Capture a reproducer/core/backtrace under recovery guards before proposing a fix; passing new tests does not repair it                                                         |
| Matrix-only packages/FP8 hardware throughput                    | Hardware boundary                                                      | Native gfx1013 has no MFMA/WMMA; do not advertise another architecture or enable unsupported instructions                                                                      |

The concrete maintenance needed now is documentation repair: the support matrix
still named N2.14 as next and kept the fixed A01 group-1 convolution defect in
its current boundary row. The current cursor and affected capability rows are
updated, with historical evidence retained.

Proceed to **N3.1: current Qwen3.5 training-step attribution**, before selecting
another optimization. Use the actual local base model and installed application
hooks with synthetic input IDs and new synthetic adapters; never open user
training data/checkpoints. Check profiled/unprofiled loss, all adapter gradients,
optimizer update/state and RNG, then collect a short native GPU trace and warm
unprofiled timings. Prior AMP/compiled-block/checkpoint-resume tests are retained
rather than rerun as a new milestone. Profile overhead is measured separately;
synthetic batch1/accumulation1 timings are not the user's accumulation16 training
throughput. Defaults, kernels and libraries remain unchanged.

The [N3.1 report](bc250_training_n31_results/README.md) now records passing65/1024
training replay and dense-mm dominance. Next N3.2 audits the dominant BF16 GEMM
shape and selection before any optimization or default change. Broad N2 and N3 remain open; no percentage or fixed
remaining subtask count follows from this review.
