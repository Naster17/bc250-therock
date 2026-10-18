# Torch / ROCm gfx1013 support matrix — 2026-10-18

This is the current capability audit and work queue requested before N2.7. It
extends the [inventory](bc250_memory_inventory.md); it does not restart completed
N1/N2 or imply that every Torch operation/shape/type was tested. Runtime evidence,
exact errors, build flags, ELF dependencies and all3520 dispatcher names are in
[the audit evidence](bc250_torch_support_audit/README.md). Dispatcher registrations
are not architecture coverage or a support percentage.

## A09 update

**A09 native Torch GPU profiler — 2026-10-18.** User selected A09 ahead of
unfinished A08. Torch `4af4a1508a6ec0b7b2b29f5e0115f8477c4d1bb5` adds tests on
unchanged native source71a0426; TheRock runtime helpers `4c2edb785` enable
USE_KINETO=1/KINETO_BACKEND=rocm with the pinned rocprofiler-sdk backend.
Seven profiler tests require actual GPU kernels/copies, positive durations,
CPU/GPU correlation, FP32/BF16 output/gradient references, allocation/free events,
scheduled capture, custom streams and CPU-only selection. Profiling is explicit;
ordinary training does not automatically collect traces or become faster.
Candidate short-window setup/teardown-inclusive wall overhead was~4.70x;
not application throughput. Existing convolution/sparse/training/A06/A07/model/
resume gates pass within their recorded scopes. [Evidence and usage](bc250_profiler_a09_results/README.md).
Hardware counters, broad simultaneous/external profiling, compile/graphs,
long traces and cross-boot absolute-clock calibration remain unqualified.
A08 remains next, then N2.7. Historical Python CPU fault stays unresolved and
separate from fixed VM-heap/CSR corruption. No .105/kernel/BIOS/image change.

## Scope of the latest fixes

The Qwen3.5 helpers from implementation690042b55 are **application scoped**.
`finetune-n7/replica/train.py` installs native frozen causal convolution in its
Qwen3.5 module and chunk64 loss on its model instance. Other scripts/models and
`torch.nn.functional.conv1d`/cross_entropy are unchanged. The PEFT target-resolution
resume fix is in this trainer. Reuse is possible through explicit helper calls;
automatic general Torch routing is not implemented. A05 below adds trainable
convolution gradients through a separate opt-in API; this older trainer is unchanged. The earlier C++ `1Torch` diagnostic fix does apply to all callers
using the patched .104 Torch runtime. None of these changes updates .105.

## A06 correction and A07 update

**A06 correction and A07 head256 qualified on .104 — 2026-10-18.** Torch
`71a0426566bc9b952d9aa9b41f4a9d79ce46e0e8`, runtime helpers `e4f4b3854`.
Low-precision/autocast full gated-delta uses reference-ordered GPU preparation and
scan; non-autocast FP32/native standalone preparation remains. All372 nonzero
adapter gradients match exactly in BF16 actual2B65/1024 fixtures. A07 extends
explicit attention through D256: scalar tiled FP32/FP16/inference, BF16 D>128
training math-forward/native-LSE+tiled-backward hybrid.36 operator/dispatch tests,
42 A06 tests, four exact autocast cases, three actual-model train/checkpoint cases
and exact fresh/resume CLI checks within math and tiled pass. Actual-model A07
worst gradient relativeL2 is2.17%/1.31%, within unchanged4%;12 head256 calls prove
forward+checkpoint recomputation. The fixture now explicitly calls `.train()`;
earlier actual-model tests that only enabled checkpointing did not establish its
execution. BF16 hybrid lowers isolated peak memory but is slower; forward temporary
allocations remain quadratic. Math stays default; `--attention-backend tiled` is
an explicit new-run option with a distinct fingerprint. No automatic speedup claim.
Full FLA/dropout/compile/graphs/higher-order remain unqualified. [Report and evidence](bc250_a06_a07_results/README.md).
Next primary **A08**, then A09, then N2.7. Historical Python CPU segfault remains
unresolved and distinct from the fixed VM-heap/CSR corruption. Earlier failed A06
review and original timing evidence below remain historical and are superseded
only within this corrected scope.

## A01 update — 2026-10-16

A01 is implemented and installed in the current development prefix. ROCr329eaf7
corrects the demonstrated native gfx1013 shader/KFD clock epoch mismatch;
MIOpen/libraries08ba6f3 includes applicable dilated GEMM workspace in size queries.
Installed15 direct API configurations and30 Torch convolution output/gradient
cases pass, plus event/shader-counter and core regressions. See the
[A01 report and limits](bc250_miopen_a01_results/README.md).
A02 is now completed within its recorded reduction/recurrent scope (see below);
A03 now supplies the bounded database seed and capability-aware probe below; modern plugin kernels remain absent B07. Prior GPU-event speed ratios
need requalification; no new performance claim. Baseline audit counts below
remain historical, rather than being silently recomputed from overlapping tests.

## A02 update — 2026-10-16

Current-prefix MIOpen d375d8a2 includes native gfx1013 wave32 legacy CK reductions
and corrected BF16 storage/compute epilogues, with a versioned reduction cache key.
Installed54 numerical reductions,3 refusal contracts and12 recurrent output/all-
gradient configurations pass, plus convolution/event/core regressions. The
[A02 report](bc250_miopen_a02_results/README.md) retains failures and scope limits.
BF16 RNN is unsupported. A03 is completed below; exact next A04.

## A03 update — 2026-10-16

The current prefix contains genuine gfx1013_20 system FindDb16 keys/58 checked
immediate solutions, one searched PerfDb record and41 compiled kernel records.
Read-only replay with empty user caches passes, as does an unseen-shape fallback.
Default CK plugin probing respects build/device capability; explicit development
paths retain warnings. The modern plugin is still absent. These are bounded seed
assets, not full architecture tuning. [A03 evidence and limits](bc250_miopen_a03_results/README.md).
Exact next A04, Torch hipSPARSELt discovery/integration.

## A04 update — 2026-10-16

Torch8cdfcc8 now finds the installed hipSPARSELt package and enables native scalar
gfx1013 compress/matmul plus HIP Python version/search bindings. Availability now
follows compiled support. Candidate/installed20 tests pass, including120 small
CPU-reference matmuls, pack decoding, typed bias/alpha, layouts/custom streams,
public FP16/BF16 wrappers and refusals. Installed convolution/core/attention smoke
regressions pass; original Torch and frozen fallback preserved. Same-dtype
FP32/FP16/BF16/INT8 only; sparse autograd/compile/graphs/performance unqualified.
[A04 evidence, deployment and limits](bc250_torch_a04_results/README.md). Next A05.

## A05 update — 2026-10-16

Installed Torch6181d00 exposes `torch.nn.bc250_training_ops`: reusable explicit
causal convolution with native dx/dw/db and chunked causal loss, without a model
installer.69 installed tests pass CPU references, AMP casting/scaled update,
checkpoint recomputation, exact tiny-model AdamW resume, fallbacks and guards.
Core/A04 regressions pass; binaries, shared consumer bundle and original trainer
are unchanged. [A05 API/evidence/limits](bc250_training_a05_results/README.md).
Eager first-order only; no global routing, large-model speedup or compile/graph
qualification. Exact next A06, gated-delta reference profiling/native substage.

## Sparse/lint follow-up — 2026-10-16

Installed Torch6ed3d12 closes first-order native sparse backward and dense source-
weight conversion gaps; public transpose backward reconstructs on GPU.38 CPU-
reference tests pass plus A04/A05/core/conv regressions. Verified lint tooling
initializes locally;12 targeted configured static/format checks pass, with an
opt-in source-only spin flag. This does not claim full-project typing/CI.

Paired12 eager forward and12 training fixtures are measured: some small prepacked
forward wins, but packing adds cost and sparse training is1.92–3.36x slower here.
Default dense routing stays unchanged. [Scope/results](bc250_sparse_followup_results/README.md).
Higher-order/forward AD/compile/graphs/integer gradients/broad shapes remain unqualified.
Exact next A06 stays unchanged.

## A06 update — 2026-10-17

Installed Torchba6cbcd adds opt-in `torch.nn.bc250_gated_delta`: native wave32
Q/K normalization/scaling, beta gating, padding/layout and cumulative decay with
all five first-order input gradients. ATen solves/matmuls/chunk scan remain.
40 installed CPU-reference tests, tiny LoRA/checkpoint pairs, actual2B BF16/rank16
65/1024-token comparisons of372 adapter gradients and A05/core regressions pass.
Actual1024-token paired synthetic forward/backward7.653->7.390s(~3.6%) and
~220MiB lower peak; no universal/convergence claim. BF16 is not bitwise equal;
max model adapter-gradient relativeL2~3.23%. Existing trainer remains unchanged,
so it still prints the gated-delta reference warning. Further native fusion,
packed/cache/higher-order/compile/graphs and FLA package integration remain open.
[API, exact timings, limits and recovery](bc250_gated_delta_a06_results/README.md).
Exact next A07: head256 tiled SDPA output/QKV-gradient references; keep math default.

## What the current audit established

- Historical audit baseline Torch4ee6571, ROCm libraries a6f7d3d, Triton ee65ff3b, immutable image5df0a58,
  kernel0957f1/image84995f1. TheRock10.2, HIP7.17.0, MIOpen3.6.2 and Torch2.13-dev
  are different component version schemes, not evidence of an ABI mismatch.
- 21 libraries examined:19 installed DSOs load successfully; RCCL and the gfx1013
  MIOpen CK grouped-conv plugin are absent. No missing mandatory loader dependency
  was observed. Loading a DSO does not establish operator correctness.
- 51 unique targeted configuration probes after retaining/correcting harness
  mistakes:35 bounded passes;6 reproduced MIOpen failure variants belonging to
  two defect families;10 unavailable/configured-off/refused feature cases. These
  include contract and availability probes, not51 independent numerical kernels.
- Kernel logs unchanged in every batch;600s guards ACKed. Collector exit0 means
  collection completed, not that all probes passed. No segfault/hardware fault,
  runtime installation, source rebuild, prefix selection or boot change this audit.

HIP reports20 multiprocessors because CLR's enabled WGP mode divides the40 physical
CUs by2 (`rocdevice.cpp:1129`). MIOpen uses that HIP attribute in its database key.
Thus `gfx1013_20` is the actual key; renaming another GPU's database or changing
it to40 would not supply qualified tuning data.

## Existing capabilities and integration boundaries

**PASS** means the listed bounded evidence; **PARTIAL** requires the stated limits.
The following rows are not an exhaustive API or dtype promise.

| Area | Torch route / dependency | Current evidence | Boundary |
| --- | --- | --- | --- |
| HIP allocation/copy/stream | ROCr, CLR, amdhip64, c10_hip | Native gfx1013; prior allocation/stream/churn and current probes | SDMA/experimental heap off; VMM/reset/fault migration limits remain |
| Pointwise/reductions/indexing | ATen HIP, rocPRIM/hipCUB/Thrust | Existing P1/P2 suites; current normalization/embedding/optimizer checks | Larger dtype/layout/indexing contracts unqualified |
| Dense FP32/FP64/FP16/BF16 GEMM | hipBLAS/rocBLAS, default | All four current output/input-gradient references pass; earlier broad fixtures retained | Scalar backend, no Tensor Core throughput claim |
| Optional dense Lt | hipBLASLt | Explicit hipblaslt selection output/gradients pass; N1.8 bounds retained | Remains opt-in; INT8/FP8 gaps below |
| Conv1D/2D/3D | MIOpen | Group2 FP32 ranks1–3 and FP16/BF16 2D output/dx/dw pass | Group1 training defect A01; not blanket convolution support |
| Depthwise / transpose conv | MIOpen or ATen selection | BF16 K4/C64/L65 depthwise and FP32 transpose-conv gradients pass | Does not qualify the missing CK plugin or all depths/layouts |
| Norms | ATen/MIOpen by selection | Batch/layer/group norm output/input gradients pass | A02 native legacy CK reductions now pass bounded FP32/FP16/BF16 references; other compute/NaN/index contracts unqualified |
| Pooling/resize/activations | ATen HIP | Current pool/resize backward and existing NHWC padding fix references | More shapes/types/antialias modes unqualified |
| Embedding, cross entropy, CTC | ATen HIP | Current CPU-float64 forward/gradient references pass | Ordinary CE still materializes full FP32 logits in Transformers |
| RNG | hipRAND/rocRAND + Torch | Current exact GPU RNG save/replay; P2 direct/RNG suites | Not all generators/distributions |
| FFT | hipFFT/rocFFT | FP32/FP64 and FP16 power-of-two8x16 rfft2 output/backward pass | BF16 rejected B01; wider half plans not qualified |
| Dense solve/SVD/lstsq | hipSOLVER/rocSOLVER/rocBLAS | Current solve, singular values+gradient, full-rank gels references pass | Rank-deficient drivers and every solver algorithm not qualified |
| Hermitian eigenvectors | Native rocSOLVER | N2.1 retained22 cases through257 and gradients | Extreme/larger/two-stage cases unqualified |
| Nonsymmetric eig | hipSOLVER DnXgeev -> host LAPACK | N2.2 host compatibility16; N2.3–6 GPU prototype bounded | Production solve is host arithmetic; native N2.7 still queued |
| Structured sparse matmul | Torch hipSPARSELt | A04 forward plus follow-up38 floating first-order gradient tests; public FP16/BF16 matmul/linear source-weight/input/bias gradients | Scalar native packs; integer backward/higher-order/graphs/compile unqualified. Training slower in measured small fixtures; keep explicit |
| Sparse COO/CSR/CSC/BSR matmul | hipSPARSE/rocSPARSE + ATen | Current CSR/CSC/BSR output refs and retained COO/CSR/triangular suites | General sparse solve absent B02; not every format/dtype/backward |
| SDPA math | ATen HIP + BLAS | Current forward/QKV gradients and prior long/model/AMP gates | Quadratic attention memory; default retained |
| Optional tiled SDPA | Patched scalar Triton helper, explicit D<=256 |36 A07 tests and actual2B nonzero-adapter train/checkpoint gates | BF16 D>128 training uses math-forward/tiled-backward hybrid: quadratic forward temporary, linear saved state. Dropout/compile/graphs remain unsupported; math default. |
| Gated-delta preparation | torch.nn.bc250_gated_delta; app hook preserved |42 tests, four exact autocast cases, BF16 actual2B65/1024 all372 gradients exact | Low-precision full operation uses reference-ordered GPU ATen preparation/scan. Native FP32/direct preparation remains; complete FLA/native solve/scan fusion absent. |
| Qwen3.5 frozen conv / loss | Application helper | Previous51 operator gates, real2B1024-token update and exact CLI resume | Only the inspected trainer by default; A05 |
| AMP / fused AdamW | ATen HIP + generic optimizer kernels | Current finite FP16 scaled update and CPU-matched fused AdamW; prior full AMP/resume | Valid FP32 loss promotion; no universal optimizer qualification |
| torch.compile | Dynamo/Inductor/patched Triton3.8 | Current fullgraph norm/GELU output/gradient ref; prior dynamic math attention | Fused SDPA/custom tiled graphs unqualified |
| HIP graphs | Torch CUDA namespace -> HIP | Current fixed-buffer add capture/replay, previous direct graph fixtures | Does not qualify every op/allocation or custom attention |
| ROCTX / external tracing | roctracer/rocprofiler | Prior API/kernel/copy correlation evidence | Torch Kineto/SDK integration now bounded-qualified A09; counters/PC sampling remain unqualified |
| hipTensor / hipDNN / CK generic | Installed libraries, no pinned ATen integration found | DSOs load; prior direct native library/provider tests | Not mandatory dependencies for Torch einsum/convolution; installing them alone changes no Torch route |

## Ordered work queue

**Feasibility is an engineering assessment**, not an implemented or performance
claim. A native scalar implementation can be feasible when a vendor matrix engine
is unavailable. Each item needs its own bounded CPU/output-gradient gate before
runtime deployment. Keep item IDs stable so subsequent requests can name them.

| ID | Missing feature or defect | Evidence / current impact | Hardware-feasible next action |
| --- | --- | --- | --- |
| **A01 — completed** | Convolution timing and dilated workspace | Native gfx1013 shader/KFD epoch mismatch caused unstable/negative HIP event times; MIOpen query excluded dilated GEMM. ROCr329eaf7/libraries08ba6f3 installed;15 direct API +30 Torch convolution configurations pass CPU references,32 stable event pairs and six raw-counter timing references. | Bounded current-prefix fix; see [report](bc250_miopen_a01_results/README.md). Absolute trace alignment, opposite epoch offsets/reset/reboot and universal convolution/performance remain unqualified. A02 completed below; continue A04. |
| **A02 — completed** | Native MIOpen legacy CK reductions and recurrent backward | Libraries d375d8a2: native wave32/RDNA1 scalar profile; BF16 mapping/raw transfers/conversions and compute-precision scaling; versioned cache key. Installed54 numerical reductions +3 unsupported-compute refusals and12 FP32/FP16 LSTM/GRU/RNN output/all-gradient fixtures pass. | Applies to current-prefix MIOpen callers; BF16 RNN and wider NaN/index/compute/recurrent contracts unqualified. [Report](bc250_miopen_a02_results/README.md). A03 supplies a bounded DB seed/probe correction; modern plugin remains absent B07; no general speedup claim. |
| **A03 — completed** | Genuine native MIOpen seed databases and capability-aware plugin probing | Libraries fcf7a714 installed:16 FindDb keys/58 checked solutions, one searched PerfDb record and41 KDB records; normal-path read-only replay has zero new user kernels. Unseen fallback and regressions pass. | Small geometry/version/device scope only; wider tuning and modern CK plugin B07 remain unqualified. [Report](bc250_miopen_a03_results/README.md). Continue A04. |
| **A04 — completed** | Native Torch hipSPARSELt integration and follow-up | Torch6ed3d12;20 forward tests plus38 first-order gradient tests, including dense source conversion/transpose/linear. Targeted lint passes;12 forward and12 training timings recorded. | Same-dtype native floating first-order training; integer/higher-order/compile/graphs unqualified. Sparse training slower in measured small fixtures; keep explicit. [Follow-up](bc250_sparse_followup_results/README.md). Next A06 after completed A05. |
| **A05 — completed** | Reusable opt-in causal conv/loss | Torch6181d00;69 tests: CPU output/dx/dw/db/loss, AMP/checkpoint/update and exact tiny-model AdamW resume. | Explicit torch.nn.bc250_training_ops calls; original trainer unchanged. Higher-order/compile/graphs/large-model/performance unqualified. [Report](bc250_training_a05_results/README.md). Next A06. |
| **A06 — corrected, bounded** | Native preparation and precision-safe full operation | Torch71a0426;42 operator tests, four autocast cases, all372 nonzero BF16 model gradients exact65/1024 | Default app hook corrected; BC250_GATED_DELTA=0 remains reference opt-out. Full FLA/deeper native fusion unqualified. [Report](bc250_a06_a07_results/README.md). |
| **A07 — head256 completed, bounded** | Explicit D<=256 tiled/hybrid attention |36 tests; actual2B training/checkpoint gates and exact per-backend fresh/resume CLI continuation | Math default. BF16 hybrid lower peak/slower in measured fixtures; quadratic forward temporaries. Dropout/compile/graphs remain separate. [Report](bc250_a06_a07_results/README.md). |
| **A08** | Dense INT8 -> INT32 `_int_mm` | Valid32x32x32 reaches hipBLASLt heuristic and returns HIPBLAS_STATUS_NOT_SUPPORTED; current Lt scalar type subset has no INT8. Sparse INT8 N1.7 is a different route. | Add bounded dense scalar INT8/INT32 Lt path or explicit native Torch alternative, with overflow/accumulation/layout/alpha-beta/epilogue tests. No matrix instruction assumption. |
| **A09 — completed, bounded** | Native Torch GPU profiler via existing Kineto/SDK | Separate USE_KINETO=1 build; seven native trace/reference/lifecycle tests and existing numerical/model/resume gates | Explicit torch.profiler contexts; CPU/GPU kernels/copies/memory correlation. Hardware counters, broad concurrency, compile/graphs and long tracing unqualified. [Report](bc250_profiler_a09_results/README.md). |
| **B01** | BF16 FFT API input rejected | `torch.fft.rfft2(BFloat16)` rejects, whereas FP16 tested power-of-two plan works | Explicit FP32 promotion helper with complex64 result/gradient contract is feasible; no native complex-BF16 type or universal FFT speed claim. Core API change optional. |
| **B02** | Public general sparse linear solve unavailable on ROCm | `torch.sparse.spsolve` requires CUDA cuDSS in this source; direct rocSPARSE triangular solves are not a general factorization solver | Design a real ROCm solve/factorization integration or clearly bounded documented fallback. Qualify residuals, singular/info and gradients where promised. Large sparse memory/performance cannot be inferred from dense fallback. |
| **B03** | Distributed build, Gloo/RCCL absent | USE_DISTRIBUTED/GLOO/NCCL off; no librccl.so. One GPU on .104; .105 remains protected. | CPU process-group/software transport first; then native gfx1013 RCCL applicability and isolated local process tests. Multi-board/P2P/XGMI require separate hardware/authorization and cannot be qualified here. |
| **B04** | torchvision / torchaudio not installed in reviewed runtime | Package discovery absent, not a core Torch loader failure; RPP/FFTS are separate libraries | Build matching pinned extensions, test needed scalar ATen/C++ ops; keep optional codecs/CPU paths explicit. No assumption that RPP/media blocks automatically accelerate them. |
| **B05** | torchao / bitsandbytes / packed low-bit model kernels not qualified | Packages absent from reviewed runtime; scalar dense INT8 route also missing A08 | Select one actual inference/training quantization recipe, audit ROCm/triton instructions/packing and gradients, port scalar dequantize+GEMM if needed. Native tensor-core FP8/INT4 throughput is unavailable. |
| **B06** | ONNX export/runtime packages absent | onnx/onnxruntime not in reviewed runtime; unrelated to eager Torch availability | Optional export and runtime build/EP audit with version-specific model checks. Not necessary to repair ordinary LoRA training. |
| **B07** | MIOpen CK grouped-convolution plugin | Still absent by architecture/build capability. A03 skips expected default probe; explicit path retains diagnostics. | Optional scalar/generic plugin port; working HIP-direct/legacy reduction routes remain distinct. No kernels are supplied by a flag or diagnostic change. |
| **N2.7** | Native nonsymmetric eig remains a bounded slow prototype | N2.6 completed; production DnXgeev uses host LAPACK; GPU arithmetic prototype has host scalar control | Resume the existing Schur substage profile/device-iteration roadmap after the selected Torch/library blockers, without duplicating N2.1–6. |

A01–A07 are completed within their recorded bounded scopes; A07 dropout/compile/
graph contracts and full native FLA remain open extensions. A09 was selected and completed ahead of A08; A08 is the
remaining primary item before N2.7. B-items are optional
workloads/integration expansions rather than mandatory missing Torch dependencies.
N2.7 stays queued while we work through the selected primary items; this audit did
not silently begin their implementation. Small targeted tests cannot prove that
no other hidden defects exist; the retained runner makes new probes reproducible.

## Hardware limits and unresolved stability

Native MFMA/WMMA/Tensor Cores cannot be added by a library. TF32 hardware support
reports false; `_scaled_mm` FP8 is gated to other architectures. Scalar FP8
conversion/dequantized computation may be possible, but enabling a newer GPU name
or matrix-only package would be incorrect. SDMA remains off and experimental VM
heap disabled; GPU reset/fault migration/media exposure/counters remain separately
unqualified. The unknown Python CPU segfault remains open and distinct from the
already-fixed CSR/VM-heap corruption, the bounded A01 corrections and the
separately corrected A02 legacy-reduction failures.

MIOpen's database roles are separate: FindDb saves algorithm-search outcomes,
PerfDb stores tuned parameters, and KDB/user kernel cache stores compiled kernels.
Missing precompiled kernels can incur compilation rather than making all math
unavailable. This distinction follows AMD's [FindDb documentation](https://rocm.docs.amd.com/projects/MIOpen/en/docs-7.2.1/conceptual/finddb.html),
[performance database](https://rocm.docs.amd.com/projects/MIOpen/en/docs-6.4.1/conceptual/perfdb.html)
and [kernel cache documentation](https://rocm.docs.amd.com/projects/MIOpen/en/docs-7.1.0/conceptual/cache.html).
The local pinned source and installed audit remain authoritative for gfx1013.
