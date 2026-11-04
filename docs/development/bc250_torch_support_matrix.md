# Torch / ROCm gfx1013 support matrix — 2026-10-21

This is the current capability audit and work queue requested before N2.7. It
extends the [inventory](bc250_memory_inventory.md); it does not restart completed
N1/N2 or imply that every Torch operation/shape/type was tested. Runtime evidence,
exact errors, build flags, ELF dependencies and all3520 dispatcher names are in
[the audit evidence](bc250_torch_support_audit/README.md). Dispatcher registrations
are not architecture coverage or a support percentage.

## Current cursor after B02.1 / broad support

User direction2026-10-20: prioritize broad hardware-feasible ROCm/PyTorch and
library/API support. Qwen/LoRA is an optional test fixture, not the scope boundary;
previous current-LoRA dependency classifications do not remove general gaps.

N3.10 model-independent compiled training passes56 composite CPU-reference/GPU
checks across residual dense/gated MLPs, grouped/dilated CNN+GroupNorm and training
BatchNorm, FP32/FP16-BF16 autocast and two layouts/shapes. Full outputs/input/all
parameter gradients, CPU AdamW and running buffers pass; four profiler replays
show actual Triton/native GPU kernels. No production defect, install or speed
claim. [Report](bc250_compiler_n310_results/README.md). Max-autotune/dropout/
dynamic reuse/compiled optimizers/full applications remain separate.

**B04.1 installed:** pinned torchvision0.28.0+gfx1013/Pillow13 in a separate
runtime image, existing Torch preserved.67 candidate+67 installed checks cover
native NMS/ROI/deformable-convolution outputs/first-order gradients, CPU-reference
transforms, CPU codecs and synthetic ResNet18. Fixed tolerances, actual GPU ROI
trace. Indirect WebP loader issue repaired through per-codec relative RUNPATH;
old image/release and failed evidence retained. [Report and use/rollback](bc250_vision_b041_results/README.md).
No GPU decoding/BF16 custom operators/video/universal correctness or speed claim.
**B04.2 installed:** torchaudio2.11.0+gfx1013, existing Torch retained;142
candidate/142 installed/48 post-user-reboot composite checks pass. GPU DSP, IIR
output/waveform/coefficient gradients, RNNT cost/logits gradients and exact CTC
alignment qualify bounded scopes. Demonstrated CLR cold-hostcall transfer-queue
CPU crash repaired; historical unknown CPU fault stays unresolved.
[Report/use/recovery](bc250_audio_b042_results/README.md).
**B04.3 installed:** CPU TorchCodec0.17.0+gfx1013/FFmpeg6.1.1 provides public
audio load/save;136 candidate+136 installed file/reference/GPU-resample-gradient/
retained audio/vision/core checks pass. ROCm/Torch DSOs unchanged; user application
excluded as requested. [Report/use/recovery](bc250_codec_b043_results/README.md).
**B04.4 installed:** CPU video55/audio46/core7 checks pass in candidate and
installed runtime (108 each); exact single-frame VFR gap seeking repaired using
the scanned presentation index. Independent lossless pixel/PTS/encoder references
and explicit HIP spatial forward/backward pass. [Report](bc250_video_b044_results/README.md).
**B04.5 installed:** CPU JPEG/PNG/WebP decoding and PNG/JPEG encoding enabled;
203 candidate+203 installed image/video/audio/vision/core checks pass, independent
pixels and explicit GPU spatial forward/backward. Private missing libwebpdemux
dependency supplied with notices; GPU codecs/AVIF/GIF/HEIC remain disabled.
[Report/use/recovery](bc250_image_b045_results/README.md).
**B01.1 installed explicit fallback:** bc250_fft0.1.0 casts only BF16 input to
FP32 on its current device.327 final candidate+327 installed CPU-reference/gradient/
DFT/native GPU trace/retained FFT checks pass. No native BF16 FFT, CPU fallback
or global torch.fft/AMP change. [Report/use/recovery](bc250_fft_b011_results/README.md).
**B02.1 installed explicit fallback:** bc250_sparse0.1.0 solves small canonical
CSR systems through GPU densification/LU with first stored-value/RHS gradients,
size/storage admission and finite/singular/residual checks.381 candidate+381
installed checks pass, including unchanged fixed CPU-double/analytic/finite-
difference/AMP/stream/invalid-input gates and a135-kernel trace. Direct CSR parts
support real/complex value gradients; the CSR wrapper supports real values and
all four RHS dtypes. No native public API/scalable sparse factorization/CPU solve/
performance claim. [Report/use/recovery](bc250_sparse_b021_results/README.md).
**B03.1 installed:** native Gloo/TensorPipe/libuv build, local CPU process groups,
31 two-rank store/collective/first-gradient/DDP checks; full candidate and installed
regression gates recorded in [the report](bc250_distributed_b031_results/README.md).
**B03.2 qualified on installed runtime:**41 host-staged GPU tensor/first-gradient/
FP32-FP64 DDP checks plus67 retained checks pass; two ranks share one GPU.
No runtime code/default changes. [Report](bc250_distributed_b032_results/README.md).
**B03.7 isolated RCCL guards/profile:**76 new/1441 total checks pass. Torch14cf
refuses unsupported FP8 arithmetic and preserves selected bit copies; generic
registration and state-only suspend/resume readiness checked. Daily runtime
unchanged/NCCL off. No multi-device/symmetric/offload-payload/performance claim.
[Evidence and profile](bc250_rccl_b037_results/README.md).
**Next B03.8:** bounded opt-in runtime packaging/DSO loading and ordinary-launch
output/backward checks, retaining daily Gloo/NCCL-off and rollback.


N3.9 restores the qualified N3.7 provenance integration and A07 attention CLI
in the current trainer. The ops helper's printed-status edit is preserved;
math and the installed N3.2 preset remain defaults. Actual masked1024/rank16/
accum16 fresh/new resume/legacy/installed replay matches all372 adapters/full
state/loss/evaluation. Short tiled fresh/resume is exact within that backend.
31 composite checks,175 application tests in candidate and installed copies,
four CPU contracts local+.104 pass; seven600s guards ACK/dmesg identical.
Current trainer now enforces selected runtime identities on bound checkpoints.
Whole-source hashes include cosmetic changes: an owned old N3.7 checkpoint
refuses the preserved status edit before weights, without mutation. No silent
checkpoint migration or speed/convergence claim. [Report](bc250_training_n39_results/README.md).

N3.8 vocabulary forward/input-backward layouts are CPU-FP64 qualified:
all14 admitted scalar solutions plus default per role, including matched eager
Torch atomics/workspace policy. Experimental -339/-533 rows remain inactive;
installed N3.2 preset is preserved. Whole-model timings remain variable.
[Report](bc250_training_n38_results/README.md). Its application drift observation
is historical and superseded by N3.9 restoration; its failed host OOM is retained.

N3.6 deployed inner CLI1024/rank16/accum16 masked-target replay passes:
all372 adapters/full optimizer/scheduler/RNG/order/progress/loss/evaluation exact,
48 training micros/three evaluation forwards, seven checks. No production change
or new speed/convergence claim. [Report](bc250_training_n36_results/README.md).
N3.7 added supplemental checkpoint numerical-runtime/preset/hook provenance;
the original fingerprint schema stays compatible. N3.9 restores and requalifies this integration.

N3.5 current-preset1024/rank16/accumulation16 qualified:372 gradients versus
CPU FP64 sum worstrelativeL2=1.17e-8, CPU AdamW/scheduler replay and exact owned
fresh-process parameter/gradient/optimizer/RNG continuation pass. Fixture mount/
CPU-oracle state isolation fixed; no runtime change or new speed claim.
[Report](bc250_training_n35_results/README.md). N3.6 now completes actual deployed
CLI owned1024/rank16/accum16 integration; prior small CLI remains distinct.

N3.4 square2048 mm audit complete: two layouts/all14 admitted solutions each
pass CPU references. Faster-346 fails the full-model gradient contract7.65%>4%;
exact captured-output alternatives are flat/slower. **No override deployed or
new qualified speedup**; N3.2 preset remains active. [Report](bc250_training_n34_results/README.md).
N3.5 now qualifies that helper accumulation/resume case; N3.6 covers actual CLI
integration. Prior42-token/rank4 CLI evidence remains distinct.

N3.3 optimized1024-token synthetic model exact replay passes loss/all372 gradients/
AdamW/RNG; median5.443s, mm75.39% of kernel sum. Reverse projection26.43% audited:
two layouts/14 admitted solutions each CPU-reference checked; best isolated1.09x,
no demonstrated full-model benefit or deployment. Current N3.2 preset unchanged.
[Report](bc250_training_n33_results/README.md). N3.4 has now completed that square-projection audit and rejected the candidate.

N3.2 is installed: rocBLAS/Tensile source8389197c1 repairs cold override loading;
exact-layout BF16 NT2048x1024x6144 solution-490 is automatic through current dev
launchers with pinned identities and BC250_GEMM_TUNING=0 opt-out. CPU GEMM,
all372 model gradients, CPU AdamW and installed gates pass. Synthetic whole-step
ABBA1.52–1.56x, max gradient relativeL2=2.03%\<4%, reduction order changes; no
bitwise continuation/convergence/sustained-throughput claim. Other native ROCm
libraries remain319a235d6. [Report](bc250_gemm_n32_results/README.md).
N3.3 has now completed that profile/selection audit within the scope above.

N2.1–N2.18 are complete within their recorded scopes. Installed Torch remains
`dc8fea0` in WS/pytorch-sparse-n213, source/test `2b449c4`; other native libraries
`319a235d6`; rocBLAS alone rebuilt at `8389197c1`. N2.13 sparse pivot marking and
the A09 CPU-first profiler repair remain installed. N2.14 FFT output/backward,
N2.15 RNG replay/contracts, N2.16 reductions/scans and N2.17 sorting/selection
have bounded CPU-reference evidence. N2.18 directly qualifies scalar CK DL FP32
grouped forward128 +8 refusals; it adds no Torch/MIOpen route or speed claim.

The [practical N2 review](bc250_n2_practical_review.md) separates missing optional
APIs from the current LoRA path. No additional mandatory N2 implementation was
identified before N3.1 training-step attribution. B01/B02/B07 and other residual
coverage remain open; a proposed N2.19 backward batch was not assigned/executed.
Historical next N2.7/N2.13/N2.14 and profiler-open cursors below do not override
this section or flow.md. N3.1 has now passed current-model65/1024 exact training
replay and kernel/shape attribution; [results](bc250_training_n31_results/README.md).
Next N3.2 targets the dominant BF16 GEMM selection. The board was already on r2
at entry; prior r3 qualifications remain historical, not a current-boot assertion.

## A08 update

**A08 completed2026-10-18: native dense INT8->INT32.** Libraries
`319a235d6c6496f0da37be44dfa60eaf31a29f82`, runtime helpers
`8f0dfb671e88b31bc3f5bb4a4958acefdafde556`; Torch4af4a15 unchanged.
The native gfx1013 Lt backend admits I8 A/B, I32 C/D/scalars, COMPUTE_32I,
modulo32 arithmetic, row/column/N/T/padded/offset/batched/broadcast layouts,
host/device scalar alpha/beta and DEFAULT/BIAS/RELU/RELU_BIAS. Existing
`torch._int_mm` works without Torch rebuild/launcher hooks.99 direct exact CPU
cases+11 refusals,16 public Torch configurations/out/custom streams+five
refusals pass; GPU trace confirms gemm_int8.1628 existing floating Lt cases,
core7/profiler7/sparse38/conv30/A0642/A0736/actual2B3 pass.12 guards ACK/600s/
dmesg unchanged,61 protected hashes unchanged. DSO+kpack paired recovery in
WS/recovery/dense-int8-a08-before, SOURCE_DENSE_INT8_A08.json in WS/current.
[Report/usage/timings](bc250_dense_int8_a08_results/README.md). Small eager timing
is mixed versus FP32 GEMM+cast; no universal or LoRA speedup. No quantization
recipe, packed low-bit/QLoRA, integer autograd or compile/graph qualification.
Other scales/compute modes (including public pedantic), pointer vectors and
nonlinear/aux epilogues remain refused. No active job/container/guard deadline;
disk~1.10GiB free, preserve existing sources/builds/caches. No kernel/.105/default
BLAS/SDMA0/heap-off/image change. Historical CPU Python fault remains separate
from fixed VM-heap/CSR. A01–A09 bounded scopes complete; next **N2.7**, existing
small Schur/device-side QR iteration roadmap with default host dispatch retained.

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
| Pointwise/reductions/indexing | ATen HIP, rocPRIM/hipCUB/Thrust | Existing P1/P2 consumers; N2.16 native1536 FP32/64 reduction/scan/custom cases, N2.17 native1960 sort/select +Torch62 incl54 finite backward | Wider types/operators/iterators/concurrency/graphs/performance unqualified |
| Dense FP32/FP64/FP16/BF16 GEMM | hipBLAS/rocBLAS, default | All four current output/input-gradient references pass; earlier broad fixtures retained | Scalar backend; N3.2 exact-layout BF16 preset/loader repair, no Tensor Core or universal throughput claim |
| Optional dense Lt | hipBLASLt | Explicit hipblaslt selection output/gradients pass; N1.8 bounds retained | Floating Lt remains opt-in; dense INT8/I32 now A08-qualified, FP8 gaps remain |
| Conv1D/2D/3D | MIOpen | A01 fixes group1 timing/workspace; installed30 output/dx/dw configurations and15 direct APIs, earlier group2 references retained | Wider tuning/types/layouts remain unqualified; no blanket convolution claim |
| Depthwise / transpose conv | MIOpen or ATen selection | BF16 K4/C64/L65 depthwise and FP32 transpose-conv gradients pass | Does not qualify the missing CK plugin or all depths/layouts |
| Norms | ATen/MIOpen by selection | Batch/layer/group norm output/input gradients pass | A02 native legacy CK reductions now pass bounded FP32/FP16/BF16 references; other compute/NaN/index contracts unqualified |
| Pooling/resize/activations | ATen HIP | Current pool/resize backward and existing NHWC padding fix references | More shapes/types/antialias modes unqualified |
| Embedding, cross entropy, CTC | ATen HIP | Current CPU-float64 forward/gradient references pass | Ordinary CE still materializes full FP32 logits in Transformers |
| RNG | hipRAND/rocRAND + Torch | N2.15 Torch state80/statistics22/contracts20/fresh continuation1; direct native68/P2 RNG28 retained | Wider generators/orderings/statistical suites/graphs/concurrent mutation/cross-version replay unqualified |
| FFT | hipFFT/rocFFT | N2.14 complex96/real72 output+weighted gradients, independent DFT/finite differences/reuse; direct16 plans/32 executions, FFT30 retained | Native BF16 rejected; B01.1 opt-in FP32 helper qualified. In-place/planar/callbacks/giant/wider half plans/performance unqualified |
| Dense solve/SVD/lstsq | hipSOLVER/rocSOLVER/rocBLAS | Current solve, singular values+gradient, full-rank gels references pass | Rank-deficient drivers and every solver algorithm not qualified |
| Hermitian eigenvectors | Native rocSOLVER | N2.1 retained22 cases through257 and gradients | Extreme/larger/two-stage cases unqualified |
| Nonsymmetric eig | hipSOLVER DnXgeev -> host LAPACK | N2.2 host compatibility16; N2.3–12 isolated GPU prototype cap16/separated spectra | Production solve is host arithmetic; native full prototype still slower; broader spectra/batching/gradients unqualified |
| Structured sparse matmul | Torch hipSPARSELt | A04 forward plus follow-up38 floating first-order gradient tests; public FP16/BF16 matmul/linear source-weight/input/bias gradients | Scalar native packs; integer backward/higher-order/graphs/compile unqualified. Training slower in measured small fixtures; keep explicit |
| Sparse COO/CSR/CSC/BSR matmul | hipSPARSE/rocSPARSE + ATen | Current CSR/CSC/BSR output refs and retained COO/CSR/triangular suites | Native general sparse API absent; B02.1 small explicit fallback separate; not every format/dtype/backward |
| Explicit small CSR general solve | bc250_sparse -> GPU scatter/dense LU/adjoint | B02.1:253 new cases plus128 retained pass candidate and installed; four dtypes, fixed-sparsity values/RHS first gradients, indices/layouts/AMP/streams/refusals/trace | n<=1024, RHS<=64, major-storage estimate<=64MiB; O(n²)/O(n³), host status sync; no native torch.sparse.spsolve, scalable sparse factorization, higher gradients/compile/graphs/performance claim |
| Sparse CSR/BSR triangular solve | generic hipSPARSE SpSV/SpSM and block APIs | N2.13:433 public +96 direct fixtures; real/complex/indices/layouts/streams, exact pivot NaNs | Public BSR64 downcasts within tested bounds; direct raw singular policy, general sparse factorization/B02, solve autograd/compile/graphs/performance unqualified |
| SDPA math | ATen HIP + BLAS | Current forward/QKV gradients and prior long/model/AMP gates | Quadratic attention memory; default retained |
| Optional tiled SDPA | Patched scalar Triton helper, explicit D<=256 |36 A07 tests and actual2B nonzero-adapter train/checkpoint gates | BF16 D>128 training uses math-forward/tiled-backward hybrid: quadratic forward temporary, linear saved state. Dropout/compile/graphs remain unsupported; math default. |
| Gated-delta preparation | torch.nn.bc250_gated_delta; app hook preserved |42 tests, four exact autocast cases, BF16 actual2B65/1024 all372 gradients exact | Low-precision full operation uses reference-ordered GPU ATen preparation/scan. Native FP32/direct preparation remains; complete FLA/native solve/scan fusion absent. |
| Qwen3.5 frozen conv / loss | Application helper | Previous51 operator gates, real2B1024-token update and exact CLI resume | Only the inspected trainer by default; A05 |
| AMP / fused AdamW | ATen HIP + generic optimizer kernels | Current finite FP16 scaled update and CPU-matched fused AdamW; prior full AMP/resume | Valid FP32 loss promotion; no universal optimizer qualification |
| torch.compile | Dynamo/Inductor/patched Triton3.8 | N3.10 composed CNN/dense/gated/BatchNorm fullgraph output/all-gradient/CPU optimizer and GPU trace gates; prior dynamic math attention | Fused SDPA/custom tiled graphs unqualified |
| HIP graphs | Torch CUDA namespace -> HIP | Current fixed-buffer add capture/replay, previous direct graph fixtures | Does not qualify every op/allocation or custom attention |
| ROCTX / external tracing | roctracer/rocprofiler | Prior API/kernel/copy correlation evidence | Torch Kineto/SDK integration now bounded-qualified A09; counters/PC sampling remain unqualified |
| hipTensor / hipDNN / CK generic | Installed libraries, no pinned ATen integration found | Prior N1/provider/CK Tile tests; N2.18 scalar CK DL FP32 grouped forward128 +8 refusals | Direct coverage does not change Torch routes; CK backward/plugin/other types remain unqualified. Not mandatory Torch einsum/convolution dependencies |

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
| **A07 — head256 completed, bounded** | Explicit D<=256 tiled/hybrid attention |36 tests; actual2B training/checkpoint gates and exact per-backend fresh/resume CLI continuation | Math default. N3.9 restores the CLI option lost at N3.8 entry; helper remains available. BF16 hybrid lower peak/slower in measured fixtures; quadratic forward temporaries. Dropout/compile/graphs remain separate. [Report](bc250_a06_a07_results/README.md). |
| **A08 — completed, bounded** | Native dense INT8 -> INT32 Lt and Torch `_int_mm` | Libraries319a235;99 direct CPU-reference cases+11 refusals,16 Torch cases/out/streams+five refusals; overflow and above-FP32 exact-range cases pass | I32 scalar/modulo32 route, four integer epilogues. Existing Torch shape restrictions retained. Quantization/QLoRA/packed/compile integration unqualified. [Report](bc250_dense_int8_a08_results/README.md). |
| **A09 — completed, bounded** | Native Torch GPU profiler via existing Kineto/SDK | Separate USE_KINETO=1 build; seven native trace/reference/lifecycle tests and existing numerical/model/resume gates | Explicit torch.profiler contexts; CPU/GPU kernels/copies/memory correlation. Hardware counters, broad concurrency, compile/graphs and long tracing unqualified. [Report](bc250_profiler_a09_results/README.md). |
| **B01 — explicit fallback qualified** | Native BF16 FFT still rejects; opt-in bc250_fft0.1.0 installed |327 final candidate+327 installed checks: BF16-to-FP32 compute, C64/F32 outputs, rounded BF16 leaf gradients, CPU/DFT references and native GPU trace | No global API/AMP/native BF16/CPU fallback or performance claim. Wider shapes/graphs/compile/higher gradients remain separate. |
| **B02 — explicit bounded fallback qualified** | Native public general sparse solve still rejects ROCm; opt-in bc250_sparse GPU dense-LU fallback installed | B02.1 candidate/installed381 each; canonical n<=1024/RHS<=64/major dense estimate<=64MiB, first value/RHS gradients, residual/pivot/refusal/trace checks | No scalable sparse factorization/global API change; memory estimate is not total peak. [Report](bc250_sparse_b021_results/README.md). |
| **B03 — local CPU and bounded GPU readiness qualified** | Native Gloo/libuv; B03.2 host-staged GPU collectives and FP32/64 GPU DDP on one GPU/two ranks | CPU31/GPU41 references; B03.2 total108 checks, unchanged runtime | B05.3 matched rccl-single layout repair1732/24+371/5 passes on .104; daily NCCL off. [Use/rollback](bc250_rccl_b038_results/README.md). GPU send-recv/RPC/non-loopback/checkpoint/multi-device/scaling unqualified; .105 unchanged. [Report](bc250_distributed_b032_results/README.md). |
| **B04 — bounded common media installed** | Native vision/audio and CPU TorchCodec audio/video/JPEG/PNG/WebP decode plus JPEG/PNG encode | B04.5 203 candidate+203 installed reference/HIP interop checks; prior bounded scopes/seek repair retained | Optional AVIF/GIF/HEIC/GPU codecs and wider format/layout/timing coverage remain separate; no universal media claim. |
| **B05 — bounded W8 package installed on .104** | Explicit `bc250_quant 0.1.0`: frozen per-channel W8 module/recipe, GPU output/input/bias first derivatives, versioned state and FP32 scale-preserving moves | B05.6:817 candidate+817 ordinary installed checks, both daily/RCCL cohorts, actual B05.5 checkpoint compatibility, retained FFT/sparse/core/consumers. [Package/use/rollback](bc250_quant_b056_results/README.md). Native layout fixes remain in both a524 cohorts. | TorchAO/bitsandbytes/NF4/packed INT4/full QLoRA absent/unqualified; native direct backward refuses. Explicit conversion only, no model-quality/peak-memory/speed claim. Prior bounded helper timing [B05.4](bc250_quant_b054_results/README.md), not packaging performance. Broad quantization remains open; B06.1 export/CPU integration now qualified separately. |
| **B06 — bounded export/CPU runtime installed** | Modern ONNX export from CPU/gfx1013 FP32 tensors; explicit ORT CPU inference and host→Torch GPU interoperability | B06.1:181 candidate+181 installed checks; dense/CNN/gated normalization graphs, independent CPU/NumPy references, downstream HIP first gradients, retained W8/compiler/consumers. [Use/recovery](bc250_onnx_b061_results/README.md). | No ONNX GPU provider/autograd/training/W8 export claim. Legacy ROCm EP removed upstream1.23; B06.2 native-header/configure audit52 checks complete; next B06.3 strict-JIT/header readiness and budgeted graph build. Azure advertised but unqualified. |
| **B07** | MIOpen CK grouped-convolution plugin | Still absent by architecture/build capability. A03 skips expected default probe; explicit path retains diagnostics. | Optional scalar/generic plugin port; working HIP-direct/legacy reduction routes remain distinct. No kernels are supplied by a flag or diagnostic change. |
| **N2.3–12** | Native nonsymmetric eig remains an isolated bounded prototype | Native GEHRD/Q/QR/control/recovery and explicit sessions qualified within cap16/separated scope; production host LAPACK retained | Further eig expansion/optimization optional; return to broader N2 coverage after the completed A09 profiler-order repair. |

A01–A09 are completed within their recorded bounded scopes, including the later
CPU-first profiler repair. B-items remain optional workload/integration
expansions. N2.1–N2.18 are bounded milestones, not exhaustive library support.
N3.1–N3.10/B04.1–B04.5/B01.1 bounded scopes are recorded; current cursor is in flow.md; next B06.3 strict-JIT and minimal-graph readiness.
Any proposed preset still needs owned trainer qualification. Small targeted
tests cannot prove that no hidden defect exists; the evidence and retained
runners make new workload-specific probes reproducible.

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

## B06.12 practical closure (2026-10-29)

Fresh modern static FP32 CPU/GPU-origin Torch exports pass through public
`bc250-ort` in daily and RCCL profiles:112 checks/24 GPU evaluations/12 actual
GPU-only model profiles with four reference routes. No backend/default change.
[Scope and optional gaps](bc250_onnxruntime_b0612_results/README.md).
Next B07.1 audits CK grouped-convolution plugin admission before choosing a port;
existing HIP-direct/N2.18 scalar coverage is retained.

## B07.1 CK plugin admission (2026-10-29)

Existing native scalar DL FP32 factory matches the MIOpen forward interface:
nine installed archive instances,192 CPU-reference GPU executions across two
profiles and24 total checks. API5 requires66 entry points/internal C++ ABI.
Plugin remains absent; no routing/default change. [Audit and B07.2 private
prototype plan](bc250_ck_plugin_b071_results/README.md).

## B07.2 private forward plugin (2026-10-29)

Isolated FP32 DL plugin passes64 forced MIOpen solver137 GPU calls,64 absence
refusals,304 direct contracts/eight Torch output-first-gradient fixtures;38 checks.
All66 ABI exports, six privately filtered unique IDs. Installed defaults unchanged.
[Scope, failures, private candidate and B07.3](bc250_ck_plugin_b072_results/README.md).

## B07.3 private forward contracts (2026-10-29)

Private DL admission now refuses unsupported scalars/transpose/invalid geometry;
returned invokers check runtime descriptors/pointers/scratch before dispatch.
All six IDs pass512 CPU-reference GPU calls;244 API/492 runtime refusals preserve
buffers. Retained public64 and Torch gradients pass;40 final checks. No default
change or performance claim. [Scope and next paired timing](bc250_ck_plugin_b073_results/README.md).

B07.4: private public-Immediate DL default loses to HIP-direct on six bounded NCHW
FP32 fixtures (1.13–2.57× slower across both profiles/orders);30 checks pass.
No production admission. [Method/evidence](bc250_ck_plugin_b074_results/README.md).
Next B07.5 compares private per-ID configurations before packaging.

B07.5: all six private DL IDs CPU-reference checked in NCHW/NHWC. Selected
block256 configurations beat cached HIP-direct1.09–1.14× on two moderate fixtures;
four other fixtures still lose.30 checks pass; no default change.
[Method/evidence](bc250_ck_plugin_b075_results/README.md). Next B07.6 qualifies
existing Search/PerfDb and fresh-process public route replay in isolated DBs.

B07.6: genuine native Search/PerfDb selection and fresh-process public137 replay
qualified in private databases; two fixtures1.085–1.135× versus public HIP-direct.
Invalid-ID copies fall back correctly;36 checks pass. No default/source change.
[Method/evidence](bc250_ck_plugin_b076_results/README.md). Next B07.7 is an isolated
identity-pinned relocatable opt-in artifact, qualified before installation.

B07.7: exact-ABI opt-in artifact r2 is relocatable in the qualified runtime,
two extracts/daily+RCCL pass50 checks/1168 convolution references. Initial
exec-after-HIP warning preserved; same-process runner final gates clean. Not
a kernel root-cause fix or installed default. [Use/evidence](bc250_ck_plugin_b077_results/README.md).
Next B07.8 is versioned explicit installation and ordinary activation/rollback.

B07.8: explicit `bc250-ck` installed on .104 with ABI/runtime identity, non-merging
example seeding and tested unregister/register recovery.75 checks/10 guards;586
native convolution references plus16 ordinary Torch output/all-first-gradient
fixtures pass across installed/plain profiles. Native defaults unchanged.
[Use/recovery](bc250_ck_plugin_b078_usage.md), [evidence](bc250_ck_plugin_b078_results/README.md).
Next B07.9 audits existing scalar FP16 factory/accumulation before isolated proof.

B07.9: isolated native FP16 DL factories in two group layouts pass656 GPU CPU
references/928 admission refusals, float accumulator source and144 discriminator
calls (subset). Missing installed group-major factory supplied by private upstream
object; archive unchanged.56 checks pass. `bc250-ck` remains FP32-only.
[Scope/evidence](bc250_ck_plugin_b079_results/README.md). Next B07.10: private
FP16 MIOpen facade and dtype-aware invocation contracts before public admission.

B07.10: isolated matching-MIOpen FP16 forward facade at libraries742ee1e2c0e
passes724 half/580 retained float references, typed pointer/descriptor/workspace
contracts, mixed-dtype cached invokers and retained backward;66 checks/eight
guards. Installed FP32 option and defaults unchanged; no performance claim.
[Scope/evidence](bc250_ck_plugin_b0710_results/README.md). Next B07.11: paired
FP16 timing/native tuning replay before versioned packaging.

B07.11: six public FP16 fixtures/native tuning/replay qualified against an isolated
HIP-direct precision repair (gfx1013 forward FP32 accumulation). The original218
legacy half accumulator fails the strict bound; normal installed code unchanged.
CK gains1.1706–1.1782× on one fixture, one ties, four lose.444 CPU checkpoints/
80 top-level checks plus24 rebuilt-native nested checks; no global speed claim.
[Method/evidence](bc250_ck_plugin_b0711_results/README.md). Next B07.12: coherent
identity-pinned optional CK+repaired-MIOpen artifact and relocation qualification.

B07.12: coherent FP32/FP16 CK+repaired-MIOpen archive qualified in two build-free
extracts (spaces)/daily+RCCL:78 top checks/48 nested,1496 half+1176 float native
references plus586 installed float,24 Torch output/all-first-gradient fixtures.
27 actual runtime pins/eight relative ELF RUNPATHs; original candidates absent.
Installed FP32/defaults unchanged. [Artifact/use/evidence](bc250_ck_plugin_b0712_results/README.md).
Next B07.13: versioned explicit install/activation/identity/rollback under EX lock.
