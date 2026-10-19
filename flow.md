# BC-250 project memory: read first after resume or compaction

Updated 2026-10-19 (local; checkpoints explicitly use UTC). This is the
**canonical working memory**, not a release note.
Detailed hardware, implementation history, evidence and resolved defects:
[bc250_memory_inventory.md](docs/development/bc250_memory_inventory.md).
Old milestone documents are evidence snapshots; their unchecked boxes and active
release claims must not override this cursor and current manifests.

## Resume cursor

- **N2.18 completed, 2026-10-19:** TheRock qualification `746528fa6`.
  Direct installed scalar CK DL FP32 grouped forward convolution:128 independent
  CPU-double reference cases plus8 vector-shape refusals pass. Plain/Add+ReLU,
  groups1/3, four1x1/3x3 shapes, stride/dilation/asymmetric padding, M/N tails,
  valid contiguous/strided spatial layouts, exact dyadic/fixed-tolerance other
  inputs, default/private streams,256-byte guards/stride holes/input preservation.
  Max abs error4.255575e-6. Retained FP32/FP16/INT8 GEMM3/core7 pass; no production
  defect, new Torch routing/operator, library/header install or speed claim.
  Initial r1 global StreamConfig namespace compile error, r2 guard startup on
  nondefault kernel, r3 invalid packed-layout fixture all retained; corrected
  r4 passes. Four build outputs preserved, ccache20GiB/OMP4/one job/native gfx1013,
  no mapped GPU/network. Test launcher now accepts explicit retained r2/r3
  --guard-release, defaults to r3, rejects mismatch before launch, reports early
  guard failure. Original kernel guard/recovery semantics unchanged; helper backup
  WS/recovery/ck-dl-n218-before. Failed GPU fixture plus3 final guards ACK/600s;
  dmesg identical across all4 attempts,191 protected identities/new binary sealed.
  **Observed .104 already on r2 at entry:**7.1.5-bc250rocm-r2,
  boot d0532075-88fb-4a2f-974e-f61578d20d9f. No reboot/boot-policy change this batch;
  prior r3 qualifications retained, do not describe r3 as currently running.
  Active Torch dc8fea0/source2b449c4/libraries319a235d6/source-doc c29021b40e7,
  current/previous/ROCm links/runtime/defaults/four FFT AOT DBs/.105/BIOS unchanged.
  No running agent job/container/guard; free~7.16GiB. CPU Python fault remains
  unresolved/separate from fixed VM-heap/CSR and test-only failures above.
  [Report](docs/development/bc250_ck_dl_n218_results/README.md); raw
  WS/logs/ck-dl-n218-*, outputs WS/build/bc250-coverage/ck-dl-n218-r{1,2,3,4},
  local ignored kernel/artifacts/ck-dl-n218. Next preservation baseline
  WS/logs/ck-dl-n218-qualification-r4/final.json plus binary_sha256.
  **Next:** audit remaining N2 gaps against current model workloads/B01/B02/B07/
  A09 evidence, then select next bounded N3 batch; avoid repeating prior AMP/
  compile/resume/CK Tile work. N2 remains broad/open, no automatic N2.19 sweep.

- **N2.17 completed, 2026-10-18:** TheRock qualification `5826261d8`.
  Native FP32/64 stable key/value sorting and selection pass rocPRIM 656 /
  hipCUB 656 / Thrust 648 =1,960 checks, plus public Torch 62 (54 finite output/
  indices/base-gradient fixtures, eight non-finite output/index fixtures).
  Sizes 0/1/33/257/4099/65537, default/private streams, guarded outputs/workspace,
  immutable inputs/indices/flags. Exact bits preserve signed zero, smallest
  subnormals, infinities and quiet-NaN sign/payloads; exact payload indices prove
  stability. Native radix pinned encoded NaN ordering is distinct from explicit
  strict-weak NaNs-last comparison (both directions) and Torch default NaN
  ordering. Five selection policies verify stable counts/bits;16 unsupported
  partial-float-bit queries rejected. Torch sort/masked_select finite backward
  matches CPU exactly, including unused strided base storage; no NaN-gradient
  claim. No production defect/new operator/default/dispatch/speed claim.
  Initial build r1 failed only new Thrust adapter's mixed-type auto declaration;
  corrected in separate r2 outputs, failed source/logs/r1 binaries retained.
  Three standalone builds ccache20GiB/OMP4/three jobs/native gfx1013, builder
  fa3e8c254, no mapped GPU/network; no library/Torch rebuild/install. P2 INT32
  primitives138/core7 retained; six GPU gates/guards ACK/600s, dmesg unchanged
  including initial build, 181 protected hashes plus new binaries sealed.
  Active native Torch `dc8fea0`, source `2b449c4`, native libraries `319a235d6`,
  source-doc `c29021b40e7`,
  links/runtime metadata/DSOs/headers/four FFT AOT DBs/image/kernel/boot/defaults/
  .105/BIOS/r2 fallback unchanged. [Report](docs/development/bc250_sorting_n217_results/README.md)
  includes explicit contracts, case counts, source/binary/build/kernel evidence.
  Outputs WS/build/bc250-coverage/sorting-n217-r{1,2}; raw WS/logs/sorting-n217-*,
  local ignored kernel/artifacts/sorting-n217. Next preservation baseline
  WS/logs/sorting-n217-qualification-r2/final.json plus binary_sha256 for new
  test products. No running agent job/container/guard; free ~7.16GiB. CPU Python
  fault unresolved/separate from fixed VM-heap/CSR and the compile error above.
  **Next N2.18:** audit CK scalar DL admission/GEMM evidence, then one admitted
  DL convolution/grouped CPU-reference qualification and targeted build; keep
  XDL/WMMA/modern MIOpen CK plugin off, preserve defaults, do not repeat completed
  CK Tile normalization/top-k. Broad N2–N8 remain open.

- **N2.16 completed, 2026-10-18:** TheRock qualification `2db3b18c1`.
  Existing installed headers pass native rocPRIM 576 / hipCUB 480 / rocThrust 480 new
  CPU-reference checks: FP32/FP64 ordinary/segmented reduce and inclusive/
  exclusive scans, nonzero init, repeated keys, empty/uneven segments, in-place
  scans, plus 288 exact custom first-index argmax checks. Sizes 0/1/33/257/4099/
  65537, default/private streams, guarded inputs/outputs/workspace and immutable
  read-only inputs/keys/offsets. Dyadic sums exact; fixed-tolerance non-dyadic
  CPU long-double refs max error 5.72e-6/8.77e-15 FP32/64. Offset APIs include
  empty segments; Thrust reduce-by-key correctly checks only nonempty runs.
  No production defect reproduced; no new Torch operator/dispatch or speed claim.
  Three standalone native test units built ccache20GiB/OMP4/three jobs (<=16),
  -O2/C++17/gfx1013, builder fa3e8c254, no mapped GPU/network. Output
  WS/build/bc250-coverage/primitives-n216-r1; sources/builds/caches/P2 binaries
  preserved. First attempt passes, P2 INT32 primitives138/core7 retained. Five
  GPU processes/guards ACK/600s; dmesg unchanged whole batch, 168 protected hashes
  match plus new binaries separately sealed. No library/Torch/header install,
  runtime metadata/link/default/image/kernel/boot/.105/BIOS/r2 fallback change;
  four FFT AOT DBs preserved. Native Torch `dc8fea0`, source tests `2b449c4`, libraries `319a235d6`,
  libraries source-doc `c29021b40e7` retained. [Report](docs/development/bc250_primitives_n216_results/README.md)
  includes source/build commands, counts/errors/guards/protected and binary hashes.
  Raw WS/logs/primitives-n216-*; ignored local kernel/artifacts/primitives-n216.
  Next preservation baseline WS/logs/primitives-n216-qualification-r1/final.json
  plus binary_sha256 when protecting new test artifacts. No running agent job/
  container/guard; free ~7.19GiB. CPU Python fault unresolved/separate from fixed
  VM-heap/CSR. No broad precision/determinism/performance/consumer-autograd claim.
  **Next N2.17:** audit P2 sorting/selection, then bounded floating stable key/
  value sorting and selection with explicit ordering/special-value contracts,
  CPU references, guards/private streams and targeted builds. N2–N8 remain open.

- **N2.15 completed, 2026-10-18:** TheRock qualification `0e4557ee5`;
  libraries source-doc revision `c29021b40e7`, native libraries still `319a235d6`.
  No RNG arithmetic defect reproduced. Torch state/layout80, moment/range22,
  offset/lifecycle/stream/dropout-trace20 and fresh-process continuation1 pass.
  FP16/BF16/FP32/64, complex normal and INT32/64; guarded strided views, empty/
  odd/tail sizes, seeds>2^63, offsets>2^32/2^40, state rejection/legacy bytes,
  default isolation/fork exception restoration and ordered private streams.
  Five GPU kernels traced including dropout forward/backward and Philox normal;
  dropout CPU mask/scale refs pass. Mixed GPU/default/explicit+CPU+Python fresh
  continuation matches10 fields. Direct native68 pass: five pseudo/four Sobol
  variants, reset/rebinding, exact prefix relationships and CPU Philox words
  through rocRAND/hipRAND. Initial native59/60: stale header incorrectly said
  LFSR113 rejects offsets; implementation/upstream tests deliberately support
  them. Corrected comments only (C API vs C++ wrapper), added9 LFSR cases incl
  high-offset non-aliasing. MTGP32 refusal retained. Patch0032/source before/after
  hashes and failed helper/result preserved; source comments staged on .104 with
  recovery WS/recovery/rng-n215-source-before. Installed headers/DSOs/defaults/
  prefix metadata unchanged; no rebuild/cache reset. Retained RNG28/core7 pass;
  seven passing containers plus initial assertion failure, eight guards ACK/600s,
  dmesg byte-identical whole batch,153 protected identities/four AOT DBs match.
  Active Torch `dc8fea0` / source tests `2b449c4`, WS/pytorch-sparse-n213,
  current/previous/ROCm links/image/kernel/boot/.105/BIOS/r2 fallback preserved.
  [Report](docs/development/bc250_rng_n215_results/README.md) includes exact cases/
  commands/source hashes/trace/failed contract; raw WS/logs/rng-n215-* and ignored
  kernel/artifacts/rng-n215. Next preservation baseline
  WS/logs/rng-n215-qualification-r2/final.json. No running agent job/container/
  guard; free~7.21GiB. No speed/randomness-certification/universal correctness
  claim. CPU Python segfault unresolved/separate from fixed VM-heap/CSR and this
  explicit assertion. **Next N2.16:** audit P2 rocPRIM42/hipCUB48/Thrust48, then
  bounded floating/segmented reduction/scan CPU references, guards and private
  streams with targeted ccache builds. Broad N2–N8 remain open; B01 separate.

- **N2.14 completed, 2026-10-18:** TheRock qualification `44568c2e2`.
  Existing native FFT implementation passes; no production defect reproduced,
  binary/build/default/prefix changes or performance claim. Active native Torch
  `dc8fea0` / source tests `2b449c4`, libraries `319a235d6` remain installed.
  New public complex96 + real72 fixtures each pass CPU-double output/weighted
  backward, actual GPU strides/private stream, input immutability and guarded
  strided out. Prime/composite/odd, crop/pad, conjugate/broadcast, reordered and
  four-transform-dimension decomposition covered. Three independent DFT/finite
  difference/cache-reuse checks pass; trace34 kernels includes native forward/
  backward rocFFT. Direct hipFFT16 plans/32 executions pass caller workspace,
  advanced strides, batches3 and synchronized stream rebinding; raw native gaps/
  input may legally be overwritten, unlike Torch's stronger preservation contract.
  Retained FFT30/core7 pass. Initial48/96 fixture failures were non-leaf GPU
  double tensors with absent .grad; detach correction passes, no tolerance change.
  Failed run/helper preserved. Seven guards ACK/600s, six successful processes,
  dmesg byte-identical whole batch,145 protected identities and four persistent
  AOT databases unchanged. No installation/cache reset/native build/reboot;
  current Torch WS/pytorch-sparse-n213, ROCm/previous links/image/kernel/boot/
  .105/BIOS/r2 fallback/defaults preserved. [Report](docs/development/bc250_fft_n214_results/README.md)
  includes cases/commands/source hashes/trace/failure/preservation; raw
  WS/logs/fft-n214-* and ignored kernel/artifacts/fft-n214. Next preservation
  baseline WS/logs/fft-n214-qualification-r2/final.json. No running agent job/
  container/guard; free~7.22GiB. Python CPU segfault unresolved/separate from
  fixed VM-heap/CSR; test exceptions above are distinct. BF16 promotion B01,
  native in-place/planar/giant plans/callbacks/graphs/higher-order gradients and
  broad performance unqualified. **Next N2.15:** audit existing P2 RNG28 and
  checkpoint/state evidence, then bounded generator offset/state replay and
  private-stream sequencing through Torch/rocRAND with exact replay/analytic
  distribution contracts. Do not require CPU/GPU RNG bitstreams to match.
  Broad N2–N8 remain open; further tiny-eig work stays optional.

- **N2.13 completed and installed, 2026-10-18.** Native Torch `dc8fea0`
  (implementation `a434d03`), final tests `2b449c4`; TheRock build `bbf556448`,
  final controllers `39ea58574`. Public CSR generic SpSV/SpSM and BSR block solves
  qualified across four real/complex types, public 32/64-bit indices, vector/three
  RHS, upper/lower/transpose/unit flags, layouts, guarded outputs and streams.
  Initial nonsingular 256 cases passed; exact/missing pivot probes failed 16/32,
  expanded 39/80 because native division avoidance can leave finite placeholders.
  ROCm-only 2D Torch postcheck now marks exact zero/missing pivot rows as NaN.
  One GPU diagonal-search launch, actual stream, no host reads/dense conversion/
  extra workspace; bit-exact zero test, no near-zero/conditioning claim. Unit
  bypass and legacy empty-A NaNs retained. Other singular rows unspecified;
  direct native APIs unchanged; general sparse factorization/spsolve B02 separate.
  Candidate/installed each pass 433 new public groups, 96 direct generic fixtures/
  192 solves (N/T/adjoint, host/device alpha, analysis reuse), retained sparse41,
  core7 and CPU-first profiler. Final pivot group verifies real NaN writes into
  strided sentinel buffers on a private stream; test-only post-build revision is
  separate from native GIT_REVISION. BSR public int64 still downcasts within
  tested bounds; no giant native64 BSR, triangular autograd, batching or broad
  performance claim. Only libtorch_hip.so changes; CPU profiler fix retained.
  Independent WS/pytorch-sparse-n213 build uses required HIP generation then pip
  editable/no-isolation, ccache20GiB/jobs12/link1/gfx1013; old builds/sources/caches
  and shared immutable third_party retained. 22 guards ACK/600s, two numerical
  baseline failures preserved, dmesg unchanged whole batch; 121 prior identities
  verified, changed active runtime/GPU aliases pinned through recovery. Only dev
  Torch link/runtime metadata updated; ROCm DSOs/packs/current/previous/rocm links,
  image/kernel/boot/BIOS/.105/defaults/r2 fallback unchanged. Recovery
  WS/recovery/sparse-n213-before, SOURCE_SPARSE_N213.json in WS/current. Next
  preservation baseline WS/logs/sparse-n213-qualification/final.json plus its
  candidate_sha256/runtime_sha256; prior active GPU alias hashes are historical.
  [Report](docs/development/bc250_sparse_n213_results/README.md), compressed traces/
  full build log, cases/commands/source/generated hashes; raw WS/logs/sparse-n213-*,
  ignored local kernel/artifacts/sparse-n213. No active agent job/container/guard;
  free about 7.23GiB: capacity check before future builds. Python CPU segfault
  remains unresolved/separate from fixed VM-heap/CSR; these are numerical pivots.
  **Next N2.14:** audit P2 FFT30, then bounded batched/strided complex plans and
  odd-length real transforms with CPU outputs/gradients; preserve rocFFT AOT
  cache. BF16 promotion B01 remains separate. Broad N2–N8 still open.

- **A09/N4 profiler ordering repaired/installed2026-10-18:** native Torch
  b823a1b, test qualificationda45369; TheRock build669baa5d7/controller39f3ed14b.
  ROCm first Kineto init retains GPU capability when a GPU is visible, even for
  CPU-selected first capture; per-capture activity filtering remains. No visible
  GPU retains CPU-only backend. CPU-first warm/cold and GPU-first now each produce
  13 kernels/two copies per GPU window; CPU-selected captures omit GPU records
  even while GPU work runs. Exact profiled/unprofiled outputs+both gradients,
  CPU float64 refs, correlations/durations and nondefault stream pass. Hidden HIP
  device CPU capture passes without Torch CUDA lazy init (not a physical CPU-only
  host/driver-uninitialized claim). Candidate and installed each pass four orders,
  API7/core7; fresh installed reactivation also passes. Only libtorch_cpu.so changes,
  GPU arithmetic DSO byte-identical; no LoRA/training speedup claim. Independent
  WS/pytorch-profiler-order built pip-only after required out-of-place HIP generation,
  ccache20GiB/jobs12/link1/native gfx1013; old A09 build and shared immutable
  sparse-followup/third_party retained.21 guards ACK/600s,18 successful process
  gates incl expected-defect reproductions; three failed fixtures retained (small
  copy trace expectation and two SDK-invalid -1 visibility settings, corrected
  to256KiB/valid nonmatching HIP ordinal). Native patch/bounds unchanged.
  First final preservation check automatically rolled back after installed gates:
  expected old hash on intentionally switched CPU alias. Controller now pins that
  old identity through recovery; fresh reactivation/seal pass.104 protected old
  identities match; dmesg unchanged entire batch. Development torch link/runtime
  metadata updated only; ROCm DSOs/packs/current/previous/rocm links/image/kernel/
  boot/BIOS/.105/defaults unchanged; r2 fallback retained. Recovery
  WS/recovery/profiler-order-before has old link/exact runtime bytes/old Torch link;
  SOURCE_PROFILER_ORDER.json in WS/current. Next preservation baseline is
  WS/logs/profiler-order-qualification/final.json:104 old/recovery identities plus
  candidate_sha256/new runtime_sha256; old N2.12 active alias hashes are historical.
  Native GIT_REVISION=b823a1b;
  subsequent test-only qualificationda45369 recorded separately; no metadata-only
  Torch rebuild. [Report](docs/development/bc250_profiler_order_results/README.md)
  includes both-order/failed traces, source/hash/command/rollback evidence.
  Raw WS/logs/profiler-order-*, ignored kernel/artifacts/profiler-order. No running
  agent job/container/guard; free~10.13GiB. Python CPU segfault unresolved/separate
  from fixed VM-heap/CSR; explicit SDK import aborts above are distinct too.
  **Next N2.13:** audit existing generic hipSPARSE/Torch CSR/BSR triangular vector/
  multi-RHS paths; CPU-reference missing complex/index/layout/stream/zero-pivot
  cases and fix demonstrated defects. Retain P2 legacy real csrsv16; general
  sparse factorization/spsolve B02 separate. Further tiny-eig optimization optional;
  N2–N8 broad workstreams remain. Profiler counters/daemon/external concurrency/
  compile/graphs/long trace/cross-boot clocks remain N4, not claimed fixed.

- **N2.12 completed2026-10-18:** TheRock codeb1c94bddc; explicit
  right_eigenvectors(recovery_backend="device") uses one32-thread native HIP
  triangular solve, complex64/128 cap16. N2.6 reference remains default/oracle;
  all finite/unitary/separation/rescaling/unitnorm/residual gates retained.
  Warm-session profile recovery~2.47–5.19ms supported fusion.128 retained groups,
  six CPU fixtures/36 backend evaluations,32 all-order direct cases,two near-gap
  rescaling stresses,six sentinel ABI cases,six ABI refusals,four public/output
  refusals pass. Default recovery exact N2.11; real native GPU kernel traced.
  Paired A1/B1/B2/A2 recovery1.32–2.32x/whole warm candidate1.17–1.34x in six
  fixtures; recovery still~1.90–2.10ms, host full eig~0.213–0.408ms much faster
  than full native~3.75–11.79ms. Scalar reads remain18, not total sync count.
  No general/model/LoRA gain or production GPU eig integration. One-unit
  hipcc/ccache20GiB/OMP4 build only; Torch/installed ROCm DSOs/defaults/prefix
  metadata/links/images/kernel/boot/.105 unchanged.104 protected hashes match,
  dmesg unchanged entire batch; nine guards ACK/600s, eight process gates pass.
  **New separate A09/Kineto defect:** first CPU-only profile can leave later GPU
  traces empty. Initial required trace check failed and is preserved; final
  fresh-process first CPU+CUDA capture proves native kernel. Runtime defect is
  unresolved, not hidden by a claim of profiler repair. Source points to one-time
  cpuOnly profiler registration; existing A09 tested opposite ordering only.
  [N212.md](docs/development/bc250_solver_results/N212.md), n212/ evidence/hashes/
  failed+passing traces/commands; raw WS/logs/solver-*n212*, ignored local
  kernel/artifacts/solver-n212. Prior helper/oracle WS/recovery/solver-n212-before
  and schur_n211_reference.py. No active job/container/guard; disk~13GiB free.
  Python CPU segfault stays unresolved/separate from fixed VM-heap/CSR.
  **Next:** A09/N4 CPU-first→GPU profiler reproducer and reviewed initialization
  fix in separate candidate, both-order trace/numerical gates before deployment;
  then broader N2 sparse/FFT/RNG coverage triage from existing audit. N2.1–12
  bounded milestones complete; broad N2–N8 remain seven workstreams, not seven
  small tasks or a fixed percentage. Larger/batched/close spectra/balancing/
  gradients/general native eig convergence remain unqualified; further small-eig
  optimization is optional, not required for ordinary training.

- **Storage refresh completed2026-10-18**, explicit user-authorized obsolete
  generated-artifact cleanup, code780e67ceb. Recovered12,839,837,696bytes=11.96GiB,
  free1.08->13.04GiB/used100->97%. Unlinked8,801 old Torch .o/.a/.so files under
  WS/pytorch-{work,a04,sparse-followup}/build,21 old Triton bin/test compiled files,
  90 synthetic CLI adapter/state/tokenizer files from five explicit test labels,
  and removed9 stopped project build containers. Exact plan/journal preserved.
  Sources/generated sources/build metadata/directory paths/links intact; retired
  generated targets require rebuilding. Current A09 full build, shared immutable
  sparse-followup/third_party, active ROCm build/caches and all installed old/current
  DSOs retained. No package/image/global prune: all image IDs/tags and user neural-p1
  container unchanged; real user models/data/checkpoints, all recovery copies,
  source archives, releases/compat installs, .105/user src/boot fallbacks preserved.
  101 protected product/config hashes unchanged, no current/previous/runtime link
  or metadata change. Current-before/current-after/fallback-after plus current/
  fallback fresh Triton constexpr-specialization gates all exit0:8 groups each,
  40 executions incl output/gradient/SDPA/FFT/RNG/consumer imports and GPU Triton.
  Five guards ACK/600s; dmesg byte-identical across entire batch. No native build/
  cache reset/default/kernel/boot/BIOS change; CPU Python fault stays unresolved/
  separate from fixed VM-heap/CSR. [Cleanup report](docs/development/bc250_storage_cleanup_20261018/README.md)
  includes compressed exact plan/journal, final identities and gate evidence.
  Raw WS/logs/storage-refresh-20261018 and storage-refresh-*; ignored local
  kernel/artifacts/storage-refresh-20261018. No active agent job/container/guard.
  **Next remains N2.12:** warm-session recovery profile and bounded native triangular
  back-substitution; retain Python/N2.6 reference/fixed numerical+failure gates/
  cap16/production host LAPACK. Scientific code807642198/Torch4af4a15/libraries319a235
  and r3 boot e3ed129b-9c41-4d0c-ab84-5ab54720bb8f unchanged.

- **N2.11 completed2026-10-18:** TheRock code807642198; explicit caller-owned
  GehrdSession reuses bindings/native handle for one PID/host-thread/device/GPU
  stream. No global cache/default change; current-stream checked/rebound, owned
  GEHRD packed/tau copies retained, close synchronizes pinned stream then destroys
  exactly once. Initial split: create~3.13–3.18ms vs binding0.11–0.13ms/alloc-copy
  0.06–0.08ms/destroy~0.13ms.128 retained groups, six full CPU/backend fixtures,
  three lifetime groups and16 refusal assertions pass: mixed-dtype reuse,
  close-event/double-close, post-GEHRD exception cleanup, two private stream
  sessions/six interleaved submissions, thread/stream/device/closed/busy/PID guards.
  PID guard injected, not actual fork; device metadata check, not multi-GPU test.
  Default reference exact N2.10; fresh/reused H/Q bitwise equal on fixtures.
  Seven guards ACK/600s/dmesg unchanged,82 protected hashes unchanged. Paired
  A1/B1/B2/A2 three-sample preparation4.25–4.51->0.34–0.64ms (7.03–12.52x),
  whole candidate1.28–1.95x, six warm-session fixtures with N2.8 QR/device Q.
  Init3.32–3.41ms/close excluded from warm pairs; cold one-call create+close
  prep4.24–4.51ms has no such gain. Host full eig0.219–0.408ms remains faster;
  no general/model/LoRA/Torch-wide gain. N2.9 regression remains separate.
  Python-only, no native build/ccache/DSO/Torch/image/prefix metadata/kernel/
  boot/default/.105 change. New gfx1013_gehrd_session.py must accompany candidate.
  Prior helper WS/recovery/solver-n211-before; immutable schur_n210_reference.py.
  [N211.md](docs/development/bc250_solver_results/N211.md), n211/ evidence/hashes/
  trace/commands; raw WS/logs/solver-*n211*, ignored kernel/artifacts/solver-n211.
  No active agent job/container/guard; disk~1.08GiB free: check before builds.
  Python CPU fault unresolved/separate from fixed VM-heap/CSR. No broad concurrency/
  batching/gradient/compile/graph/general convergence/production GPU eig claim.
  **Next N2.12:** profile warm-session right-vector recovery and prototype bounded
  native triangular back-substitution for separated Schur forms; retain Python/
  N2.6 reference/fixed separation-rescaling-unitnorm-residual/CPU gates/cap16/
  production host LAPACK default.

- **N2.10 completed2026-10-18:** TheRock codecc8a05d3c; explicit q_backend="device"
  on isolated hessenberg/schur forms GEHRD reflector-Q with one32-thread HIP
  launch, complex64/128 cap16. Native GEHRD/owned packed copy/current-stream
  handle lifetime unchanged; default reference Q and production host LAPACK
  unchanged. Initial split at16: GEHRD~0.37–0.40ms/Q~1.87–1.91ms/total~6.2ms;
  remaining setup/teardown not yet attributed.128 retained groups,24 independent
  complex/zero/mixed CPU reflector cases+guards/nondefault streams, six full CPU
  eigenpair fixtures with both Q backends across reference/N2.8/N2.9 QR pass.
  Default reference exactly matches immutable N2.9; H exact across Q backends.
  Zero/scalar/unknown-backend controls and real GPU Q kernel trace pass. Paired
  A1/B1/B2/A2 three-sample full preparation1.11–1.39x, whole N2.8-control candidate
  1.05–1.12x in six fixtures; order16 Q diagnostic1.87->0.17/1.83->0.19ms. Not
  kernel-only/general/model/LoRA gains; host full eig~0.222–0.429ms still faster.
  N2.9 mixed timing/regression remains separate. Seven guards ACK/600s/dmesg
  unchanged,79 protected runtime/source/recovery hashes unchanged. One-unit
  hipcc/ccache20GiB/OMP4 build, no Torch/ROCm DSO/image/prefix metadata/kernel/
  boot/default/.105 mutation. Prior helper WS/recovery/solver-n210-before,
  immutable WS/helpers/schur_n29_reference.py; all prior binaries retained.
  [N210.md](docs/development/bc250_solver_results/N210.md), n210/ results/trace/
  hashes/commands, raw WS/logs/solver-*n210*, ignored kernel/artifacts/solver-n210.
  No active agent job/container/guard; disk~1.08GiB free: capacity check before
  builds. Python CPU fault unresolved/separate from fixed VM-heap/CSR. No general
  convergence/batching/gradient/compile/graph/concurrency/production GPU eig claim.
  **Next N2.11:** split remaining library binding/handle create-destroy/tensor
  allocation prep costs; implement one measured reuse/cache correction with
  explicit lifetime/current-stream gates, owned GEHRD inputs/cap16/fixed numerical/
  failure gates/Python oracle/host production eig retained.

- **N2.9 completed2026-10-18 (bounded prototype, mixed latency):** TheRock
  code13f91e094; explicit schur(qr_backend="device_chunked",chunk_size=16), valid
  chunk1..32. Reuses N2.8 control core, stops immediately on terminal/failure;
  exact total QR budget, ceil((budget+1)/chunk) launch bound, separate T/Q input
  buffers.128 retained N2.3–5 groups,74 Python-oracle control cases,100 chunk
  boundaries, four long-chunk failures, seven public option refusals and six CPU
  eigenpair fixtures pass. Results/state bitwise exact across chunks1/2/8/16/32
  in those full fixtures and repeated chunk1 boundary cases; reference unchanged.
  Default16 status copies8/20/35->1/2/3 c64,11/26/46->1/2/3 c128. Paired
  A1/B1/B2/A2 three-sample whole ratios1.034/1.046/0.984 and1.040/1.047/0.951
  at4/9/16: small gains4/9, regression16, not a general speedup. Host full eig
  ~0.224–0.426ms remains much faster. Padded-Q experiment passes but does not
  help; retained WS/recovery/solver-n29-padded, selected original/unpadded
  source/binary restored under lock and verified against passed hashes.12 guards
  ACK/600s/dmesg unchanged across both variants;76 protected runtime/source/
  recovery hashes unchanged. Two one-unit hipcc builds/ccache20GiB; no native
  Torch/ROCm DSO/image/prefix metadata/kernel/default/boot/.105 change. Prior
  source/binary WS/recovery/solver-n29-before and immutable schur_n28_reference.py
  retained. [N29.md](docs/development/bc250_solver_results/N29.md), n29/ results/
  trace/commands/final identities; raw WS/logs/solver-*n29*, ignored local
  kernel/artifacts/solver-n29. No active agent job/container/guard; disk~1.09GiB
  free: capacity check before builds. CPU Python fault unresolved/separate from
  fixed VM-heap/CSR. No general convergence/batching/gradient/compile/graph/
  production GPU eig or model/LoRA performance claim; reference and production
  host LAPACK remain defaults. **Next N2.10:** split current GEHRD/reflector-Q
  preparation cost and prototype bounded device reflector-Q accumulation,
  retaining native GEHRD/Python+N2.8+N2.9 controls/cap16/fixed gates/host default.

- **N2.8 completed2026-10-18:** TheRock code67e5d1ef8; explicit isolated
  schur(qr_backend="device_control") fuses trailing deflation, nearest/exceptional
  complex shift, reused cooperative QR/similarity and finite/128eps fill status.
  At most one QR/launch, bounded budget+1 launches; five-int D2H status/launch.
  Cap16/complex64+128/normalization/iteration budget/deflation-before-budget/
  separation/residual/refusal gates unchanged. Reference helper and production
  host-LAPACK defaults unchanged.128 retained N2.3–5 groups +74 direct oracle/
  boundary/branch/nonfinite/fill/nondefault-stream cases +six CPU eigenpair
  fixtures pass. Default reference exactly matches immutable N2.7 on six fixtures.
  First direct harness ignored nonunique QR phases near singular shifts; fixed
  phase-aligned T/Q+independent reconstruction with original bounds, no solver
  change or tolerance increase. Failed log retained. Seven guards ACK/600s/
  dmesg unchanged. Paired A1/B1/B2/A2 three-sample synchronized Schur1.80–2.73x
  and whole-candidate1.56–2.21x versus N2.7 in six fixtures; whole8.88–20.35ms
  still slower than host eig~0.23–0.42ms. No universal/model/LoRA speedup claim.
  Order16 ATen scalar reads159/207->8, but explicit D2H statuses35/47 remain;
  do not describe scalar counts as total synchronizations.70 protected runtime/
  source/recovery/metadata hashes unchanged. Targeted one-unit hipcc/ccache20GiB;
  no Torch/ROCm DSO/image/prefix metadata/kernel/boot/.105/default mutation.
  Prior source/binary WS/recovery/solver-n28-before and immutable N2.7 Python
  oracle retained. [N28.md](docs/development/bc250_solver_results/N28.md) and n28/
  contain results/hashes/trace/commands; raw WS/logs/solver-*n28*, ignored local
  kernel/artifacts/solver-n28. No running agent job/container/guard deadline;
  disk~1.09GiB free: capacity check before builds. Python CPU fault unresolved,
  separate from fixed VM-heap/CSR. No general convergence/batching/gradient/
  compile/graph/production GPU eig contract. **Next N2.9:** bounded multi-iteration
  device chunk to reduce repeated status transfers; retain exact total QR budget,
  terminal/failure semantics, Python/N2.8 oracle/fixed gates/cap16/host eig default.

- **N2.7 completed2026-10-18:** TheRock code100f43e2a; isolated native HIP
  complex64/128 Householder QR+full similarity/Q accumulation, order2..16,
  explicit schur(qr_backend="device").32-thread shared-memory block/current
  stream; default helper reference and production host-LAPACK eig unchanged.
  Normalization/shift/deflation/budget/fill/separation/residual gates unchanged.
  28 direct CPU QR/invariant/stream cases, six CPU eigenpair fixtures,128 retained
  N2.3–5 groups pass; budget0/1 partial/refusal checks and real GPU kernel trace.
  Final reference outputs/counts exactly match N2.6 on six fixtures. Initial
  correct serial prototype slower at9/16; retained in WS/recovery/solver-n27-serial.
  Paired A1/B1/B2/A2 three-sample synchronized Schur medians reference12.37–52.45ms
  ->device10.02–36.02ms (1.24–1.46x). Host full eig~0.22–0.40ms remains faster;
  no whole-eig/model/LoRA/universal speedup claim. Separate order16 device phases
  preparation~6.1ms, QR~3.38/6.41ms, other control/scaling/checks~22.46/28.24ms;
  instrumented sums include forced sync and differ from uninstrumented timing.
  Eight guards ACK/600s/dmesg unchanged;64 prior protected source/runtime hashes
  unchanged. No prefix metadata/DSO/Torch/image/kernel/default/boot/.105 mutation.
  Targeted one-unit hipcc build with ccache20GiB; no Torch build. Source/helpers/
  final hashes/report [N27.md](docs/development/bc250_solver_results/N27.md), raw
  WS/logs/solver-*n27*, ignored kernel/artifacts/solver-n27. Source saved locally
  in Git; prior N2.6 helper retained on host. No running agent job/container/guard;
  disk~1.09GiB free: capacity check before builds. Historical Python CPU fault
  remains unresolved/separate from fixed VM-heap/CSR. No batching/gradients/general
  convergence/production GPU eig qualification. **Next N2.8:** bounded device
  shift selection/deflation/status around cooperative QR to target remaining
  control overhead; retain Python oracle/fixed gates/cap16/host production eig.

- **A08 completed2026-10-18:** libraries319a235d6c6496f0da37be44dfa60eaf31a29f82;
  runtime helpers8f0dfb671e88b31bc3f5bb4a4958acefdafde556; Torch4af4a15 unchanged.
  Existing native gfx1013 Lt now supports signed INT8 A/B, INT32 C/D/scalars and
  COMPUTE_32I with unsigned modulo32 accumulation/scalar/bias arithmetic. Row/col,
  N/T, padding/offsets/strided+broadcast batches, host/device scalars and DEFAULT/
  BIAS/RELU/RELU_BIAS qualified. Integer alignment/span-alias guards added; zero
  scalar no-read behavior retained. Existing torch._int_mm works without Torch
  rebuild or launcher hook, preserving its M>16/K,N positive multiples8/out rules.
  99 direct CPU cases+11 refusals and16 public Torch/out/custom-stream cases+five
  refusals pass, including odd16,790,289 above FP32 exact range and both overflow
  directions. Profiler confirms native gemm_int8. Existing floating403+561+664
  cases pass; old floating gemm/launch token streams unchanged. Core7/profiler7/
  sparse38/conv30/A0642/A0736/actual2B3 pass;372 nonzero adapters and12 head256
  forward/recompute calls retain A06 exact BF16/A07 prior bounded agreement.
  Three five-pair eager timing fixtures versus resident FP32 GEMM+cast are mixed:
  M=N=K32/128/256 INT8~0.01473/0.01528/0.05640ms vs0.02179/0.03440/0.05550ms.
  Event envelopes include submission, baseline has an extra conversion kernel;
  no kernel-only/universal/LoRA/quantized-model speedup claim. No quantization
  recipe, packed low-bit/QLoRA, integer autograd or compile/graph integration.
  Initial probe requested unsupported public pedantic descriptor mode and failed
  before GEMM dispatch; retained, implementation explicitly supports plain32I.
  Other compute/scales/pointer vectors/nonlinear+aux epilogues remain refused.
  12 GPU guards all ACK/600s/dmesg unchanged. Final source/DSO/external kpack
  identities verified;61 protected old/current Torch/runtime/SDK/app/frozen-pair
  hashes unchanged. Paired recovery WS/recovery/dense-int8-a08-before; restore
  DSO+kpack together under exclusive prefix lock. Runtime/source GIT stamps
  updated only after qualification. SOURCE_DENSE_INT8_A08.json in WS/current.
  [Report/usage/trace](docs/development/bc250_dense_int8_a08_results/README.md),
  logs WS/logs/dense-int8-a08*, ignored kernel/artifacts/dense-int8-a08/remote-logs/.
  Targeted hipBLASLt+dist build ccache20GiB/jobs12; full Git/patch0031/verified
  bundle onfcf7a714 retained. Clang-format, six Python helper format checks pass;
  no full CI/type claim. No agent job/container/guard deadline; disk~1.10GiB
  free: capacity check before builds. Old source/build/cache/image/links/user src
  retained. No kernel/BIOS/boot/.105/default BLAS/attention/SDMA0/heap-off changes.
  Historical Python CPU fault stays unresolved/separate from fixed VM-heap/CSR.
  **Next N2.7:** re-read existing N2.6 profile/failure evidence, split Schur
  preparation/QR-loop cost and prototype one bounded device-side small QR/
  similarity iteration with residual/scale/separation/failure gates, retaining
  default host-LAPACK production eig. A01–A09 complete within recorded bounds;
  optional B-items, fuller attention/profiling/quantization contracts remain open.

- **A09 completed2026-10-18 (user selected ahead of unfinished A08):** Torch
  `4af4a1508a6ec0b7b2b29f5e0115f8477c4d1bb5` adds tests on unchanged native source
  71a0426566bc9b952d9aa9b41f4a9d79ce46e0e8; helpers
  `4c2edb78553a8fe993a209d41539eba2c5966317` (full revision in SOURCE_PROFILER_A09.json). Existing pinned
  Kineto094d3c1 SDK backend enabled by USE_KINETO=1/KINETO_BACKEND=rocm and linked
  librocprofiler-sdk.so.1. No Kineto/SDK native source repair or architecture alias
  required. Separate WS/pytorch-profiler-a09 candidate built through pip editable,
  ccache20GiB/MAX_JOBS12/link1, original object cache copied and immutable third-party
  sources bind-mounted read-only from preserved pytorch-sparse-followup/third_party.
  Rebuilds require this shared source mount; no old tree/cache/image removed.
  Seven candidate/installed profiler tests pass:13 kernels/16 runtime/13 linked CPU
  events per FP32/BF16 fixture, exact profiled/unprofiled output+both gradients and
  CPU-float64 refs, two correlated H2D/D2H copies with exact roundtrip,4MiB alloc/free,
  two scheduled8-kernel custom-stream windows, CPU-only selection and bounded wall
  overhead. Candidate/installed short-window median ratio4.70x/4.63x includes setup/
  teardown, not LoRA throughput; profiling is explicit and adds cost.
  Nine candidate gates pass: profiler7/core7/sparse38/A05=69/A06=42/A07=36/conv30/
  actual2B3/CLI2. Four installed gates pass: profiler7/core7/actual2B3/CLI2. Model
  checks retain all372 nonzero adapter grads, exact A06 BF1665/1024, bounded A07
  2.17%/1.31% with12 D256 forward/checkpoint calls. Each backend fresh/resume exact
  across adapters/optimizer/scheduler/RNG/progress; user checkpoints/data untouched.
  14 GPU guards all ACK/600s/dmesg unchanged. Initial marker test confused CPU/GPU
  annotations sharing a name; explicit CPU-category check fixed harness, no wider
  tolerance. Controller schema mismatch after successful model test corrected;
  existing passed/ACKed gates reused, failed intermediate report retained.
  Exclusive-lock current/torch -> WS/pytorch-profiler-a09, old full source/build
  pytorch-sparse-followup retained.47 protected old/runtime/SDK/app hashes unchanged;
  candidate CPU/Python DSOs change, libtorch_hip.so remains byte-identical8ffd32ee.
  Recovery WS/recovery/profiler-a09-before; SOURCE_PROFILER_A09.json in WS/current.
  [Report/traces/usage](docs/development/bc250_profiler_a09_results/README.md), logs
  WS/logs/profiler-a09-* and ignored kernel/artifacts/profiler-a09/remote-logs/.
  Build container bc250-profiler-a09-build retained stopped; no running agent job,
  GPU container or guard deadline. Disk~2.03GiB free: capacity check before builds.
  Full local Git, patch0008/test bundle on71a0426 preserve tests; feature build flag
  is in build_torch_a09.sh. Older SDPA3-file reconstruction gap retained. Nine
  targeted spin/lintrunner -a checks pass, no full CI/type claim. General and host
  BC250_TRAINING_OPS.md usage updated. No ROCm DSO/image/kernel/boot/BIOS/.105/default
  attention/SDMA0/heap-off/dense BLAS/user src change. Hardware counters, broad
  concurrency/external profiler combinations, compile/graphs, long tracing and
  cross-boot clock calibration unqualified. CPU Python segfault remains open and
  distinct from fixed VM-heap/CSR. **Exact next A08:** inspect dense INT8->INT32
  _int_mm/Lt code/audit, capacity plan and bounded native route with CPU integer,
  overflow/layout/scalar/epilogue references; then N2.7. A08 was not completed.

- **A06 precision correction + A07 head256 completed2026-10-18:** Torch
  `71a0426566bc9b952d9aa9b41f4a9d79ce46e0e8`; installed runtime helpers
  `e4f4b3854` (full revision in SOURCE_A06_A07.json). FP16/BF16/autocast full
  gated-delta now preserves reference GPU preparation/scan ordering; non-autocast
  FP32/native direct preparation retained.42 A06 tests + four exact autocast cases
  pass; actual2B training/checkpoint/nonzero-B all372 BF16 adapter gradients exact
  at65/1024. This supersedes the failed BF16 qualification below within this scope.
  A07 explicit D<=256/tile16 above128: scalar tiled FP32/FP16/inference; BF16 D>128
  training uses math-forward/native-LSE+tiled-backward hybrid.36 tests pass; actual
  model worst adapter relativeL2~2.17%/1.31%65/1024 within unchanged4%;12 D256 calls
  verify six forward + six checkpoint recomputations. Shared fixture now .train():
  earlier actual-model tests that only enabled checkpointing were eval and did not
  prove recomputation; prior operator and real CLI evidence remains separate.
  Five paired attention fixtures through4096: BF16 hybrid lower peak but slower
  (1024:27.42->32.98ms,156->112MiB extra peak). Forward temporary allocations remain
  quadratic, saved state linear. Math remains default; no universal/model speedup.
  App's independent A06 hook preserved/corrected and banner updated. Inner/outer
  trainer --attention-backend tiled is opt-in for a new output/run; default math
  preserves legacy fingerprint schema, tiled records a distinct backend. Exact
  synthetic fresh/resume continuation passes within both backends, all372 adapter
  tensors plus optimizer/scheduler/RNG/progress. User checkpoint500 unchanged;
  this does not prove bitwise continuity across old buggy code/backend changes.
  Final exclusive-lock deployment eight gates pass:42/36/four autocast/three
  actual model/two CLI backend/69 A05/seven old-attention/seven core groups.
  Five installed source hashes match,39 protected DSO/DB/kpack/frozen/consumer
  hashes unchanged.32 GPU guards across candidates/retries all ACK/600s and unchanged
  per-run dmesg; four failures retained (eval harness, pure BF16 model threshold,
  stale trainer assertion, moved chat file). A separate r2 guard refusal occurred
  before GPU execution and rolled back. Found .104 booted r2 at resume; restored
  documented one-shot r3, base SSH/watchdog/cooling and full GPU/VMM gates pass;
  boot e3ed129b-9c41-4d0c-ab84-5ab54720bb8f. r2 remains persistent fallback.
  Recovery WS/recovery/a06-a07-before-r4; earlier original/r2/r3 attempt directories
  retained. SOURCE_A06_A07.json in WS/current, logs WS/logs/a06-a07-deployment-r4
  and a06*/a07*. [Results/evidence](docs/development/bc250_a06_a07_results/README.md).
  Full local Git, patch0007 and verified incremental bundle onba6cbcd preserved;
  older SDPA three-file reconstruction gap remains. Nine targeted spin/lintrunner
  -a checks pass; no full CI/type claim. Python/Triton source-only, no native build;
  existing Triton caches retained, ccache untouched. Usage docs updated locally and
  /home/nik/finetune-n7/BC250_TRAINING_OPS.md. No BIOS/image/DSO/default BLAS/SDMA0/
  heap-off/.105/user src/data change. Historical Python CPU segfault unresolved,
  distinct from fixed VM-heap/CSR corruption. No running agent job/container/guard
  deadline. **Exact next A08:** inspect existing dense INT8->INT32 _int_mm/Lt code
  and audit evidence; implement one bounded native route with CPU integer refs,
  overflow/layout/scalar/epilogue contracts and targeted ccache build settings.
  Then A09, then N2.7; full native FLA, dropout/compile/graphs remain extensions.

- **A06 recheck2026-10-17: BF16 LoRA qualification reopened; production unchanged.**
  The original actual2B test initialized LoRA B=0, so its372 named gradients included
  zero A gradients. New seed61718/std0.001 nonzero-B fixtures exercise372 nonzero
  finite A/B gradients each. FP32/65 worst/global relativeL2~9.54e-6/3.81e-6 passes;
  BF16/65 worst4.00217%, BF16/1024 worst5.42540% exceed the unchanged4% per-adapter
  limit (global1.574%/0.870%). Do not call broad/resumed BF16 LoRA parity complete.
  40 earlier independent operator tests remain bounded evidence, not invalidated.
  Exact ATen preparation control in the same composition gives zero model-gradient
  difference at65; native-forward/exact-backward and inverse swaps support FP32
  preparation rounding/derivative-order amplification through BF16 as the next
  investigation. Restoring original ATen scan structure still fails4.002%/5.573%;
  candidate retained but NOT installed. Original ba6cbcd/module9ecb6f9 unchanged.
  Forward phase1.341ms of27.384ms(~4.9%); backward19.836ms(~72.4%);196 batched-mm
  calls unchanged. Native event-instrumented total26.594ms; coarse event envelopes,
  not kernel-only time. Five actual nonzero-B1024 pairs median7.778/7.317s, pair
  ratios0.9887–1.0756; diagnostic timing on a failed numerical gate, not deployment
  or universal gain evidence. [Review](docs/development/bc250_gated_delta_a06_review/README.md)
  contains exact outcomes, counters, failed guards and source hashes.
  Six review guards ACK/600s/dmesg unchanged53f83f7b; three retained numerical/
  instrumentation failures.40/41 prior protected hashes match. Independent app edit
  observed in finetune-n7/replica/bc250_ops.py at00:08:37UTC: eddfea40->d058b14a,
  now enables A06 by default via BC250_GATED_DELTA=1. Origin not established; edit
  preserved. Our test uses separate pinned old helper. For new ordinary training,
  use inner bc250-train launcher option --env BC250_GATED_DELTA=0 before the script;
  host shell assignment alone is not inherited. Current app integration is unqualified.
  No prefix/module/DSO/kernel/default/boot change by this review; .105/fallback and
  user src/data unchanged. Raw WS/logs/gated-delta-a06-review-* and ignored local
  kernel/artifacts/gated-delta-a06-review/. No active job/container/guard deadline.
  Exact next **A06 numerical parity**, before A07: reproduce seed61718 nonzero-B
  65/1024, investigate reference-ordered Q/K normalization/reductions/derivatives
  and decay scan rounding, keep4% model gate and independent operator refs. Full
  native solve/scan fusion and FLA remain unqualified; A07–A09/N2.7 queued. Historical
  Python CPU fault remains separate/unresolved; VM-heap/CSR corruption remains fixed.

- **A06 completed2026-10-17 (first native substage):** Torch
  ba6cbcd2fafac1a6503f1f580033fbf72ad2a4f9 / TheRock helperse61d609d3658ee984907841f4fe1dc8ecdf66baa
  installed .104. Explicit torch.nn.bc250_gated_delta fuses Q/K normalization/scaling,
  beta*K/V, padded/chunked layout and cumulative log-decay; native scalar wave32
  forward/backward covers all Q/K/V/g/beta. ATen triangular solves, GPU dense matmuls
  and sequential chunk scan remain. Not a complete FLA backend or automatic trainer
  replacement.40 candidate/installed CPU-float64 stage and independent recurrent
  reference tests pass, plus69 A05 and seven core groups. Tiny FP32/BF16 LoRA with/
  without checkpointing and actual Qwen3.5-2B BF16/rank16/checkpointed65+1024tokens
  pass all372 adapter-gradient comparisons; max relativeL2~3.23%, not bitwise equal.
  Three paired synthetic1024-token forward/backward medians7.653->7.390s(~3.6% faster)
  and peak11,976,548,352->11,745,457,152bytes(~220MiB lower). Not dataset/optimizer/
  convergence/resume or universal speed evidence. Installed four operator fixtures
  show prep1.34–1.88x/full1.02–1.12x ratios; exact protocol/sample data in
  [report](docs/development/bc250_gated_delta_a06_results/README.md).
  Twelve guards all ACK/600s/dmesg unchanged53f83f7b, including three retained
  model-adapter flag refusals.210 cached candidate/installed variants audit native
  gfx1013/wave32/no-MFMA/WMMA. Spin nine targeted static/format checks and lintrunner
  -a pass; no full-project typing/CI claim. Exclusive-lock additive module install,
  41 protected hashes unchanged; DSOs, shared consumers and original trainer/helper
  unchanged. Recovery WS/recovery/gated-delta-a06-before; SOURCE_GATED_DELTA_A06.json
  in WS/current. Module SHA9ecb6f9014bc139e54cffe0fbb2c80f782035e57a3149e3c9a6b31fe445e8b2b.
  Full local Git/incremental patch0006/bundle on6ed3d12 preserved; old SDPA3-file
  reconstruction gap remains. No C++ build/image/kernel/reboot/default change;
  existing Triton cache retained and ccache untouched. Raw WS/logs/gated-delta-a06-*
  and local ignored kernel/artifacts/gated-delta-a06/. Disk8.4GiB free. No active
  agent job/container/guard deadline. Existing LoRA command retains the reference
  warning until explicitly integrated; packed/cache/higher-order/compile/graphs and
  further native solve/scan fusion remain unqualified. Historical Python CPU fault
  remains unresolved/distinct from fixed VM-heap/CSR. .105/BIOS/r2fallback/SDMA0/
  heap-off/math/default dense BLAS/user src/data unchanged. Exact next **A07**:
  inspect current tiled SDPA D<=128 guards and head256 Qwen3.5 geometry; implement
  and qualify D256 output/QKV gradients with bounded GQA/mask/all-masked/LoRA refs,
  keeping math default. Dropout/compile/graph contracts remain separate. A07–A09
  before N2.7; K6 completed.

- **Sparse/lint follow-up completed2026-10-16:** user-requested A04/A05 gap closure
  before A06; Torch6ed3d12b184a27d21c0fa41d43269ff6bd17f1e0 / TheRock runtime
  helpers4f478cc1d6acd9aa901f011a17d7c79856dccd36 installed .104. Raw native
  FP32/FP16/BF16 compress/mm first-order gradients now cover retained slots, dense
  input, bias and scalar/vector alpha. Public FP16/BF16 conversion preserves dense
  source-weight edge; transposed row-sparse weights reconstruct on GPU for dense
  GPU backward, avoiding incorrect transpose pruning. Retained zeros and absent/
  undefined optional tensors handled. Public wrapper initialized lazily to preserve
  Torch import order. Candidate/installed38 tests pass CPU references:18 raw,
  16 public/linear rectangular/frozen/trainable,3 gradient-subset groups(5 each),
  retained-zero group. Installed20 A04/120 matmuls,69 A05, seven core and30 conv
  regressions pass.12 forward +12 training fixtures also pass CPU references.
  15 guards including retained import/optional failures ACK/600s/dmesg identical
  53f83f7b. Reversible exclusive-lock current/torch switch to WS/pytorch-sparse-followup;
  full original A05 WS/pytorch-a04 and old WS/pytorch-work preserved.31 protected
  runtime/DB/kpack/original/frozen/trainer/consumer hashes unchanged. Recovery
  WS/recovery/sparse-followup-before; SOURCE_SPARSE_FOLLOWUP.json in WS/current.
  libtorch_hip SHA8ffd32eeaa6548ef896a5f56773255a26a985398a4a7bf811463bcb1dc043b90.
  Verified lint tools initialize locally in isolated uv caches;12 targeted configured
  static/format checks and required lintrunner -a pass. Spin0.18/lintrunner0.12.7;
  PYTORCH_LINT_SKIP_CODEGEN=1 avoids separate codegen. Not full-project type/CI claim.
  A05 function/class ASTs unchanged by formatting; import order canonicalized.
  Performance measured, not a blanket speedup: prepacked eager forward dense/sparse
  0.279–1.585 across12 fixtures, compression separate; forward+backward sparse
  1.92–3.36x slower than dense in12 small fixtures. Keep ordinary dense BLAS/training
  unchanged and structured sparse explicit. GPU backward uses dense intermediates
  and metadata sync; no hardware sparse accelerator or application speedup claim.
  Raw valid floating first-order scope only; integer/higher-order/forward AD/compile/
  export/graphs/broad shapes/concurrency/stability remain unqualified. Sources,
  exact lint/timing protocol, errors and evidence in [follow-up report](docs/development/bc250_sparse_followup_results/README.md).
  Raw WS/logs/sparse-followup-* and local ignored kernel/artifacts/sparse-followup/.
  Build r1 concrete-size codegen and r2 C++ operator/ADL failures retained; r3/r4
  success, final bc250-sparse-followup-build-r4/image28ad36e6/pip-only/native/noGPU/
  ccache20GiB/jobs12/link1. Log WS/logs/sparse-followup-build-r4.log. No active agent
  build/test/container/guard deadline. Disk8.7GiB free; avoid unnecessary clones.
  Full local Git, patch0005/reverse-apply/verified bundle on6181d00 preserved; older
  SDPA3-file reconstruction gap remains. Kernel/image/boot/BIOS/.105/user src/data/
  SDMA0/heap-off/math/default rocBLAS/r2fallback unchanged. Historical Python CPU
  fault remains unresolved/distinct from fixed VM-heap/CSR. Exact next **A06**:
  inspect existing gated-delta/FLA reference/model shapes, profile bounded forward/
  backward substages, implement one scalar/wave32 stage with independent output and
  Q/K/V/gate gradients. A06–A09 before N2.7; K6 completed.

- **A05 completed2026-10-16:** Torch6181d00b5beb8cc9fd6800da7a19c8abf756b30d,
  TheRock helpers869463155599a1c836bef10fb6089cbaa511058c installed .104.
  New opt-in torch.nn.bc250_training_ops exports causal_conv and chunked_causal_loss,
  without model/version hooks. Native gfx1013 K1–4 FP32/FP16/BF16 forward/dx plus
  new deterministic FP32 partial-reduction dw/db; strided/custom-stream and autocast
  handling; unsupported valid conv falls back to ATen. Chunked loss recomputes FP32
  probabilities, validates labels/counts/kwargs; original low-precision logits remain.
  Final candidate/installed69 TestCase tests pass:48 all-gradient conv,3 no-bias,
  4 AMP/checkpoint/scaled-update,12 loss, dispatch/8-refusal group and tiny-model
  CPU loss/all-gradient plus exact AdamW/parameter/GPU-RNG serialized resume.
  Installed seven core groups and A04 sparse20/120-matmul regressions pass. Six
  GPU guards all ACK/600s/dmesg identical53f83f7b;30 protected hashes unchanged.
  Only one Python module atomically added to active WS/pytorch-a04 under exclusive
  lock; Torch/ROCm DSOs and shared/frozen consumer bundle unchanged. Recovery
  WS/recovery/training-a05-before; SOURCE_TRAINING_A05.json in WS/current.
  Old finetune-n7 trainer/bc250_ops.py remain unchanged; new API requires explicit
  import/calls. No automatic general conv/loss replacement, FLA, higher-order,
  compile/graphs/export, large-model or performance/stability claim. Fixed8x AMP
  scaling test does not qualify dynamic GradScaler overflow/growth. Existing real
  Qwen/CLI51-gate evidence belongs to the old helper, not this module. Source/hash/
  usage/evidence in [A05 report](docs/development/bc250_training_a05_results/README.md), raw logs
  WS/logs/training-a05-* and local ignored kernel/artifacts/training-a05/.
  Full Torch Git/incremental patch0004/verified bundle preserved; prior SDPA3-file
  bundle gap remains. Lint initialization still blocked, attempts retained; syntax/
  whitespace pass. No active jobs/container/guard deadline. Disk15.9GiB free.
  Kernel/image/boot/BIOS/.105/user src/data/default dense BLAS/SDMA0/heap-off/math/r2
  fallback unchanged. Historical Python CPU fault distinct from fixed VM-heap/CSR.
  Exact next **A06**: inspect recorded Qwen3.5 gated-delta reference/model shapes,
  profile bounded forward/backward substages; implement one scalar/wave32 substage
  with independent output and Q/K/V/gate gradient refs before deployment.
  A06–A09 before N2.7; K6 completed.

- **A04 completed2026-10-16:** Torch8cdfcc8e0feb3689a1ec1d292539af083b13ea94,
  TheRock helpers b5879aefa7990928270645676e66cc40d972de6b installed on .104.
  Fresh candidate configure finds the existing hipSPARSELt package; no discovery
  source workaround needed. Native gfx1013 gate/FP32, HIP Python version/search,
  truthful compiled availability, initialized search fields and pack/layout/device/
  bias/alpha guards implemented. Candidate and installed20 tests pass:120 small
  CPU-reference matmuls, independent pack decoding, FP16/BF16 public wrappers,
  search and14 refusals. Installed30 convolution cases, seven core groups and one
  tiled SDPA output/QKV-gradient smoke pass. Eight guards including retained test
  failures ACK/600s, dmesg unchanged53f83f7b.18 protected ROCm/DB/kpack/original-Torch/
  frozen-fallback hashes unchanged. Exclusive-lock atomic current/torch switch to
  WS/pytorch-a04; original WS/pytorch-work fully preserved. Recovery metadata in
  WS/recovery/torch-a04-before. SOURCE_TORCH_A04.json in WS/current; source/hash/log
  scope in [A04 report](docs/development/bc250_torch_a04_results/README.md). Native scalar same-dtype
  FP32/FP16/BF16/INT8; valid1:2/2:4 inputs, compression is not pruning; GPU scalar/
  row-vector alpha, typed/FP32 bias. Default dense rocBLAS unchanged. No sparse
  hardware speedup, sparse-autograd/compile/graphs/large-shape/stability claim.
  Retained path/CPU-alpha/version assumptions and stale-HIPify/live-script build
  failures; final r4 pip build succeeded with ccache20GiB/jobs12/link1. Lint blocked
  by missing offline initialization; attempts logged, no lint pass. Full Torch Git,
  incremental A04 patch/bundle retained; old P2 SDPA3-file bundle gap remains.
  Historical Python CPU fault distinct from fixed VM-heap/CSR. No active jobs or
  guard deadline; retained build containers stopped. Disk16.1GiB free. Kernel/image/
  boot/SDMA0/heap-off/math defaults/r2fallback/BIOS/.105/user src/data unchanged.
  Exact next **A05**: inspect existing Qwen3.5 causal-conv/loss helpers and51 prior
  operator/model/resume gates; reusable opt-in API, trainable conv gradients and
  CPU/checkpoint/AMP references. A05–A09 before N2.7; K6 completed.
- **CURRENT A03 completed2026-10-16:** TheRock source/helpers
  7d9176a45a2ae40766290b0a440c468cfcb3a164, libraries
  fcf7a7142af8c258164e106e66bf893c7f563d15, installed .104. Optional modern CK
  loader respects build flag/existing device whitelist; explicit plugin path still
  warns on missing library. No plugin kernels added (optional B07 remains).
  Genuine gfx1013_20 system FindDb16 keys/58 CPU-checked immediate solutions,
  one searched ConvHipDirectFwd PerfDb record and41 compiled kernel records with
  exact geometry/type/source/hash provenance. Unforced normal alternatives checked
  after isolated search. Read-only default-path replay15+1 configurations/58
  solutions passes with empty user FindDb/PerfDb and zero new user kernels; logged
  searched-parameter load. Unseen geometry fwd/dx/dw passes and creates3 user
  records; system payload hashes unchanged.30 Torch conv +A02 smoke3 +seven core
  groups pass; all16 GPU guards ACK/600s/dmesg same53f83f7b, including retained
  initial WAL/read-only failure. Packaging now consistent SQLite backup/DELETE
  journal; original caches/packages retained. Initial deployment precheck rejected
  Info-level empty-user-cache notices before mutation; warning-severity check fixed.
  Under exclusive runtime lock, only MIOpen DSO and3 native DBs/manifest added;
  installed SHA3a751adc149778f061198014d6f086a75c33e7640b9473dd75bf14464652afce.
  11 protected HSA/HIP/Torch/BLAS/kpack/frozen/shared-heuristic hashes unchanged.
  Recovery WS/recovery/miopen-a03-before, SOURCE_MIOPEN_A03.json in WS/current.
  Report docs/development/bc250_miopen_a03_results/README.md; raw WS/logs/miopen-a03-*
  and local ignored kernel/artifacts/miopen-a03/. Bundle WS/build/bc250-a03-db-package-r3
  and local ignored copy preserved; text assets/manifest/hashes committed.30 library
  patches exact reconstruction. Restored historical MIOpen bias/fusion inventory
  row accidentally overwritten by A02 doc edit. Prior Torch SDPA3-file patch-bundle
  gap retained; full sources/Git preserved. Small seed, no universal tuning/speedup/
  cross-board/compiler/coverage claim; not all compiled search candidates CPU checked.
  A01 clock/trace/reset/reboot and event-speed-ratio limits retained. Python CPU
  fault remains unresolved/distinct from fixed VM-heap/CSR. Torch4ee6571/Triton
  ee65ff3b/ROCr329eaf7/image5df0/kernel0957/image84995/boot a287fd16/SDMA0/heap-off/
  math/r2fallback/BIOS/.105/user src/data unchanged. No active build/test/container/
  recovery deadline. **Exact next A04:** inspect patched Torch hipSPARSELt CMake
  package discovery/version gates, targeted ccache build (12 jobs/link1), then real
  compress/sparse-mm metadata/bias/scalar/stream CPU-reference contracts. Keep
  default dense BLAS. A04–A09 before N2.7; K6 completed; don't repeat earlier work.
- **CURRENT A02 completed2026-10-16:** TheRock source/helpers
  d17427bfeff6ab7204855a753a883c9a66fbf51b; libraries
  d375d8a2ea7280170455d133058c3cf8e16f58b6 (native-profile2093090048f), installed .104.
  Original direct sums/LSTM backward failure reproduced; native bundled legacy CK
  target accepted with RDNA1 descriptor0x31014000/scalar FMA, packed-dot asm off,
  explicit XDLOPS refusal; wave32 metadata verified. Additional BF16 mapping/raw
  buffer/load conversion and compute-precision alpha/beta epilogues corrected.
  Reduction cache key version2 prevents reuse of old embedded-header kernels;
  preserves caches. Retained intermediate mapping/buffer/numeric/stale-cache
  failures, no hidden tolerance change. Final candidate69/69; installed54 numerical
  reduction cases +3 unsupported BF16 compute refusals and12 FP32/FP16 LSTM/GRU/
  tanh-RNN outputs/all gradients pass. Thread/wave/block/multiblock/scales/beta/
  guards/strided custom stream/FP32 min-max indices covered. Installed30 conv,
  32 stable positive event pairs and seven core reference groups pass.14 total
  GPU/diagnostic guards ACK/600s/dmesg identical; baseline collector0 had3 failures.
  Only MIOpen DSO atomically replaced under exclusive prefix lock with installed
  RPATH; SHA f4b1a10ce20bdbdf0e6ac90168fb50945b35cf8e26385b66f3ee387aa4507b3c.
  Eight protected HSA/HIP/Torch/BLAS/kpack/frozen-fallback hashes unchanged.
  Recovery WS/recovery/miopen-a02-before; SOURCE_MIOPEN_A02.json in WS/current.
  Report docs/development/bc250_miopen_a02_results/README.md, raw WS/logs/miopen-a02-*
  and ignored local kernel/artifacts/miopen-a02/host-logs.29 library patches exact
  reconstruction; prior Torch3-file SDPA patch-bundle gap separate/preserved.
  No performance/universal coverage claim; BF16 RNN and wider NaN/index/compute/
  recurrent contracts unqualified. A01 clock/reset/absolute trace limits and event
  speed-ratio remeasurement remain. Python CPU fault unresolved/distinct from fixed
  VM-heap/CSR. Torch4ee6571/Triton ee65ff3b/ROCr329eaf7/image5df0/kernel0957/image84995/
  boot a287fd16/SDMA0/heap-off/math/r2fallback/BIOS/.105/user src/data unchanged.
  No active agent build/test/container/recovery deadline.
  **Exact next A03:** inspect actual gfx1013_20 FindDb/PerfDb/kernel-cache records;
  bounded genuine tuning entries with device/version/shape provenance, replay and
  packaging; review capability-aware optional CK plugin probing. Missing system DB/
  plugin warnings remain A03/B07. Never fabricate DBs or rename other-GPU records.
  A03–A09 before N2.7; K6 completed; do not repeat earlier milestones.
- **CURRENT A01 completed2026-10-16:** source/helper380a403a1bc17fa561345846fb4d92a6f57c3237,
  systems329eaf726632ab0136924804cbc1321ae09fc126,
  libraries08ba6f386ff661e720873779fc417b4246787d9c; installed on .104.
  Direct C API reproduced failure independent of Torch.32/32 retained HIP event
  pairs unstable/nonpositive before; KFD GOLDEN_TSC and shader s_memrealtime
  measured approximately100MHz but ~14,360s epoch offset on this boot. ROCr's
  existing Windows offset correction extended only to native gfx1013 on Linux,
  detected when a completed timestamp lies >1s in future; mutex/other GPU paths
  preserved. Absolute CPU/GPU trace alignment, opposite epoch/reset/reboot and
  other boards unqualified. No time clamp/solver force/architecture spoof.
  Separate MIOpen obsolete dilated-GEMM workspace exclusion removed; cold query
  480bytes, warm hybrid zero-workspace solution remains valid. Candidate15 direct
  API +30 Torch conv configurations pass, installed same15+30 with fresh private
  DBs pass; f32/f16/bf16 ranks1–3/groups1–2/layout/gradient/transpose/control scopes.
  Installed32 event pairs stable/positive, six1/2/5ms raw shader-clock references
  ~10–18us event overhead/stable after100ms, seven core groups pass. No universal
  correctness/performance claim; earlier GPU-event speed ratios including N1.8
  need remeasurement, numerical and host-wall-time evidence remain separate.
  All21 diagnostic/GPU guards ACKed/600s/dmesg identical, including retained
  defect/loader/DB-permission failures and terminated extra-completion diagnostic
  137. Qualified HIP source restored/rebuilt; only HSA/MIOpen DSOs installed with
  proper RPATHs, seven protected HIP/Torch/BLAS/kpack/fallback hashes unchanged.
  Under exclusive prefix lock; rollback WS/recovery/miopen-a01-before, installed
  SOURCE_MIOPEN_A01.json; source export component revision labels updated,
  superbuild baseline7f73 retained/effective port380a recorded separately.
  Report docs/development/bc250_miopen_a01_results/README.md, raw WS/logs/miopen-a01-*
  and local ignored kernel/artifacts/miopen-a01/host-logs.27 library/four system
  patches reconstruct exactly; legacy P2 Torch patch bundle still lacks three
  later SDPA file deltas (separate packaging gap; full Torch Git/source preserved).
  Torch4ee6571/Triton ee65ff3b/image5df0/kernel0957/image84995/boot a287fd16,
  SDMA/heap-off/math/r2fallback/BIOS/.105/user src/data unchanged. No active agent
  build/test/container/deadline. Python CPU fault remains unresolved, distinct
  from fixed VM-heap/CSR and A01. Missing plugin/system DB warnings remain A03/B07.
  **Exact next A02:** inspect MIOpen bundled legacy CK reduction configuration/
  wave32 feature gates, reproduce direct reduction and LSTM backward; implement
  native gfx1013-safe plumbing or portable fallback, CPU fwd/gradient comparisons.
  A02–A09 primary queue before N2.7; do not repeat N2.1–6 or A01 baseline work.
- **CURRENT storage cleanup completed 2026-10-16**, implementation/evidence
  `a26b5621b666d08dc18257208a6a68d21b7bb21e`. User explicitly authorized obsolete
  project artifact removal; this supersedes blanket retention only for inspected
  retired items. .104 recovered31,282,634,752bytes (29.13GiB): free12.2->41.3GiB,
  used97->91%. Retired candidate priority2-20261014 and initial20261012 rocm/torch/
  cache payloads (6dirs),56 stopped project containers,2 obsolete images,33
  synthetic validation file entries and2 orphan private launcher contexts. No
  global prune/volume/package removal. Runtime metadata renamed runtime.retired.json
  beside RETIRED.json; numerical reports/logs retained, deleted test checkpoints
  require regeneration under a new helper label. User data/checkpoints untouched.
  `current` unchanged ->dev/priority1; `previous` atomically consolidated to tested
  frozen20261013-4b000d542, same as before-dev-current. This independent payload/
  image and current/python-consumers dependency remain. No active links into
  retired payloads.14 core runtime/config/DSO hashes unchanged; external .kpack
  hashes separately match N1.8/frozen sealed manifests (executed helper initially
  omitted packs via wrong relative path; retained source corrected, see report).
  Three GPU gates cleanup-{fallback-before,current-after,fallback-after}:exit0,
  seven bounded CPU-reference/contract groups each,21 group executions total;
  600s guards ACKed, dmesg byte-identical across entire cleanup. No universal
  correctness/stability claim. Source/plan/journal/results and scope exclusions:
  docs/development/bc250_storage_cleanup_20261016/README.md, matching WS/logs.
  User neural-p1 container/image unchanged. Sources/builds/ccache/real models,
  compatibility installs/links, user src/, .105, r2 boot fallback/BIOS retained.
  Local Alpine build belongs to separate gfx1103 setup with running container:
  excluded; initial unprivileged attempt failed before deleting local build files.
  Torch4ee6571/libraries a6f7d3d/Triton ee65ff3b/kernel0957/image84995/boot a287fd16,
  SDMA/heap-off/math defaults unchanged. No running agent build/test/container or
  recovery deadline. MIOpen A01/A02 and historical Python CPU fault remain open,
  distinct from fixed VM-heap/CSR. **Exact next A01:** minimal direct-MIOpen C-API
  workspace-query/invocation versus handle/stream profiling isolation, CPU fwd/dx/dw
  references; fix only demonstrated cause. Audit primary queue before N2.7.
- **CURRENT Torch support audit completed 2026-10-16**, source/evidence revision
  `7e39ed699289a60d6a9605ed22ff4493446540f2`. User asks to audit Torch/dependencies,
  identify hardware-feasible gaps and follow a shared table BEFORE N2.7. Latest
  Qwen3.5 conv/loss/resume fixes are app scoped; earlier C++1Torch diagnostic fix
  remains runtime-wide. No production dispatch/DSO/prefix/build/boot changes here.
  Canonical table docs/development/bc250_torch_support_matrix.md has stable primary
  A01–A09, optional B01–B07 IDs; work-items.json and raw evidence in
  docs/development/bc250_torch_support_audit/. All3520 dispatcher names recorded,
  registration is not support proof.21 DSOs examined/19 load; RCCL and gfx1013 CK
  grouped-conv plugin absent. No observed missing mandatory loader dependency.
 51 unique targeted configurations after retained harness corrections:35 bounded
  passes,6 reproduced MIOpen failure variants across two defect families,10
  unavailable/configured-off/refused paths. Collector exit0 is NOT all-probes-pass.
  A01 group1 FP32 conv B2/C4/L19,W6x4x3,stride2,pad2,dilation2: inference fwd passes;
  combined grad-input/weight fwd passes then bwd fails. Input-only/weight-only
  requires-grad variants fail already in fwd.480-byte workspace/provided0 and
  elapsed<=0 recur; scoped GEMM-off removes workspace warning but does NOT fix.
  A02 NEW separate LSTM fwd CPU match/bwd HIPRTC failure: bundled legacy CK
  reduction config lacks gfx1013 macro. Standalone CK Tile/optional modern CK
  grouped-conv plugin are different copies; no architecture spoof/blanket flag.
  A03 no gfx1013 FindDb/PerfDb/KDB, only3 shared heuristic files; user ufdb/ukdb
  exist. HIP20 WGP multiprocessors matches40 physical CUs; gfx1013_20 key correct.
  A04 Torch cached hipsparselt_DIR-NOTFOUND although package/DSO now installed;
  direct N1 API pass does not mean Torch integration. A05 reusable conv/loss API,
  A06 gated-delta port, A07 head256/attention contracts, A08 denseINT8 Lt (valid
 32x32x32 NOT_SUPPORTED), A09 Kineto; optional FFT BF16/sparse solve/distributed/
  ecosystem packages below. FP16 power-of-two FFT/CSC/BSR/fused AdamW/AMP/compile/
  graph pass bounded gates; ordinary native operations retained. No blanket
  all-operators/all-hidden-defects or performance claim. Harness M16 INT8, AMP
  loss dtype, CPU tiled-oracle and one syntax failure retained, not platform bugs.
  Labels torch-support-{audit,refined,conv-isolation-r2} completed with findings;
  all four batch guards ACKed/600s/dmesg identical under runtime lock, including
  syntax-failed conv-isolation. No crash/core, first batch diagnostic core cap only.
  SOURCE_TORCH_SUPPORT_AUDIT.json retains source/runtime identities. Torch4ee6571/
  libraries a6f7d3d/Triton ee65ff3b/kernel0957/image84995/boot a287fd16 retained;
  SDMA/heap-off/math defaults/r2fallback/BIOS/.105/user src/ unchanged. No active
  agent build, test, container or recovery deadline. **Exact next A01:** direct
  MIOpen C-API minimal reproduction; separate workspace size-query/invocation from
  handle/stream profiler timing, CPU fwd/dx/dw references, fix only demonstrated
  cause. A02 legacy reduction follows separately; selected primary queue before
  N2.7. Historical Python CPU fault remains distinct from fixed VM-heap/CSR and
  both reproduced MIOpen failures.
- **CURRENT Qwen3.5 training follow-up completed 2026-10-16:** user supplied an
  old log and freed .104; this supersedes the N2.7 cursor until training follow-up
  is done. Implementation `690042b55f33b4a6fd67c4c48d54ff8d7313b64a` adds scalar
  Triton frozen causal depthwise conv K<=4 (FP32/FP16/BF16 forward/dx/SiLU) and
  chunk64 FP32 causal loss with backward recomputation. Full BF16 logits remain.
  Default only in finetune-n7 replica/train.py / Transformers5.19.0 / native gfx1013;
  unsupported/trainable conv weights retain original, math SDPA/head256 retained.
  Source/patch/tests local build_tools/bc250/qwen35_training* + deploy/validate helpers;
  deployed replica/bc250_ops.py SHA eddfea40c7fb2f0fd6dcfac72d67077918d1c61730ceba60d6e3f872f5e1f6d8.
  51 final operator/reference/stream/storage/refusal checks pass;4 small LoRA pairs.
  Real text-only Qwen3.5-2B BF16 rank16/alpha32:65-token loss match,372 adapter
  gradients max relativeL2~1.12%;1024-token checkpointed backward/AdamW update pass,
  whole peak~11.15GiB without a paired full-model baseline. Isolated1024/vocab248320
  loss extra peak2.84->0.65GiB; no full-model speed/memory/convergence claim.
  Fresh CLI exposed PEFT minimized suffix target names rejected by old raw string
  comparison; now require exact resolved module set/rank/alpha.2 fresh synthetic
  updates/eval plus checkpoint1 resume:372 exact saved tensors and exact optimizer/
  scheduler/RNG/progress;4 mismatches refused. Initial CLI failure retained.
  Final labels qwen35-{ops-final,loss-memory,model-tiny,model-actual,cli-resume-fixed,
  resume-guards}:exit0/dmesg identical/600s guards ACKed under exclusive prefix lock.
  Original user log continued251/saved250; allocator warnings were recoverable
  attempts, not a demonstrated fatal OOM. Final CLI causal/MIOpen/OOM warnings absent;
  gated-delta/FLA reference warning remains valid. Global MIOpen elapsed/workspace
  defect NOT fixed; Python CPU fault unresolved, distinct from fixed VM-heap/CSR.
  Evidence docs/development/bc250_qwen35_training_results/README.md + matching WS/logs;
  SOURCE_QWEN35_TRAINING_OPS.json; WS/recovery/qwen35-training-before and local ignored
  kernel/artifacts/qwen35-training-reference retain caller originals/final sources.
  Usage updated, app opt-out documented. No native rebuild; existing Triton cache,
  Torch4ee6571/libraries a6f7d3d/kernel0957/image84995, prefix/DSOs/boot unchanged.
  SDMA/heap-off/r2 fallback/BIOS/.105/user src/ unchanged. No active agent build,
  test, container or recovery deadline. **Exact next:** profile reference gated-
  delta forward/backward at synthetic Qwen3.5 geometry and select one bounded native
  gfx1013 substage with independent output/gradient references; isolate global
  MIOpen root cause separately. N2.7 Schur iteration remains queued after this.
- **CURRENT N2.6 completed2026-10-16:** measured small candidate before changing
  code. Order16 recovery~37ms/376 ATen scalar reads versus Schur41/52ms and host
  eig~0.35/0.41ms. Vectorized right recovery across all columns, n-1 row updates
  instead n(n-1)/2; global checked gap proves active pivots, inactive pivots set1,
  per-column homogeneous scaling and final finite/norm/residual guards retained.
  No tolerance/gap/iteration changes. Scalar reads34/124/376 ->18, ATen calls
  377/1437/4349 ->226/331/478 at4/9/16. Same-input A1/B1/B2/A2,3 synchronized
  wall samples/pass, six c64/c128 order4/9/16 basis10 fixtures: recovery1.53–8.44x,
  whole1.07–1.72x; raw controls retained, no general/model speed claim. Native
  whole still16–55ms versus host0.20–0.40ms, so production host dispatch unchanged.
  Unchanged N2.4/N2.5 gates21+40+30+16=107 pass plus timing fixture CPU references.
  Labels solver-{n24,scale,boundary,failure}-regression-n26 and solver-paired-n26
  exit0/dmesg identical/600s guards ACKed under lock; initial profile-before retained.
  Report docs/development/bc250_solver_results/N26.md and n26/, .104 same WS/logs.
  Prior helper WS/recovery/solver-n26-before; no native compile or Torch/DSO/
  prefix metadata/boot-policy change, no active agent job/container/deadline.
  Torch4ee6571/libraries a6f7d3d/kernel0957/image84995; SDMA/heap-off, math defaults,
  r2/BIOS/.105 unchanged. Python CPU fault/MIOpen issue remain open. **Exact next
  N2.7:** split Schur preparation/QR-loop cost and prototype one bounded device-
  side small QR/similarity iteration for remaining launch/scalar-sync overhead;
  retain Python reference/failure gates and production host dispatch.
- **CURRENT N2.5 completed2026-10-15:** demonstrated candidate scale defect:
  seeded baseline6 had2 unit passes, tiny c64/c128 positive info4 at256steps and
  large c64/c128 nonfinite/fill errors. Added GPU component normalization before
  GEHRD/QR; restore original-unit T on success/partial result with finite check.
  Shared safe normalization preserves right-vector 8sqrt(eps) gap policy; integer
  bit checks refuse all-subnormal inputs even if floating reductions flush zero.
  Exact zero/scalar paths retained; nonrepresentable Schur output refused.
  Same seeded baseline6 now6/6; new scale40 (c64 1e-25..1e25, c128 1e-200..1e200,
  orders4/9 basis1/10), boundary30 (18accept/12refuse, also passed before), failure/
  control16 and unchanged N2.4 regression21 all pass. Max original residual
  ~4.00e-7/7.52e-16; no tolerance/iteration-limit changes or performance claim.
  Final labels solver-{scale-n25-fixed,boundary-n25-fixed,failure-n25,
  n24-regression-n25,scale-n25-replay} exit0/dmesg identical/600s guards ACKed under
  lock; failed before baseline retained separately. Report/hashes/evidence:
  docs/development/bc250_solver_results/N25.md and n25/, same .104 WS/logs labels;
  prior helper WS/recovery/solver-n25-before. No native rebuild, default Torch/
  DSO/prefix metadata/kernel/boot-policy change; no active agent job/container/
  deadline. Production eig remains host, cap16 prototype uses GPU arithmetic/
  host scalar control. General dynamic range/conditioning/balancing/subnormals/
  batching/gradients/performance unqualified; CPU fault/MIOpen issue open, SDMA/
  heap-off/math defaults/r2/BIOS/.105 unchanged. **Exact next N2.6:** bounded
  end-to-end/component cost and host-sync profile versus host eig with correctness,
  then one measured overhead-reduction step, keeping production dispatch unchanged.
- **CURRENT N2.4 completed2026-10-15, isolated native right eigenvectors:** added
  right_eigenvectors(SchurResult) to gfx1013_schur_candidate.py. GPU dimensionless
  triangular back-substitution with homogeneous rescaling, pivot/finite/residual
  checks, GPU Q transform/unit normalization. Rejects positive info, malformed
  T/Q and repeated/close scaled gaps<=8sqrt(eps); no pivot perturbation/host solve.
  Scalar zero supported; cap16/dtypes complex64/128; production eig remains host.
  New21 groups pass first run:12 end-to-end orders4/9/16 basis1/10 CPU+analytic
  spectra/vector projectors/original residuals, nondefault streams/input+Schur
  guard preservation, analytic nonnormal2x2, repeated/Jordan/close/zero refusals,
  above-floor control, zero/one-step nonconvergence, scalar and7 malformed cases.
  Max original residual~3.84e-7/7.79e-16, CPU projector~8.21e-6/2.02e-14. Label
  solver-eigvectors-n24 exit0/dmesg identical/600s guard ACKed under lock. Report:
  docs/development/bc250_solver_results/N24.md and n24/; .104 same WS/logs label.
  Prior helper retained at WS/recovery/solver-n24-before. No native rebuild or
  production Torch/DSO/prefix metadata/boot-policy change; source Torch4ee6571,
  libraries a6f7d3d and kernel0957/image84995 retained. No active agent job/container/
  deadline. Host scalar sync remains; extreme scales/balancing/conditioning,
  repeated vectors/batching/gradients/graphs/performance unqualified. CPU fault
  and MIOpen issue remain open; SDMA/heap-off, math defaults, r2/BIOS/.105 unchanged.
  **Exact next N2.5:** bounded scale ladder and spectral-separation boundary with
  nonnormal/refusal controls; fix demonstrated scaling/residual defects before
  any opt-in integration, leaving production host dispatch unchanged.
- **CURRENT N2.3 completed2026-10-15, isolated native Schur prototype:** audited
  native GEHRD/QR pieces; no pinned rocSOLVER geev/hseqr/lahqr/gees. Added
  build_tools/bc250/gfx1013_schur_candidate.py: native C/Zgehrd, device reflector-Q
  reconstruction, bounded explicit shifted GPU QR and Schur accumulation, cap16.
  Python loop/scalar convergence controls synchronize host; matrix arithmetic
  stays GPU, no CPU eig fallback. Positive info preserves partial T/Q on budget
  exhaustion; Q is Schur vectors, NOT right eigenvectors. Production eig remains
  N2.2 host compatibility; no Torch/DSO/default-prefix mutation or native rebuild.
  Initial17 and final21 groups pass; final adds ±i rotation/scalar cases and
  budget1 partial iteration. Main complex64/128 orders4/9/16 basis1/10 CPU+analytic
  spectra, Hessenberg/Schur reconstruction/orthogonality, guards/nondefault stream;
  budget0/1 nonconvergence, repeated triangular and9 argument rejections checked.
  Max Schur reconstruction~1.35e-6/2.90e-15; no speed/general convergence claim.
  Labels solver-schur-n23{,-final} exit0/dmesg identical/600s guards ACKed under
  lock. Evidence docs/development/bc250_solver_results/N23.md and n23/, same
  .104 WS/logs labels; Torch4ee6571/libraries a6f7d3d/kernel0957/image84995 unchanged.
  No active agent job/container/deadline; balancing/extreme scaling/larger cases,
  vectors/batching/gradients/concurrency remain unqualified. Python CPU fault and
  MIOpen issue unresolved; SDMA/heap off, math defaults, r2/BIOS/.105 unchanged.
  **Exact next N2.4:** guarded triangular back-substitution and GPU Q transform
  for right eigenvectors from converged small Schur forms; separated-spectrum
  CPU/analytic residual/direction checks and explicit repeated/near-singular/
  nonconverged refusal, keeping production dispatch unchanged.
- **CURRENT N2.2 completed2026-10-15, host compatibility ONLY:** pinned Torch
  complex nonsymmetric eig selects hipSOLVER DnXgeev, whose implementation copies
  to host and calls LAPACK cgeev/zgeev, then copies outputs to GPU. This is NOT a
  native GPU eigensolver. Native N2.1 Hermitian eigh remains a separate result.
  New16 groups pass first run: complex64/128 batch2 dimensions9/33/65, known
  separated complex spectra with basis condition1/20, padded/transposed guards,
  nondefault streams, eig/eigvals CPU+analytic spectra, residual/projector checks,
  two eigenvalue-gradient groups and two empty-shape groups. Max residual
  c64~3.17e-7/c128~6.72e-16. Label solver-eig-n22 exit0/dmesg identical/600s guard
  ACKed under lock. Report: docs/development/bc250_solver_results/N22.md; mapped
  hipSOLVER/OpenBLAS hashes, runtime identity, logs/results retained locally and
  .104 WS/logs. No backend defect reproduced, code/selection/runtime patch or
  native rebuild. Torch4ee6571, libraries a6f7d3d, kernel0957/image84995 unchanged.
  No active agent job/container/deadline. Native nonsymmetric eig remains open;
  no performance/universal correctness claim. Python CPU fault/MIOpen failure
  unresolved; SDMA/heap off, math defaults, r2/BIOS/.105 unchanged.
  **Exact next N2.3:** audit native Hessenberg/QR/Schur primitives and select the
  smallest reviewed GPU implementation/convergence step; retain host compatibility
  until independent native eigenpair and failure-contract gates pass.
- **CURRENT N2.1 completed2026-10-15:** audited existing P2 solver32 and pinned
  Torch4ee6571/ROCm-libraries a6f7d3d consumers; qualified existing native complex
  Hermitian eigh eigenvectors rather than adding a duplicate backend. New22
  checks: complex64/128, batch2 sizes17/65/129/257, both triangles, padded strides/
  guard preservation/ignored-triangle NaNs, nondefault stream, size33 repeated
  eigenspace projectors and two separated-spectrum CPU gradient comparisons.
  Source default HETRD blocked switch256;257 covered, two-stage11000 unqualified.
  Max residual c64~1.15e-6/c128~2.16e-15; no speed/stability/universal claim.
  Existing32 solver regression passes unchanged. Final labels solver-eigh-n21-blocked
  and solver-p2-regression-n21 exit0/dmesg identical/600s guards ACKed under lock.
  Earlier18-case gate passes too; initial reference-check broadcasting failure
  retained, corrected without tolerance changes, not a backend defect. No native
  implementation/selection/runtime change or rebuild. Report/evidence:
  docs/development/bc250_solver_results/README.md; local helpers and matching
  .104 WS/logs. Existing image/module/Torch/DSOs, SDMA/heap-off, math defaults,
  r2 fallback/BIOS/.105 unchanged; Python CPU fault and MIOpen failure unresolved.
  No active agent job/container/deadline. **Exact next N2.2:** audit nonsymmetric
  eig coverage and pinned dispatch, qualify one bounded complex eigenpair gap
  with CPU/residual references, and change backend only for demonstrated failure.
- **CURRENT K6 handoff2026-10-15:** user explicitly resumed K6, superseding the
  earlier deferral below. K6 complete for the recorded bounded scope; no runtime,
  kernel code, power-policy or boot change. `kernel/r3/handoff/README.md` contains
  exact configs, reviewed six-patch series, build/sign/install safeguards and
  one-shot SSH/watchdog/cooling/GPU/recovery procedure. Temporary-index replay
  exactly matches image84995f1 tree after2 patches and module0957f1 tree after6.
  Local/current source archive+fan source and21 artifacts have matching local/
  .104 hashes; manifests/evidence under `kernel/r3/handoff/`, ignored artifacts
  under `kernel/artifacts/k6-7.2.10/`, .104 KWS/handoff/k6/{artifacts,docs}.
  Signed upstream verification retained; archive excludes generated module signing keys.
  Read-only .104 audit and fresh loaded identity check pass; same boot
  a287fd16-c97b-4d88-8055-b0553a2f7631, image84995f1/module0957f1, active watchdog/
  SSH/cooling, empty next_entry, r2 default. No new build/GPU test/reboot; prior
  bounded VMM/SVM/HPD/consumer evidence retained, no bit-identical build claim.
  Telemetry/per-core fidelity, scheduler-stop/resume and hardware-hang limits
  remain; Python CPU fault and separate MIOpen failure remain unresolved. SDMA/
  experimental heap off, math attention default, BIOS/.105/fallback unchanged.
  No active agent job/container/deadline. **Exact next: N2.1** audit existing32
  solver cases and pinned Torch/rocSOLVER/hipSOLVER consumers; choose one missing
  difficult/complex operation with CPU/residual references before implementing.
- **Historical user-priorities1–4 batch2026-10-15:** user initially deferred N2/K6.
  Priorities1/2/4 bounded-qualified; priority3 investigated/instrumented, NOT fixed.
  Torch`4ee65717d69e9f51021272e3f37d0f342f68fc08` (optimization43cb299 ancestor).
  Longer CPU/GPU attention output+gradient references:11 cases through4096tokens,
  head128/GQA. Real existing Qwen3-1.7B BF16 rank4 q/v LoRA reference passes at512/
  1024 synthetic tokens; max per-adapter gradient relativeL2~0.041 versus math.
  No user data/checkpoint changes or convergence/full-model fine-tuning claim.
- **Priority2:** causal row/key bounds skip mathematically masked tiles; tile32
  only causal heads>=64, other tile16 paths retained. Final same-input paired BF16
  head128 causal kernel ratios1.67–2.05 versus prior implementation in measured
  fixtures; FP32/control noise retained, no universal/whole-model speedup claim.
  Real Qwen1024 whole-model peak remains5,818,112,000 allocated bytes for both;
  isolated attention memory savings do not eliminate logits/loss-dominated peaks.
  Math stays default. Autocast now handles eligible mixed Q/K/V dtypes with gradients.
- **Priority3 unresolved:** recovered ONE historical Python IP0x3b3e6 event (many
  later logs duplicate it), no retained original core/stack. Eight fresh debug-
  allocator solver/FFT/CSR/COO/GEMM processes pass1920 checks; not a crash repair.
  Launcher defaults PYTHONFAULTHANDLER=1; empty-value user opt-out checked. Diagnostic
  core limit256MiB only, no global core-policy change. Distinct newly observed
  mixed-operator failure: MIOpen EvaluateInvokers elapsed<=0 / GemmFwdRest workspace
  480/provided0, before first conv/sparse; exit1, no segfault/dmesg delta. Retained
  and unresolved; do not claim original full55 operator suite passed this batch.
- **Priority4:** actual train_lora resume RNG ByteTensor failure reproduced and
  repaired; adapters now trainable on resume, scheduler/Python/data-order state
  saved, legacy format resumes/updates with missing-state replay warnings. New
  matching-config tiny CLI fresh/resume passed, old-format fixture passed too.
  Full-parameter small Qwen AMP batch2/length128 FP32/FP16 math and length256 BF16
  math/tiled: CPU gradients,2updates/no scaler skips, fresh-process exact next
  loss/model/optimizer/scheduler/scaler+CPU/GPU/Python RNG probes pass. Compiled
  fullgraph/dynamic math-attention block:6 FP32/FP16/BF16 output/gradient cases at
  batches2/3 lengths128/96 pass. Tiled compile/graphs/dropout remain unqualified.
  False TF32 Tensor Core warning corrected using native is_tf32_supported API.
- **Evidence/state/EXACT NEXT:** docs/development/bc250_training_expansion_results/README.md,
  reports/logs and build_tools/bc250 helpers/caller patch; .104 WS/logs matching
  labels. Final attention/AMP/compile/app/legacy/sparse labels exit0,600s guards
  ACKed,dmesg identical; failed mixed/initial AMP/app attempts preserved. Python-only
  changes reuse existing caches; no new native build. Recovery WS/recovery/
  {attention-priorities,training-expansion,tf32-diagnostic,training-expansion-metadata}-before;
  SOURCE_TRAINING_EXPANSION records revisions/hashes. No agent job/container/deadline
  remains; source/kernel0957f1aa4, r3/r2fallback, BIOS/governor/.105 unchanged.
  Next isolate the newly demonstrated small-convolution timing/workspace failure
  and capture a real CPU crash/core/backtrace before proposing its fix. Keep
  historical CPU fault separate from proven CSR corruption/safe-pool bypass and
  kernel VMM/SVM repairs. N2/K6 remain deferred, not silently restarted.
- **Usage-documentation follow-up2026-10-15:** added dated attention instructions
  to docs/development/bc250_torch_usage.md, including default warning/API fixes,
  math default, application-only --attention-backend tiled selection, Transformer
  registration and qualification limits. Linked detailed evidence; no code/runtime
  change or GPU rerun. No agent jobs; attention/K6/N2.1 next actions unchanged.
- **Previous N3 attention batch2026-10-15, bounded opt-in:**
  patched Torch`bda0831ba49ef9d0ba2968a800642e5abaec5dd5`. Three native scalar
  Triton kernels implement tiled online-softmax forward/recomputed backward;
  no full score/probability matrix, matrix instructions or default SDPA override.
  FP32/FP16/BF16, GQA, broadcast boolean/additive masks, zero fully masked rows,
  finite Q/K/V, D<=128/L<=4096, eager first-order only. No dropout/mask gradients/
  compile/graph or full-model qualification. Explicit Transformer attention+mask
  registration and caller --attention-backend tiled; math remains default.
- **Warning/caller corrections:** built-in C++ probe passed !debug as warning text,
  producing1Torch during successful fallback. Now ordinary checks are quiet;
  explicit unsupported-backend diagnostics and failure retained. This does not
  enable built-in efficient/flash/CK backends. User finetune-n7 train/chat migrated
  to nn.attention.sdpa_kernel(MATH), backward included for checkpoint recomputation,
  broad warning filters removed. Reviewed caller patch/local helpers retained.
- **Attention evidence:** `docs/development/bc250_attention_results/README.md`;
  .104 WS/logs/sdpa-tiled-candidate-r5:18 CPU-float64/GPU-math output+QKV-gradient
  cases, padded small Llama/checkpoint test. Installed integration-r4:7 groups,
  actual Qwen1.7B head128/Hq16/Hkv8 geometry at65tokens; small random Qwen3 BF16
  LoRA2updates/CPU gradients/cache decode; diagnostics/streams/caller checks.
  Unchanged19-case eager Torch regression passes. Final batches exit0/dmesg
  unchanged/600s guards ACKed. Earlier harness/reference/install failures preserved.
  Extra allocated FP32 B1/H2/D32 forward+backward peak:512/1024/2048 tokens
  tiled0.63/1.26/2.52MiB vs math8.38/32.75/129.50MiB. Tiled slower in these one-event
  timings; no speedup/general memory/stability claim. Dense input masks can be quadratic.
- **Attention deployment/job status:** exclusive update_dev_prefix lock used for
  source/library/caller mutations and direct-Podman tests; ccache20GiB/12Torchjobs.
  HIP SDP-utils+regenerated HIPHooks TUs and libtorch_hip rebuilt; installed DSO
  SHA578a3583e78ad0009779d77bd65591fcf8119dec441ea81dde7be56d9b9c11d5.
  New module SHA227a68174927397cf01132421f7d5c7dd7c2a65ba05bbc4babc651addde2bbc3.
  Recovery WS/recovery/sdpa-{probe,callers,metadata}-before; SOURCE_SDPA records
  source/revision hashes. Existing images/releases/checkpoints/caches retained.
  User's training/chat exited normally before tests; no agent job/container/deadline
  remains. Kernel0957f1aa4/r3/r2default/governor/BIOS/.105 unchanged.
- **Exact next:** attention scope is opt-in only; qualify longer head128 references
  and representative full-model memory/time/training before default integration,
  then optimize the demonstrated scalar bottleneck. K6/N2.1 remain independent
  roadmap items; telemetry correction optional for current use. Unresolved Python
  CPU segfault remains separate from fixed VM-heap corruption and VMM/SVM repairs.
- **ACTIVE: K1–K6 kernel workstream; K5/SVM passed; HPD fixed; K6 remains; telemetry correction optional.**
  2026-10-15: .104 now runs`7.2.10-bc250rocm-r3` after six one-shot boots;
  persistent GRUB default remains`bc250-rocm-r2-kernel`, next_entry empty. Native
  GPU loaded,40CU/80SIMD, wave32/KFD100103; live quirks3/0/3. No active build,
  GPU gate, container or unacknowledged recovery deadline at batch end. User's
  training/download finished; user explicitly returned.104 to development use.
  .105 unchanged. Root orchestration SSH session9359 is idle, not durable state.
- **CURRENT HPD/Display Core milestone2026-10-15:** module/source
  `0957f1aa4717bb5ea19545bdcb01d6899e94bb9f`, patch0006. Init incorrectly called
  all6 HPD+6 HPDRX enums, but active DCN201 table implements only two pairs. Query
  active table before bulk clears; skip dummy/empty entries, retain actual clears,
  connector enablement and invalid-call ASSERT/error diagnostics. No dc=0 or
  guessed pin counts. Baseline8 dummy-ack+8 dummy-set+8 failed-clear errors and
  two IRQ WARNs →0 each after guarded reboot/load. Temporary DRM KMS debug4
  traced exactly sources1,2,7,8; restored0. DCN2.0.1 initialized, amdgpu.dc=-1,
  same DP-1/DP-2 disconnected connectors, taint4160 with WARN bit clear. Physical
  hotplug/scanout/audio/resume and other ASICs unqualified. Existing2 SDMA0 fence
  fallback-expiry messages/workqueue-name truncation remain; no wholly clean-boot
  or SDMA qualification claim.
- **Current identity/evidence:** r3-only signed updates/amdgpu.ko.gz SHA
  `6ebd8393a9c7d25cd810fd89d4ef5defa187980cae64da0d37ee0d975715e8d1`,
  build/installed/loaded ID`b97bca56793aecea35967385d0c68d4186ecd121`.
  Kernel image84995f1/buildID57f7c52/initramfs/firmware unchanged. Prior source/
  module independent`KWS/recovery/hpd-before`; r2/default/.105 retained. Boot
  `a287fd16-c97b-4d88-8055-b0553a2f7631`; base/cooling and360s GPU guard ACKed.
  Unchanged Lt664/SparseLt150/hipTensor316/Torch19 and two fresh memory processes
  each24 allocation+16 VMM rounds pass, workload dmesg delta0. Existing SVM source
  hashes preserved. Host`SOURCE_HPD.json`, source/GIT_REVISION0957f1aa4;
  repository`kernel/r3/evidence/hpd-capability/`, `hpd-*` helpers/manifest/README.
  PID6463 modules build ended0,16jobs/20GiB ccache; public DC header dependency
  rebuild,`logs/hpd-modules-build.log`, `hpd-module-install.log`. No active build,
  agent-owned test/container or deadline. DRM debug/injection parameter both0;
  external governor now recorded in the read-only audit below.
- **CPU unlock persistence request2026-10-15:** read-only .104 check confirms
  eight cores/16threads online and OpenRC default unlock service enabled/started.
  Existing script reapplies volatile mask, deliberately never auto-reboots (prior
  bootloop described in its comments); cold-start enumeration needs a later warm
  reboot, not newly cold-cycle-tested. Requested upstream uses same class of
  volatile unlock and points to BIOS DXE for cold-boot persistence. No duplicate
  mailbox unlock, reboot, service/policy change or firmware flash; .105 unchanged.
  Evidence/local script+service snapshots `kernel/r3/evidence/core-unlock-audit/`.
  User explicitly chose2026-10-15 to leave BIOS untouched and retain the existing
  service/warm-reboot arrangement. Firmware modification is out of current scope;
  r2/SSH cannot rescue pre-OS failure. DMI P3.00 does not prove stock BIOS.
  No active jobs. K6/N2.1 retained; telemetry correction is optional for current use.
- **CURRENT read-only telemetry audit2026-10-15 completed:** production
  kernel/module0957f1aa4/image84995f1/prefix/boot unchanged, same boot above.
  Newly observed external governor PID5265 (`cyan-skillfish-governor-smu`) has
  fix-metrics/fix-freq enabled and bind-mounts tmpfs over GPU metrics AND hwmon
  freq1_input. Kernel policy auto does not imply stock clocks. Existing governor/
  overlays/config/clock/voltage/fan policy untouched. amdgpu_top PID5728 was also
  present initially, absent later; not stopped by this audit. No agent-owned jobs
  remain. GuardPID7594 ACKed/ended; rootSSH9359 idle. No kernel build/reboot.
- **Passed collection/reference:** original governor-held sysfs fd6 (device0:22,
  overlay0:25) read without unmounting; kernel-translated v2.2, NOT raw PMFW.
 84 samples~5Hz:4s idle,8s native integer GPU workload plus startup,4s cooldown;
 131,072 exact CPU-reference checks, kernel logs byte-identical. During own load,
  governor-supplied clock1000–2000MHz (median2000) versus original kernel/DRM/
  DPM mostly2–7MHz, max47; average kernel456–953MHz also untrusted. Overlay
  clock views agreed84/84; shared kernel clock views agreed74/84 sequential
  samples, NOT independent clock validation. Actual attained clock unresolved.
  Physical8core/16thread topology confirmed; per-core temperatures0/2 report
 0.63–1.88/0.62–3.56°C,1/3 zero,6/7 temperature/clock65535. Original GPU activity
 65535 and DRM load errno22; overlay utilization is governor-derived.
- **Temperatures/provenance:** reported edge51→68°C, original GFX51.25→68.25°C,
  CPU Tctl52.625→67.125°C in this short load. Hwmon edge truncates centidegrees
  to integer°C (observed difference−0.75..0°C). Fan actually reads CPU Tctl;
  existing automatic PWM90→175/RPM~2k→3.5k, remaining high during CPU cooldown.
  No absolute calibration/max-thermal claim. Running governor binarySHA00cd65a6...
  differs from retained target/releasee8028ca2...; reference Git240b0df/Cargo0.4.0
  lacks observed frequency-overlay code. Running binary/source NOT qualified.
  Reference archive`kernel/artifacts/telemetry-readonly/governor-reference-240b0df.tar`
  retained locally (ignored artifact/SHA in evidence); original repo/history intact.
- **Evidence/EXACT NEXT:** `kernel/r3/telemetry/README.md`, sampler/load/summarizer;
  `kernel/r3/evidence/telemetry-readonly/` raw JSONL/config/script/identity/results.
  Host`KWS/logs/telemetry-readonly`, `telemetry-compile.log`, `telemetry-guard.log`;
  one-TU gfx1013 probe compiled with20GiB ccache, binary`KWS/telemetry-reference-load`.
  Pin running governor source/overlay semantics and kernel/daemon SMU mailbox
  coordination, THEN minimal guarded read-only GetGfxFrequency getter validation.
  Do not blindly apply legacy packaging0005: physical core count alone does not
  prove the hybrid map; instantaneous power is labeled average; query failure
  falls back to the suspect clock slot. Keep per-core remapping/power semantics
  separate until offsets/units proven. K6 then N2.1 remain; SDMA0/heap-off, r2/
  .105 and prior SVM/fault/lockdep/physical-display limits retained. Python CPU
  segfault separate/unresolved. No speedup/sustained claim from this8-second test.
- **Historical SVM validation milestone2026-10-15:** source/module
  `41db89c103b183c82c05bbe099ea96ad1ddee8d1`, patch0005. Map AND validation-path
  unmap collect PT fences/site bits under BO reservations, then release reservations,
  wait and invalidate under notifier lock; caller-held mmap/SVM/migrate locks
  preserve lifetime. Drain earlier completed work even when a later VMA fails.
  Fence-tracking allocation failure falls back to the prior in-place wait, without
  DQM invalidation under BO reservations. No OOM/fence-error injection claim.
  Also fixed validation PROT_NONE byte/page bounds using each VMA's inclusive
  page indices: exact same probe traced0 target-page unmaps before,2 after;
  no GPU access to protected page attempted; restore references passed.
- **Previous SVM validation identity/results (superseded above):** r3-only signed updates/amdgpu.ko.gz SHA
  `d47974d6d7eef898b50a4fdf03edb716e8cd13dfd7f2373b495f273727e5f07f`,
  build/installed/loaded ID`7ee5573fc124ceb0edad1bc662feb44ebdbf129f`.
  Kernel IMAGE84995f1/buildID57f7c52, image/initramfs/firmware unchanged. Prior
  module/source independently retained`KWS/recovery/svm-map-before`; original
  stock/K5/SVM modules and r2/default/.105 preserved. Boot
  `84debfc4-5f83-413b-9656-6a5fd3140a3a`; base/cooling and360s GPU guard ACKed.
  Two concurrent processes each54 GPU CPU-reference checks, including24 multi-VMA
  updates+24 CPU discard/restores, passed with unchanged workload dmesg. Both
  partial/full CPU-unmap -EIO regressions pass again; unchanged full Lt664/
  SparseLt150/hipTensor316/Torch19 and two memory processes each24 allocation+
  16 VMM rounds pass, delta0. Two intentional trace+two injection lines accounted
  separately. Dynamic debug restored; one-shot injection parameter verified0.
- **Previous validation jobs/evidence/next (HPD now fixed above):** PID7547 modules build ended0,16jobs/20GiB ccache;
  `logs/svm-map-modules-build.log`, `svm-map-module-install.log`. No active build,
  test/container/GPU user or deadline. Host`SOURCE_SVM_MAP.json`, source/GIT_REVISION
 41db89c10; helpers`svm-map-*`, repository`kernel/r3/evidence/svm-map-deferred/`,
  patch0005 and source manifest record replay/identity. No lockdep/universal deadlock,
  OOM/fence-error/partial-map-failure lifetime, failed scheduler-stop/resume,
  XNACK/retry-fault or multi-GPU qualification; no performance/sustained claim.
  Next analyze the two existing Display Core IRQ load WARN call chains/tables;
  preserve warning evidence and avoid blindly suppressing WARNs/disabling display.
  Then trusted telemetry/K6; N2.1 next library item. Retain r2 default, SDMA0 and
  experimental heap off; separate Python CPU segfault remains unresolved.
- **Historical SVM CPU-unmap milestone2026-10-15:** module/source
  `67d0f189570c831efa17b524c080de07bbec485a`, patch0004. Single-device BC-250
  CPU-unmap notifier acquires queue eviction BEFORE PT invalidation; successful
  invalidation balances it, returned errors after a confirmed stop retain it
  until process exit. Full-range mapped_to_gpu clears only after success. Existing
  opt-in unmap bit controls this guard; other ASIC/multi-device behavior unchanged.
  Root-only0600 one-shot`bc250_svm_unmap_fail_pid` defaults0; initial-namespace TGID
  must match and queues must have successfully quiesced before consumption.
- **Previous CPU-unmap module/boot (superseded above):** signed r3-only updates/amdgpu.ko.gz SHA
  `c1d0e5a887bc3d2019472b9758cf57c920e9ef9544b3b47f6ed1a2d620f5e868`,
  build/installed/loaded ID`c99a2c6dc9074760efe06d074ec0248ce5e8739b`;
  kernel image still84995f1/buildID57f7c52, image/initramfs unchanged. Previous
  K5 module independent copy`KWS/recovery/svm-before/amdgpu.ko.gz`, original stock
  r3 and r2/default retained. New boot`eb2cc9df-5df6-4250-a0ec-1af4de205f93`;
  base/cooling ACK,360s SVM guard and180s full-injection guard ACKs completed.
  Source/GIT_REVISION67d0f1895;`SOURCE_SVM.json` records image/module delta.
  Pid7101 incremental modules build finished0,16jobs/20GiB ccache;
  `logs/svm-modules-build.log`, `svm-module-install.log`. No active build/test/
  container/GPU user or recovery deadline. Injection parameter verified0.
- **SVM evidence:** initial six GPU CPU-reference checks passed on K5 module.
  New module: partial AND full CPU-unmap injection gates passed. Each target
  process completed five prior SVM references and a normal GPU-write control,
  then received one modeled -EIO at the unmap runlist point; PID matched, hook
  consumed, exactly one expected error log. Queued GPU write remained unexecuted
  through200 one-millisecond host polling samples, process exited0; six-check
  fresh-process SVM gates passed before/after each injection. No new ordinary
  kernel lines. Unchanged full Lt664/SparseLt150/hipTensor316/Torch19 gate passed,
  two memory processes each24 allocation/pinned/stream/event+16 VMM rounds,
  workload dmesg delta0. Intentional error lines retained separately, not hidden.
  Host`logs/svm-{baseline,cpu-unmap-gate,cpu-unmap-full-injection-gate,
  cpu-unmap-full}`, repository`kernel/r3/evidence/svm-cpu-unmap/`; new
  `svm-{reference.cpp,validate-on-host.py,module-on-host.sh,gpu-phase-on-host.sh,
  identity-on-host.py,source-manifest.json}` provide replay/provenance.
- **CPU-unmap scope/previous next (validation locking superseded above):** qualify only returned CPU-unmap errors AFTER confirmed
  quiescence. Failed scheduler stops cannot veto blockable MMU reclamation by
  returning false (core warns/continues); hardware HWS hangs, resume failures,
  retry-fault migration/XNACK and multi-GPU remain unqualified. Audit SVM MAP
  runlist flush under BO reservations and move post-PT invalidation outside those
  reservations while preserving notifier/range lifetime ordering, then short
  concurrent CPU-compared GPU gates. Validation-context flushing is now moved outside reservations above; other
  SVM callers and failure paths are not universally certified lock-order safe. Follow with Display Core warning analysis,
  telemetry and K6; N2.1 remains next library item. Keep r2 default, SDMA0 and
  experimental heap off. Unknown Python CPU segfault remains separate/unresolved.
- **Historical K5 milestone2026-10-15:** explicit fixed-VA VMM stale translations repaired
  on .104. Kernel source/module revision`8f3c0d352643378629e5a73f4b9df96b7d38d94a`,
  patch0003; kernel IMAGE remains84995f1, same config/image/initramfs/firmware.
  HIP and direct ROCr isolation each failed8/26 checks before repair, while
  distinct-VA anchors passed; ordinary KFD mapping churn made both26/26 pass.
  GEM_VA bypassed existing KFD map/unmap runlist invalidation. New BC-250 compute
  path waits PT completion, releases BO reservations before DQM lock, invokes
  opt-in runlist flush, then publishes timeline completion/returns flush errors.
  Original failures preserved, not weakened/skipped. Upper SVM containment is
  still separate/unqualified; no injected-failure or raw concurrent DRM timeline
  ordering claim.
- **Historical installed K5 module (retained; superseded by SVM module above):** r3-only`updates/amdgpu.ko.gz`, compressed SHA
  `451b8a3d05cdf494b67546f9074c86f4d931481a3efd7f7b64e7bad48611dadb`,
  build/installed/loaded ID`12ad61569200a95f1d626396fca3692753231707`.
  Original r3 stock module, image/initramfs, r2 artifacts/default and .105 retained.
  K5 boot`e2a06828-208c-47f7-9838-8a5be9f7bfd5`; base/cooling readiness ACK
  and fresh360s GPU guard ACK completed. No active build/container/GPU gate/deadline.
  Single .ko build compiled both units but failed modpost on companion DRM symbols;
  incremental`make ... -j16 modules` succeeded using16jobs/20GiB ccache, only new
  AMDGPU installed. PIDs25806(failed)/26295(success) ended. Exact logs
  `KWS/logs/k5-{amdgpu-build,modules-build,module-install}.log`.
- **K5 passed unchanged full gate:** Lt664, SparseLt134 numerical+16 API,
  hipTensor304 API+12 GPU, Torch19 CPU references; two fresh processes each
  24 malloc/free+pinned alias/streams/events and16 explicit VMM remaps. Same
  immutable runtime/DSO+kpack/harness and original memory binary. Separate HIP
  and direct ROCr each26 GPU/CPU comparisons pass WITHOUT diagnostic churn;
  workload dmesg byte-identical. Two old Display Core load WARNs remain. No
  whole-Torch performance, sustained stability, universal VMM or arbitrary hang
  recovery claim. Python CPU segfault and experimental VM-heap defect unresolved
  and distinct. SDMA0/experimental heap off unchanged.
- **Current evidence/replay:** `kernel/r3/evidence/k5-{summary,gem-va-full,
  gem-va-isolation,running-identity}.json`, pre/post isolation logs; helpers
  `k5-{build,module,gpu-phase,isolation,identity}-on-host.*`, patch0003 and
  `k5-source-manifest.json`. Host`SOURCE_K5.json` records module delta;
  `source/GIT_REVISION`8f3c0d3, original image/source manifest84995f1 retained.
  `KWS/logs/k5-gem-va-{full,isolation}`, `k5-gem-va-console.log` and guard log.
  K5 follow-up CPU-unmap containment is now qualified in the bounded SVM scope
  above; use that current cursor for remaining locking/failure limits and next work.
- **Source/build:** signed stable`v7.2.10`, base
  `7b20d05fd38de8e0be168956f6fd241d750c7d69`, Greg fingerprint
  `647F28654894E3BD457199BE38DBBDC86092693E` verified. Separate repo
  `kernel/linux-r3`, original image source`84995f1a85188cde56910729214b0a68a7e402da`, initial two reviewed
  patches in`kernel/patches/r3/`: CU/PASID/TTM quirks, complete probe register
  restoration, runlist caller error propagation. SVM upper notifier/unmap paths
  still discard some existing returns/pre-mark unmapped: CPU-unmap returned-error containment now bounded-qualified above;
  retry-fault/failed-stop handling remains unqualified. Eight-core telemetry/clock trust still open.
  .104`KWS=/home/nik/bc250work/kernel-r3-7.2.10`; native16jobs/20GiB ccache build
  finished22:26:33UTC Oct14. Original PIDs12625/25715 ended; source/build/stage,
  caches and logs retained. Fan source`kernel/board-fan` revision`9086701` imported
  read-only from.105; ABI compiled/signed. Initial fan signing path failed; fixed
  source/scripts/sign-file→build/scripts/sign-file, repaired without rebuilding
  kernel; original failure/unsigned module retained,`logs/fan-sign-repair.log`.
  Signing private key remains remote build/certs,600/root; never copy it.
- **Artifacts/identity:** new image SHA
  `4531223f35bd9f313a242c2aa5d7bece0b4ffd4161d28b41271914a9b54d4558`,
  initramfs SHA`0e8444c328d070e0f6d422514116e19ccb13feb9e44d0bb4b289f0d1ea3eeb11`.
  `kernel/r3/evidence/candidate-inspection.json`:499 initramfs entries, exact
  watchdog hook, storage drivers, AMDGPU deferred, no private signing keys;
  new modules have matching release/signing key. r2 image/initrd/module/default
  hashes unchanged. Loaded kernel build ID`57f7c52d9c7ac0ac21c67d6e6efc3e08c12bbfde`
  and ORIGINAL AMDGPU`e2b6524374abbb8852759eb9c61857f56222793a` matched initial build; superseded by K5 module above; evidence
  `running-identity.json`, not bit-reproducible-build proof. Actual r2 firmware
  snapshot privately pinned via firmware_class.path; its2 SDMA entries are Navi12
  bytes under cyan names, differing from root cyan files. Root firmware unchanged,
  ROCr SDMA0 and experimental CLR VM heap default off. r2 loaded/initrd/installed
  module identity already closed; do not repeat the initial provenance audit.
- **K4 recovery exercised:** r2 feeder SIGSTOP under exclusive runtime lock,
  countdown to19s observed, then SSH loss/new boot ID/r2 return; flag bootstatus0
  does not confirm reset cause. Separate r3 idle-GPU guard deliberately expired
  after15s and emergency-restarted to persistent r2; SSH/new boot ID/default and
  consumed one-shot verified. Returned to r3, base/cooling ACK and24-round GPU
  smoke passed again. Builtin SP5100 early init handoff, nowayout1, root feeder
  120s, candidate panic60/oops/lockup policy confirmed. Candidate guard starts
  after bootmisc/sysctl, reapplies policy after Alpine panic120 default. Empty
  next_entry= is accepted; nonempty pending selection rejected. No guarantee for
  pre-watchdog hangs or arbitrary wedged GPU/reset recovery. No daily default
  promotion. Guard dry-runs alone were not treated as hardware qualification.
- **Historical pre-fix K5 comparison: BOTH full gates FAILED explicit VMM.**
  Identical immutable runtime/N1.8 DSO+kpack/harness on r2/r3: Lt664 references,
  SparseLt134 numerical+16 API, hipTensor304 API+12 GPU,19 eager CPU-compared Torch
  checks (bmm/fp32/fp16/bf16, FFT, Cholesky residual, COO/CSR, Adam) passed. Each
  workload kernel-log delta0; two pre-existing Display Core IRQ WARNs remain at
  AMDGPU load on both kernels. Malloc/free24 rounds, pinned GPU alias, streams/
  events/guards pass; post-recovery r3 basic smoke another24 rounds, delta0.
  Full gate fails after those consumers: fixed VA rebound to handle0 reads exact
  handle1 initializer BEFORE update (got01238878 vs01234567, round0/index0).
  Independent diagnostic kernel→ordinary buffer copy reproduces on BOTH r2/r3;
  cause/layer subsequently isolated and repaired in K5 above. Original/diagnostic failures preserved; not skipped
  or labeled full pass. This VMM defect is distinct from fixed CSR/VM-heap
  corruption AND unresolved historical Python CPU segfault; neither repaired
  by the new kernel. No whole-Torch performance or sustained correctness claim.
- **Evidence/next:** `kernel/r3/evidence/{kernel-comparison,r2-baseline-full,
  r3-candidate-full,watchdog-r2-hardware,deadline-fallback,running-identity,
  post-recovery-basic}.json`, diagnostic logs and README. Host`KWS/logs/` retains
  all commands/results: build-initial,fan-build/install-inactive, r2-baseline,
  r2-vmm-diagnostic,r2-baseline-full,r3-candidate-full,r3-vmm-diagnostic,
  r3-deadline-practice,r3-post-recovery; candidate.json records booted bounded
  scope/full VMM failure (superseded by SOURCE_K5.json and new candidate module fields). Initial test queued behind user's training was canceled
  BEFORE lock acquisition/feeder stop; restarted only after user finished jobs.
  Historical next isolation steps are completed by K5 above. The failed baseline
  remains preserved; use the new cursor for current module status and next action.
- **Historical initial kernel review (superseded by K1/K3 checkpoints above):**
  preliminary read-only r2 audit complete;
  see `docs/development/bc250_kernel_audit_20261014.{md,json,patch}`. Both hosts run
  the recorded7.1.5-r2 image; installed AMDGPU module hashes match. Archive-derived
  AMD subtree comparison:7 active files/136 additions/3 removals,2762 identical
  files and5 inactive `.orig` backups. Upstream signatures/full tree not audited.
  Findings: four ignored runlist-flush error returns; CU probe modes leave RLC
  changed; retained eight-core telemetry patch absent; two pre-existing Display
  Core boot warnings. No kernel cause for unresolved Python fault established.
  At that initial checkpoint there were no kernel/module/boot/runtime changes
  or builds. Upstream/full-tree and module identity work since completed above;
  use the current K3/K4 cursor, not this historical next step. N2.1 retained.
- **Latest completed library milestone:** N1.8 hipBLASLt short small/skinny GEMM dispatch implemented,
  installed on .104 and bounded-qualified. N1.1–N1.7 retained in their scopes.
- **Last qualified code:** root `0115c01f06f8654b26b5a665286069caf89eda2d`,
  libraries `a6f7d3d8ba7a1d022628d8c3fdcc54235551a9b8` (patches0025–0026).
- **N1.8 contract:** K<=64 and either min(M,N)<=4 or both M,N<=32 with M*N<=512
  use one32-lane scalar subgroup/output, eight outputs/block. Physical D order
  guides output lanes; no LDS/barriers in this specialization. FP32 tree reduction
  changes rounding from ascending-K FMA; same/mixed f32/f16/bf16 and existing
  scalar/bias/activation/AUX/DGELU epilogues retained. Public algorithm unchanged;
  remaining shapes keep tiled/register-tile dispatch. Subgroup grids above INT_MAX
  also fall back to the validated tiled grid; no oversized launch was executed. No matrix engine claim.
- **Passed:**664 new CPU-double-referenced GPU cases +403 scalar/561 extended Lt
  regressions. Orders/all transposes, padded/broadcast batches, host/device and
  zero scalars, compatible in-place C/D, K0/1/15/16/17/63/64/65 and area512/544,
  skinny tails/cross-batch blocks and immutable guards. Kernel logs identical1009
  lines; no general bitwise/nonfinite or sustained qualification.
- **Timings:**36 CPU-checked deterministic C++ fixtures ×A1/B1/B2/A2; five GPU-event
  samples of32-launch HIP graphs/pass. Selected skinny ratios1.14–1.86x and
  selected small1.21–1.61x; controls0.998–1.006x (roughly flat); max repeated-pass
  median drift0.24%. Column-major NN, batch1, host scalars/default epilogue only.
  These are graph-replay kernel costs, not uncaptured application or Torch speedups.
  Clock telemetry remains untrusted; default Torch BLAS stays rocBLAS.
- **Evidence/jobs:** `WS/logs/n18-lt-r4/result.json`, repository
  `bc250_scalar_library_results/n18-{lt,lt-attempts,lt-performance,lt-pack,reconstruction}.json`.
  Targeted hipBLASLt+dist,16jobs/20GiB ccache/exclusive lock; final build11.8s,
  numerical3.7s/scalar2.3s/extended2.1s, each36-case timing pass1.0–1.2s. No running
  jobs/gates. Initial one-thread/output candidate passed504 cases but was slower
  (0.46–0.76x selected ratios); rejected. Cooperative r2 passed504+964 checks,
  but32x32 regressed0.91–0.96x: final area<=512 restriction retains its tiled path. r3 passed664+964; final
  review added the subgroup grid-bound fallback and r4 repeated the full gate.
- **Recovery finding:** BLAS DSO references external `.kpack/blas_lib_gfx1013.kpack`.
  Initial attempt restored only DSO; A2 aborted134 on missing old GPU symbol.
  Recovered exact baseline pair from immutable P2 candidate (DSO hash matched),
  preserving experimental files. Runner now backs up/restores/hashes BOTH files;
  successful A2 proves paired baseline execution before reinstating final pair.
  Older N1.1–N1.7 failure reports validated DSO restoration only: do not claim
  complete runtime rollback from those fields or reuse old runners unchanged.
  Their successful final GPU gates remain evidence. This host-side code-object
  mismatch is distinct from unresolved Python CPU segfault and fixed pool/VM issue.
- **Provenance:** `SOURCE_N18.json` verifies12 source files on N1.7/P2 plus both
  active BLAS hashes: DSO `91c65ae424a19b58edb9210d033e0c5d2ce72f1781fd283f5204371c2f654ff6`;
  pack `a431881f2b826ee3eedb65fa6f08b2c2ba6242a62f9c4671f811f7ceb018dde5`.
  Pack payload comparison:1 hipBLASLt entry changed,68 entries unchanged (including
  all67 rocBLAS entries). Independent recovery DSO `WS/recovery/n18-lt-before.so.1.5`
  and pack `WS/recovery/n18-blas_lib_gfx1013-before.kpack` retained. All26 patches
  reconstruct exact tree; .105/frozen candidates/kernel/boot/Torch unchanged.
- **Next: N2.1, solver difficult/complex coverage and selection audit.** Read the
  existing32 P2 solver cases, residual/gradient/condition10,000 evidence, patched
  Torch linalg consumer and rocSOLVER/hipSOLVER dispatch before choosing a missing
  complex/difficult operation. Use short independent CPU/reference/residual gates;
  implement only a demonstrated missing path/defect. Any prefix mutation needs
  paired DSO/kernel-pack recovery. Sparse widened INT8 outputs/beta-vector and
  other optional N1 extensions remain separately open, not hardware-impossible.

At resume: read this file, inspect local Git status/log and `.104` runtime/source
manifests, then read the inventory section for the selected task. Check evidence
before deciding something is missing. If state differs, reconcile it; do not
silently replace the plan or downgrade already qualified work.

## Goal, authorization and working agreements

Build the most functional, reliable and efficient **native gfx1013 ROCm/Torch**
stack the BC-250 can support: compiler/runtime, feasible libraries, training,
profiling/debug tools, communication, packaging and measured performance.
Do not spoof another GPU or pretend missing matrix instructions exist. A scalar
implementation of a mathematical operation may be feasible even when an upstream
matrix backend is not. CPU fallbacks must be labeled.

- User preference2026-10-15: update/commit canonical flow once at final response
  or meaningful handoff; use commentary for progress, avoid repeated routine doc commits.
- English communication/code/docs. Work locally, transfer explicit changed files,
  build/test/install on `.104`. User authorizes development/kernel experiments;
  retain SSH/boot fallback. No push/publish without explicit instruction.
- `.105` is the preserved old-stack reference. Read-only comparison is authorized;
  do not upgrade it until `.104` is stable and the user authorizes migration.
- Preserve releases, sources, caches, build objects, images, models/checkpoints,
  kernel/initramfs/GRUB recovery and unrelated untracked `src/`.
- Use bounded meaningful CPU-compared GPU tests. Reuse objects/ccache, group commits
  at milestones, keep routine documentation short. No automatic long soaks.
- Do not run `fetch_sources.py`, reset submodules, expunge or prune as a resume step.
- SSH keys work. Doas credentials are transient session information; **never write
  passwords, tokens or raw credential-bearing prompts into memory/source/logs**.

## Current state and paths

`WS` = `/home/nik/bc250work/rocm-10.2-gfx1013` on `nik@192.168.1.104`.
Workspace root `/home/nik/Documents/Vibe/bc250stack`; TheRock root
`/home/nik/Documents/Vibe/bc250stack/therock` (old path remains a compatibility
symlink), branch
`users/Naster17/rocm-10.2-gfx1013`. `.104` host is Alpine, ROCm userspace is glibc
inside rootful Podman; this is not the separate native-musl gfx1103 project.

| Item                     | Current identity                                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| Qualified root code      | `0115c01f06f8654b26b5a665286069caf89eda2d`                                                                                |
| rocm-libraries           | `a6f7d3d8ba7a1d022628d8c3fdcc54235551a9b8`, ordered patches0001–0026                                                      |
| rocm-systems             | `dbdb7b8d1c56316d873a9b90e548e5a72853496d`, patches0001–0003                                                              |
| Torch                    | `4f13bf86476252e3d57b17d1c72f8512814400e9`, patches0001–0002; editable pip                                                |
| Triton                   | `ee65ff3b85f99d52e678b252a99c8425fa46e329`, patches0001–0004; qualified wheel retained                                    |
| LLVM                     | Clean upstream `7667f52dd687899256a3285ab0bce3c53bc73394`                                                                 |
| Active stack             | `/opt/bc250-stack/current` → `/opt/bc250-stack/dev/priority1`; legacy directory name, **P2 + N1.1–N1.8 dev profile**                  |
| ROCm/Torch/cache         | `WS/build/dist/rocm`, `WS/pytorch-work`, `WS/cache/dev-user` (uid/gid1000)                                                |
| Runtime image ID         | `5df0a58392de13916819f6d84342b0338ae2c2bbb2f3c21a1f3ad38be2724a59`                                                        |
| Runtime/build tags       | `localhost/bc250-stack:10.2-gfx1013-priority1-candidate`; `localhost/bc250-rocm-build:10.2-ccache`                        |
| Source exports           | Gitless `WS/source`, `WS/pytorch-source`, `WS/pytorch-work`; GIT_REVISION + `SOURCE_N18.json` delta on N1.7/P2                |
| Independent P2 candidate | `/opt/bc250-stack/candidates/priority2-20261014`; passed bounded checks, not daily-release activation                     |
| Recovery userspace       | `/opt/bc250-stack/before-dev-current` → frozen `10.2-gfx1013-20261013-4b000d542`; initial a640c9296 release also retained |
| Kernel/boot              | `.104` experimental `7.2.10-bc250rocm-r3`; persistent default/retained fallback r2; new artifacts `KWS/artifacts`, old files `bc250-rocm-lab` |
| Memory backup            | `WS/agent-memory/latest` (documentation only; preserves qualified source snapshots)                                       |
| Training artifacts       | `WS/training-model`, `training-env-stable`, `training-validation`, `logs/training-memory/`                                |

Manifests and immutable image IDs outrank mutable tag names. Component version
strings differ legitimately (TheRock10.2, HIP7.17, Torch 2.13). Source archive names
and generated Torch Git labels may retain older baselines; do not relabel them.
The P2 source transfer verified51 files; N1.1 adds7; N1.2–N1.7 each add8 verified changed files; N1.8 adds12 and BLAS DSO/pack hashes. Candidate payload verified191
independent shared-object/kernel-pack copies. Detailed hashes are in manifests.

## Unified local workspace

The new parent workspace has its own AGENTS.md/README and component manifest.
`torch/` and `triton/` link to our patched sources inside TheRock. Your fresh
upstream checkout is retained under `references/pytorch-upstream`; it is not an
upgrade/replacement of the qualified Torch. Old `Vibe/therock` and `Vibe/pytorch`
paths remain symlinks to preserve absolute build paths. Do not delete them yet.

`kernel/source` is the already-patched Linux7.1.5 r2 source recovered read-only
from .105, with `kernel/configs/r2.config`, packaging repository, legacy scripts
and provenance. Kernel binary hash matches both hosts. Compiled objects/private
signing keys/generated initramfs archives are excluded. Do not apply legacy
patches twice. Existing board driver/CU-unlock repositories are copied under
`board-support`. Organization did not change either remote runtime or boot state.

## Hardware and indispensable constraints

`.104` has eight CPU cores/16 threads, **40CU/20WGP/80SIMD**, qualified wave32,
64KiB LDS, KFD100103/HIPgfx1013:xnack−, PCI1002:13fe. HIP multiprocessor count20
is correct for WGP reporting. GTT15,938,355,200 bytes≈14.844GiB and systemRAM
≈14.858GiB share physical GDDR6; they are not two independent memory budgets.

Keep SDMA disabled (`HSA_ENABLE_SDMA=0`) and experimental VM heap default off;
VMM/async pools remain available. Current GPU recovery parameter is0. Watchdog
SP5100 TCO service is active (10s/120s), but actual hang/reset recovery is not
qualified. No usable MFMA/WMMA or packed-dot backend. Wave64/XNACK+ and hardware
media/counter/debug modes are separate qualification, not inferred from enums.
No cpufreq governor exists on this host; GPU clock telemetry conflicts. Fan curve
reads CPU Tctl, not GPU edge. Full measured hardware/recovery facts are in inventory.

## Completed work: do not restart it

| Batch                     | Qualified work / evidence anchor                                                                                                                                                                                                                                                                |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Core and expanded release | Native HIP/RTC/Torch/Triton; BLAS/solver/FFT/RNG/sparse; grouped MIOpen convolution, primitives, OpenCL/OpenMP, RPP, hipTensor generic, hipDNN providers, hipFile compatibility, profiler tracing. Historical13 component/56 eager/18 compiler gates; independent frozen600s soak40,339 cycles. |
| Training/files            | Real SmolLM2-135M full FP32 and AMP LoRA FP16/BF16, optimizer/scheduler/scaler/RNG checkpoint/resume; 10GiB GPU+ 768 MiB CPU pressure, streams/graphs/processes. Persistent host paths, mount modes, downloads/DataLoader/pip/stdin/shell/interruption qualification already exists.            |
| Scalar milestone          | Real native hipBLASLt/hipTensor/hipSPARSELt backends, CK/RPP/generic/Torch coverage; Torch NHWC average-pool gradient fix and separate sparse Lt kernel-pack ownership.                                                                                                                         |
| P1                        | Lt C++/host-grouped consumers and epilogues; CK247 numerical+400 rejection cases; hipDNN100 graph cases; Torch64 including36 observed real Lt calls. Mutable dev stack activated.                                                                                                               |
| P2                        | hipTensor166, hipSPARSELt234, hipBLASLt964 API, math149, primitives138 GPU cases;71 performance cases. Exact library/system/Torch patch reconstruction; independent candidate and installed launcher each passed20 cycles with sparse/compiled/FFT/checkpoint checks.                           |
| N1.1                      | hipTensor unilateral A/B sums and mixed shared reductions:253 numerical +20 descriptor cases; alias/zero-scalar/span guards,16-mode union, nine algorithm/workspace combinations, small-fixture timings. Patch00018; native scalar routing. |
| N1.2 | Same-type C32F/C64F hipTensor contractions with identity/NEG/CONJ:426 numerical +24 descriptor cases,273 real regressions, scalar ABI/zero-scalars/alias/span/plan-lifetime guards; patch0019. |
| N1.3 | hipTensor plan null/enum/attribute-size guards, atomic setters and owning failure cleanup; honest no-cache/JIT/autotune scalar policy.304 API +12 numerical checks,273 real/450 complex regressions; patch0020. |
| N1.4 | Sparse Lt matching-count zero-stride broadcasts for A/B/C, one physical sparse pack, single-writer pruning/compression and span alias guards.604 numerical +5 descriptor cases,234 regressions; patch0021. |
| N1.5 | Native GPU sparse Lt per-row bias: same-type/FP32, batch broadcast/strides, before activation, plan snapshots and size/type/span/alias guards.526 numerical +41 API checks,234 scalar/609 broadcast regressions; patch0022. |
| N1.6 | Native device FP32 per-row alpha vector shared across batches, plan flag snapshots and span/alignment guards.404 numerical +34 API checks;234 scalar/609 broadcast/567 bias regressions. Beta-vector enable explicitly unsupported; patch0023. |
| N1.7 | Native INT8 input/output, bounded INT32 accumulation, FP32 epilogue/bias, nearest-even saturation.134 numerical +16 API and234/609/567/438 regressions; K<=262140, widened outputs unimplemented; patch0024. |
| N1.8 | Short small/skinny Lt subgroup dispatch:664 new +964 regression cases,36 A/B graph timing fixtures; selected1.14–1.86x, controls~flat; paired DSO/kpack recovery; patches0025–0026. |
| Performance               | Matched `.104`/`.105` comparisons already done: most GEMM/copy differences small, variable small-conv results. Separate same-board P2 large LtGEMM≈2.1–2.4x faster; selected large compiled HIP-graph fusion≈4x. No whole-Torch or universal speedup claim; default BLAS unchanged.             |

Reports: `docs/development/bc250_scalar_library_results/priority{1,2}-*.json`,
`bc250_performance_results/`, host `WS/logs`. Detailed coverage limits, patch
index, training/model pins and old defects are in the inventory. Counts overlap.

## Known pitfalls and resolved problems

- **Proven old CSR corruption:** experimental VM heap overwrote live rocSPARSE
  analysis metadata. Safe compiled CLR default fixed repeated CG gates (40 and
  five-process200); it did not repair the experimental heap. Preserve this fix.
- **Separate later segfault:** Python CPU execution fault at0x3b3e6, cause unknown.
  Do not identify it as the proven CSR defect or claim bounded reruns solved it.
- **Prefix concurrency:** launcher holds shared `.runtime.lock`; run every host
  prefix update through `helpers/update_dev_prefix.py -- COMMAND...` (exclusive).
  Arbitrary manual writes bypass this protection. Never test against staging libs.
- **Torch generation:** before each source rebuild, run upstream
  `python tools/amd_build/build_amd.py --out-of-place-only`, then
  `python -m pip install -e . -v --no-build-isolation`. Cached CMake can leave HIP
  sources stale; don't rebuild solely to change generated version metadata.
- **Runtime rollback:** pair every changed DSO with its external kernel pack; back up independent copies and verify both hashes under the exclusive lock. DSO-only restoration is incomplete. N1.8 discovered this; older rollback helpers need this fix before reuse.
- **Targeted builds:** prefer component+dist, including
  `hipBLASLt+dist hipTensor+dist hipSPARSELt+dist`. Global dist can rebuild unrelated
  MIOpen. Keep16 normal jobs, tensor12/providers6/Torch 12/link 1 and20GiB ccache.
- **Flag names:** use `THEROCK_GFX1013_SCALAR_BACKENDS=ON` with exclusive gfx1013;
  similarly named `THEROCK_ENABLE_...` cache entries are not this option.
- **Already fixed:** Triton RDNA1 buffer flags/low-precision lowering, RPP
  spectrogram/decibel handling, Torch pooling, CK/header tracking, FFTS linking,
  launcher path/read-only/shell/consumer pins. Read inventory before redoing them.
- **ABI details:** Lt SWISH beta is fixed 1; MIOpen bias uses beta 0; mixed fusion
  intermediates remain rejected. SparseLt compression is backend-specific.
- **Testing:** require real execution/counts, CPU references, unchanged meaningful
  tolerances andguards. Clone float64 CPU references before gradient mutation.
  Do not disable/skips tests to turn red into green.
- **History:** native Alpine branch testedgfx1103, not BC-250. Historical frozen
  long-soak results do not qualify the currentmutableprefix forlongstability.

## Ordered full-port plan: stable task IDs

P0/P1/P2 milestones are completed **within their recorded scopes**. Remaining
work is below; do not revive their old unchecked checklist lines as new tasks.
Work through one measurable subtask at a time and keep these IDs stable.

| State | ID                   | Work and acceptance boundary                                                                                                                                                                                                                                                                                                                                                                        |
| ----- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [x]   | N1.1 | hipTensor unilateral contraction modes completed within the bounded scope above. Preserved zero-scalar, span and alias guards; CPU-referenced types/layouts/plan-lifetime gates and small-fixture timing passed. |
| [x]   | N1.2 | Native same-type C32F/C64F hipTensor contractions, identity/NEG/CONJ, bounded numerical and real-regression gates completed as above; mixed/widened complex and other unary functions remain rejected. |
| [x]   | N1.3 | hipTensor plan API validation, owning failure cleanup and supported scalar preference acceptance/rejection qualified in the bounded scope above. Scalar cache/JIT/autotuning/ranked alternatives remain explicitly unsupported. |
| [x]   | N1.4 | hipSPARSELt matching-count zero-stride broadcast inputs, single physical sparse compression/pruning and span-alias guards qualified above. Incompatible counts/multi-batch zero-stride outputs remain rejected. |
| [x]   | N1.5 | Native hipSPARSELt row bias pointer/stride/type contracts and epilogue qualified above. Device bias memory remains caller-owned; plans capture settings. |
| [x]   | N1.6 | Native alpha-vector path and explicit scalar-beta/unsupported beta-vector policy qualified within the bounded scope above. Beta-vector implementation remains separately open; no universal vector-mode claim. |
| [x]   | N1.7 | Native signed INT8 input/output with bounded INT32 accumulation, FP32 scalar/vector/bias epilogue and nearest-even saturation qualified above. Widened outputs remain unimplemented. |
| [x]   | N1.8 | Short small/skinny Lt subgroup dispatch qualified above; tiled path retained outside measured selection. No default Torch BLAS change or whole-Torch speedup claim. |
| [x]   | N2.1 | Audit P2 solver32 cases and pinned Torch/rocSOLVER/hipSOLVER consumers; qualify one missing difficult/complex operation with CPU/residual references and fix only demonstrated defects. |
| [x]   | N2.2 | Complex nonsymmetric eig16 groups qualified through existing host-LAPACK compatibility; native GPU eig remains missing. Pinned dispatch audited, no backend defect reproduced. |
| [x]   | N2.3 | Isolated native small Schur prototype cap16: GEHRD plus bounded GPU shifted QR, host scalar control, explicit nonconvergence;21 groups pass. No right eigenvectors or production eig integration. |
| [x]   | N2.4 | Isolated native GPU right eigenvectors from small Schur forms;21 groups pass including analytic/CPU residual/direction and repeated/close/nonconverged refusal. Production eig unchanged. |
| [x]   | N2.5 | Repair demonstrated tiny/large magnitude failures; seeded6 replay,40 scale,30 separation,16 failure/control and21 N2.4 groups pass. Experimental cap16/default eig unchanged. |
| [x]   | N2.6 | Profiled small native versus host, vectorized recovery;107 groups and6 paired fixtures pass, recovery1.53–8.44x/whole1.07–1.72x in fixtures. Native remains slower than host; default unchanged. |
| [x]   | N2.7 | Isolated native cooperative small QR/similarity step;28 direct+128 retained groups+six paired CPU fixtures pass. Schur1.24–1.46x in fixtures, host eig remains faster; defaults unchanged. |
| [x]   | N2.8 | Device shift/deflation/status around cooperative QR;128 retained+74 direct+six CPU/paired fixtures pass. Whole prototype1.56–2.21x versus N2.7 in fixtures; host default remains faster/unchanged. |
| [x]   | N2.9 | Bounded chunks1..32 reduce status copies;128 retained+74 direct+100 chunk+six CPU fixtures pass. Latency mixed/regresses16; padded experiment not selected, defaults unchanged. |
| [x]   | N2.10 | Native bounded reflector-Q formation;128 retained+24 independent reflector+six CPU/backend fixtures pass. Preparation1.11–1.39x/whole1.05–1.12x in fixtures; defaults unchanged. |
| [x]   | N2.11 | Explicit pinned-stream GEHRD sessions;128 retained+six CPU/backend fixtures+three lifetime groups/16 refusals pass. Warm prep7.03–12.52x/whole1.28–1.95x in fixtures; cold/default/host limits recorded. |
| [x]   | N2.12 | Native bounded triangular recovery;128 retained+36 CPU/backend+32 direct+two rescaling+six ABI cases pass; fixed gates/defaults retained. Recovery1.32–2.32x/whole1.17–1.34x in six fixtures; host remains faster. Separate CPU-first profiler defect found. |
| [x]   | N2.13 | Generic sparse triangular coverage and exact-pivot NaN marking installed:433 public +96 direct generic fixtures, retained sparse41/core7/profiler pass in candidate and installed runtime. General sparse spsolve B02 remains separate. |
| [x]   | N2.14 | FFT complex96/real72 output+weighted backward+guarded out, three DFT/finite-difference/reuse checks,16 direct plans/32 executions and FFT30/core7 pass; no production defect or binary/default change, AOT cache preserved. BF16 promotion B01 separate. |
| [x]   | N2.15 | Torch state80/statistics22/contracts20/fresh continuation1, direct native68 and RNG28/core7 pass. Stale LFSR113 offset comments corrected; implementation/binaries/defaults unchanged; four AOT DBs preserved. |
| [x]   | N2.16 | Native floating/segmented/custom reductions/scans: rocPRIM576/hipCUB480/Thrust480, P2 primitives138/core7 pass; fixed CPU references/guards/streams, no production defect or default/binary change. |
| [x]   | N2.17 | Native floating stable sorting/selection: rocPRIM656/hipCUB656/Thrust648; Torch62 incl54 finite backward cases; P2 primitives138/core7 pass. Exact ordering/payload/count/guard checks; no production defect or installed change. |
| [x]   | N2.18 | Scalar CK DL FP32 grouped forward:128 CPU-reference cases +8 refusals; GEMM3/core7 retained. Explicit r2/r3 guard selection repaired; no production CK/runtime/default change or speed claim. Next review N2 practical gaps before selecting another N3 batch. |
| [ ]   | N2                   | Broader scientific/primitives/RPP coverage: solver difficult/complex cases and selection, sparse formats/analysis/solves, FFT plans/layouts, RNG distributions, floating/custom/segmented algorithms, admitted CK DL conv/grouped operations. Qualify actual operations, not architecture lists.                                                                                                    |
| [ ]   | N3                   | Torch functional/performance expansion: larger meaningful training sequences/batches, full-model AMP, compiled/eager gradient equivalence, optimizer/checkpoint resume, quantization/QLoRA applicability, feasible scalar tiled attention and hipDNN mixed/fused graphs. Keep mathematical SDPA working; unsupported matrix backends stay off.                                                      |
| [ ]   | N4                   | Trusted read-only telemetry, RDC applicability, debugger consumers/suites, additional tracing and physical counter mapping. A09 CPU-first ordering repaired with bounded both-order traces; physical telemetry/clock/counter and advanced profiler work remain. Dangerous debugger tests require recovery first.                                                                                                                                                   |
| [ ]   | N5                   | Build native RCCL/rocSHMEM/MPI local readiness; feasible generic/software transports with GDA/SDMA/XGMI/P2P claims off until demonstrated. Single-GPU readiness is not full collective coverage. `.105` migration/two-board distributed tests wait for explicit later authorization and stability.                                                                                                  |
| [ ]   | N6                   | Guarded kernel/runtime investigations: experimental VM-heap repair, SDMA, reset/fault recovery, XNACK/wave64 where supported, advanced profiling. Preserve known boot/stack defaults; isolated opt-in gates and independent recovery required.                                                                                                                                                      |
| [ ]   | N7                   | Remaining optional feasibility: host media/HIP interoperability, actual media rings, hipThreads/Fortran execution, hipify-clang, hipFile registration/concurrency/direct-storage applicability, rocRoller/emulation and other pinned projects. Explicit unsupported/not-applicable decisions where hardware/API prevents a path.                                                                    |
| [ ]   | N8                   | Final integration/release: source reconstruction, independent relocatable packaging, licenses/tools/plugins/kernel packs, selected Torch regressions, agreed bounded stability/training/memory gates, kernel-log review, recovery verification and truthful operation matrix. Longer runs only when relevant/user-agreed.                                                                           |

Kernel workstream added at the user's request (2026-10-14):

| State | ID | Work and acceptance boundary |
| ----- | -- | ---------------------------- |
| [x] | K1 | Verified latest stable upstream Git tag/source and signing identity; compare legacy source/config/modules and retain provenance/fallbacks. |
| [x] | K2 | Minimal series reviewed/implemented and normal paths bounded-tested; diagnostic register restore corrected; CPU-unmap returned-error containment bounded-qualified by patch0004; failed-stop/map-locking/telemetry remain open. |
| [x] | K3 | Separate r3 image/modules/config built, signed/staged/installed and artifact-inspected on .104; r2 retained unchanged. |
| [x] | K4 | Bounded r2 watchdog countdown/SSH return and r3 deadline fallback/SSH return exercised; two r3 base boots/cooling ACKs; early-hang/arbitrary GPU-hang recovery unqualified. |
| [x] | K5 | GEM_VA runlist invalidation repaired; unchanged full GPU/Torch/VMM gate and independent HIP/ROCr references pass on signed r3 module8f3c0d3. Historical r2/r3 failures preserved; bounded scope only. |
| [x] | K6 | Record qualified scope/artifact hashes, source patch series and reproducible build/boot/recovery instructions; integrate only supported settings and retain r2 fallback. |

Every component ultimately needs a documented disposition: qualified native path,
qualified explicit fallback, or a concrete hardware/API limitation. Missing matrix
instructions themselves are **not pending implementation**. Do not compute a
completion percentage from boxes; an infinite API-combination sweep is not a plan.

## Memory maintenance rule for every future agent

At final response or a meaningful handoff, update this file once (user preference
2026-10-15: do not rewrite/commit flow for routine progress). Preserve durable
job/status evidence if interrupted; use commentary during work. Record: taskID/subtask, actual source and
installed revisions, exact changed behavior, passed/failed evidence andscope,
unresolved failures, next concrete action, running container/PID/log/exit paths,
resource settings andany user scope changes. Update the affected inventory row
when a capability/decision changes. Preserve historical reports/failures; don't
rewrite the entire history or make a commit after every command.

Temporary tool cell/session IDs and `/tmp` transfer helpers are not durable state.
Record replayable repository/host commands and artifact paths. Never record
credentials. Mark complete only when implemented, staged and numerically executed
for the stated scope. New plans/assumptions must be labeled, not quietly promoted
to hardware facts. If interrupted, write **in-progress/unknown**, not passed.

Before repeating work, answer: what already exists, which source/report proves it,
and what new change/uncovered case/failure requires another build or test?
