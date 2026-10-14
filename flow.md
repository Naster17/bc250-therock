# BC-250 project memory: read first after resume or compaction

Updated 2026-10-15 (local; checkpoints explicitly use UTC). This is the
**canonical working memory**, not a release note.
Detailed hardware, implementation history, evidence and resolved defects:
[bc250_memory_inventory.md](docs/development/bc250_memory_inventory.md).
Old milestone documents are evidence snapshots; their unchecked boxes and active
release claims must not override this cursor and current manifests.

## Resume cursor

- **Live checkpoint 2026-10-14 21:48 UTC (user requested before compaction):**
  K3 build PID12625 alive after27min, phase`build`, fan waiter25715 alive;
  log advancing through AMDGPU/network/USB/filesystem modules, no build failure
  observed,28.3GiB free. Same command/log/status/stage paths below. No install,
  reboot or GPU execution; running kernel remains r2, persistent default r2.
  Next independent preparation: make GPU runner reject new kernel faults/WARNs
  and validate early-init hook/installer guards locally. Then finish staged
  kernel/fan, exercise hardware watchdog recovery on r2, guarded r2 references,
  and only then one-shot candidate boot/base SSH and controlled GPU gates.
  Preserve generated remote signing private key; never include it in evidence.
- **Recovery preparation batch (build still running):** GPU runner now saves
  durable phase checkpoints and rejects new kernel faults/WARNs or ring rotation;
  existing boot warnings remain in before-log. Ten local log-review cases passed.
  Actual host init copy passes generated-hook shell/order checks; three guard
  dry-run cases recorded in`kernel/r3/evidence/recovery-preparation.json`.
  Prepared/transferred`test-watchdog-on-host.py` (verified feeder SIGSTOP plus
  detached165s same-boot/PID fail-safe CONT) and`ack-base-boot-on-host.py`
  (root watchdog/nowayout, cooling/fan, NIC, consumed one-shot, AMDGPU deferred).
  These live helpers NOT executed; recovery/boot/GPU still unqualified.
  Native signing key permissions verified600/root, stays remote. CPU~62°C,
  fan~3500RPM during build. Memory checkpoint backed up on.104 at
  `WS/agent-memory/8c6efe00e-kernel-build` and`latest` points there; sources intact.
  Follow-up review: temporary GRUB env proved consumed one-shot may remain as
  empty`next_entry=`; installer/watchdog-test/base-ACK now reject nonempty values,
  accept empty. Early init requires nowayout and real character device, creating
  it from sysfs major/minor if devtmpfs falls back to tmpfs. Syntax passed.
  SVM upper MMU-notifier/unmap callers still discard some existing helper returns
  and pre-mark ranges unmapped: no end-to-end failed-flush lifetime containment
  claim; investigate separately before fault/migration experiments. Latest build
  check22:00UTC:~10,500 objects/1.3GiB output, phase build, no observed error.
  Alpine `/usr/lib/sysctl.d/00-alpine.conf` overrides panic to120; candidate-only
  OpenRC guard now starts after bootmisc/sysctl, arms its deadline FIRST, then
  restores panic60/oops/softlockup/hardlockup policy for r3 only. Base ACK verifies
  these live values. Existing r2 services/settings unchanged. GPU runner records
  immutable image/source identities and BLAS DSO/kpack hashes, exact664/150/316
  consumer markers and19 Torch references; VMM unsupported is explicitly
  unqualified rather than counted as executed. Live GPU execution still pending.
- **ACTIVE user priority: K1–K6, clean upstream BC-250 kernel workstream**
  (authorized2026-10-14), alongside ROCm/Torch. K1 source fetched: signed stable
  `v7.2.10`, base`7b20d05fd38de8e0be168956f6fd241d750c7d69`; Greg fingerprint
  `647F28654894E3BD457199BE38DBBDC86092693E` verified via kernel.org WKD/published
  fingerprint. Official tag/verification artifacts in `kernel/verification/`.
  New separate repo`kernel/linux-r3`, branch`users/Naster17/bc250-r3-7.2.10`,
  sourceHEAD`84995f1a85188cde56910729214b0a68a7e402da`; two reviewed patches under
  `kernel/patches/r3/`: explicit CU/PASID/TTM quirks, full probe register restore,
  runlist callers propagate errors through existing partial-operation error paths.
  No failed-flush injection qualification. Eight-core telemetry remains separate.
  Full93,517-file r2 comparison confirmed only7 active AMD changes +5 backups
  against archived baseline; archive signatures not checked by that comparison.
- **RUNNING native kernel build (.104):** root PID12625, command
  `bash /home/nik/bc250work/kernel-r3-7.2.10/helpers/build-on-host.sh`, log
  `/home/nik/bc250work/kernel-r3-7.2.10/logs/build-initial.log`, status JSON
  `.../build-status.json`. O=`.../build`, private staging=`.../stage`, source
  verified tar/hash in`kernel/r3/source-manifest.json`;16jobs/existing20GiB ccache.
  r2-config seed, debug symbols disabled for31.6GiB disk budget; builtin SP5100,
  nowayout,180s first-open bound and panic/lockup reboot policy for candidate.
  Signing key is generated remotely in build/certs only; never copy private keys.
  Local clone and full-source audit jobs finished; authenticated root SSH session
  68487 remains open for orchestration. No GPU test, install or reboot yet.
- **K3/K4 preparation checkpoint:** host fan-build waiter PID25715, log
  `.../kernel-r3-7.2.10/logs/fan-build.log`; rebuild/sign separate nct6687 module
  only after kernel phase`staged`, with the new build's private key kept remote.
  Fan source imported read-only from.105 into new repo`kernel/board-fan`,
  revision`9086701`; original fan Makefile install/uninstall recipes not executed.
  R3 memory/reference probe compiled (initial void** API compile correction logged
  separately); no GPU execution yet. Candidate-only boot guard dry-run checks
  passed: valid ACK, stale ACK rejected/expiry, wrong release inactive. Hardware
  watchdog expiry/SSH recovery NOT yet exercised; do not call recovery qualified.
  Helpers under`kernel/r3/`: early initramfs watchdog handoff, boot ACK deadline,
  inactive candidate installer and serial GPU/Torch runner; not installed yet.
  Firmware snapshot: actual r2 initramfs contains8 cyan entries, of which the2
  SDMA blobs exactly match root Navi12 firmware and differ from root cyan blobs.
  Pinned selected bytes in host`.../firmware-from-r2`; initial candidate uses that
  override to avoid changing kernel and firmware together. Root firmware untouched;
  ROCr SDMA remains disabled. Inspection handled usr/lib→lib path normalization;
  initial parser/compile failures retained in hostlogs, no GPU fault involved.
- **R2 identity gap closed during K1:** retained initramfs and installed AMDGPU
  payload SHA`cf6c4c0ec2bae53da7c5ad800719997deb9b2f1ccd90f1da882a76d35bd1e611`
  match; loaded/installed/archive build ID all`4bbb2004029bff0b0e9f8e60185968ebd073c27e`.
  Evidence`kernel/r3/evidence/r2-module-identity.json`; not a bit-reproducible build
  proof. Firmware comparison now normalizes initramfs usr/lib to root/lib and
  checks decompressed payloads; only2 SDMA entries differ, exactly Navi12 blobs.
  `kernel/r3/evidence/r3.config` and config diff retain actual migrated settings.
  Native build still PID12625, fan waiter25715; patched TTM compiled, other modules
  still compiling. No install/boot/GPU test yet; no actual watchdog expiry gate.
- **Immediate next:** finish build; rebuild retained NCT6687 cooling module from
  .105 read-only source import`kernel/board-fan` against r3. Check source modules/
  firmware/initramfs, prepare one-shot entry with persistent r2 default. First
  boot should defer AMDGPU auto-loading until SSH, then controlled module/GPU
  gates under watchdog/deadline recovery. Current SP5100 watchdog stops during
  reboot: current daemon alone does not guard early boot; establish initramfs/
  base-boot guard before boot test. Preserve all r2/ROCm files and .105 unchanged.
  No blanket recovery guarantee without out-of-band access; no deliberate GPU
  reset/SDMA/fault experiments. N2.1 remains next library item after kernel batch.
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
- **Current task:** N1.8 hipBLASLt short small/skinny GEMM dispatch implemented,
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
| Kernel/boot              | `7.1.5-bc250rocm-r2`; live boot files in `/home/nik/bc250work/bc250-rocm-lab`, backups `WS/recovery/`                     |
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
| [ ]   | **N2.1 — next** | Audit P2 solver32 cases and pinned Torch/rocSOLVER/hipSOLVER consumers; qualify one missing difficult/complex operation with CPU/residual references and fix only demonstrated defects. |
| [ ]   | N2                   | Broader scientific/primitives/RPP coverage: solver difficult/complex cases and selection, sparse formats/analysis/solves, FFT plans/layouts, RNG distributions, floating/custom/segmented algorithms, admitted CK DL conv/grouped operations. Qualify actual operations, not architecture lists.                                                                                                    |
| [ ]   | N3                   | Torch functional/performance expansion: larger meaningful training sequences/batches, full-model AMP, compiled/eager gradient equivalence, optimizer/checkpoint resume, quantization/QLoRA applicability, feasible scalar tiled attention and hipDNN mixed/fused graphs. Keep mathematical SDPA working; unsupported matrix backends stay off.                                                      |
| [ ]   | N4                   | Trusted read-only telemetry, RDC applicability, debugger consumers/suites, additional tracing and physical counter mapping. Resolve clock inconsistencies before explaining small speed changes; dangerous debugger tests require recovery first.                                                                                                                                                   |
| [ ]   | N5                   | Build native RCCL/rocSHMEM/MPI local readiness; feasible generic/software transports with GDA/SDMA/XGMI/P2P claims off until demonstrated. Single-GPU readiness is not full collective coverage. `.105` migration/two-board distributed tests wait for explicit later authorization and stability.                                                                                                  |
| [ ]   | N6                   | Guarded kernel/runtime investigations: experimental VM-heap repair, SDMA, reset/fault recovery, XNACK/wave64 where supported, advanced profiling. Preserve known boot/stack defaults; isolated opt-in gates and independent recovery required.                                                                                                                                                      |
| [ ]   | N7                   | Remaining optional feasibility: host media/HIP interoperability, actual media rings, hipThreads/Fortran execution, hipify-clang, hipFile registration/concurrency/direct-storage applicability, rocRoller/emulation and other pinned projects. Explicit unsupported/not-applicable decisions where hardware/API prevents a path.                                                                    |
| [ ]   | N8                   | Final integration/release: source reconstruction, independent relocatable packaging, licenses/tools/plugins/kernel packs, selected Torch regressions, agreed bounded stability/training/memory gates, kernel-log review, recovery verification and truthful operation matrix. Longer runs only when relevant/user-agreed.                                                                           |

Kernel workstream added at the user's request (2026-10-14):

| State | ID | Work and acceptance boundary |
| ----- | -- | ---------------------------- |
| [x] | K1 | Verified latest stable upstream Git tag/source and signing identity; compare legacy source/config/modules and retain provenance/fallbacks. |
| [ ] | K2 | Review/reimplement only needed gfx1013 patches; correct diagnostic register restoration and analyze TLB error/locking/lifetime behavior; telemetry separately. |
| [ ] | K3 | Build separate r3 image/modules/config with ccache/resource/disk bounds on .104; inspect artifacts and initramfs without replacing r2. |
| [ ] | K4 | Establish and check one-shot boot, persistent r2 default, early panic/watchdog and SSH recovery; bounded candidate boot only after recovery readiness. |
| [ ] | K5 | Bounded memory/remap/queue/process correctness and qualified ROCm/Torch references on the candidate; review kernel logs, preserve failures and compare r2. |
| [ ] | K6 | Record qualified scope/artifact hashes, source patch series and reproducible build/boot/recovery instructions; integrate only supported settings and retain r2 fallback. |

Every component ultimately needs a documented disposition: qualified native path,
qualified explicit fallback, or a concrete hardware/API limitation. Missing matrix
instructions themselves are **not pending implementation**. Do not compute a
completion percentage from boxes; an infinite API-combination sweep is not a plan.

## Memory maintenance rule for every future agent

At the end of each meaningful batch, and before handoff/compaction if work is
still running, update this file once. Record: taskID/subtask, actual source and
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
