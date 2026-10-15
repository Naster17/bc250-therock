# BC-250 project memory: read first after resume or compaction

Updated 2026-10-15 (local; checkpoints explicitly use UTC). This is the
**canonical working memory**, not a release note.
Detailed hardware, implementation history, evidence and resolved defects:
[bc250_memory_inventory.md](docs/development/bc250_memory_inventory.md).
Old milestone documents are evidence snapshots; their unchecked boxes and active
release claims must not override this cursor and current manifests.

## Resume cursor

- **ACTIVE: K1–K6 kernel workstream; K5/SVM passed; HPD fixed; read-only telemetry audit complete; clock/provenance correction next.**
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
| [x] | K2 | Minimal series reviewed/implemented and normal paths bounded-tested; diagnostic register restore corrected; CPU-unmap returned-error containment bounded-qualified by patch0004; failed-stop/map-locking/telemetry remain open. |
| [x] | K3 | Separate r3 image/modules/config built, signed/staged/installed and artifact-inspected on .104; r2 retained unchanged. |
| [x] | K4 | Bounded r2 watchdog countdown/SSH return and r3 deadline fallback/SSH return exercised; two r3 base boots/cooling ACKs; early-hang/arbitrary GPU-hang recovery unqualified. |
| [x] | K5 | GEM_VA runlist invalidation repaired; unchanged full GPU/Torch/VMM gate and independent HIP/ROCr references pass on signed r3 module8f3c0d3. Historical r2/r3 failures preserved; bounded scope only. |
| [ ] | K6 | Record qualified scope/artifact hashes, source patch series and reproducible build/boot/recovery instructions; integrate only supported settings and retain r2 fallback. |

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
