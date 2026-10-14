# BC-250 project memory: read first after resume or compaction

Updated 2026-10-14. This is the **canonical working memory**, not a release note.
Detailed hardware, implementation history, evidence and resolved defects:
[bc250_memory_inventory.md](docs/development/bc250_memory_inventory.md).
Old milestone documents are evidence snapshots; their unchecked boxes and active
release claims must not override this cursor and current manifests.

## Resume cursor

- **Current task:** N1.4 hipSPARSELt zero-stride broadcast contracts implemented,
  installed on .104 and bounded-qualified. N1 remains open; bias is next.
- **Last qualified code:** root `b823c390c5d1c8c1873fdd1fedfb33b0b93e40d2`,
  libraries `40e2c04af95e6f9f95001aaec2a7e5b48b590924` (patch0021).
  N1.3 `04aac4838`/`18983cf7497` and earlier milestones remain prior evidence.
- **N1.4 behavior:** matching logical A/B/C/D batch counts retained (upstream
  contract). Zero stride broadcasts any A/B/C combination from one physical
  matrix; sparse pruning/compression processes it once and stores one values/
  metadata pack reused by all output batches. This avoids repeated sparse writes
  and incorrect per-batch compressed metadata indexing. Multi-batch zero-stride
  D, incompatible counts, negative/overlapping positive strides remain rejected.
  Partial compression, input/D and C/D span aliases are rejected; valid exact
  in-place pruning retained. Compression remains backend-specific1:2/2:4.
- **Passed:**604 CPU-double-referenced GPU cases +5 descriptor rejections, then
  the existing234 sparse API regressions. All7 A/B/C broadcast masks, sparse A/B,
  both orders/all transposes, f32/f16/bf16, strip/low-precision TILE and partial
  tiles, six activations; independent packed decoding, descriptor/plan compression
  equality, sparse dense buffers poisoned before GEMM, exact padding/pack guards.
  Logical batches3 and selected batch1/stride0 cases executed. Kernel logs
  identical1009 lines. No general performance or hardware sparse speedup claim.
- **Evidence/jobs:** `WS/logs/n14-sparselt/result.json`, repository reports
  `bc250_scalar_library_results/n14-{sparselt,reconstruction}.json`.
  Targeted hipSPARSELt+dist,16jobs/20GiB ccache under exclusive runtime lock;
  build6.9s, broadcast3.1s, sparse regression1.6s. First attempt passed all gates.
  No running jobs or outstanding gates; no Torch/hipTensor rebuild required.
- **Provenance/recovery:** `SOURCE_N14.json` verifies8 changed files on N1.3/P2;
  active sparse library SHA256 `12c57c5ebd8b34cf973dcfb4bbe8f2d708c04a3b641dcde5a9e16d932c175b89`.
  Independent prior sparse library `WS/recovery/n14-sparselt-before.so.0.1`
  (backup filename; runtime SONAME file is libhipsparselt.so.0.2), SHA256
  `b89043c024448068eb6137e223fbe68b738c0bd954593cb098d2a67adf8aa41f`.
  All21 library patches reconstruct exact tree. .105/frozen candidates and
  kernel/boot files unchanged; no reboot or host configuration changes this batch.
- **Next: N1.5, hipSPARSELt bias.** Inspect public BIAS_POINTER/BIAS_STRIDE/BIAS_TYPE
  contracts and upstream row/batch mapping (header comments alone are ambiguous),
  then implement a bounded scalar bias epilogue with CPU references for the
  supported real types/layouts/broadcasts and span/alias/lifetime guards. Do not
  revive already-qualified batching or claim new types/scale/split-K support.

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
| Qualified root code      | `b823c390c5d1c8c1873fdd1fedfb33b0b93e40d2`                                                                                |
| rocm-libraries           | `40e2c04af95e6f9f95001aaec2a7e5b48b590924`, ordered patches0001–0021                                                      |
| rocm-systems             | `dbdb7b8d1c56316d873a9b90e548e5a72853496d`, patches0001–0003                                                              |
| Torch                    | `4f13bf86476252e3d57b17d1c72f8512814400e9`, patches0001–0002; editable pip                                                |
| Triton                   | `ee65ff3b85f99d52e678b252a99c8425fa46e329`, patches0001–0004; qualified wheel retained                                    |
| LLVM                     | Clean upstream `7667f52dd687899256a3285ab0bce3c53bc73394`                                                                 |
| Active stack             | `/opt/bc250-stack/current` → `/opt/bc250-stack/dev/priority1`; legacy directory name, **P2 + N1.1–N1.4 dev profile**                  |
| ROCm/Torch/cache         | `WS/build/dist/rocm`, `WS/pytorch-work`, `WS/cache/dev-user` (uid/gid1000)                                                |
| Runtime image ID         | `5df0a58392de13916819f6d84342b0338ae2c2bbb2f3c21a1f3ad38be2724a59`                                                        |
| Runtime/build tags       | `localhost/bc250-stack:10.2-gfx1013-priority1-candidate`; `localhost/bc250-rocm-build:10.2-ccache`                        |
| Source exports           | Gitless `WS/source`, `WS/pytorch-source`, `WS/pytorch-work`; GIT_REVISION + `SOURCE_N14.json` delta on N1.3/P2                |
| Independent P2 candidate | `/opt/bc250-stack/candidates/priority2-20261014`; passed bounded checks, not daily-release activation                     |
| Recovery userspace       | `/opt/bc250-stack/before-dev-current` → frozen `10.2-gfx1013-20261013-4b000d542`; initial a640c9296 release also retained |
| Kernel/boot              | `7.1.5-bc250rocm-r2`; live boot files in `/home/nik/bc250work/bc250-rocm-lab`, backups `WS/recovery/`                     |
| Memory backup            | `WS/agent-memory/latest` (documentation only; preserves qualified source snapshots)                                       |
| Training artifacts       | `WS/training-model`, `training-env-stable`, `training-validation`, `logs/training-memory/`                                |

Manifests and immutable image IDs outrank mutable tag names. Component version
strings differ legitimately (TheRock10.2, HIP7.17, Torch 2.13). Source archive names
and generated Torch Git labels may retain older baselines; do not relabel them.
The P2 source transfer verified51 files; N1.1 adds7; N1.2/N1.3/N1.4 each add8 verified changed files. Candidate payload verified191
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
| [ ]   | **N1 / N1.5 — next** | hipSPARSELt bias pointer/stride/type contracts and bounded native bias epilogue, with CPU-reference layout/broadcast/alias tests. Remaining sparse types/vector scaling and Lt small/skinny shapes/epilogues follow. |
| [ ]   | N2                   | Broader scientific/primitives/RPP coverage: solver difficult/complex cases and selection, sparse formats/analysis/solves, FFT plans/layouts, RNG distributions, floating/custom/segmented algorithms, admitted CK DL conv/grouped operations. Qualify actual operations, not architecture lists.                                                                                                    |
| [ ]   | N3                   | Torch functional/performance expansion: larger meaningful training sequences/batches, full-model AMP, compiled/eager gradient equivalence, optimizer/checkpoint resume, quantization/QLoRA applicability, feasible scalar tiled attention and hipDNN mixed/fused graphs. Keep mathematical SDPA working; unsupported matrix backends stay off.                                                      |
| [ ]   | N4                   | Trusted read-only telemetry, RDC applicability, debugger consumers/suites, additional tracing and physical counter mapping. Resolve clock inconsistencies before explaining small speed changes; dangerous debugger tests require recovery first.                                                                                                                                                   |
| [ ]   | N5                   | Build native RCCL/rocSHMEM/MPI local readiness; feasible generic/software transports with GDA/SDMA/XGMI/P2P claims off until demonstrated. Single-GPU readiness is not full collective coverage. `.105` migration/two-board distributed tests wait for explicit later authorization and stability.                                                                                                  |
| [ ]   | N6                   | Guarded kernel/runtime investigations: experimental VM-heap repair, SDMA, reset/fault recovery, XNACK/wave64 where supported, advanced profiling. Preserve known boot/stack defaults; isolated opt-in gates and independent recovery required.                                                                                                                                                      |
| [ ]   | N7                   | Remaining optional feasibility: host media/HIP interoperability, actual media rings, hipThreads/Fortran execution, hipify-clang, hipFile registration/concurrency/direct-storage applicability, rocRoller/emulation and other pinned projects. Explicit unsupported/not-applicable decisions where hardware/API prevents a path.                                                                    |
| [ ]   | N8                   | Final integration/release: source reconstruction, independent relocatable packaging, licenses/tools/plugins/kernel packs, selected Torch regressions, agreed bounded stability/training/memory gates, kernel-log review, recovery verification and truthful operation matrix. Longer runs only when relevant/user-agreed.                                                                           |

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
