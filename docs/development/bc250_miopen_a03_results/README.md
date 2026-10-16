# A03: genuine native MIOpen databases and optional-plugin probing

Implemented and installed on .104 on 2026-10-16. Libraries
`fcf7a7142af8c258164e106e66bf893c7f563d15`; source/helpers TheRock `7d9176a45`
(initial packaging implementation `452ac72e7`). Torch4ee6571, ROCr329eaf7,
Triton ee65ff3b, image5df0a58, kernel0957f1/image84995f1, boot a287fd16 and
SDMA0/heap-off/math defaults are unchanged. The frozen software fallback, r2 boot
fallback, BIOS, .105 and user files remain intact.

## Changes

The optional grouped-convolution loader previously attempted a plugin at handle
creation even with `MIOPEN_USE_COMPOSABLEKERNEL=OFF`. It now skips the probe when
the build disables CK or the existing CK solver device whitelist excludes the
GPU. Explicit `MIOPEN_CK_LIB_PATH` still permits development plugins and retains
real loading/symbol/version diagnostics. Native gfx1013 remains native; no plugin
was fabricated or enabled. The absent modern CK plugin remains optional B07,
separate from the working A02 bundled legacy CK reductions.

Collected new private databases using native GPU convolution calls with CPU-double
references. Every solution returned by the immediate API was executed and checked.
One small FP32 ConvHipDirectFwd search produced a real parameter record; a subsequent
unforced normal Find checked its available alternatives before packaging. Production
selection is unforced. Existing historical caches were inspected and preserved,
not promoted wholesale: [old cache inventory](old-cache-inventory.json).

| Installed asset | Bounded contents |
| --- | --- |
| gfx1013_20.HIP.fdb.txt |16 exact problem keys, 58 checked immediate solutions |
| gfx1013_20.db.txt |One searched ConvHipDirectFwd performance configuration |
| gfx1013_20.kdb |41 compiled native kernel records, including search candidates |
| BC250_A03_MANIFEST.json |Source, input geometry, dtype, IDs, hashes and qualification limits |

`gfx1013_20` is HIP's actual WGP-mode database key for this unlocked 40-CU board;
it was not renamed from another GPU. Text records preserve genuine solver data.
The kernel DB is a consistent SQLite backup of the new user kernel cache, with
journal mode converted to DELETE for read-only installation; code blobs retain
their original hashes. The original WAL cache remains untouched. A03's initial
WAL-mode package failed read-only replay, was retained, and was never installed.

## Evidence and scope

| Gate | Result |
| --- | --- |
| Plugin probe |Baseline emits missing-plugin warning; candidate is quiet by default; explicit invalid plugin path still warns and handle creation succeeds |
| Collection |15 geometry/dtype/direction cases, 55 immediate solution checks pass; separate searched solver and then unforced normal case with three solutions pass |
| Final package replay |15 +1 configurations, 58 solutions pass CPU references; empty user FindDb/PerfDb directories, zero new user kernels; system PerfDb load of searched parameters logged |
| Installed normal paths |Same 58 solution checks pass with default system directory and no candidate preload/system-path override; fresh user caches remain empty |
| Unseen shape |Three forward/dx/dw cases pass through normal fallback and write three genuine new user FindDb records; system package hashes unchanged |
| Regressions |30 Torch convolution cases, A02 three-case reduction/LSTM smoke and seven core reference groups plus pinned consumer imports pass |

Fixtures: NCHW 3x3 convolution with FP32/FP16/BF16 B2/C4/H13/W11/K8;
FP32 groups2 B2/C8/H9/W7/K8; FP32 dilated B2/C4/H1/W19/K6, kernel1x3,
stride1x2/pad0x2/dilation1x2; searched FP32 B1/C2/H5/W5/K2 3x3. Each main fixture
covers forward, input gradient and weight gradient; the searched fixture is forward.
The unseen FP32 B1/C3/H7/W9/K5 3x3 fixture was deliberately not packaged.
Inputs and upstream gradients are quantized to their tested storage dtype before
CPU-double references. Tolerances: FP32 4e-4, FP16 .006, BF16 .04; individual errors
are recorded in result JSON. These counts overlap earlier milestones.

The kernel cache includes compiled search candidates; it does not establish that
every blob/configuration was numerically checked. This is a small seed database,
not architecture-wide tuning, a universal operator guarantee or a speedup claim.
Unknown geometries retain ordinary search/compile fallback. Existing user records
keep normal MIOpen precedence. Other boards, compiler versions, layouts and wider
solver/type contracts remain unqualified. A01 clock/absolute-trace/reset limits
and prior GPU-event speed-ratio remeasurement remain. Historical Python CPU fault
is unresolved and distinct from fixed VM-heap/CSR corruption.

All16 GPU gates ACKed 600s guards and have identical kernel logs, including the
failed initial SQLite replay; see [gates.json](gates.json). Every before/after log
SHA256 is `53f83f7bbee6c060e3153e7ae95a4ff069bf9627113591c39cdddb9ab5f4b973`.
An initial deployment precheck incorrectly treated expected Info-level notices
for empty private user caches as system warnings; it stopped before changing the
prefix. The retained correction checks Warning severity. Actual missing/corrupt
system-DB diagnostics are not globally suppressed.

## Deployment and reproduction

Targeted MIOpen build: 16 jobs, existing 20GiB ccache, no GPU mapping into builder.
Deployment held the exclusive prefix-update lock, staged only MIOpen's runtime
component with installed RPATH, atomically replaced its versioned DSO and added
only the three native DB files plus manifest. Automatic rollback restores A02's
DSO/runtime metadata and removes only the newly created package files if a gate
fails. Rollback was not required. Independent backups remain at
`WS/recovery/miopen-a03-before`. Eleven protected HSA/HIP/Torch/BLAS/kpack/frozen-
fallback/shared-heuristic hashes remain unchanged.

Installed DSO SHA256:
`3a751adc149778f061198014d6f086a75c33e7640b9473dd75bf14464652afce`.
[Deployment manifest](miopen-a03-deployment-r2/result.json) records every payload
hash and installed gate. `WS/SOURCE_MIOPEN_A03.json` is copied into the active
prefix; component export GIT_REVISION updated with its original retained.
The unchanged superbuild baseline remains separate from the effective port revision.

Source helpers: `validate_miopen_a03.py`, `package_miopen_a03.py` and
`deploy_miopen_a03_on_host.py` under `build_tools/bc250/`. Guard command/environment
records retain exact collection, search and replay invocations. Packaging is
reproducible from those GPU-generated private caches; no empty/fake records are
created. Selected text assets and full package provenance are in [database/](database/).
The complete binary bundle is retained on .104 at
`WS/build/bc250-a03-db-package-r3/` and locally under ignored
`kernel/artifacts/miopen-a03/bc250-a03-db-package-r3/`; binary hashes are committed.
All30 library patches reconstruct exactly; [reconstruction](reconstruction.json).
The prior Torch SDPA patch-bundle gap is separate; full qualified Git/source remains.

Full logs: .104 `WS/logs/miopen-a03-*`, locally ignored
`kernel/artifacts/miopen-a03/host-logs/`. Source trees, caches and prior packages
remain. No agent build/test/container/deadline remains.

Applies automatically to bc250-torch/bc250-train and other callers loading the
current development MIOpen. Other prefixes/venvs and frozen fallback are outside
this deployment. Exact next **A04**: inspect the existing hipSPARSELt Torch CMake
package discovery/version gate, then targeted patched-Torch integration and actual
compress/sparse-mm CPU-reference contracts. A04–A09 precede N2.7; K6 stays completed.
