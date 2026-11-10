# PLAN-07 / D46: Torch patch completeness and SDK artifact provenance

Completed 2026-11-10; **PLAN-07 remains open** for portable source/dependency
closure. No native build, code generation, GPU execution, prefix activation or
runtime improvement is claimed. .105 was unaccessed/unchanged.

## Torch reconstruction repaired

The old0001/0002 series reproduces a tree missing exactly the historical three
SDPA files relative to4ee6571: `sdp_utils.cpp`, `compile_fx.py`, and
`torch/nn/attention/_gfx1013.py`. The files were already present in current Git;
no source implementation was missing there. See [old-gap receipt](historical-gap.json).

The separate [complete series](../../../patches/bc250-pytorch-complete/README.md)
now reproduces the exact complete tracked trees from baselinec779020:

| Export | Qualified revision | Reconstructed tree |
| --- | --- | --- |
| Native source, patch0001 | a5249802372882dedccb91d3fd4d73be9fcdf2bd | 3e3edf7f1de1fd1fe4730f9ec4ed30dd038b2112 |
| Source head, plus patch0002 | 8d8b18df558cb478727ba2179f2e01dfce20ea40 | 33a862732a52c8825a3d1f9c0b609d06fba0df65 |

The35-path manifest distinguishes baseline/native/head bytes. Existing incremental
patches and all Git histories are preserved; never combine old and complete series.
The source-head agent fix remains a pinned D09 bridge in production; the native
installed source revision is a524980. Nothing was upgraded or rebuilt by exporting
these patches.

Two private shared bare/index replays on the workstation pass complete tree
equality. An applicable-but-altered patch produces a different tree and is refused.
These checks require existing baseline objects and do not constitute a portable
object bundle. Torch is a shallow repository with23 reachable commits. D47 must
address independent source availability, not assume shared Git objects are public.

On .104, a new private directory replays the35 affected paths from a baseline
subset and both patches with exact SHA checks at each step. This is focused host
execution, not a full-tree host reconstruction. The original installed source,
builds, caches and Git/index are untouched. Initial unprivileged invocation was
refused by runtime-lock permissions before creating output; the authorized root
invocation acquired the SH runtime lock and passed. No GPU reservation was needed
for this read-only CPU audit.

Both installed profiles match30/35 raw native paths; the same five C++ paths show
CUDA→HIP code-generation changes. A separate retained reference overlay reproduces
all five observed generated files in a scratch directory, byte-exact for both
profiles. This is not an independent HIPify execution or an additional source
patch to apply before building. Only the documented pip build owns code generation.
The profile caches confirm distributed/ROCm/Kineto on, NCCL off daily/on RCCL.

## SDK mismatch resolved without replacement

The active SDK SHAa6dc1c88 differs from retained buildbcbb20f9 exclusively in
`.dynstr`. Zero-padding/replacing the old runtime-path string with the installed
one reproduces the entire installed file byte for byte, SHA256
`a6dc1c884c4f446e23a2afc8185b26ef9fd55b57cdd1d15324d46054b902b9c2`.
All other sections, including code, are unchanged. [Host receipt](host-result.json)
records the old/new RPATH, exact offset129047 and all section hashes.

The retained build has CI=ON and compiler definitionROCPROFILER_CI=1. Its agent
parser, logging header, build-settings and options files match native1f9609fd6
Git bytes. Thus the warning-to-fatal CI mechanism has an artifact/source/config
match; this closes D45's identity uncertainty. It does not make the old unrelated
Python IP0x3b3e6 fault, KFD resume warning or clocksource timeout explained/fixed.

A [minimal SDK proposal](../../../patches/candidates/rocprofiler-sdk-d46/README.md)
handles the exact `-1` terminating token before the fatal branch. It is only
source-apply checked, not compiled, executed or installed. The actual SDK and D44
launcher refusal remain unchanged. Do not describe this proposal as a native fix
already available to users.

## Scope, evidence and next action

Eight qualified Git component pins/trees are available in
[source inventory](source-inventory.json); this is not complete dependency/license/
toolchain/generated-file reconstruction. The library base8389197 plus later MIOpen
overlaye1d8d1f must remain distinct, as must native Torch and its Python agent bridge.

124 local unit tests (four new ELF refusal/replay tests), Black, critical Ruff+F401,
syntax and source/docs diff checks pass. Unified-diff artifacts retain required
blank context markers; patch-specific whitespace settings are used for those
files, while complete source patch application uses `--whitespace=error`.
Complete tree replays, .104 focused source/overlay checks
and the applicable-wrong-tree negative control pass. No full upstream CI or new
GPU correctness/performance claim. Source-only work did not need a native rebuild.

Owned tools: `~/bc250work/tools/source-d46/`; evidence:
`~/bc250work/evidence/source-d46-r1/`; controller log:
`~/bc250work/evidence/source-d46-controller.log`. All CPU jobs finished; no recovery
guard was armed. Native prefixes/libraries/defaults/images/kernels/boot fallbacks/
user apps/finetune-n7/source trees/builds/caches/checkpoints were unchanged.
.104 boot2774d201/r2. Host runtime JSON was read as evidence, not rewritten to
erase historical fields. Current canonical docs take precedence over old labels.

Host archive local `kernel/artifacts/source-d46/104-evidence-r1.tar.gz`,973,572B,
SHA `081b595f248372adc836823ff3c3ca716045aa80ca840b8c422f8359bf3ceca5`;
same archive host `evidence/source-d46-evidence-r1.tar.gz`. Additional workstation
receipts retain the complete-tree and SDK candidate-index checks. SDK candidate
proposal is in local Git; it was not deployed into host component sources.

```sh
python3 therock/build_tools/bc250/verify_torch_d46.py --output /tmp/torch-d46.json
```

**Next PLAN-07 / D47:** create a portable Torch source snapshot/replay with explicit
shallow-baseline and submodule/dependency handling, independent of shared object
stores; inventory licenses/build inputs and matching SDK source/config. Prepare
the candidate SDK build/test admission alongside that closure. At D46 admission,
only41,287,680B remained above the5GiB disk reserve, versus17,575,368B per SDK DSO
plus compiler/link/cache/staging/rollback costs. No build is admitted until the
peak budget fits; bounded RAM scratch and verified workstation retention of owned
temporary transfers may be considered, without deleting sources/builds/caches or
reducing reserves. Four important packages remain, PLAN07–10. Historical unknown
Python/KFD/clocksource issues remain separate known limitations.
