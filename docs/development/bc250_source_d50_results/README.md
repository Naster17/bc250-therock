# D50: portable recursive Torch dependency contents

Completed 2026-11-10 for the available-content scope. PLAN-07 remains open.
No native build, GPU test, prefix update, reboot, image replacement or .105 access.
D49's repaired SDK remains active on .104.

## What is now portable

D47 exported the tracked Torch parent but only recorded dependency identities.
D50 exports the available recursive dependency contents and license texts, with
an independent replay that needs no original checkout, Git history or network.

| Item | Result |
| --- | --- |
| Direct dependency pins anchored by the D47 manifest | 37 |
| Discovered graph locations | 77 |
| Available locations / unique commit payloads | 69 / 64 |
| Tracked blob files in unique payloads | 75,522 |
| Tracked license/copying/notice-named files | 107 |
| Missing, unexpanded branches | 8 |

Each payload includes a tracked-tree archive and the raw pinned commit object.
The replayer checks the commit object's Git SHA-1 against its parent gitlink,
binds its tree, reconstructs the complete tree in a new private index and checks
all gitlinks, file modes, symlink bytes and tracked license entries. The D47
manifest's pinned SHA256 anchors the 37 direct dependencies. Duplicate commit
payloads are stored once. Private repositories have no inherited objects/history;
these verification indices are not assembled native build directories.

The 107 files are preserved source license material, not a completed SPDX/SBOM or
legal review. Embedded notices and differently named license files still require
the release license inventory. Unknown descendants beneath missing branches are
not counted as explored; **complete_recursive_source_closure=false** is deliberate.

## Missing branches

Exact pins/reasons are in [dependency-manifest.json](dependency-manifest.json).

| Parent | Unavailable nested checkouts |
| --- | --- |
| aiter | composable_kernel |
| flash-attention | composable_kernel, cutlass, aiter |
| mslk | composable_kernel, cutlass, googletest, hipify_torch |

Original uninitialized source paths were left untouched; no submodule reset,
fetch, deletion or speculative backend enablement occurred. Retained daily/RCCL
CMake caches have FLASH_ATTENTION=0, MEM_EFF_ATTENTION=0 and MSLK=0. These flags
are evidence about the selected builds, not proof that all optional source
branches are unnecessary for every future configuration. Source/disposition and
required build inputs must be separated before declaring PLAN-07 closed.

## Packaging defect repaired

The first workstation replay failed the exact ittapi tree check: Python's tar
`data` filter normalized symlink bytes `../../` to `../..`. The new filter retains
the original trailing-slash spelling only after the standard filter validates the
same target boundary. Other target normalization is refused, not silently accepted.
The failed replay is preserved as `kernel/artifacts/source-d50/replay-r1/` and
`replay-r1.log`. R2 passes all 64 unique payloads; .104 independently passes the
same package. A focused regression covers the literal symlink bytes.

## Gloo and build-version evidence

The existing B03.1 legacy-platform patch is included as a separate overlay,
never applied to the original local/host sources. Private replay produces Gloo
tree `913f01fbb91a02ba2e30284fbe744e6b746e66f9`; cuda.cu SHA256
`7dc2ce53bf518ba455b0e47e7b2e09976261db008b7d2a83669f02c9451f997f`
exactly matches the retained .104 `pytorch-work/third_party/gloo/gloo/cuda.cu`.
This is source reconstruction, not a new native compilation or GPU proof.
TensorPipe and its initialized recursive dependencies, including pinned libuv,
are among the verified contents.

[Raw host result](host-result.json) binds both retained CMake caches and generated
`torch/version.py`: version `2.13.0a0+gfx1013`, HIP `7.17.0`, ROCm `10.2.0`,
`git_version='Unknown'`. Do not use that last field as source provenance: native
source a524980 and later Python source8d8b18d are separately pinned in the workspace
manifest. No installed version file was rewritten to invent a native rebuild.
For a future portable build, explicitly record PYTORCH_BUILD_VERSION and
PYTORCH_BUILD_NUMBER plus source/tree IDs; the pinned generator reads those before
PKG-INFO/version.txt. Original environment values are not recoverable merely from
this generated file. Toolchain, Python wheel/build dependencies, generated inputs
and full build recipe closure remain pending. No code generation was invoked.

## Tests, resources and artifacts

140 local tests pass, including six new commit/tree/payload/ambient-Git/path/
symlink regressions. Updating a modified archive's SHA still fails the pinned
commit/tree gate. Black, critical Ruff+F401 and diff checks pass. The independent
metadata reader binds archived driver/manifest, raw CMake/version/Gloo bytes and
resource samples; three altered driver/Gloo/CMake exports are refused.
[Metadata replay](verified-export.json), [negative controls](negative-controls.json).
The full source replay and metadata verification are distinct checks.

.104 CPU replay cost54.477s, not build/training performance. UID1000/CPU affinity0–3,
2GiB address-space and120 CPU seconds per inherited process,420s wall deadline;
100ms sampling and owned process-group stop below3GiB host/5GiB disk reserves.
These are process limits, not a cgroup aggregate-memory guarantee. Minimum sampled
host available14,889,078,784B; disk21,882,839,040B. SH runtime lock only for retained
metadata/provider audit; no GPU reservation or reboot guard was needed.
Controller30288 finished; no owned job remains. Existing caches/builds/sources,
boot fallbacks and user applications were preserved; .105 unchanged.

Portable package, workstation `kernel/artifacts/source-d50/portable-dependencies-d50-r1.tar`,
host `~/bc250work/tools/source-d50/portable-dependencies-d50-r1.tar`:258,723,840B,
SHA256 `1c624f9d672ef3798a72b9974a6778b3a58c36bcafeefc4d06a04da0781e5540`.
[Receipt](package.json), [local full replay](local-replay.json).
Host package/execution: `~/bc250work/tools/source-d50/dependencies-r1/` and
`~/bc250work/evidence/source-d50-r1/{replay,replay.json,result.json,worker.log}`;
controller log `~/bc250work/evidence/source-d50-controller.log`.

Host metadata archive copied to `kernel/artifacts/source-d50/104-evidence-r1.tar.gz`:
82,862B SHA256 `06262318fe63377a24f9ba82682ac47c0a348f8195ae934cb4165683c1eb96b6`.
Its full dependency contents/private replay are retained separately, not hidden
inside this small metadata archive. The task-owned /tmp transfer was copied and
SHA-verified into the durable tools directory; only that duplicate was removed.

```sh
# Extract package into a new parent directory first, then:
python3 /new/package/dependencies-r1/portable_dependencies_d50.py replay \
  --package /new/package/dependencies-r1 --destination /new/private/replay \
  --output /new/replayed.json

python3 therock/build_tools/bc250/verify_source_d50_export.py \
  kernel/artifacts/source-d50/104-evidence-r1.tar.gz \
  --package-receipt kernel/artifacts/source-d50/package.json --output /tmp/d50.json
```

Next D51: classify the eight missing branches against the exact selected native
build configuration, package required missing pins using private object stores,
and close the recorded build-version/toolchain/Python-wheel input inventory.
Keep .105 unchanged. Historical Python CPU/KFD/clocksource causes remain unknown;
D49 fixed the distinct SDK sentinel crash, and D50 makes no new crash/stability claim.
