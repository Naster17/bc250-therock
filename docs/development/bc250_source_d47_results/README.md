# PLAN-07 / D47: portable Torch parent sources and SDK build admission

Completed 2026-11-10. PLAN-07 remains open for recursive dependency/build-input
closure. .104 executes the complete parent-source replay without access to the
workstation's Git objects. Native Torch/ROCm/SDK/kernels/defaults remain unchanged.

## Portable artifact

`kernel/artifacts/source-d47/portable-torch-source-d47-r1.tar` is66,703,360B,
SHA256 `ffb18da3ea5e79349965c467d1ef2194a9bbe2863a93d01e08f1f6245d8e5143`.
It contains the complete tracked baseline parent source, both D46 patches,
manifest, standalone stdlib/Git replayer and usage instructions. Main LICENSE is
included and verified as Git blobc23172f7aff0254e4f0f163fb2e6e355cbaec5f4.

The replayer extracts into a new destination, initializes only that private Git
index/object store, reconstructs37 gitlink identities and verifies all21,491 parent
blob files through complete tree equality at every stage:

| Stage | Provenance revision | Exact tree |
| --- | --- | --- |
| Baseline | c779020750d7dc99a39ea8c104ef8df1fc9d0755 | abdd861bee54ca83fd7ed9a742dd9db1b7536831 |
| Native | a5249802372882dedccb91d3fd4d73be9fcdf2bd | 3e3edf7f1de1fd1fe4730f9ec4ed30dd038b2112 |
| Source head | 8d8b18df558cb478727ba2179f2e01dfce20ea40 | 33a862732a52c8825a3d1f9c0b609d06fba0df65 |

The new repository has no object alternates, shallow boundary or commit history.
Its HEAD is unborn: this is a verification index, not a configured build tree.
Original commit identities are recorded as provenance. A future native build needs
explicit build-version/export metadata and dependency/configuration setup; do not
invoke pip directly in this verification directory. The original shallow source
repository, its histories and working tree were preserved.

```sh
tar -xf kernel/artifacts/source-d47/portable-torch-source-d47-r1.tar -C /new/package/location
python3 /new/package/location/portable-torch-r1/portable_torch_d47.py replay \
  --package /new/package/location/portable-torch-r1 \
  --destination /new/private/replay --output /new/replay.json
```

These paths are examples; package parent paths must exist and the replay
destination must not exist. Requires Python3.12+ and Git supporting SHA-1 objects.
No original repository, network fetch or source history is required for replay.
The payload SHA and three tree identities are checked before accepting results.

## Dependency boundary and tests

All37 direct dependency commit pins are available in the local initialized
repositories, with heads matching parent pins. The manifest records53 license/
copying/notice file identities from those pinned trees. Dependency contents are
not included in the parent artifact. There are27 immediate nested gitlinks across
those dependencies; recursively exporting/validating their contents and license
texts remains closure work. This is explicit metadata, not an offline-build-ready
whole-stack claim. Preserve the existing Gloo legacy-platform overlay and nested
TensorPipe/libuv provenance from B03.1 when assembling the build inputs.

Workstation full replay passed. .104 r1 replay also passed; a code review then
extended ambient-Git-environment isolation to every Git invocation (including tree
listing/export), with a regression for foreign GIT_DIR/GIT_INDEX_FILE. R1 source
copies/outputs remain preserved. Final r2 replays the same payload with the hardened
driver and passes again. R2 CPU execution12.398s includes read-only SDK admission;
this is test cost, not build/training performance. Lowest sampled host available
15,200,382,976B, disk24,443,985,920B. Disk increased during the batch as the user
freed space; this task deleted no files or caches.

CPU replay ran as UID1000, affinity0–3,2GiB address-space and120 CPU seconds per
inherited process,180s wall deadline. These are process limits, not a cgroup-wide
memory claim. Parent sampled100ms and stops only its own process group if host
available falls below3GiB or free disk below5GiB. No GPU mappings, recovery reboot
guard or native build was involved. Read-only SDK metadata uses the SH runtime lock.

128 unit tests pass, including four new payload/tree/archive-escape/ambient-Git
regressions. An actual altered-but-applicable source patch with its payload SHA
updated is still refused by the pinned native tree. Black, critical Ruff+F401,
syntax and source/docs diff checks pass; no full upstream suite or GPU claim.
Source-package and host metadata receipts agree on driver SHA and all three trees.

## SDK candidate admission

The retained compile command and native link argv are now captured as structured
arguments. All181 explicit link inputs are accounted for:178 exist in the host
build export, while `libpthread.a`, `libdl.a` and `librt.a` exist in the matching
glibc image1bb4ac0e. The Alpine host's initial path check correctly reported those
three as missing. One stdlib-only container inspected them with image pulls and
network disabled, read-only root, no GPU mappings,1CPU/256MiB/64PIDs, then exited.
This is input verification, not a compiler or ABI execution test.

[Admission](sdk-build-admission.json), [argv/input ledger](sdk-recipe.json),
[image proof](sdk-image-inputs.json). The original argv target the retained build;
**do not execute them directly**. Candidate source/object/depfile/library outputs
must be redirected to an owned directory, with original source/build mounted
read-only in the recorded image. Prefix the compiler with ccache: compile_commands
omits the launcher while the retained Ninja command includes it. Keep20GiB ccache
configuration/statistics, jobs1/link1,4GiB candidate memory and a conservative4GiB
extra disk envelope. Recheck actual resources before starting; retain3GiB host and
5GiB disk reserves and an owned-stop policy. D47 begins no SDK build or activation.

Final preflight: idle except allowed native cyan-skillfish service, watchdog active,
no Docker/Podman workloads. Controllers1130/2722 finished; no owned build/test job
or armed guard remains. .104 boot2774d201/r2 unchanged; .105 unaccessed/unchanged.
No prefix, native source, library, image, boot fallback, app, cache, build or
checkpoint was replaced. D44 refusal remains active on .104.

## Evidence and next action

Host source package: `~/bc250work/tools/source-d47/portable-torch-r1/`.
Private replays/evidence: `~/bc250work/evidence/source-d47-r1/` and `source-d47-r2/`.
R1 includes SDK compile/link/image proofs; r2 is the final hardened source replay.
Local [package receipt](package.json), [manifest](source-manifest.json),
[host result](host-result.json), [negative control](negative-control.json).
Host metadata archive (full source/replay directories omitted):
`kernel/artifacts/source-d47/104-evidence-r1.tar.gz`,98,529B, SHA256
`29dc89e752b280504fa5aeb1a6bd0608bd4e0771b6bf9199f2c597c3720809e7`.
Same host `evidence/source-d47-evidence-r1.tar.gz`; both source replays are retained.

**Next D48:** build the isolated native SDK sentinel candidate from the verified
source/recipe after fresh ownership/resource checks; redirect every output and
preserve CI checks. Qualify direct HIP/CUDA/ROCR `-1` import/device-count behavior,
valid0 selection, GPU output/backward and CPU-first/native profiler traces in both
profiles. Bind actual loaded DSO paths/hashes, retain independent recovery for GPU
tests, and keep the active SDK/D44 refusal unchanged until qualification and prefix
update checks pass. Recursive dependency/source/license/build-version closure
continues under PLAN-07; four important packages remain. The historical Python
CPU IP0x3b3e6, KFD mm/queue and .105 clocksource causes remain open and distinct
from the specific SDK import defect and already fixed VM-heap/D05/D21 defects.
