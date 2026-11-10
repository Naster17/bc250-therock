# D52: preserved builder and fresh selected source/recipe snapshot

Completed 2026-11-10 for preservation/assembly scope. Active runtimes and .105
unchanged. PLAN-07 remains open for matching external SDK inputs and build closure.

## New portable artifacts

[Exact receipts](artifacts.json), [verification](verified-results.json).

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| Original builder OCI archive | 682,451,968 | c57dbec6f76fc0f651eca005435320f47092f8f4dc478026053d058390998f88 |
| Gitless source/recipes | 346,534,969 | 6867ed719c201cf7bc368800aa64c5e16a2e3a4c3c03485ad502684e2feb5723 |
| Portable stdlib verification/extraction tools | 4,936 | f06f5b23578e7223bbeb299cbfdf1d1b6409e1f9de75d5899da01b9f6b48116e |
| Raw metadata export | 860,815 | c36a2f45c766f5adfe160f0112a2f2b697d8e9fea1df277885adc792e66e1225 |

Workstation artifacts are in `kernel/artifacts/source-d52/`, intentionally ignored
by Git. The source/helpers/evidence receipts are committed; binary preservation
archives are not publication approval or complete license closure.

The OCI archive preserves config/image identity
`28ad36e62398875f0ad77112b1d0f2b1cd0b3b55560d9cc38c6908cada8caa06`.
All14 stored layer hashes and their uncompressed diff IDs match. Both .104 and
workstation independently stream-verify the real archive. The existing Podman
image ID was checked unchanged. No load, pull, retag or image replacement occurred.
HIP Clang and ROCm libraries mounted from the external build prefix are **not**
embedded in this image; D51's image/toolchain distinction is retained.

The fresh assembly replays the D47 parent and D50 available dependencies, validates
D51 input evidence, materializes69 dependency locations and checks copied working
bytes, modes and symlinks. Canonical Git trees remain separately verified:
source head8d8b18d/tree33a86273 on native C++ basea524980/tree3e3edf7f. The private
Gloo overlay tree913f01fb is preserved. [Assembly proof](assembly-proof.json).
Eight excluded branch directories remain empty; arbitrary configurations are not
admitted. Original source/build/cache trees were never used as write targets.

The archive has100,099 source entries:21,501 parent and78,598 dependency blobs
including repeated locations. With recipes,100,116 entries have manifest-backed
byte/mode/link checks. Private Git indices/histories remain in the owned proof
folders and are excluded from the portable snapshot. The actual archive was
verified locally, extracted into a new path containing a space, and every extracted
entry rechecked against its manifest: [extracted result](extracted-bytes.json).

## Assembly defects found and repaired

The first copy check failed ideep/mkl-dnn's `cmake/template.vcxproj.user`. Its
verified archive working bytes are331 bytes, while Git's canonical blob is324:
the pinned attributes specify text/eol=crlf. The corrected checker requires copied
bytes to match the already tree-verified working snapshot, while preserving the
independent canonical Git tree checks. Parent working bytes are captured before
materialization and compared afterward. Nothing normalizes or patches that source
file. R1 remains preserved; R2 passes all69 locations and the complete parent.
A regression rejects normalizing the copied CRLF bytes to LF.

The UID1000 worker now has its correct home and an isolated XDG config path, avoiding
irrelevant root Git attributes warnings. A read-only diagnostic initially hit
Git's ownership guard; repeating it as the owning UID confirmed the attributes.
No global safe.directory/config exception was added.

The initial link-rule inventory missed targets ending immediately before the Ninja
colon. A parser regression covers this; corrected R2 records38 native library link
rules per profile, excluding phony aliases. Initial R1 inventory is preserved.

## Fresh recipes and retained input inventory

The old B05.2/B05.3 scripts required a pre-existing CMake cache. New prospective
`recipes/build-fresh-{daily,rccl-single}.sh` remove that copy and refuse an existing
cache before any ccache/build work. The RCCL recipe uses explicit `/workspace/rccl`
in place of the old host-bound install path and checks its CMake package directory.
They retain native gfx1013, safe backend settings,20GiB ccache/no reset, four jobs/
one linker, explicit build-version/number and the authorized pip-only build.

Both scripts pass `bash -n` inside the preserved build image, read-only/no GPU.
They were **not executed**: there was no pip install, configure, HIP generation,
compiler invocation or native rebuild. Historical caches/graphs under the profile
recipe subdirectories are evidence, not caches to copy into a fresh build.

Retained compilation-unit inventory: all3,239 daily and3,241 RCCL input paths exist
and have recorded SHA256 values;260 per profile lie under their build directories.
Include-root inventories contain103/105 paths; declared native link-rule counts38
each. Mapping uses the actual longest matching original container mounts, including
read-only third-party overlays. These are current retained-file observations,
not historical compiler-byte identity, a complete transitive header dependency
walk, or complete linked ELF closure. No `.ninja_deps`/all-system-header claim.

## Resources and evidence

153 local tests pass, nine new cases covering OCI config/diff IDs/unreachable
blobs, source mode/content/EOL drift, nested boundaries, fresh recipes, link-rule
parsing and exact Gitless extraction. Black, critical Ruff+F401 and diff checks
pass. Metadata is cross-checked against the independently verified actual OCI and
source archives; synthetic changed-identity/diff/content cases are refused.
The archived pack runner precedes a nonbehavioral unused-import cleanup.

Image export admission used `2 * image_size + 4GiB source assembly + 5GiB reserve`;
original image size1,946,482,726B and initial host free21,360,730,112B passed.
Export: EX board reservation+SH runtime lock, CPU0–1/GOMAXPROCS2, own scratch,
420s save deadline/100ms3GiB host+5GiB disk checks. No aggregate export memory cap
is claimed. CPU assembly and packaging: UID1000/affinity0–3/2GiB address space+
180CPU seconds per inherited process/480s wall/100ms reserves/owned PG stops.
These are process limits, not cgroup aggregate limits. No GPU mappings/reboot guard
was needed. Image export36.491s, final assembly99.852s, packaging146.687s are job
costs, not training/build performance measurements. Exact sampled minima are in
verified-results.json; cgroup peak/paging I/O were not measured.

Host locations under `~/bc250work/`:

- `evidence/source-d52-image-r1/native-builder-28ad36e6.oci.tar` and result/save log.
- `evidence/source-d52-assembly-r1` failed proof and encoding diagnostic, preserved.
- `evidence/source-d52-assembly-r2/assembled/{torch,dependency-proof,recipes}`;
  source checks/assembly.json/result.json/worker.log are retained.
- `evidence/source-d52-inputs-r1/r2/result.json` retained unit/include/link inventory.
- `evidence/source-d52-package-r1/torch-source-d52-r1.tar.gz` plus package/result/
  recipe-syntax JSON and worker log.
- `tools/source-d52/`, including executed-r1 helpers and portable-tools archive.
- `evidence/source-d52-evidence-r1.tar.gz` exports metadata/tools only, not large
  source/image/proof contents. Controllers7158/7744(failed)/8899/10275 finished;
  logs `source-d52-{image,assembly,assembly-r2,package}-controller.log`.

No owned workload remains. Active SDK2a975370, boot2774d201/r2, native board
controls/watchdog/boot fallback and unrelated applications remain unchanged;
.105 was not accessed. Historical Python CPU/KFD/clocksource causes remain open,
separate from fixed SDK sentinel/VM-heap/D05/D21 defects.

## Replaying artifacts and next action

Extract the tiny portable-tools archive into a new directory, check external
receipts, then use its standalone Python3.12+ tools:

```sh
python3 portable-tools/verify_oci_d52.py native-builder-28ad36e6.oci.tar --output image.json
python3 portable-tools/source_kit_d52.py verify --archive torch-source-d52-r1.tar.gz --output source.json
python3 portable-tools/source_kit_d52.py extract --archive torch-source-d52-r1.tar.gz \
  --destination /new/private/source --output extracted.json
```

Next D53: stage the matching external ROCm/compiler/RCCL/toolchain inputs with
storage admission and immutable identities, then perform a clean-build resource
admission for this fresh source/recipe kit. Do not execute archived historical
cache-dependent scripts or overwrite active prefixes. PLAN-07 payload/license/
clean-reproduction closure remains open; four important packages PLAN07–10 remain.
