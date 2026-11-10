# D53: external compiler/ROCm/RCCL payload preservation

This batch exports the current retained native build inputs for D52's source kit.
It does not rebuild Torch, change an active runtime, qualify relocation, or close
PLAN-07. .105 is untouched.

[Receipts](artifacts.json), [verified results and resource admission](verified-results.json).
The real archive passes all28,008 entry checks:24,522 files,3,262 directories,
224 symlinks,10,726,325,706 file bytes; no lexical external/missing links.
Compressed payload2,467,945,440B SHA256
`76a018fae10972070ff414dbe5ef7c55e94f2674076cb86b8d69fd317db28d98`.
The independent archive manifest exactly equals the separately exported host
manifest. HIPClang SHA55833a05 matches D51; RCCL SHAca790dc7 matches the retained
B03/B05 input. Retained SDK SHAa6dc1c88 remains distinct from active D49 SHA2a975370.
There are52 license/copying/notice-named entries, not a compliance conclusion.

160 local tests pass; Black and Python compilation pass. Standalone portable
tools3,964B and raw host metadata1,331,009B have independent SHA256 receipts.
The host's executed verifier predates the final gzip-trailer-only verification
hardening; its **create** behavior is unchanged. The exact executed helper is
preserved as `external_kit_d53-executed-r1.py` in the metadata archive. The final
committed verifier independently checks the real payload, including gzip CRC.
Initial `status.json` is a launch snapshot; final `result.json` says passed.
Controller14238/child14239 finished. Export289.099s, transfer277.299s, not speed
claims; sampled host minimum15,165,878,272B and board disk12,425,912,320B.
No owned job, guard or container remains. Boot2774d201/r2 and D49 selection unchanged.

The original builder image was already preserved in D52; repeating that export
would add no coverage. The missing input was its external read-only build mount.
This batch includes the complete retained `build/dist/rocm` prefix, B03.4 RCCL
install, original Torch CMake toolchain and both ccache helper scripts. Paths in
the inventory describe original inputs, not additional implicit files to fetch.

The archive is streamed through an owned FIFO from .104 to the workstation. No
second 10 GiB SDK tree or archive is placed on the board. Every regular file has
its size, SHA256 and permission bits recorded; directories and symlink spelling
are explicit. Hardlinked input files are stored as independent regular-file
payloads. The exporter inventories inputs again after writing and requires exact
agreement. The workstation separately verifies actual archive content against
that manifest; receipts bind the resulting compressed archive.

This is current retained-byte preservation. It does not establish that these
bytes were present during the historical native build. The compiler comparison
is to D51's explicitly current observation. The repaired active D49 profiler SDK
is a separate qualified override; this archive preserves the retained build SDK
and does not silently replace it with that override.

The verifier reports links outside the archive namespace or without inventory
entries. Presence of all lexical link targets is not proof that every dynamic
ELF dependency, generated include, CMake lookup or compiler invocation succeeds
in a relocated build. Existing image-supplied GCC, CMake, Python distributions
and system libraries remain represented by D52's original OCI image. Binary
payloads and license-named files are preservation evidence, not publication or
complete source/license clearance.

Resources: EX cooperative board reservation and SH current runtime lock; no GPU
nodes, prefix write, native compiler invocation, ccache mutation or recovery
reboot. UID1000 worker uses CPUs0–1, 2 GiB address space and 600 CPU seconds per
inherited process; controller enforces 900 seconds wall, sampled every100ms,
3 GiB host available and 5 GiB board-disk reserves, stopping only its owned
process group. These are not aggregate cgroup limits. Workstation transfer has
an independent900s/5GiB disk guard and a16GiB initial admission. Export time is
job cost, not a performance claim. Cgroup peaks and paging I/O are unmeasured.

The proposed *sequential one-profile* clean-build budget is40GiB:3GiB source/
HIPify,8GiB native objects/codegen,4GiB link/install scratch,20GiB isolated ccache
growth and5GiB preserved reserve. It reuses a read-only SDK without an extraction
duplicate. These are proposed allocation ceilings, not measured sufficient peak
requirements. The current12.4GB free fails this admission; no configure, HIPify,
codegen, pip install or build was launched. Original retained build directories
are only about1.9GiB/profile, which is not a clean-build peak bound. Memory and
closure gates also remain separate.

The initial tiny-fixture tests caught `TarInfo.mode` retaining file-type bits
before serialization; the exporter now explicitly compares permission bits.
No full host export was attempted before that correction. Seven regressions cover
real bytes/hardlinks/link reporting, corrupted payload, changed permissions,
duplicate members, missing manifest entries, gzip trailer damage and budget reserve boundaries.

Local payloads and raw evidence: `kernel/artifacts/source-d53/` (ignored by Git).
Host helpers: `~/bc250work/tools/source-d53/`; metadata/logs:
`~/bc250work/evidence/source-d53-r1/` and `source-d53-controller.log`.
The large payload exists only on the workstation; original inputs remain on .104.
Standalone verification:

```sh
python3 external_kit_d53.py verify --archive external-inputs-d53-r1.tar.gz --output verified.json
```

Next D54: audit the saved external-input/image ELF, CMake and link closure with
bounded CPU-only probes, then establish a storage-admitted isolated clean-build
layout. Reuse the already preserved image/source/payload; do not re-export them
or launch a full build against the current failed budget. Preserve original
sources, caches, prefixes, .105, applications and fallbacks. A revised admitted
layout may reuse the existing ccache with an explicit incremental-growth budget
rather than allocate a new20GiB cache; keep its configured20GiB limit/no reset.
Do not confuse that capacity limit with a measured mandatory build allocation. Historical Python
CPU/KFD/clocksource causes remain open and separate from repaired defects.
