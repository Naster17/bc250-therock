# .104 cleanup and release development baseline

The cleanup reclaimed **44.20 GiB** on .104. Measured free space rose from
11,902,251,008 B to 59,362,246,656 B before the final documentation snapshot.
The full removal lists and measurements are in [verified-results.json](verified-results.json).

| Retired material | Count | Recovery |
| --- | ---: | --- |
| Unused intermediate ELF objects from superseded builds and completed tests | 18,695 files | Source, configuration and compiler cache retained for recompilation |
| Redundant source proof copies | 5 trees | Verified source-proof archive, including private Git metadata |
| Closed staging, evidence, helper and loose-file roots | 654 roots | Verified staging archive |
| Superseded documentation snapshots | 57 | Verified docs archive |
| Legacy ROCm workspace copies and launcher fixtures | 21 roots | Verified workspace archive |
| Duplicate builder OCI export and finished transfer staging | 4 entries | Workstation OCI and original Podman builder retained |

The 654 roots included 26 directories under `rocm-10.2-gfx1013/tmp`, 115
closed evidence directories, 56 superseded helper directories and 457 loose
evidence files. The main workspace pass also retired old `staging`, progress
snapshots, copied source fragments and synthetic launcher fixtures.

The archives are in `kernel/artifacts/storage-cleanup-20261110/` on the
workstation. [artifacts.json](artifacts.json) records each archive SHA256.
Their manifests record the original host roots and path identities. The staging
archive stores equal payload bytes once and uses checked tar hardlink references.
Copy recovered source files to independent working files before editing. Legacy
links retain their spelling. Review external targets before restoring a tree.
These recovery archives are separate from the source files intended for Git.

## Host layout

- `rocm-10.2-gfx1013/tmp/` contains a usage note and is available for fresh scratch.
- `tools/source-d52/package-r1/` holds the verified Torch source kit. The old
  evidence path resolves through a compatibility link.
- `evidence/maintenance/20261110/` contains the grouped receipts, tools and
  retirement journals. Large JSON records are gzip-compressed and verified.
- The distributed registry reports and D35/D41 numerical reference tensors
  remain at their established paths. Native source, build, cache, application and
  recovery trees stay available at the paths used by existing callers.

## Source reconciliation

The tracked-file audit checked the .104 TheRock, ROCm Libraries, ROCm Systems,
two Torch and Triton exports against their recorded Git baselines and current
local commits. [source-audit.json](source-audit.json) contains counts and the
installed host matrix.

The audited ROCm differences resolve to existing Git objects or 23 materialized
LFS assets whose sizes and SHA256 values match their pointers. The loose FFTS,
CLR, ROCm device and rocPRIM copies have local Git identities. The staged PLAN02
MIOpen solver input matches the local solver file byte for byte. The three audio
patches match commits 8f0f90f, 045081c and 1ae7f5d by stable patch ID. The old
B05.6 staging helper is preserved with its original image pin as historical
material.

Both Torch profiles produced the same raw variant list. After accounting for
known Git blobs and CRLF conversions, 22 matching variants show HIP translation
in the reviewed diffs. Their exact bytes are in the code-variant archive for
D54 comparison. Keep the authored sources in Git and reproduce the generated
changes through the recorded HIPify build route. The audit also records missing
baseline paths. Independent clean-build closure remains D54 work.

## Checks and recovery records

The live dependency audit inspected 69 runtime ELF objects, 43 SDK ELF objects,
340 retained Ninja/CMake files and the installed registries. The retirement
checks also inspected mounts, open file descriptors, mappings, working
directories and retained symlink references. No selected root was held by those
checks. Deletion used the exclusive board reservation and runtime lock.

Fifty-nine protected file identities and the prefix links matched through
retirement and reorganization. Six CK/MIGraphX/ORT registration checks passed
across daily and RCCL. `bc250-torch --help` and `bc250-distributed --info`
passed. The workstation suite passed 166 tests. The batch performed archive,
identity and launcher checks without a native rebuild or GPU workload.

The first staging export stopped at a historical FIFO. The second used a root
reader for protected guard records, strict directory traversal, explicit FIFO
metadata and deduplicated payloads. It verified 126,368 entries and
33,321,570,810 logical file bytes in a 4,469,298,417 B archive. The earlier
source-proof verification also exposed a manifest limit and a premature gzip
trailer check. Explicit manifest admission and waiting for producer exit
resolved those cases. The final exporter flushes the trailer before its second
inventory. Exact executed helpers and failed-attempt records remain in the
maintenance journals.

Implementations are afdac891a and e3f8c899d. The staging exporter ran as
13799/13800 and finished with its input inventory intact. Retirement workers
16474, 16472 and 17580 finished. .104 was observed on r2 boot 5c7ba874 after an
intervening reboot between the two cleanup passes.

Both hosts report the same runtime image and Torch a524980. The checked HIP,
MIOpen and Torch CPU/HIP library hashes match by profile. .104 uses the qualified
D49 profiler SDK repair. .105 retains its separately qualified SDK and was
audited read-only. A common SDK deployment needs target qualification before
promotion. D54 remains the next build milestone. The reclaimed space clears
the old 40 GiB disk allocation, with ELF/CMake/link closure and build-resource
admission still pending.
