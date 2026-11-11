# PLAN-08 / D60: .105 deployment qualification

Completed 2026-11-12 on bc500 (.105). The D58/D59-qualified release is now
**deployed, qualified and active on .105**, with a proven live rollback cycle.
[Receipts](artifacts.json), [replay summary](verified-results.json),
[evidence](../../../../kernel/artifacts/release-d60).
No kernel/boot change; the r3 kernel and all boot-recovery assets are
untouched. Helper revisions be6b17388 → ebcdaff63; 173 local tests pass.

## Release and staging

`10.2-gfx1013-20261111-138a66f52-host105` under
`/opt/bc250-stack/releases/`. .105 had only ~16 GiB free vs the 18.3 GB
release, so staging follows the documented D01/r4 inode pattern:

- **rocm**: 24,442 files hardlinked from the retained immutable base
  `10.2-gfx1013-20261023-host105/rocm` (24,440 byte-identical) plus 222
  symlinks; the two repaired files detached as independent copies —
  `libMIOpen.so.1.0` from the r4 candidate (same hash) and
  `librocprofiler-sdk.so.1.4.2` transferred from .104. All 24,442 hashes
  verified against the D58 manifest; detached files have `nlink == 1`.
- **Native change to .105**: the rocprofiler sentinel upgrades from the
  pre-D49 `a6dc1c88…` to the D49 repair `2a975370…` (the repair had only been
  activated on .104). MIOpen is unchanged (`381eac05…`).
- torch trees, cache, consumers, metadata and manifests transferred directly
  .104→.105 (7.78 GB, two chunks); profile `.runtime.lock` files hardlink the
  migration-control inode (7733260) for cooperative locking; zero external
  symlinks; runtime SHAs match the D59-qualified identities.

## Helper and qualifier update

The old .105 launcher predates rccl-single support (no rccl mount /
LD_LIBRARY_PATH branch), so it could never launch the release's rccl profile.
The reviewed current helper (`9c8b3c78…`, rccl support + D59 allowlist) was
deployed to `/usr/local/bin/bc250-torch` and the migration toolset, with the
originals backed up (`2da29b2c…`). The qualifier gained an rccl-aware revision
(`qualify_release_d60_host105.py`): selects `runtime_profile` from the manifest
and passes `--nccl-available` to the b031 distributed gate for rccl torch.

## Target qualification (before activation)

| Round | Profile | Result |
| --- | --- | --- |
| r1 | daily | **656 passed** (original helper) |
| r1 | rccl | failed — old helper lacks the rccl mount (`librccl.so.1` import) |
| r2 | rccl | core 7 passed; distributed failed — b031 needs `--nccl-available` |
| r3 | rccl | **656 passed** (updated helper + rccl-aware qualifier) |
| r3 | daily | **7 core passed** (updated-helper re-smoke) |

Every gate: boot-guard armed/acknowledged, dmesg identical, zero leftover
containers, 4 CPU / 10 GiB / 12 GiB swap envelope, 540 s gate timeout.

## Activation, smoke, live rollback cycle

Activated atomically under the migration-control EX lock: `current` and
`profiles/rccl-single` pointers + the d01 registry rebound
(registry SHA `ea7314b0…`); recovery record written. Ordinary-caller smoke:
`D60-DAILY-GPU-OK`, `D60-CONSUMERS-OK` (torchvision 0.28.0+gfx1013),
`D60-RCCL-GPU-OK`, all torch 2.13.0a0+gfx1013.

Live rollback cycle: restored the r4 selection from the recovery record,
smoked it (`D60-ROLLBACK-R4-OK`), reactivated the release (`--reActivate`,
`D60-REACTIVATE-OK`). The release is the final active selection.

**Incident/lesson**: the first cycle attempt ran the r4 smoke *inside* the EX
control lock; the mutable r4 launcher takes an SH lock on the same inode and
blocked ("waiting for the developer runtime lock"). The owned runner was
killed and the smokes re-run outside the lock. Rule: never run mutable-cohort
launches while holding the EX control lock.

## Preservation

Retained base rocm, M105.1 release + image (`506f3b26…`), boot-health package,
grub r2 fallback, production healthcheck/data, native board controls and user
applications untouched; the old r4 cohort and D01 release remain available for
rollback; no reboot (r3 kernel, boot `08ecaaaa…` throughout). 23 evidence
files transferred and SHA-verified; full gate logs and helper backups retained
on .105 at `evidence/release-d60/`.

## Next

D61 finalization: known-limit matrix, SBOM/license/source-recipe bundles,
final receipts, and the promotion/publication decision. Publication requires
explicit user instruction. After PLAN-08: PLAN-09 (compatibility) and PLAN-10
(performance).