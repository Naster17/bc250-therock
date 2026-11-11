# Verified-redundancy space reclamation — 2026-11-12

Follow-up to the [2026-11-12 cleanup](../bc250_cleanup_20261112_results/README.md).
The cleanup retired October scratch into `kernel/artifacts/workspace-tmp-20261112/`
(capacity-neutral rename) and flagged the bulk for a future verified-redundancy
pass. The user explicitly approved that pass ("reclame and clear some space").
This report records the member-by-member verification and the deletion.

## Method (read-only verification first)

1. **Content-hash comparison.** Every one of the 68,444 scratch files was
   sha256-hashed and compared against a global index of the 25 same-named
   per-milestone stores (8,471 store files / 5,198 unique hashes / 5.96 GB).
   Script: [`compare_redundancy.py`](../../../kernel/artifacts/space-reclaim-20261112/compare_redundancy.py);
   full audit: [`redundancy-report.json`](../../../kernel/artifacts/space-reclaim-20261112/redundancy-report.json).
   - 3,070 files (659 MB) byte-identical to store content → already preserved.
   - 65,374 files (5.08 GB) unique to the scratch → classified below.
2. **Classification of unique content.** All unique files were grouped by path
   and inspected. They are working intermediates, not reviewed work product:
   - extracted **source-reconstruction checkouts** (abseil, pybind11, migraphx,
     rocm-cmake, onnxruntime upstream trees, incl. `.git`);
   - **build / cache-data / lint-venv / mirror** trees and nested `b065/b066/b067`
     working trees inside `migraphx-b064`;
   - **re-exported working tarballs** (`source*.tar.gz`, `migraphx.bundle`,
     `pybind11.tar`, `mirror.tar.gz`);
   - **disposable helper scripts / drafts** (`sync_memory.py`, `docs.py`,
     `verify*.py`, `README-draft.md`, `next.txt`, `block-*.sh`).
3. **Store-preservation confirmation.** For every entry the store preserves the
   authoritative material: git bundles (`abseil.bundle`, `protobuf.bundle`,
   `pybind11.bundle`, `migraphx-repair.bundle`, `ort-repair.bundle`,
   `migraphx-rocblas-default.bundle`), source tarballs (`onnxruntime-source.tar.gz`,
   `raw/sources/migraphx.tar.gz`), repair patches, preservation/reconstruction
   records (`external-preservation*.json`, `source-reconstruction.json`,
   `source-bundles.json`, `source-audit.json`), qualification logs and
   verification/archive JSON. Store tarball members were inspected to confirm the
   payloads. PLAN-07 (D47–D56) independently proved the stack reconstructs from
   these preserved inputs.
4. **Unique data preserved.** The only unique records not represented in any
   store were the October D01 host inventories. They were copied into
   `kernel/artifacts/distributed-d01/` and sha256-verified **before** deletion:
   `104-inventory.json` (b5dea07c…), `105-inventory.json` (72ddd4e7…),
   `delta.json` (63df593a…). Receipt:
   [`preserved-inventories.json`](../../../kernel/artifacts/space-reclaim-20261112/preserved-inventories.json).

## Action

Deleted all 25 scratch entry dirs (68,444 files / 5,736,296,381 B nominal).
The archive now holds only its catalog + docs snapshot
(`manifest.json`, `docs-final.json`, `docs-final.tar.gz`); the catalog was updated
to record this pass.

## Space outcome (honest accounting)

- `du` of `kernel/artifacts/` fell **69.16 GB → 63.43 GB** (−5.73 GB), confirming
  the scratch content was removed.
- `df` free space rose **~7.5 GiB → ~8.75 GiB** (net **~1.87 GiB** of blocks).
- The filesystem is **btrfs** (data ratio 1.00, dup metadata, fully allocated).
  The scratch shared CoW/reflink data extents with retained material, so deleting
  it released only uniquely-owned blocks; the remaining extents stay referenced by
  preserved copies. The earlier "~5 GB" estimate assumed the scratch owned all its
  blocks; actual net block reclaim is ~1.9 GiB. **No unique material was lost.**
- No btrfs administration (balance / snapshot removal) was performed — out of
  scope and risky on the user's workstation.

## Preserved (not touched)

All 25 per-milestone stores and their authoritative source/repair/preservation
payloads; all other `kernel/artifacts/` recovery stores; the D01 inventories
(relocated into the D01 store); all source/build/cache and rollback trees; all
user material. Only verified-redundant scratch was deleted.

## Verification

- Member-by-member sha256 audit retained (15.4 MB report).
- D01 inventory relocation hash-verified before deletion.
- Store integrity spot-checked after deletion (b064=192, b065=153, b0610=606,
  b0711=378 files; D01 image.tar/userspace.tar.gz/manifest intact).
- Receipts: [`deletion-receipt.json`](../../../kernel/artifacts/space-reclaim-20261112/deletion-receipt.json),
  [`journal.json`](journal.json).