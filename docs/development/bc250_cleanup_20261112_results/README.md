# Workspace and host cleanup — 2026-11-12

Pedantic order-and-cleanliness pass after PLAN-08 completion. Read-only audit
first, then only safe retire/delete actions; all rollback assets, user-owned
material, source/build/cache trees and historical evidence preserved.

## Audit findings

- **Git**: workspace and therock fully committed; only user-owned untracked
  `install-arch.sh` (workspace) and `therock/src/` remain (untouched per
  AGENTS.md). `kernel/source` and `references/pytorch-upstream` clean.
- **Workspace `tmp/`**: 5.74 GB of October B06/B07/D01 task scratch whose
  reviewed outputs are already preserved in the per-milestone
  `kernel/artifacts/` stores.
- **Hosts**: `.104` `rocm-10.2-gfx1013/tmp/` already empty (README only); no
  running containers on either host; stack images intact in root storage
  (runtime `1bb4ac0e`, builder `28ad36e6`, historical milestone images — all
  rollback-referenced). `.105` had one leftover D60 transfer file.
- **User material**: both hosts' home directories hold many user applications
  (kinetic, llama.cpp, hashcat, models, etc.) — excluded entirely.

## Actions

| Action | Detail | Verification |
| --- | --- | --- |
| Retire workspace scratch | `tmp/` (25 entries, 68,444 files, 5,736,296,381 B) → `kernel/artifacts/workspace-tmp-20261112/` via same-filesystem rename; `tmp/` left with a README | file count + total bytes identical before/after; catalogued in `manifest.json` |
| Delete .105 transfer leftover | `evidence/release-d60/transfer/librocprofiler-sdk.so.1.4.2` (17.6 MB) + `rocm-base.done/.err` markers; removed empty `transfer/` | release copy hash `2a975370…` verified identical before deletion; file also preserved on .104 |
| Group .104 D59 evidence | 9 attempt dirs + 9 controller logs → `evidence/release-d59-attempts/`; passing-run controller log → `evidence/release-d59/` | 18 entries grouped; D59 report/receipt path references updated |

## Preserved (not touched)

- `kernel/artifacts/` recovery stores (67 GB) — the designated preservation
  archive, referenced by cleanup reports and rollback records.
- All source/build/cache trees (`therock/`, `rocm-10.2-gfx1013/{source,build,
  cache}`, retained `pytorch-*` rollback trees) — absolute paths embedded in
  launchers/manifests; layout doc forbids broad rename.
- Historical `.105` evidence layout (D05–D43 controller logs/tarballs) —
  embedded references; not broadly reorganized.
- All releases/rollback assets: .104 D49 cohort + historical releases; .105
  r4 + D01 + retained base + M105.1 boot-health; all stack images.
- `patches/`, `board-support/`, `references/pytorch-upstream`, user apps/data.

## Space note

The scratch retirement is capacity-neutral (same-filesystem rename into the
ignored archive store); the active workspace is clean. The workstation has
~7.5 GB free. The bulk intermediates now in
`kernel/artifacts/workspace-tmp-20261112/` (migraphx durable/cache-data,
onnxruntime mirror/lint-venv) are candidates for a **future verified-redundancy
deletion pass** to reclaim ~5 GB; they were retained rather than guessed
obsolete, per the layout discipline. That pass needs member-by-member
comparison against the per-milestone stores and explicit approval.

## Verification

Both repos committed; docs snapshot synced; both hosts idle on the release;
no owned processes/containers left; task scratch (`.qwen/tmp/`) cleared.