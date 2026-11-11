# PLAN-08 / D58: self-contained release assembly

Completed 2026-11-11 on .104. Harness 9b6f569a8
(`assemble_release_d58.py`, script SHA `bdc700cd…`), run under the EX
prefix-update lock via `helpers/update_dev_prefix.py` (helper SHA
`8ec1a4c9…`, byte-identical to the current repo version).
[Receipts](artifacts.json), [replay summary](verified-results.json),
[assembly receipt](../../../../kernel/artifacts/release-d58/assembly.json).
**Assembly only — no activation.** `current` and all live profiles are
untouched; the release is `validated: false` pending D59. .105 unaccessed.

## Release

`/opt/bc250-stack/releases/10.2-gfx1013-20261111-138a66f52`

- Materialized from the qualified `sdk-d49-r1` cohort: image `1bb4ac0e…`,
  torch revision `a524980…`, both `daily` and `rccl-single` profiles.
- Layout: `rocm/`, `cache/`, `python-consumers/` shared at top level;
  `profiles/{daily,rccl-single}/` with fully copied torch trees
  (`pytorch-quant-b052/b053`), the rccl library for rccl-single, the gemm
  tuning csv, SOURCE_* provenance, native-overrides and adapted
  `runtime.json`. Profile links to shared subtrees are **relative**
  (`../../rocm` etc.) — the D57 path-dependency gap is closed.
- `runtime.json` (top + per profile): release version set,
  `validated: false`, `mutable: false` (immutable release), full
  provenance history preserved, rccl environment/identity block preserved.
- **18,295,550,735 bytes / 196,964 files** in 267.726 s; copied sizes equal
  source sizes exactly; free disk 55.0 → 37.1 GiB (5 GiB floor enforced).

## Verification

- Preflight: root on bc250, `current` target asserted, zero running
  containers, all 10 source sentinels hash-verified before copying, disk
  admission (needed + 5 GiB floor).
- Post-assembly: all 10 release sentinels re-verified by the harness, plus
  an independent workstation recheck of the 4 torch-library hashes against
  the release matrix (4/4).
- **Self-containment scan: zero violations** — every symlink in the release
  resolves inside the release tree.
- Six per-file SHA manifests (rocm 24,442; cache 75,743; python-consumers
  7,714; daily torch 44,527; rccl torch 44,537; rccl 1 file) live with the
  release; their digests are bound in `assembly.json` and were re-read from
  the host and matched.
- `current` asserted unchanged before/after; lock released; no owned
  processes remain.
- Local checks: 171 `bc250*_test.py` tests OK (incl. the new
  `adapt_runtime` regression), black applied.

## Reboot event

.104 rebooted immediately before the D58 launch (SSH refused, then
`up 0 min`; boot `aafacc82…` → `be4ef4dd…`). Post-reboot verification
passed before proceeding: r2 kernel 7.1.5-bc250rocm-r2, 10/10 cohort
sentinels, `current` unchanged, watchdog + cgroups started, only the
cyan-skillfish GPU client, no containers, 59.1 GB free. Cause unknown; no
owned jobs were running at the time. The r2 fallback boot path worked as
designed.

## Next

D59: build-free extraction/activation of this release at an independent
test path on .104 — ordinary `bc250-*` caller tests, selected
forward/backward/compiler/distributed/checkpoint regressions from owned
fixtures, identity refusals and actual rollback, then restore `current`.
After that D60 (.105 deployment qualification) and D61 (finalization).
Publication only on explicit user instruction.