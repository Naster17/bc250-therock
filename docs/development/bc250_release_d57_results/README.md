# PLAN-07 closure and PLAN-08 / D57: release inventory and assembly plan

Completed 2026-11-11. CPU-only workstation work plus read-only .104
inspection; no build, no container, no GPU workload, no host mutation,
.105 unaccessed. [Receipts](artifacts.json),
[host inventory](../../../../kernel/artifacts/release-d57/host-inventory.json).

## PLAN-07 closed for its stated scope

User-approved scope decisions (2026-11-11, "continue"):

1. **Triton**: provenance accepted as preserved (pin
   `ee65ff3b85f99d52e678b252a99c8425fa46e329`, four patches in
   `patches/bc250-triton`, qualified Triton 3.8 wheel SHA `2d65d6fe…`); a
   Triton rebuild demonstration is not required for release.
2. **Library deltas**: the D56 documented boundary is accepted; an
   object-level retained-rebuild follow-up is not required before release.
   The release ships the qualified deployed cohort, not rebuilt binaries.

With D47–D56, PLAN-07's completion evidence is satisfied: independent
clean-directory reconstruction matches required source hashes and
configuration (D52/D54/D55/D56); license/dependency/generated-file
provenance and all required patches are present (D47/D50/D51/D52/D56); no
build paths or undeclared workstation mounts are required (D54/D55 offline
builds). PLAN-07 is marked complete for this scope.

## Release inventory (D57)

Read-only findings, full detail in
[host-inventory.json](../../../../kernel/artifacts/release-d57/host-inventory.json):

- Qualified cohort: `sdk-d49-r1` candidate on .104 (daily + rccl-single),
  image `1bb4ac0e…`, torch revision `a524980…`, kernel
  7.1.5-bc250rocm-r2; .105 runs the matched `miopen-plan02-r4` cohort.
- **Core assembly gap**: both live profiles are symlink assemblies pointing
  at nik-home working trees (`torch -> pytorch-quant-b052/b053`,
  `cache -> …/cache/dev-user`), `dev/` paths (rccl, gemm csv) and the
  20261013 release's `python-consumers`. A release must materialize a
  self-contained, path-independent tree.
- Layout precedent: `releases/10.2-gfx1013-20261013-4b000d542` (cache,
  python-consumers, rocm, torch, runtime.json, validation, review).
- Registries present: migraphx, miopen-dl, ort-migraphx. Callers:
  bc250-torch/-train/-distributed/-ck/-ort/-migraphx plus board controls.
- `preservation.json` records the old SDK symlink layout needed for
  materialization; `launcher.json` binds revision 761ec5803 with passing
  validation.
- The release matrix currently records only 5 sentinel files per profile;
  the release needs complete SHA manifests.

## Assembly plan (bounded milestones)

Version scheme: `10.2-gfx1013-<YYYYMMDD>-<shortrev>` under
`/opt/bc250-stack/releases/`, following the 20261013 precedent.

| Milestone | Scope | Failure criteria |
| --- | --- | --- |
| **D58 assembly** (.104, EX prefix-update lock) | Materialize self-contained release tree from the qualified cohort: copy (not symlink) torch/rocm/rccl/cache-seed/python-consumers into the versioned dir; complete per-file SHA manifests; bind SOURCE_* provenance, registries, native-overrides, runtime.json; include kernel/fallback instructions, docs, D56 license manifest, source-recipe references (D47–D53). No activation; `current` untouched. | Any unresolved home/dev symlink in the release tree; any manifest hash mismatch; any mutation of live profiles/sources/caches; disk floor breach (release size measured first; ~57 GiB free). |
| **D59 independent-path qualification** (.104) | Build-free extraction/activation of the release at an independent test path; ordinary bc250-* caller tests; selected forward/backward/compiler/distributed/checkpoint regressions from owned fixtures; identity refusals (wrong cohort/version) and actual rollback; restore `current` afterwards. | Any caller/regression failure; refusal gaps; rollback that does not restore the exact prior state; GPU ownership/preflight violations. |
| **D60 .105 deployment qualification** | Apply the release via the HOST105_UPDATE workflow under .105's recorded lock (explicitly authorized deployment host); matched caller/regression subset; preserve r2 boot fallback, production healthcheck package/data, native board controls, unrelated user applications. | Any deviation from the recorded .105 lock/workflow; healthcheck or boot-fallback regression; touching excluded user applications. |
| **D61 release finalization** | Known-limit matrix, SBOM/license bundle, source-recipe bundle, boot fallback documentation, final receipts and memory; release marked qualified. Publication only on explicit user instruction. | Missing license/SBOM entries; undocumented limits; any claim beyond qualified evidence. |

Standing rules for all milestones: reservation + fail-closed preflight for
GPU work; 3 GiB host / 5 GiB disk floors; owned-container/job cleanup;
preserve histories/caches/checkpoints/old path links; no destructive source
reset; ccache 20G/no reset; finetune-n7 excluded; English docs; milestone
commits and flow.md updates; docs snapshot sync at handoff.

## Next

D58 release assembly on .104 (mutating; requires the EX prefix-update lock
and explicit approval). First D58 action: measure the materialized release
size and confirm disk admission before copying.