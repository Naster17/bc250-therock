# PLAN-08 / D61: release finalization and .104 promotion — PLAN-08 complete

Completed 2026-11-12. **Release `10.2-gfx1013-20261111-138a66f52` is now the
active selection on both hosts**, with the full finalization bundle.
[Receipts](artifacts.json), [replay summary](verified-results.json),
[bundle + evidence](../../../../kernel/artifacts/release-d61).
Publication was **not** performed and requires explicit user instruction.

## Finalization bundle

- **SBOM** (`sbom.json`): image `1bb4ac0e…`, torch `a524980…`
  (2.13.0a0+gfx1013), triton pin `ee65ff3b…` + qualified wheel `2d65d6fe…`,
  ROCm libraries/systems revisions, SDK sentinels, consumer wheels
  (torchvision 0.28.0, torchaudio 2.11.0, torchcodec 0.17.0, ONNX stack),
  opt-in modules (bc250_fft/bc250_sparse), MIOpen repair, rocblas tuning,
  kernels (r2 on .104, r3 on .105), launcher hashes, runtime identities.
- **Known-limitations matrix** (`known-limitations.json`): hardware
  (no MFMA/WMMA, SDMA/VM-heap disabled, RAM floors, profiling boundaries),
  software subsets (scalar/eig/sparse/SDPA/quantization/FP16-PEFT),
  performance (no general two-board speedup; D43 measured costs), release
  boundaries (immutable release, rebuilt-vs-deployed delta, Triton scope,
  no long soak, finetune-n7 exclusion), deployment/rollback assets. Corrects
  the stale "mutable developer prefix" runtime string for the release.
- **Source-recipe index** (`source-recipes.json`): the D47–D56 preserved
  reconstruction stores with key hashes (D52 kit `6867ed71…`, builder image
  `28ad36e6…`, six `70e8a77b…`), build recipes, the D56 license bundle
  (87 wheel licenses), Triton pin/patches/wheel, kernel recipes. Each store
  carries its own per-file SHA receipts; the index does not duplicate them.

## .104 promotion

`promote_release_d61_on_host.py` (ed8143651) switched `current` and
`profiles/rccl-single` to the release and rebound the d01 registry
(`813c8d02…`) atomically under the board EX reservation + current-prefix EX
runtime lock (inode checked), with fail-closed preflight. A recovery set
(before/after pointers + registry snapshots) is retained; `--rollback`
restores the D49 cohort. Smokes ran after the lock released (D60 lesson).

Post-promotion verification through the public callers:
`D61-DAILY-GPU-OK`, `D61-CONSUMERS-OK` (torchvision 0.28.0+gfx1013),
`D61-RCCL-GPU-OK` (all torch 2.13.0a0+gfx1013), and the **220-check quant
regression passed**.

## PLAN-08 completion

| Milestone | Evidence |
| --- | --- |
| D57 | release inventory + assembly plan |
| D58 | self-contained assembly: 18.3 GB / 196,964 files, zero external symlinks, full SHA manifests |
| D59 | .104 independent-path qualification: 326 checks, identity refusals, protected checks, actual rollback |
| D60 | .105 deployment: 656-check suites on both profiles, rccl-capable helper update, atomic activation, live rollback + reactivation |
| D61 | finalization bundle + .104 promotion + verification |

Both hosts run the same versioned release; all rollback/boot assets preserved
(.104: D49 cohort + historical releases; .105: r4 + D01 + retained base +
M105.1 boot-health). 173 local tests pass; 10 D61 evidence files SHA-verified.

## Next

PLAN-09 (modern model/environment compatibility matrix) then PLAN-10
(measured training performance) — two important packages remain. Optional
work (B07.13 CK FP16, N4 observability, N2/N6/N7 extensions) stays separate.
Publication/distribution of the release requires explicit user instruction.