# PLAN-08 / D59: independent-path release qualification

Completed 2026-11-11 on .104 (attempt 10). Harness `qualify_release_d59_on_host.py`
(revisions caef114c5 → bafbc577e), helper allowlist extension in `bc250_torch.py`.
[Receipts](artifacts.json), [replay summary](verified-results.json),
[evidence](../../../../kernel/artifacts/release-d59).
**Release `10.2-gfx1013-20261111-138a66f52` is now qualified**
(`validated: true`, immutable). The live selection was restored to the D49
cohort afterwards; promotion is a D61 decision. .105 unaccessed.

## Method

Under the board EX reservation + current-prefix EX runtime lock (inode
checked), the harness activated the release through the public mechanism:
`current` and `profiles/rccl-single` pointers switched, distributed registry
`d01.json` rebound, and the helper updated with an **additive** allowlist
(`RELEASE_D59_RUNTIME_SHA256`) binding the release's validated runtime
identities; the SDK-sentinel and alias checks are unchanged. The release's
`runtime.json` was set to its final validated form (hashes bound in the
allowlist before activation). Every gate ran through the ordinary public
callers with the established health envelope: fail-closed preflight, armed
boot-guard with acknowledgment, dmesg before/after identity, zero leftover
containers, 3 GiB memory / 5 GiB disk floors, 540 s per-gate timeout.

## Results (attempt 10, 121.776 s)

| Gate | Caller | Checks |
| --- | --- | --- |
| quant-b052 | bc250-torch | 220 |
| rccl-daily + consumers | bc250-train | 5 |
| rccl-single + consumers | bc250-train | 6 |
| compiler dense/gated/cnn/batchnorm | bc250-torch | 4 × 14 |
| distributed-b031 (single-host Gloo) | bc250-torch | 31 |
| checkpoint-owned | bc250-torch | 8 |

**326 checks passed.** Negative-visibility CPU-only import
(`HIP_VISIBLE_DEVICES=-1`) admitted on the bound release; identity refusals
verified (RCCL env override, unknown profile). Protected checks passed
(source sentinels, boot id, historical release untouched). The selection was
then **rolled back** to D49 (pointers + registry restored) and a
post-rollback caller smoke passed on the D49 cohort through the new helper.
The additive helper stays deployed (it binds both cohorts); the release
keeps `validated: true` — earned by the full pass, retained on disk.

## Checkpoint fixture replacement

The planned N3.9 continuity gate
(`validate_checkpoint_continuity_n39.py`) proved **inadmissible**: it depends
on `/home/nik/finetune-n7`, the user application excluded by workspace
preservation rules. It was replaced with the owned, self-contained
`validate_checkpoint_release_d59.py` (pure torch, no consumers): GPU
training steps, bitwise model/optimizer state restoration, exact resumed
continuation against an in-process reference, provenance-identity refusal,
and truncated-checkpoint refusal (8 checks).

## Attempt history (all failures restored or refused before change)

1. Harness `--mount` flag syntax bug → fixed (d7c2c63cf).
2. Preflight refused: user `kinetic` GPU app → user stopped it.
3. Checkpoint fixture needs the consumers env (`bc250-train`) → fixed (211dbe50d).
4. dmesg noise from user `llama-gguf` 25–26 GB allocation attempts (fixture passed).
5. Preflight refused: transient MemAvailable 7.6 GiB < 8 GiB during reclaim.
6. n39 fixture's excluded-app dependency surfaced → replaced (60cc90a6d);
   host rebooted afterwards (boot `73817cc1…`, r2 kernel, state verified safe).
7. dmesg noise from user `llama-bench` 24 GB allocation attempt (fixture passed).
8. Owned fixture RNG-state bug (`map_location` remap) → fixed (32b94e5c4).
9. Owned fixture tamper test unreliable (mid-file byte flip) → truncation (bafbc577e).
10. **Passed.**

The rotating gate failures in attempts 4/7 were environment interference,
not release defects — the fixtures themselves passed and the kernel-log
invariant correctly refused. After repeated conflicts the user dedicated
.104 to stack work. Failed-attempt evidence is retained on the host
(`evidence/release-d59-attempts/`); 98 passing-run files were transferred and
SHA-verified. 173 local tests pass (incl. the new allowlist regressions).

## Next

D60: .105 deployment qualification via the HOST105_UPDATE recorded
lock/workflow (preserve r2 boot fallback, production healthcheck, native
board controls, unrelated user applications). Then D61: known-limit matrix,
SBOM/license/source-recipe bundles, final receipts, and the promotion
decision. Publication only on explicit user instruction.