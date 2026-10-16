# BC-250 storage cleanup — 2026-10-16

The user explicitly authorized removal of obsolete project installations,
containers/images and temporary artifacts, superseding earlier blanket retention
for the audited obsolete items. Active sources/builds/caches, genuine user data,
required recovery fallbacks and other projects remain protected.

Recovered **31,282,634,752 bytes (29.13 GiB)** on .104, measured by available-space
change. Root filesystem went from about12.2GiB free/97% used to41.3GiB free/91% used.
`applied.json` is the exact deletion journal; `plan.json`, `inventory-before.json`,
`final-path-check.json` and the runtime gates retain the decision/evidence.

| Removed | Scope |
| --- | --- |
| Superseded P2 standalone candidate payload | `/opt/bc250-stack/candidates/priority2-20261014/{rocm,torch,cache}` |
| Redundant initial frozen-release payload | `/opt/bc250-stack/releases/10.2-gfx1013-20261012-a640c9296/{rocm,torch,cache}` |
|56 stopped experiment containers | Explicit `bc250-*` IDs from inspected plan; no volume/global prune |
|2 obsolete image records |4e305e1d6353 (initial frozen image), cd9f0f9bd955 (old7.2.4-r1.2 image); shared layers retained by surviving images |
|33 generated validation/fixture files | Only explicit training-validation and our Qwen synthetic test directories; checkpoint weights/state and duplicate tokenizers. About1.999GB regular-file logical data; symlink entries are not additional physical savings. |
|2 orphan launcher contexts | Exact old `/tmp/bc250-torch-context-*` directories, older than1h with no process references; contents neither read nor archived |

Original runtime manifests are now `runtime.retired.json` beside `RETIRED.json`;
validation and historical evidence remain in each retired prefix. They are no
longer executable standalone stacks. Original source histories/revisions remain
in the canonical source repositories, not dependent on these copied payloads.
Synthetic test JSON/config/logs remain; old deleted fixture checkpoints must be
regenerated through the test helpers under a new label, not treated as user data.

## Kept and verified

- `current` still selects `/opt/bc250-stack/dev/priority1`, with the same working
  ROCm/Torch/cache links. Current Python consumers still use the frozen20261013
  bundle. No Torch/ROCm DSO or production behavior change.
- `previous` now selects the newer tested independent
  `releases/10.2-gfx1013-20261013-4b000d542`, same as `before-dev-current`. This
  consolidates redundant software fallbacks. Kernel r2 boot fallback is unchanged.
- 14 active/fallback runtime/config/DSO hashes compared before and after match.
  External BLAS `.kpack` files were additionally checked against qualified N1.8
  and the frozen release's historical sealed hash list; see `pack-verification.json`.
  The retained cleanup helper now includes the correct `rocm/.kpack` path directly
  in its fingerprint list. Exact executed helper is retained locally under ignored
  `kernel/artifacts/cleanup-20261016/cleanup.executed.py` and .104 cleanup logs.
- Active and fallback each passed seven bounded CPU-reference/contract groups:
  FP32/BF16 GEMM output/input gradients, math SDPA D32/D256 output/QKV gradients,
  layernorm, FFT gradients and exact RNG replay, plus reviewed consumer imports.
  Fallback checked before removal and again via `previous` afterward; current
  checked afterward.21 group executions total, no broad numerical/stability claim.
- All3 runtime gates exit0,600s guards ACKed. Kernel log byte-identical across the
  entire cleanup, SHA53f83f7bbee6c060e3153e7ae95a4ff069bf9627113591c39cdddb9ab5f4b973.
  Same boot a287fd16-c97b-4d88-8055-b0553a2f7631, watchdog/SSH available.
- User `neural-p1` container and image6a30463535e5 retained with unchanged inspected
  container identity/state. User finetune-n7 models/data/checkpoints untouched.
  No package uninstall, user-model/cache sweep, ccache wipe or build expunge.
- Canonical local/remote sources, incremental ROCm/Torch/Triton/kernel builds,
  working compiler caches, wheels, training base model, release metadata and all
  compatibility links retained. Host `/opt/rocm` is referenced by user shell
  configuration; legacy `/opt/bc250` and `/opt/torch-venv` remain as compatibility
  installations rather than guessing that other user workloads no longer need them.
  Builder/runtime images referenced by existing helpers and their ancestors remain.
- Local `/home/nik/Documents/Vibe/bc250stack` inspected. Its Alpine build/images
  are associated with a separate gfx1103 setup, including a running container;
  they are outside this BC-250 cleanup. An initial unprivileged attempt stopped
  on the first root-owned directory without deleting build files; scope corrected
  before privilege use. Nested SPIRV-Headers Git history and all local builds,
  caches and other projects preserved. See `local-scope-decision.json`.
- .105, BIOS, governor, SDMA/heap-off/math defaults, kernel source/image and boot
  policy unchanged. No active agent build/test/container/recovery deadline remains.

Code: `build_tools/bc250/cleanup_obsolete_storage.py` is a restricted one-shot
planner/applicator, applied under `update_dev_prefix.py`'s exclusive lock. It uses
explicit paths/IDs, live-process/container checks, tested fallback prerequisite,
atomic previous-link replacement, per-action journal and protected hash checks.
No `system prune`, forced image removal or volume deletion. Completed container
log tails retained in .104 `WS/logs/cleanup-20261016/retired-container-logs/`, with
potential credential lines redacted; environments were not exported.

This cleanup changes no Torch support result. MIOpen A01/A02 and the separate
Python CPU fault remain unresolved; the fixed VM-heap/CSR issue remains distinct.
Exact next action returns to **A01** in the
[Torch/library queue](../bc250_torch_support_matrix.md): minimal direct-MIOpen
workspace/profiling isolation. N2.7 remains queued behind the selected primary work.
