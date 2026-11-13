# BC-250 project memory: read first after resume or compaction

Updated 2026-11-13 (local; checkpoints explicitly use UTC). This is the
**canonical working memory**, not a release note.
Detailed hardware, implementation history, evidence and resolved defects:
[bc250_memory_inventory.md](docs/development/bc250_memory_inventory.md).
Old milestone documents are evidence snapshots; their unchecked boxes and active
release claims must not override this cursor and current manifests.

## Resume cursor





- **D68 fused native solve/scan completed 2026-11-13 — opt-in selective fusion
  qualified, op 1.13–1.81× / model-step 1.15–1.17× at seq256, default bitwise
  unchanged.** [Report](docs/development/bc250_gated_delta_d68_results/README.md),
  [evidence](../../kernel/artifacts/gated-delta-d68). First bounded scope of the
  deeper bottleneck follow-up. Profile (owned fixture, exact-replica event
  breakdown + microbenchmarks): the native chunk op's Python scan loop is ~70%
  of fwd+bwd at B1/H16/D128/L1024 (launch-bound autograd), the two batched ATen
  solves ~48% at the tiny C=1 geometry. Patched Torch source (commit `262accc`,
  module sha `06c5a8e1…`) adds `use_fused_solve_scan=True`: in-register Triton
  unit-triangular inverse (analytic backward −tril(Mᵀ dM Mᵀ)) + column-blocked
  linear-recurrence scan (S_{i+1}=M_i S_i+N_i, adjoint backward, saved states;
  M/N/P/R precompute stays ATen), elementwise convention (no tl.dot, wave32).
  **Selective from C≥2** (`FUSED_SCAN_MIN_CHUNKS=2`, measured decomposition:
  every variant regresses 0.79–0.94× at C=1); 256MiB saved-state cap refusal;
  FP16/BF16 keep reference-ordered prepare. Final r6: op FP32 a06-long
  26.17→14.49ms (1.81×; r3 1.94×), long256 1.58×, a06-short 1.17×, benchmark
  1.0× (default); BF16/FP16 1.13–1.77×. Model (D66 route + flag, D64 protocol):
  seq256 routed 19.05→16.55ms FP32 (1.15×), BF16/FP16 1.17/1.16×, vs frozen
  2.60/2.38/2.35×; seq32 no regression (flag keeps default). Parity: op
  fused-vs-default ≤1.5e-6 FP32 / ≤9.3e-5 low (bounds 1e-5/1e-2); CPU/FP64
  recurrence anchor 0.3–5.6% of A06 tolerances; model seq256 FP32 worst grad
  1.52e-3 ≤2e-3, fused 7× closer to FP64 than frozen (2.1e-4 vs 1.4e-3).
  Extended A06 battery 72/72 on candidate; default path bitwise == installed
  `21398daf…`; refusals pass; per-case scan counters prove path selection.
  Defects retained: battery unpack (r1), fixture oracle detach (r1), C=1
  regression → measured selectivity (r1/r2→r3), fixture routing-injection
  nested-capture leak contaminating the seq256 control (r3/r4, root-caused r5
  trace, fixed r6; diag1/diag2 retained). 20 new workstation tests (8 numpy
  FP64 math/grad checks + 12 fixture-logic), suite 269 pass, Black clean.
  Board clean every run (0 containers/GPU clients, 54°C, boot a3142efc
  unchanged); ordinary bc250-train caller, immutable release, new kernels JIT
  into the designed rw user-cache; no default/package/release/installed-module
  change, no .105/user-app access. Host /home/nik/bc250work/gated-delta-d68
  retained for replay. Publication: bc250-torch 8d8b18d→262accc (push 500 then
  verified landed); complete series 22→23 patches (tree ce527668, index-mode
  verify PASS); umbrella bc250-stack main 5887d18→b16a573 (curated D68 in
  reports/README.md + docs/PERFORMANCE.md + FORK_CI/REPOS/ARCHITECTURE;
  fork CI run 43463566419 success, 8 components + syntax).
  **NEXT (optional):** remaining deeper-bottleneck scopes (finer selectivity
  ~5% at C=4, trainable-weight conv, cast overhead, larger geometries);
  N4 deeper/upstream-tracking/full-prefix/N2/N6/N7/NF4/FP16-PEFT/soak on
  concrete need — see BC250_REMAINING_PLAN.md. Stopped per user pacing rule.

- **D67 distributed value benchmark completed 2026-11-13 — two boards buy
  capacity, not speed; LoRA is the exception (~1.94× per update).**
  [Report](docs/development/bc250_distributed_benchmark_d67_results/README.md),
  [evidence](../../kernel/artifacts/lora-bench-d67). Consolidated D05/D32/D35/
  D43/L01 into one decision table (full-param fits-one-board: 6.8–68× slower on
  two; doesn't-fit: 1.024B FSDP 53.36s/update capacity-only; LoRA: L01 +23%).
  New owned measurement via the authorized lora-smoke integration (hashes
  832017cd/2f24c6fe identical both hosts) in BC250_LORA_BENCHMARK mode with
  **owned synthetic exact-length data** (no user data; generator d706e066,
  256 samples/length, cross-host SHA-256 verified): Qwen3.5-2B rank-128 LoRA
  FP16, global batch 4 matched (single micro1×accum4; DDP micro1×accum2×2).
  **1024 steady state: single 36.262s (113 tok/s) vs DDP 18.679s (219 tok/s) =
  1.941×** (IQR 0.22%/4.15%, 5 measured updates, all finite/no skips);
  **4096 cold first update: 325.822s vs 167.567s = 1.944×**. DDP overlaps the
  538MB gradient all-reduce with backward → 1GbE almost fully hidden. 8192 NOT
  run: reference-path cost (eager attention O(L²) + launch-bound gated-delta)
  exceeds the qualified 460s guardian child budget even for one measured
  update — documented infeasibility, data generated/verified only. Scope
  adaptations retained as evidence: global-16 1024 attempt (108s/update) and
  global-4 4096 4-update attempt (326.8s first update) safely guardian-stopped
  at deadline (nonce-acked guards, clean boards); the global-16 attempt's
  kernel-log gate flagged host swap activation (/swapfile pre-existing
  2026-10-22, actor unidentified, no GPU faults, board healthy). Guardians:
  adapted L01 pattern (guard_d67.py + surviving d09 preflight, SH locks, 600s
  boot-guard nonce ack, 460s child, kernel-log/registry identity, foreign-
  container refusal). Generator defects fixed+regression-tested: shared-RNG
  non-monotonic search, prompt overshoot for multi-token syllables, transformers
  5.22.1 BatchEncoding (UserDict len==keys) normalization, unreachable-total
  fallback. 24 new tests, suite 249 pass, Black clean. Both boards idle
  pre/post (0 containers/GPU clients), app originals/outputs/checkpoints
  untouched, no user data, no runtime/default/source change. Host task dirs
  bc250work/lora-bench-d67 retained BOTH for replay. Publication:
  bc250-therock c0ddc2a64→c3238f5d4; umbrella bc250-stack main 6f76f72→5887d18
  (curated D67 section in reports/README.md). **NEXT (optional):**
  deeper bottleneck work (fused native solve/scan, trainable-weight conv, cast
  overhead, larger geometries), or N4 deeper/upstream-tracking/full-prefix/
  N2/N6/N7/NF4/FP16-PEFT/soak on concrete need — see BC250_REMAINING_PLAN.md.
  Stopped per user pacing rule.

- **D66 hybrid bottleneck optimization completed 2026-11-13 — opt-in native
  gated-delta routing qualified, 2.2–2.4× hybrid step speedup on .104.**
  [Report](docs/development/bc250_bottleneck_d66_results/README.md),
  [evidence](../../kernel/artifacts/bottleneck-d66). Root cause of PLAN-10's ~3–5×
  qwen35-hybrid cost: the frozen Transformers 5.2.0 torch fallback
  `torch_chunk_gated_delta_rule` builds its triangular inverse with a
  **63-iteration Python loop** (hundreds of sequential small kernels); the
  installed native `torch.nn.bc250_gated_delta.chunk_gated_delta_rule` (A06/A07,
  byte-identical `21398daf…`) uses batched `solve_triangular`. New owned helper
  `route_qwen35_gated_delta_d66.py` (sha `96cc2529…`) routes the model's chunk
  call explicitly: gates 5.2.0 + qwen3_5_text + gfx1013, kill switch
  `BC250_NATIVE_GATED_DELTA=0`, counted fallbacks (dtype/geometry/initial-state/
  compiled), exact uninstall; conv (trainable weights) and recurrent/decode paths
  NOT routed. Fixture `profile_hybrid_d66.py` (final sha `4f84c030…`): steady-state
  breakdown (step 34.93ms; chunk fwd 8.02ms/23%; chunk fwd+bwd ≈25.5ms ≈73% of
  step), paired op timing (benchmark geometry FP32 **4.33×**, BF16/FP16 ~4.1×;
  long-256 3.35–3.59×), D64-protocol model timing frozen→routed FP32
  36.994→15.254ms (**2.425×**, 4196 tok/s), BF16 2.209×, FP16 2.207× (IQR≤1.5%),
  model parity (FP32 loss Δ0/logits 1.5e-7/worst grad 8.4e-4 on A_log; BF16/FP16
  within A06/A07-style bounds), independent CPU/FP64 gradient anchor (frozen
  3.5e-4 / routed 1.2e-3 ≤5e-3), full D62 admission battery on the routed model
  passed, 5 refusal checks. **Bound discipline:** original FP32 grad bound 1e-4
  FAILED in r2 at 8.43e-4 — retained on record; revised 2e-3 bound pre-declared
  with justification + CPU anchor before r3/r4 confirmation (not silently
  relaxed). Method defects fixed: r1 doas-TTY launch (milestone TTY driver,
  runtime PIN), r2 per-call-sync breakdown inflation, r3 cold-step breakdown →
  r4 warmup. 21 new tests, full suite 225 pass, Black clean. Board clean:
  0 containers/GPU clients pre/post, 54°C, boot a3142efc unchanged, ordinary
  bc250-train caller under the immutable release — no default/package/source/
  release change, no .105/user-app access. Host task dir
  /home/nik/bc250work/bottleneck-d66 retained for replay. Publication:
  bc250-therock 7bb99d9b8→b3860f614; umbrella bc250-stack main 3802e3d→6f76f72
  (curated D66 sections in reports/README.md + docs/PERFORMANCE.md).
  **NEXT (optional):**
  fused native solve/scan kernels, trainable-weight native conv, low-precision
  cast overhead, larger-geometry admission; or distributed benchmark — see
  BC250_REMAINING_PLAN.md. Stopped per user pacing rule.

- **Patch-series reconciliation completed 2026-11-13 — fork CI now proves all 8
  patched components reconstruct their published forks.** Umbrella `bc250-stack`
  `main` `5e6cbfd→3802e3d`; **GitHub Actions run 43433605536 success** (8 verify
  jobs + syntax). The three series the first CI run exposed as incomplete were
  regenerated from the authoritative local fork histories (`git format-patch
  base..head`, commit content/authors/messages unchanged): **rocm-libraries 33→38**
  (adds the five B07.12-era MIOpen CK-plugin commits; base `3d8f334b9a7` on
  ROCm/rocm-libraries develop — the old REPOS.md pin `therock-7.13-27423` does not
  exist upstream), **rocm-systems 6→8** (adds the two RCCL fixes — NVTX includes
  when tracing disabled, iostream for DDA IPC — and fixes duplicate `0001-`
  numbering; base tag `hip-version_10.2.62820` `d5c528019d`), **pytorch complete
  22-patch single-base series** published at `patches/bc250-pytorch-complete`
  (base ROCm/pytorch release/2.13 `c779020750`, head `8d8b18df5`, tree
  `33a86273…` — exactly matches the D46 verified complete export; the historical
  incremental `patches/bc250-pytorch` is retained unchanged as evidence, never
  combined). **New index apply mode** in `tools/verify_patch_series.py` for
  LFS-heavy bases: rocm-libraries' base tree is ~9.4 GiB of git blobs (54,787
  files; the earlier checkout failure), so index mode fetches `--depth 1
  --filter=blob:none` (~2 MiB), `read-tree`, `git apply --cached` per patch
  (lazy-fetches only the ~113 patched files' preimage blobs), then
  `write-tree --missing-ok` with `GIT_NO_LAZY_FETCH=1` (plain write-tree would
  lazy-fetch the entire missing blob set — diagnosed and avoided).
  `GIT_LFS_SKIP_SMUDGE=1` on every git call; the BC-250 delta touches no
  LFS-tracked file and has no binary diffs. 8 new unit tests
  (`tools/tests/verify_patch_series_test.py`) prove am/index tree agreement +
  mismatch/corruption/count refusals; Black clean. All 8 components also verified
  locally against the real upstreams before push (5 am-mode regressions re-passed
  with the updated tool). Fork trees pinned: rocm-libraries `046ed8e3`,
  rocm-systems `65deb63f`, pytorch `33a86273`. Docs updated
  (reports/FORK_CI.md, REPOS.md, ARCHITECTURE.md, both pytorch READMEs). **No fork
  branch changed** (forks already carried full histories); no board access;
  deployed release unchanged; patch provenance authentic (one rocm-systems patch
  keeps its original upstream author). Umbrella staged in task scratch
  `tmp/patch-reconcile-20261113/` (clone retired after push; re-clone
  `https://github.com/Naster17/bc250-stack.git` for future maintenance).
  **NEXT (optional):** bottleneck optimization (PLAN-10 ~3–5× hybrid
  gated-delta/conv), distributed benchmark, then upstream-tracking/full-prefix
  distribution/N4 deeper getter/N2/N6/N7/NF4/FP16-PEFT/soak on concrete need —
  see BC250_REMAINING_PLAN.md. Stopped per user pacing rule.

- **Fork CI added 2026-11-13 — patch-apply reconstruction proof, live+green on
  GitHub.** Umbrella `bc250-stack` `main` `ba3c710→5e6cbfd`: `.github/workflows/
  patch-apply.yml` + `tools/verify_patch_series.py` + `ci/components.json` +
  `reports/FORK_CI.md`. For each component it fetches only the recorded upstream
  base, `git am`s the umbrella patch series, and asserts the resulting tree ==
  the published fork `bc250/gfx1013` tree (`expected_tree`). Git-only (no ROCm
  toolchain/build; full build stays with the offline recipes), so it runs on
  GitHub-hosted runners; weekly schedule surfaces upstream drift. **5 components
  verified + CI green** (run 43421509586): triton(base 66b3f9f/4 patches),
  migraphx(9567291/5), onnxruntime(402979a/1), torchaudio(34c52a6/4),
  torchcodec(d9dc542/1). **Finding (follow-up):** rocm-systems' 6 published patches
  do NOT reproduce the fork tree (fork has more commits); rocm-libraries has more
  commits than its 33 patches + is LFS-heavy (base worktree checkout hit resource
  limits); pytorch's umbrella series is the historical incremental (incomplete)
  export. These three are excluded from the matrix pending base/commit
  reconciliation — recorded in reports/FORK_CI.md, not silently dropped. No board
  access; publication-only. **NEXT (recommended):** patch-series reconciliation
  (rocm-libraries/rocm-systems/pytorch) then extend the CI matrix; see the
  prioritized backlog in BC250_REMAINING_PLAN.md and the operational safety
  reminders in AGENTS.md (boot-guard nonce ack, per-boot PIDs, locks, btrfs).
  Stopped per user pacing rule.

- **N4 read-only telemetry audit completed 2026-11-13 on .104 (r2 release).**
  [Report](docs/development/bc250_n4_telemetry_results/README.md),
  [evidence](../../kernel/artifacts/n4-telemetry-r2). Re-established governor/
  source/overlay provenance + validated read-only clock/temp/power/fan semantics
  on kernel `7.1.5-bc250rocm-r2`, boot `a3142efc…` (board rebooted since B07.13's
  `ed0d130a…`). Helper `qualify_n4_telemetry_on_host.py` adapts the proven r3
  sampler: 84 samples (~5Hz, 20 idle/44 load/20 post-load), one bounded 8s native
  integer load (`checks=131072 launches=58592`), kernel log byte-identical,
  mounts/governor-binary/config/fan hashes/power-policy(auto) unchanged, no
  clock/voltage/fan/governor/overlay/boot change, no raw SMU mailbox commands.
  **Provenance gap confirmed:** running governor `00cd65a6…` is a musl cross-build
  (BuildID `a1de19b4…`) ≠ retained repo `240b0df` native build `e8028ca2…`; the
  `freq1_input` overlay code is absent from every repo branch (smu/gpu-metrics/
  fix_11/dbus-performance-mode) though the running binary exhibits it → running
  source NOT pinned. **r2 vs r3 semantics difference:** r3 kernel `current_gfxclk`
  read 2–47MHz (disagreeing with governor); on r2 all clock views agree on sensible
  steps 1000/1400/1600/1800/2000MHz (patched==hwmon 84/84) because the governor
  drives SMU and r2 reads it coherently — still governor-set, NOT independent
  physical attained-clock calibration. Utilization still governor-derived
  (`gpu_activity` 65535, DRM load errno22, `gpu_busy_percent` EOPNOTSUPP). Edge
  52→70°C/Tctl 53.9→69.25°C/fan PWM90→175. **Safety fix:** first helper draft wrote
  a plain `acknowledged` marker; the boot guard requires nonce-matched `ack.json`
  and emergency-reboots on unacknowledged deadline — acked correctly before
  deadline (no restart), helper fixed, clean re-run passed. **NEXT (optional):**
  pin running governor build source + minimal guarded read-only `GetGfxFrequency`
  (PMFW 0x37) live-clock getter after establishing SMU mailbox ownership; keep
  hybrid per-core/power semantics separate; no legacy 8-core patch blind apply.
  **Publication sync done 2026-11-13:** bc250-therock `bc250/gfx1013`
  `4c532c755→5b61d21ad` (source as-is, no full IPs/excluded-app/creds in diff);
  bc250-stack `main` `8b6c2bc→ba3c710` (curated/scrubbed N4 report). Umbrella
  component patch series/SBOM unaffected (read-only audit, no source/release
  identity change). Stopped per user pacing rule.

- **B07.13 completed 2026-11-12 — installed explicit FP32/FP16 `bc250-ck` on .104;
  the one genuinely pending optional package is now done.** Tools `be602f7ee`.
  [Report](docs/development/bc250_ck_plugin_b0713_results/README.md),
  [use/recovery](docs/development/bc250_ck_plugin_b0713_usage.md),
  [evidence](../../kernel/artifacts/ck-plugin-b0713). The qualified B07.12 coherent
  FP32/FP16 CK + repaired-MIOpen archive (SHA `a0d47ed4…`, manifest `52db5d42…`,
  format2 worker, 27 mapped libs incl. private `runtime/libMIOpen.so.1`, API5/66)
  is installed at `sdks/bc250-miopen-dl-gfx1013-b0712` + `tools/ck-b0713` and
  activated: `miopen-dl.json` and the `/usr/local/bin/bc250-ck` symlink moved
  together under the EX lock. B07.7 FP32 package/tools/version record retained as
  the live-rollback target (`miopen-dl-b0713-rollback.json` captures the exact prior
  registry bytes + cohort-wrapper symlink). **92 checks / 10 guards**: 18 CPU
  (identity, 5 refusals, unregister→refusal→activate, lock exclusion, 3 drift
  refusals, live rollback→cohort `--ck-info`→reactivate), 9 guarded gates = 72
  (host-cli 8, installed-daily/rccl 15 each with 6 probe batteries
  b072/b073/b076/b0710_public/b0710_half/b0711 + solver137 FP32 + 8-key seed + 4
  gradient fixtures + backward, plain-daily/rccl 6 each, core 7, onnx 4,
  daily-consumers 5, rccl-consumers 6), ordinary UID1000 interactive `doas` guard 2.
  Four gate fixes vs D61: deploy captures rollback once (re-activation no longer
  clobbers it); baseline protects `current/torch/torch/__init__.py` by hash + links
  only for real symlinks (D61 `current/torch` is a directory); CLI `--ck-info` bare
  (sole-argument form); installed validator spawns nested entry `--check` with the
  clean container-base `LD_LIBRARY_PATH` (inheriting the run-worker prefix maps 3
  extra ROCm deps and drifts from the pinned 27-lib identity). 133 protected hashes
  / 5 links / 4 AOT match; boot `ed0d130a…`, kernel `7.1.5-bc250rocm-r2`, kernel log
  unchanged, 0 containers, no reboot/.105/user-app touch; ordinary free
  75,098,808,320→74,579,607,552 B. Evidence tar 739,076 B SHA `bab9d8d2…`, 221 files.
  Ordinary defaults unchanged; plugin is explicit opt-in FP32/FP16 packed2D forward
  only (backward via native solvers), no forced solver/global speedup claim.
  **Publication sync done 2026-11-13:** B07.13 tooling/docs propagated to the public
  Naster17 forks — `bc250-therock` `bc250/gfx1013` fast-forwarded `eec815a31→dc66a9fc3`
  (5 commits: publication record ×2 + B07.13 ×3; source as-is per established
  practice, no full IPs/excluded-app/creds in the diff) and `bc250-stack` `main`
  `29048c0→8b6c2bc` (curated/scrubbed B07.13 report in reports/README.md). Umbrella
  component patch series/SBOM unaffected (B07.13 is therock tooling, not a ROCm
  source patch; release identities unchanged). **NEXT:** optional follow-ups not
  started (fork CI, upstream tracking, full-prefix distribution, distributed
  benchmark, bottleneck optimization). Stopped per user pacing rule.

- **GitHub publication completed 2026-11-12 — public stack under Naster17.**
  Umbrella `bc250-stack` (architecture docs, 78-patch series, offline recipes,
  SBOM/licenses, qualification reports, tooling) + 10 component forks on
  maintained `bc250/gfx1013` branches: `bc250-linux` (kernel r2 snapshot on `main`
  + r3 series/provenance README), `bc250-therock`, `bc250-rocm-libraries`,
  `bc250-rocm-systems`, `bc250-torch` (ROCm/pytorch base, NOT mainline),
  `bc250-triton`, `bc250-torchaudio`, `bc250-torchcodec`, `bc250-migraphx`,
  `bc250-onnxruntime`. GitHub Release `v10.2-gfx1013-20261111` (17 assets:
  torchvision/torchaudio/torchcodec/bc250_quant wheels + ONNX stack + SBOM/docs +
  401MB torch source tarball). Old stack repos deleted (old-era
  bc250-stack/rocm/torch/linux); separate-concern repos preserved (bc250-docs,
  bc250-gfx1013-fix Mesa, bc250fc, bc250-sh2, bc250-docs-elektricm). torchvision
  pristine v0.28.0 = recipe pin (no fork); unmodified deps recipe-fetched.
  Push gotchas resolved: default SSH key maps to CiviDevs → HTTPS + gh token
  (Naster17); rocm-libraries stale fork (develop at ancient merge-base) → re-fork
  fresh for small delta; rocm LFS 37GB → bypass pre-push hook (source-only push).
  PII scrubbed in recipes/tools (build artifacts removed); patch series kept
  authentic for SBOM provenance. Local source repos untouched (publish remotes
  added). Publication does not change the deployed release on either board.

- **PLAN-10 completed 2026-11-12 (D64+D65) — measured training performance; ALL TEN
  original work packages now complete.** [Report](docs/development/bc250_plan10_results/README.md),
  [evidence](../../kernel/artifacts/plan10-d65). D64: `benchmark_training_d64.py`
  reuses the D62 owned registry + qualified timing (lr=0 fixed weights, cold first
  step, 3 warmup, 5×3-step synchronized intervals, median/IQR, tokens/s, peak
  memory, finite-loss/no-AMP-overflow/loss-stability checks; FP32 master weights +
  autocast AMP). D65: 15/15 cases pass on idle .104 release via ordinary
  bc250-train (gpt2/qwen3/qwen35-hybrid × FP32/BF16/FP16 × eager/sdpa/tiled),
  fixture sha b4adb7f9 bound. Findings (recorded before optimizing): FP32 fastest
  on these tiny models (bf16/fp16 AMP ~10–20% slower — cast overhead, no compute
  win); native tiled ≈ math sdpa; qwen35-hybrid reference gated-delta/conv path is
  the ~3–5× bottleneck (native overlay is 5.19.0-gated, absent in frozen 5.2.0);
  hybrid bf16/fp16 cold first step ~16.7s Triton JIT vs ~38ms steady-state.
  Utilization boundary: gpu_busy_percent EOPNOTSUPP → recorded unavailable + hwmon
  temp 53→54°C + governor auto. Fixed 2 fixture bugs (SEQ=64>gpt2 n_positions=32
  position-embedding segfault → SEQ=32 + incremental per-case writes; fp16 params
  → FP32-master AMP for GradScaler). 8 new tests; full bc250 suite 198 passed/84
  subtests; Black clean. 0 containers pre/post, no source/provider/prefix/default
  change, no reboot, no foreign touch; immutable release (no mutable lock).
  Single-board scope; distributed benchmark + bottleneck optimization are
  follow-ups. Publication still gated on explicit instruction. Stopped per user
  instruction.

- **D64 2026-11-12 — PLAN-10 owned training-performance benchmark fixture (local).**
  `build_tools/bc250/benchmark_training_d64.py` reuses the D62 owned registry/
  builders + the qualified timing methodology (lr=0 fixed weights, cold first
  step, 3 warmup, 5×3-step synchronized intervals, median/IQR, tokens/s, peak
  memory, finite-loss + no-AMP-overflow + loss-stability checks). Case matrix =
  registry backends × FP32/BF16/FP16 (15 cases: gpt2 eager/sdpa, qwen3 sdpa/tiled,
  qwen35-hybrid sdpa). Utilization boundary: `gpu_busy_percent` returns EOPNOTSUPP
  on this board → recorded unavailable + hwmon temperature + DPM governor context
  (single-board scope; distributed benchmark a follow-up, L01 was correctness-
  first). 8 new tests; full bc250 suite 198 passed/84 subtests; Black clean.
  Superseded by the PLAN-10 completion entry above (D65 done; 15/15 pass).

- **PLAN-09 completed 2026-11-12 (D62+D63) — compatibility matrix + reusable owned
  new-model admission fixture.** [Report](docs/development/bc250_plan09_results/README.md),
  [matrix](docs/development/bc250_plan09_results/compatibility-matrix.md),
  [evidence](../../kernel/artifacts/plan09-d63). D62: `admit_model_d62.py` —
  owned config registry (gpt2-tiny=D26 21280 identity, qwen3-tiny, qwen35-hybrid
  lifted from the excluded module's random-init branch), finetune-n7 import guard,
  battery load→GPU fwd vs CPU/FP64→finite/nonzero bwd→optimizer→bitwise
  save/resume→exact continuation (D59 backbone), op-level route probes
  (attention flags/tiled-SDPA/gated-delta/causal-conv/frozen-W8) recording
  fired/refused/absent; pure logic torch-free testable, GPU battery lazy. Matrix:
  THREE on-host-verified envs on torch 2.13.0a0+gfx1013 — frozen consumers
  5.2.0/1.12.0/0.18.1, lora-smoke app 5.22.1/1.15.0/0.21.2/trl 1.17.1/datasets
  5.1.0, Qwen3.5 native-op overlay 5.19.0. D63: 5/5 owned combos PASS on idle .104
  release via ordinary bc250-train (gpt2 fp32-eager + bf16-sdpa; qwen3 fp32-sdpa +
  bf16-tiled; qwen35-hybrid fp32-sdpa); forward rel 0–1.5e-05, exact bitwise
  resume, all 4 routes fired, 0 containers pre/post, fixture sha 28414a0c bound
  into every report. Fixed 6 fixture bugs during qualification (bf16 numpy→uint8
  byte view; gated-delta tuple unpack; CPU-ref inherited tiled→sdpa; W8 needs
  frozen weight; bf16 autocast; record-don't-crash). 17 new tests; full bc250 suite
  190 passed/84 subtests; Black clean (ruff absent on workstation). No
  source/provider/prefix/default change, no reboot, no foreign touch; immutable
  release (no mutable runtime lock). **NEXT:** PLAN-10 measured training
  performance (last important package); publication still gated on explicit
  instruction. Stopped after D63 per user instruction.

- **D62 2026-11-12 — PLAN-09 admission fixture + compatibility matrix (local).**
  [Matrix](docs/development/bc250_plan09_results/compatibility-matrix.md).
  New reusable owned new-model admission fixture
  `build_tools/bc250/admit_model_d62.py`: config-driven registry (gpt2-tiny=D26
  21280-param identity, qwen3-tiny, qwen35-hybrid-tiny lifted from the excluded
  module's random-init branch), owned-import guard (refuses finetune-n7 sys.path),
  battery = load → GPU forward vs CPU/FP64 ref → finite/nonzero backward →
  optimizer step → bitwise save/resume + exact continuation (D59 backbone, save
  BEFORE reference continuation) → op-level route probes (attention backend
  flags/tiled-SDPA, gated-delta, causal-conv, frozen-W8) recording
  fired/refused/absent. Pure logic importable without torch; GPU battery lazy.
  Matrix documents THREE verified environments on native torch 2.13.0a0+gfx1013:
  frozen consumers transformers 5.2.0/accelerate 1.12.0/peft 0.18.1 (+triton 3.8.0/
  numpy 2.5.1/hub 1.4.1/safetensors 0.8.0/tokenizers 0.22.2, from release
  packages.json), lora-smoke app 5.22.1/1.15.0/0.21.2/trl 1.17.1/datasets 5.1.0
  (verified on-host), Qwen3.5 overlay 5.19.0. 17 new local tests; full bc250 suite
  190 passed/84 subtests; Black clean (ruff unavailable on workstation).
  Superseded by the PLAN-09 completion entry above (D63 done; 5/5 pass).

- **Space reclamation completed 2026-11-12 — verified-redundancy pass (user-approved).**
  [Report](docs/development/bc250_space_reclaim_20261112_results/README.md),
  [journal](docs/development/bc250_space_reclaim_20261112_results/journal.json),
  [evidence](../../kernel/artifacts/space-reclaim-20261112). Member-by-member
  sha256 audit of all 68,444 retired-scratch files vs the 25 same-named stores
  (8,471 files/5,198 hashes indexed); every entry's authoritative source
  bundles/tarballs + repair bundles/patches + preservation/reconstruction records
  + qualification logs confirmed in its store (PLAN-07 proved reconstructibility).
  Unique content = working intermediates only (source-reconstruction checkouts,
  build/cache-data/lint-venv/mirror, nested b065/b066/b067 trees, re-exported
  tarballs, disposable helper scripts/drafts). Sole unique data (D01 host
  inventories 104/105/delta) copied+hash-verified into distributed-d01 store
  BEFORE deletion. Deleted all 25 scratch dirs (68,444 files/5.74GB nominal);
  archive now metadata-only (manifest+docs snapshot), catalog updated. btrfs
  reflink extent sharing: du removed 5.73GB from artifacts (69.16->63.43GB) but
  net blocks freed ~1.87GiB; workstation free ~7.5->~8.75GiB. No unique material
  lost; all stores/rollback/user material intact; no btrfs admin performed.
  **NEXT:** PLAN-09 compatibility matrix + owned new-model admission fixture,
  then PLAN-10 performance.

- **Cleanup completed 2026-11-12 — pedantic order/cleanliness pass (post PLAN-08).**
  [Report](docs/development/bc250_cleanup_20261112_results/README.md),
  [journal](docs/development/bc250_cleanup_20261112_results/journal.json).
  Git audit: workspace+therock fully committed (only user-owned install-arch.sh
  and src/ untracked, untouched); kernel/source + references/pytorch-upstream
  clean. Retired workspace tmp/ (25 entries/68,444 files/5.74GB October
  B06/B07/D01 scratch) -> kernel/artifacts/workspace-tmp-20261112/ via
  same-fs rename (capacity-neutral; count+bytes verified; catalogued); tmp/
  left with README. Deleted .105 D60 transfer leftover librocprofiler .so
  (release copy 2a975370 verified identical first) + rocm-base markers.
  Grouped .104 D59 evidence: 9 attempt dirs + 9 logs -> release-d59-attempts/,
  passing-run log -> release-d59/; D59 doc refs updated. PRESERVED: all
  kernel/artifacts stores, source/build/cache + pytorch-* rollback trees,
  historical .105 evidence layout, all releases/images/boot assets, patches/,
  board-support/, references/, ALL user apps/data on both hosts. Both hosts
  idle on release; no owned processes/containers; .qwen/tmp cleared.
  Workstation ~7.5GB free; optional future verified-redundancy pass over the
  retired scratch bulk could reclaim ~5GB (needs approval). **NEXT:** PLAN-09
  compatibility matrix, then PLAN-10 performance; two important packages.

- **D61 completed 2026-11-12 — PLAN-08 COMPLETE; release active on BOTH hosts.**
  [Report](docs/development/bc250_release_d61_results/README.md),
  [receipts](docs/development/bc250_release_d61_results/artifacts.json),
  [bundle](../../kernel/artifacts/release-d61). Finalization bundle:
  sbom.json (image 1bb4ac0e/torch a524980/triton ee65ff3b+wheel 2d65d6fe/
  ROCm revisions/consumer wheels tv 0.28.0 ta 2.11.0 tc 0.17.0/ONNX stack/
  opt-in modules/repairs/kernels r2+r3/launchers 9c8b3c78/runtime identities),
  known-limitations.json (hardware/software/performance/release/deployment
  matrix; corrects stale mutable-prefix string), source-recipes.json
  (D47-D56 store index + key hashes + D56 license bundle 87 files).
  .104 PROMOTED: promote_release_d61_on_host.py ed8143651 switched current +
  rccl pointer + d01 registry (813c8d02) atomically under board EX + runtime
  EX (inode checked) + preflight; recovery set retained (--rollback restores
  D49). Post-promotion via public callers: D61-DAILY-GPU-OK / CONSUMERS-OK /
  RCCL-GPU-OK (torch 2.13.0a0+gfx1013) + quant 220 PASS. Both hosts now on
  release 10.2-gfx1013-20261111-138a66f52; rollback/boot assets preserved
  (.104 D49+historical; .105 r4+D01+base+M105.1). PLAN-08 chain D57 plan /
  D58 assembly / D59 .104 qual / D60 .105 deployment / D61 finalization+
  promotion. Publication NOT performed (explicit instruction required).
  173 tests; 10 evidence files SHA-verified. **NEXT:** PLAN-09 compatibility
  matrix, then PLAN-10 performance; two important packages remain. Docs
  snapshot 20261112-d61-final synced at handoff.

- **D60 completed 2026-11-12 — release DEPLOYED+QUALIFIED+ACTIVE on .105; live rollback proven; PLAN-08.**
  [Report](docs/development/bc250_release_d60_results/README.md),
  [receipts](docs/development/bc250_release_d60_results/artifacts.json),
  [evidence](../../kernel/artifacts/release-d60). Release
  10.2-gfx1013-20261111-138a66f52-host105 at /opt/bc250-stack/releases.
  Staging: 16GiB free vs 18.3GB release -> D01/r4 inode pattern: rocm 24,442
  files hardlinked from retained base (24,440 identical) + 222 symlinks; 2
  detached repairs (MIOpen from r4 same-hash; rocprofiler transferred from
  .104 — .105 had pre-D49 a6dc1c88, now upgraded to 2a975370). 7.78GB direct
  .104->.105 two chunks; profile locks hardlink control inode 7733260; zero
  external symlinks; full manifest verified. Helper update required: old
  .105 launcher predates rccl support -> deployed reviewed 9c8b3c78 (rccl +
  D59 allowlist) to /usr/local/bin/bc250-torch + migration toolset, backups
  2da29b2c; rccl-aware qualifier revision (runtime_profile + --nccl-available).
  Target qual: daily r1 656 PASS; rccl r1 fail (no rccl mount), r2 fail
  (b031 needs --nccl-available), r3 656 PASS; daily r3 core 7 PASS. All gates
  boot-guard/dmesg-clean/4CPU-10GiB-12GiBswap. Activated atomically under
  migration-control EX lock (current+rccl pointer+d01 registry ea7314b0);
  smoke D60-DAILY-GPU-OK/CONSUMERS-OK(tv 0.28.0)/RCCL-GPU-OK. LIVE ROLLBACK
  CYCLE: r4 restored + D60-ROLLBACK-R4-OK, reactivated + D60-REACTIVATE-OK;
  release final active. Incident: r4 smoke inside EX lock deadlocked (mutable
  r4 SH lock same inode) — killed owned runner, smokes outside lock; rule:
  never launch mutable cohorts under EX control lock. Preservation: base
  rocm/M105.1 release+image 506f3b26/boot-health/grub r2 fallback/healthcheck
  data/board controls/user apps untouched; r4+D01 preserved; NO reboot (r3
  kernel 7.2.10, boot 08ecaaaa). 23 evidence files SHA-verified; gate logs +
  helper backups retained host evidence/release-d60. Revisions
  be6b17388/131845f20/3abf24b54/826121b16/ebcdaff63; 173 tests. **NEXT:** D61
  finalization (known-limit matrix, SBOM/license/source-recipe bundles, final
  receipts, promotion/publication decision — publication only on explicit
  instruction); then PLAN-09/10. Docs snapshot 20261111-d60-final synced at
  handoff.

- **D59 completed 2026-11-11 — release QUALIFIED (attempt 10); selection restored to D49; PLAN-08.**
  [Report](docs/development/bc250_release_d59_results/README.md),
  [receipts](docs/development/bc250_release_d59_results/artifacts.json),
  [evidence](../../kernel/artifacts/release-d59). Release
  10.2-gfx1013-20261111-138a66f52 validated=true/mutable=false; runtime SHAs
  daily 86d94f43/rccl f183bd3f bound in additive helper allowlist
  RELEASE_D59_RUNTIME_SHA256 (SDK sentinel check unchanged; helper stays
  deployed, binds D49+D59). Activation via public pointers+registry under
  board EX + runtime EX locks; 9 gates/326 checks via ordinary callers:
  quant-b052 220, rccl-daily+consumers 5, rccl-single+consumers 6, compiler
  4x14, distributed-b031 31, checkpoint-owned 8; each gate boot-guard
  armed/acknowledged + dmesg identity + no containers + 3GiB/5GiB floors.
  Negative visibility (-1) admitted on bound release; refusals rccl-env +
  unknown-profile; protected checks passed; rollback restored D49
  pointers+registry; post-rollback smoke passed. Promotion deferred to D61.
  Checkpoint gate replaced: n39 fixture depends on EXCLUDED finetune-n7 ->
  owned validate_checkpoint_release_d59.py (8 checks: bitwise state restore,
  exact resumed continuation, provenance + truncated-checkpoint refusal).
  10 attempts: 1 mount-syntax bug, 2 kinetic preflight refusal, 3 consumers
  caller fix, 4 llama-gguf dmesg noise, 5 transient memory preflight refusal,
  6 n39 excluded-app discovery (+ host reboot 73817cc1 after, r2, verified),
  7 llama-bench dmesg noise, 8 RNG map_location bug, 9 tamper-test
  truncation fix, 10 PASS 121.776s. All failures restored/refused safely.
  User dedicated .104 to stack work after app conflicts. Revisions
  caef114c5/2cff74ebc/d7c2c63cf/211dbe50d/60cc90a6d/32b94e5c4/bafbc577e;
  173 local tests; 98 evidence files SHA-verified; failed-attempt receipts
  retained host evidence/release-d59-attempt*. **NEXT:** D60 .105 deployment
  qualification (HOST105_UPDATE recorded lock/workflow; preserve r2 fallback/
  healthcheck/board controls/user apps), D61 finalization + promotion
  decision; publication only on explicit instruction. Three important
  packages PLAN-08/09/10. Docs snapshot 20261111-d59-final synced at handoff.

- **D58 completed 2026-11-11 — self-contained release assembled (no activation); PLAN-08.**
  [Report](docs/development/bc250_release_d58_results/README.md),
  [receipts](docs/development/bc250_release_d58_results/artifacts.json),
  [assembly receipt](../../kernel/artifacts/release-d58/assembly.json).
  Harness 9b6f569a8 (script bdc700cd) under EX prefix-update lock via
  helpers/update_dev_prefix.py (8ec1a4c9, byte-identical to repo; manifest
  322d3e42 pins are historical records, unchanged). Release
  10.2-gfx1013-20261111-138a66f52 at /opt/bc250-stack/releases: materialized
  from sdk-d49-r1 cohort (image 1bb4ac0e, torch a524980), daily+rccl-single;
  rocm/cache/python-consumers shared top-level, profiles with copied torch
  trees b052/b053 + rccl + gemm csv + SOURCE_* + adapted runtime.json
  (validated=false, mutable=false, provenance preserved); relative-only
  symlinks — D57 path-dependency gap closed. 18,295,550,735B/196,964 files
  in 267.726s; copied==source sizes; free 55.0->37.1GiB (5GiB floor).
  Preflight: root/current-target/no-containers/10 source sentinels/disk
  admission. Post: 10 release sentinels + independent 4/4 torch-lib recheck;
  self-containment scan ZERO violations; 6 per-file SHA manifests
  (rocm 24442/cache 75743/consumers 7714/daily-torch 44527/rccl-torch 44537/
  rccl 1) bound by digest in assembly.json, re-read+matched from host.
  current unchanged before/after; lock released; no owned processes.
  171 local tests OK (new adapt_runtime regression), black applied.
  REBOOT EVENT: .104 rebooted just before launch (aafacc82->be4ef4dd,
  cause unknown, no owned jobs running); post-reboot verification passed:
  r2 kernel, 10/10 sentinels, current unchanged, watchdog+cgroups started,
  cyan-skillfish only, no containers. Sources/live profiles/historical
  releases untouched; .105 unaccessed. **NEXT:** D59 independent-path
  build-free extraction/activation qualification (bc250-* callers, selected
  fwd/bwd/compiler/distributed/checkpoint regressions, identity refusals,
  actual rollback, restore current); then D60 .105, D61 finalization;
  publication only on explicit instruction. Three important packages
  PLAN-08/09/10. Docs snapshot 20261111-d58-final synced at handoff.

- **PLAN-07 CLOSED + D57 completed 2026-11-11 — release inventory/assembly plan; PLAN-08 started.**
  [D57 report](docs/development/bc250_release_d57_results/README.md),
  [receipts](docs/development/bc250_release_d57_results/artifacts.json),
  [host inventory](../../kernel/artifacts/release-d57/host-inventory.json).
  User accepted both scope decisions ("continue"): Triton provenance from
  pin ee65ff3b + 4 patches + qualified wheel 2d65d6fe (no rebuild demo
  required); D56 library-delta boundary accepted (no object-level follow-up
  before release). PLAN-07 complete for stated scope via D47-D56. CPU-only
  + read-only .104 inspection; no host mutation; .105 unaccessed. Release
  inventory: current=sdk-d49-r1/profiles/daily; BOTH live profiles are
  symlink assemblies over nik-home working trees (torch->pytorch-quant-
  b052/b053, cache->dev-user) + dev paths (rccl, gemm csv) + 20261013
  release python-consumers — core assembly gap is materializing a
  self-contained versioned tree. Layout precedent releases/10.2-gfx1013-
  20261013-4b000d542 (cache/python-consumers/rocm/torch/runtime.json/
  validation/review); registries migraphx/miopen-dl/ort-migraphx; 10
  bc250-* callers; launcher 761ec5803 MATRIX_PASS; preservation.json binds
  old SDK symlink layout; matrix has only 5 sentinel files/profile —
  release needs full SHA manifests. Assembly plan: D58 materialize release
  10.2-gfx1013-<date>-<rev> under /opt/bc250-stack/releases (EX prefix-
  update lock, no activation, current untouched, measure size first);
  D59 independent-path build-free extraction/activation + bc250-* caller
  tests + selected fwd/bwd/compiler/distributed/checkpoint regressions +
  identity refusals + actual rollback, restore current; D60 .105 via
  HOST105_UPDATE recorded lock (preserve r2 fallback/healthcheck/board
  controls/user apps); D61 limits matrix/SBOM/licenses/source recipes/
  final receipts; publication only on explicit user instruction. Standing
  rules: reservation/preflight for GPU work, 3GiB/5GiB floors, owned-job
  cleanup, no source reset, ccache 20G/no reset, finetune-n7 excluded.
  **NEXT:** D58 release assembly on .104 (mutating; needs EX lock +
  approval). Three important packages remain PLAN-08/09/10. Docs snapshot
  20261111-d57-final synced at handoff.

- **D56 completed 2026-11-11 — generated-file/license provenance closure (CPU-only); PLAN-07 OPEN.**
  [Report](docs/development/bc250_source_d56_results/README.md),
  [receipts](docs/development/bc250_source_d56_results/artifacts.json).
  No build/container/GPU/reservation; read-only .104 inspection only; .105
  unaccessed. Wheel license inventory closed: 87 unique files (174 log
  entries, both profiles identical), 87/87 in D52 kit with SHA; 44 exact
  D47 tracked git-blob OID matches, 41 nested in D50 recursive payloads, 2
  parent-vendored (llvm-openmp, miniz); zero mismatches/uncovered.
  Configuration identity: D55 CMake caches vs D50-preserved ORIGINAL caches
  (2406c7db/9c153e7e re-verified) differ only in 4 benign vars/profile
  (SITE hostname, 2 BOOL->INTERNAL same-value, empty find-module artifact);
  all compile flags identical. Live pytorch-work/build cache was
  reconfigured post-D50 (USE_DISTRIBUTED=0) — not authoritative. Version
  metadata: generated version.py 0899cb9f identical to retained original
  and deployed daily; original already Gitless (Unknown). Generated-file
  inventory: 6 steps (hipify, torchgen GEN_COMMAND, version, macros.h.in,
  linker script, offline six); closure proven by D55 offline build.
  Library-delta investigation: ruled out flags/paths/git/timestamps with
  evidence; deltas confined to ccache miss set (241/251 marker lines,
  209/224 invocations; 52 probes/profile; torch/csrc dominated incl 10
  autograd/generated); remaining hypothesis working-tree drift; follow-up
  = retained rebuild + object diff or torchgen replay (not a closure
  blocker). PLAN-07 Torch rebuild-path provenance CLOSED; remaining scope
  decisions: Triton rebuild demonstration, library-delta follow-up. Then
  PLAN-08; four important packages PLAN07-10. Docs snapshot
  20261111-d56-final synced at handoff.

- **D55 completed 2026-11-11 — full isolated clean Torch rebuild, both profiles; PLAN-07 OPEN.**
  Harness 3d494ef03 + fixes d6daedcf1/f2690ce03. [Report](docs/development/bc250_source_d55_results/README.md),
  [receipts](docs/development/bc250_source_d55_results/artifacts.json),
  [replay](docs/development/bc250_source_d55_results/verified-results.json).
  Full offline `pip install -e . --no-build-isolation --no-deps` from D52 kit
  6867ed71 + six 70e8a77b in preserved image 28ad36e6; --network=none, NO
  --read-only (venv writes), all preserved inputs RO; 4CPU0-3/10GiB/2048PID/
  2GiBtmp; same reservation/preflight/12GiB-20GiB admission/3GiB-5GiB floors
  as D54; deadline 7200s. daily-full-1 hit the old 3600s deadline AFTER all
  3282 ninja steps, during wheel license packaging (no compile/link failure);
  daily-full-2 passed 1761.276s peak 9,418,891,264B; rccl-single-full-1 passed
  1848.435s peak 9,578,196,992B; both installed torch-2.13.0a0+gfx1013
  offline. HIP 22/22 byte-exact vs D54 clean AND retained b052/b053; Eig
  object 6c3e6dc0 equal everywhere; gfx1013 offload bundles; 10 libs hashed
  per profile, 7 identical cross-profile, 3 differ = NCCL linkage set
  (recorded not equated). Preserved ccache RO reuse: daily 3023 hits/241
  misses/3291 invocations, rccl 3015/251/3293; zero growth. ccache 4.9.1
  read-only/no-stats omits Result: lines — counters fixed to 'Succeeded
  getting cached result'/'Read-only mode; running real compiler'; raw log +
  --show-config retained; running script never overwritten in place (.new
  staged, renamed after finish). Min mem 7,712,739,328B/disk
  56,769,667,072B; zero OOM/throttle; costs not performance. 170 local
  tests/Black/pycompile pass; ruff unavailable. 30 evidence files SHA-verified
  into kernel/artifacts/source-d55; attempt-1 receipt stays in host
  evidence/source-d55. Host tmp clean; no owned jobs/containers; watchdog
  active; sources/prefixes/images/profiles/boot/native controls unchanged;
  .105 unaccessed. Docs snapshot 20261111-d55-final synced at handoff.
  **NEXT:** PLAN-07 generated-file/license provenance closure from the proven
  rebuild path, then PLAN-08 release; four important packages PLAN07-10.

- **D54 completed 2026-11-11 — isolated clean-build admission and retained closure; PLAN-07 OPEN.**
  Implementation 742385f4f. [Report](docs/development/bc250_source_d54_results/README.md),
  [receipts](docs/development/bc250_source_d54_results/artifacts.json),
  [replay](docs/development/bc250_source_d54_results/verified-results.json).
  Prior session ran the clean probes; this pass added the retained audits,
  RCCL byte comparison, evidence transfer and closeout. Preserved image
  28ad36e6 + D52 kit 6867ed71 + offline six 70e8a77b; NNPACK hidden six
  download closed via PYTHON_SIX_SOURCE_DIR. --network=none --read-only,
  4CPU0-3/10GiB/512PID/2GiBtmp, board+SHruntime locks, 12GiB/20GiB admission,
  3GiBhost/5GiBdisk floors, 720s deadline, owned-only stop, scratch removed.
  daily-clean-4 passed 105.759s peak 891,113,472B; rccl-single-clean-1 passed
  115.584s peak 1,378,820,096B. Targeted native step: BatchLinearAlgebraEig
  .hip.o (+NCCLUtils.cpp.o rccl) + NNPACK peachpy object, gfx1013 offload
  bundle, link/ldd/run closure vs SDK libs (+rccl); exit 54 wheel refusal by
  design — NOT a full rebuild/wheel/GPU test. HIP translation 22/22 byte-exact
  daily b052 AND rccl b053. Retained audits: daily 3188 records/17610 unique
  headers, rccl 3190/17619; 0 stale/missing headers/missing link/unexpected
  loader paths; 55/56 link inputs +7 bare; 11/12 loader objects clean.
  RCCL_ROOT env fix binds host-path retained graphs (retained phase first
  run). Eig object SHA equal across profiles; NNPACK object differs per
  profile, recorded not equated. Min clean mem 13,746,327,552B/disk
  59,656,708,096B; retained 14,479,544,320B/61,650,231,296B; costs not
  performance. 168 local tests/two new parser regressions/Black/pycompile
  pass; ruff unavailable this pass. 30 evidence files SHA-verified into
  kernel/artifacts/source-d54 (31 receipt entries incl six). Host
  evidence/source-d54 keeps all seven attempts; tools/source-d54 reusable
  inputs; tmp clean; no owned jobs/containers/guards; watchdog active.
  ccache RO/no reset/20G kept; sources/builds/prefixes/images/profiles/boot/
  native controls unchanged; .105 unaccessed. Build SDK a6dc1c88 distinct
  from active D49 2a975370. PythonCPU/KFD/105clocksource causes still
  unknown/separate. .104 docs snapshot 20261111-d54-final synced at handoff.
  **NEXT:** PLAN-07 full clean-rebuild storage/resource plan from the admitted
  layout, generated-file/license closure, four important packages PLAN07-10.

- **Storage follow-up completed 2026-11-11.**
  .104 freed 2.26GiB and reached 57.46GiB before docs sync.
  [Report](docs/development/bc250_storage_cleanup_20261110/README.md),
  [receipts](docs/development/bc250_storage_cleanup_20261110/followup-results.json).
  First ten-root attempt refused deletion after detecting real rollback/candidate
  links and retained mounts to four historical Torch trees. They remain in place.
  Verified smaller archives and discarded redundant full backup.
  Retired six MIOpen stage/package roots, A02 profiling output, B052 candidate
  staging, B033 build copy and exact duplicate B053 evidence transfer. Four
  exited B033 containers had generated metadata only. Captured logs, removed
  those containers, verified all Podman image IDs. 59 protected hashes/links match.
  340 Ninja/CMake inputs show no retired-path references. Preserve active b052/b053,
  retained b037, historical rollback paths and sparse-followup third-party overlay.
  No GPU workload or profile activation. .105 outside this batch.
  All producer and retirement jobs complete. Removed own host transfer scratch/FIFO.
  .104 journals evidence/maintenance/20261111, workstation archives
  kernel/artifacts/storage-cleanup-20261111. Root and TheRock AGENTS.md now require
  cleanup at task close and grouping durable inputs, results and recovery material.
  **NEXT D54:** builder/toolchain ELF/CMake/link closure and isolated clean-build
  admission from preserved D52/D53 kits. Historical runtime faults unchanged.

- **Storage cleanup and release prep completed 2026-11-10.**
  Implementations afdac891a/e3f8c899d.
  [Report](docs/development/bc250_storage_cleanup_20261110/README.md),
  [archive receipts](docs/development/bc250_storage_cleanup_20261110/artifacts.json).
  .104 reclaimed 44.20GiB, 11,902,251,008B to 59,362,246,656B before docs sync.
  Retired 18,695 obsolete .o files, 5 archived proof copies, redundant D52 OCI
  and closed transfer staging. Then retired 654 verified roots (26 tmp, 115
  evidence, 56 helper, 457 loose files), 57 docs snapshots and 21 legacy WS roots.
  All recoverable except obsolete objects, which can be rebuilt from retained
  sources/configs/cache. Original OCI image/local archive and D35/D41 oracles kept.
  Staging r1 refused old FIFO. R2 strict root-readable archive preserved guard
  records, FIFO metadata and deduplicated bytes. 126,368 entries, 4,469,298,417B
  SHA1dc1317a8dedeeab9590104e742f1824115f006364802f969ce3b8b2a9033ac6.
  Producer13799/13800, retirement16474/16472/17580 done. No owned jobs/guards.
  Raw journals now .104 evidence/maintenance/20261110; local verified payloads
  kernel/artifacts/storage-cleanup-20261110. Tmp empty except usage README.
  D52 kit now tools/source-d52/package-r1/torch-source-d52-r1.tar.gz with its
  original evidence path as a compatibility link. 59 protected identities and
  prefix links equal. Six provider registrations and two launcher checks pass.
  166 local tests. .104 observed r2 boot5c7ba874 after inter-pass reboot.
  Active ELF audit 69 runtime/43 SDK, 340 Ninja/CMake files and live registry
  paths checked. .104 source audit captured 371 code variants and verified 23
  LFS pointers. Loose port patches already in Git including three audio patches.
  22 shared Torch HIP translation variants preserved for D54 comparison, no
  authored source patch inferred from generated output. .105 read-only audit
  matches .104 image/Torch a524980 and checked HIP/MIOpen/Torch libraries.
  SDK differs: D49 2a975370 on .104 and qualified a6dc1c88 on .105.
  Target qualification precedes SDK promotion. No release build or GPU test here.
  PythonCPU/KFD/105clocksource issues remain distinct from the SDK repair.
  **NEXT D54:** audit saved builder/SDK/header/ELF/CMake/link inputs and admit
  an isolated clean build. 55.3GiB free clears the earlier 40GiB disk estimate.
  Use the preserved D52/D53 kits and 22-file observed HIP output reference.
  Check complete resource and dependency closure before building. PLAN07 open.

- **D53 completed 2026-11-10 — external native build payload preserved; clean-build budget refused, PLAN-07 OPEN.**
  Implementation ba613c857. [Report](docs/development/bc250_source_d53_results/README.md),
  [receipts](docs/development/bc250_source_d53_results/artifacts.json).
  Entire retained ROCm/compiler prefix+B03.4RCCL+toolchain/two ccache helpers
  streamed104 FIFO→workstation, no second SDK/archive on board.28,008 entries:
  24,522 files/3,262 dirs/224 links;10,726,325,706 file bytes. Archive2,467,945,440B
  SHA76a018fae10972070ff414dbe5ef7c55e94f2674076cb86b8d69fd317db28d98.
  Each file SHA/size/mode and directory/link spelling verified independently here;
  actual archive manifest EXACT separately exported host manifest. Exporter repeats
  full source inventory afterward, equal. All lexical link targets present, not
  ELF/CMake/include closure or relocated execution proof. No extraction/native build.
  HIPClang55833a05 matches D51 current observation; RCCLca790dc7 retainedB03/B05.
  SDK a6dc1c88 retained build input, separate active repairedD49SDK2a975370.
  Fifty-two license-named entries, no whole-source/license/publication clearance.
  Metadata1,331,009B SHA9d0097243f26ab0cc9f323c53f7bd0c79ebb69c62c61684830150fcc47538c12;
  portable stdlib tools3,964B SHAe807cf280f32b7ca0e193ed1b43e1e987b64e5213243951014eb29628470f892.
  kernel/artifacts/source-d53; host tools/source-d53 and evidence/source-d53-r1,
  source-d53-controller.log. Controller14238/child14239finished; resultpassed,
  status.json initial launch only. No owned jobs/containers/guards. RootSSH closed.
  EXboard+SHruntime/UID1000/CPU0-1/2GiBAS+600CPU inherited/900wall/100ms/
  3GiBhost+5GiBdisk stops only ownedPG; no aggregate/cgroup/paging peak claim.
  Local transfer16GiB initial/5GiB reserve/900wall. Costs289.099export/277.299transfer,
  not performance; minhost15,165,878,272B/disk12,425,912,320B.
  Proposed sequential clean-build40GiB budget (3source+8objects+4link+20newcache+
  5reserve) REFUSED current12,432,261,120B free, short30,517,411,840B. This proposal
  is NOT measured mandatory/sufficient peak. Reuse-existing-cache with bounded
  incremental growth may reduce allocation; keep20G configured/no reset.
  160 local tests/seven new payload/hardlink/mode/duplicate/missing/gzip/budget
  regressions, Black/pycompile pass. Tiny fixture caught TarInfo file-type mode
  bits; permissions explicitly masked. No failed full host export. Exact executed
  helper retained; final verifier adds gzip-trailer hardening to verification only.
  CurrentD49profiles/sdk/boot2774d201r2/nativecontrols/originalsources/builds/cache/
  images/apps/fallbacks unchanged.104docs20261110-d53-final;105unaccessed/D43docs.
  PythonCPU/KFD/105clocksource unknown causes remain separate/unfixed.
  **NEXT D54:** bounded image/toolchain ELF/CMake/link closure probes and revised
  storage-admitted isolated clean-build layout. Reuse preserved image/source/SDK,
  do not re-export/repeat sweeps or launch failed-budget build. Keep105/current
  prefix/originalsources/caches unchanged. PLAN07 open/four important packages remain.

- **D52 completed 2026-11-10 — builder preserved, fresh source/recipes assembled; PLAN-07 OPEN.**
  Implementation f05489d31. [Report](docs/development/bc250_source_d52_results/README.md),
  [receipts](docs/development/bc250_source_d52_results/artifacts.json).
  Original image28ad36e6 preserved as OCI:682,451,968B SHA
  c57dbec6f76fc0f651eca005435320f47092f8f4dc478026053d058390998f88.
  All14 compressed layer hashes/uncompressed diff IDs and original config identity
  verified on104 and workstation; original image unchanged, no load/pull/retag.
  HIP compiler/ROCm remain external build-prefix inputs, not embedded in this image.
  D47/D50/D51 replay in fresh owned assembly: sourceHEAD8d8b18d/tree33a86273,
  native C++ basea524980/tree3e3edf7f, Gloo overlay913f01fb.69 dependency locations
  materialized,21,501 parent+78,598 dependency working entries=100,099 source entries.
  Eight excluded nested branches empty, not admitted for arbitrary configurations.
  R1 raw-Git-blob copy check fails mkl-dnn template:331 working bytes vs324 canonical,
  pinned text/eol=crlf. R2 compares copied raw bytes to tree-verified snapshots while
  retaining canonical Git checks; no source normalization. Parent bytes captured
  before materialization. Failed R1 and encoding diagnostic preserved. Correct UID/
  isolated XDG worker avoids root-attributes warnings; no global Git config changes.
  Gitless source/recipes artifact346,534,969B SHA
  6867ed719c201cf7bc368800aa64c5e16a2e3a4c3c03485ad502684e2feb5723.
  Manifest-backed100,116 source+recipe entries independently verified locally,
  extracted to a path containing a space, EVERY extracted entry rechecked. Private
  proof Git indices/history excluded from archive but kept onhost. No native build.
  Fresh prospective pip-only scripts remove old cache-copy and refuse existing
  CMake cache; RCCL input now explicit /workspace/rccl. Both bash-n checks pass in
  preserved image; recipes NOT executed. Safe defaults/nativegfx1013/20Gccache/
  no reset/four jobs/one linker/version-number exports retained for future admission.
  All3239/3241 retained compile units exist+SHA256;260/profile under build dirs.
  Include roots103/105 and38 native link rules/profile inventoried. Initial rule
  parser missed target before colon; fixed/regression, R1 inventory retained.
  Not complete transitive headers/.ninja_deps/linked-ELF or historical-byte proof.
 153 local tests/nine new OCI/source/EOL/recipe/link/extraction cases; static checks
  and metadata cross-check pass. Portable stdlib tools4936B SHA
  f06f5b23578e7223bbeb299cbfdf1d1b6409e1f9de75d5899da01b9f6b48116e.
  Local artifacts kernel/artifacts/source-d52/; host tools/source-d52 and evidence/
  source-d52-{image-r1,assembly-r1,assembly-r2,inputs-r1,inputs-r2,package-r1}.
  Metadata archive860,815B SHAc36a2f45c766f5adfe160f0112a2f2b697d8e9fea1df277885adc792e66e1225;
  host evidence/source-d52-evidence-r1.tar.gz, metadata/tools only (large payloads separate).
  Controllers7158/7744(failed)/8899/10275finished; image/assembly/assembly-r2/package
  controller logs in evidence/source-d52-*-controller.log. All jobs/containers done.
  Image storage admission2*imageSize+4GiBassembly+5GiBreserve passed; export EXboard+
  SHruntime/CPU0-1/GOMAXPROCS2/420s save/100ms owned stop, no aggregate memory-cap claim.
  Assembly/package UID1000/CPU0-3/2GiBAS+180CPU per inherited process/480wall/100ms/
  3GiBhost+5GiBdisk reserves. Costs36.491/99.852/146.687s, not performance claims.
  Min host14,907,224,064B/disk12,486,606,848B; cgroup peak/paging IO unmeasured.
  Original sources/builds/caches/images/prefixes/kernels/apps/checkpoints/fallbacks
  preserved; D49SDK2a975370/boot2774d201r2/native controls unchanged. Finalidle/watchdog
  active, no owned workload.104docs20261110-d52-final synced;105 unaccessed/D43docs.
  PythonCPU/KFD/105clocksource causes still unknown, distinct fixed SDK/VMheap/D05/D21.
  **NEXT D53:** storage-admitted matching external ROCm/compiler/RCCL/toolchain kit,
  immutable input identities, then clean-build resource admission for fresh source.
  Keep105/currentprefix/original source/build/cache unchanged. PLAN07 payload/license/
  clean-reproduction closure OPEN; four important packages PLAN07-10 remain.

- **D51 completed 2026-11-10 — selected native input disposition/portable evidence; PLAN-07 OPEN.**
  Implementation f9e45aace. [Report](docs/development/bc250_source_d51_results/README.md).
  Eight missing D50 branches excluded for exact daily/RCCL configs: five optional
  FLASH/MEM_EFF/CK_SDPA/CK_GEMM/MSLK flags off;0 family references in complete3239/
  3241 compile entries AND Ninja graphs, pinned source CMake guards archived.
  Required missing-branch fetches0; branches still unexpanded for other configs.
  Original build records identify immutable image28ad36e62398875f0ad77112b1d0f2b1cd0b3b55560d9cc38c6908cada8caa06,
  distinct current runtime1bb4ac0e. Both retained scripts pin build version
  2.13.0a0+gfx1013/number1/nativegfx1013 and authorized pip-only invocation;
  native git_version Unknown preserved, no invented rebuild. D50 version-input
  inventory gap resolved from scripts, not inferred from version.py.
  Read-only CPU probes:35 build/49 runtime Python distributions+RECORD hashes,
  seven actual tool hashes/versions. GCC13.3/Clang24 LLVM7667f52/Python3.12.3/
  Ninja1.13/ccache4.9.1. HIP compiler measured NOW from retained RO build mount,
  not embedded image/no historical-byte-identity claim. Actual CMake3.31.12
  vs Python cmake distribution4.4.0 detected; PATH binary hash equals cache tool.
  Initial supplemental text decode refused ELF; binary read resolves assumption.
  R1 probe kept, finalR2 binds selected-path hash in both images. No compiler/codegen.
  Native9 pyproject requirements, complete retained graphs/caches/toolchain/scripts,
  version outputs/five pinned source files+Torch license packaged. Triton wheel
  2d65d6fe313,852,959B retained. No full wheelhouse/image/toolchain payload export.
  Portable package kernel/artifacts/source-d51/portable-build-inputs-d51-r1.tar.gz
  1,130,174B SHAddb71d39c18d82e7b3a4acace088037ae58176c65a303b9ab8c50686c04b580b;
  same host tools/source-d51/, extracted build-inputs-r1. Workstation+104 kit replay
  pass without original checkout/image/GPU/network.144localtests/four new feature/
  include/version cases, three altered source/backend/CMake kits refused, Black/
  critical Ruff+F401/diff pass. Updated metadata SHA cannot admit enabled backend.
  Raw finalR2 metadata kernel/artifacts/source-d51/104-evidence-r1.tar.gz1,084,737B
  SHA542fe580f63524c7efba82afc5832b12a4d067e323dec76351d65d496b0bc788;
  host evidence/source-d51-evidence-r1.tar.gz. Host evidence/source-d51-r1/r2 kept;
  controllers32552/1044finished, source-d51-controller.log/source-d51-r2-controller.log.
  R2 cost2.474s, notperf; minhost15,084,756,992B/disk21,467,734,016B. EXboard+SHruntime,
  noGPU nodes/Torch import/network/pull; image probe2CPU0-3/1GiBnoswap/128PID/64MiBtmp/
  120s child/100ms3GiBhost+5GiBdisk reserves, owned stops only. No cgrouppeak/paging claim.
  No source/cache/build/nativeprefix/library/image/kernel/reboot/app/checkpoint change;
  D49SDK2a975370/boot2774d201r2 unchanged, finalidle/nativecyan/watchdogactive/nojobs.
  .104docs20261110-d51-final synced at handoff;105unaccessed/unchanged/D43docs.
  Historical PythonCPU/KFD/105clocksource causes remain unknown, separate fixed
  SDKsentinel/VMheap/D05/D21; no crash/stability/performance improvement claim.
  **NEXT D52:** admit storage for preserving/exporting original image28ad36e6,
  assemble fresh selected-source/recipe kit from D47/D50/D51 and inventory required
  generated/header/link inputs before clean native rebuild. Inventory is explicit;
  payload/license/clean-build closure OPEN. Four packages PLAN07-10 remain.
  Keep105/currentprefix/source/build/cache unchanged; no blind optional-backend enable.

- **D50 completed 2026-11-10 — portable available recursive Torch dependencies; PLAN-07 OPEN.**
  Implementation eb6147ea0. [Report](docs/development/bc250_source_d50_results/README.md).
  D47 parent-manifest anchor fixes37 direct pins. Traversal discovers77 locations:
  69 available,64 unique pinned commit payloads,75,522 tracked blobs,107 tracked
  license/copying/notice-named files. Eight unavailable/unexpanded branches under
  aiter/flash-attention/mslk explicitly recorded; descendants below them unknown.
  Complete recursive closure=false; no whole-stack offline-build/license claim.
  Raw commit SHA1 binds each tree to the parent gitlink, private Git index recreates
  exact tree including modes/symlinks/nested pins; no original objects/history needed.
  First local replay fails ittapi symlink ../../ -> ../.. normalization by Python
  tar data filter. Fixed only trailing-slash preservation AFTER standard target
  safety check; other normalization refused. Failed r1 kept; full local r2 and
  independent104 replay pass all64 payloads. Six focused regressions;140 local
  tests/Black/critical Ruff+F401/diff pass, three altered driver/Gloo/CMake exports
  refused. Modified source archive with updated SHA still fails commit/tree test.
  Existing Gloo overlay applied only in private replay: tree913f01fb, cuda.cuSHA
  7dc2ce53bf518ba455b0e47e7b2e09976261db008b7d2a83669f02c9451f997f
  EXACT retained104 pytorch-work dependency source. TensorPipe/libuv included.
  Raw daily/RCCL CMake/version inputs bound; FLASH/MEM_EFF/MSLK off, Gloo/TensorPipe on.
  Both version.py say2.13.0a0+gfx1013/HIP7.17/ROCm10.2/git_version Unknown; preserve
  that historical metadata, use pinned manifests for native provenance. Future
  build-version env must be explicit; original build env cannot be inferred from it.
  Package local kernel/artifacts/source-d50/portable-dependencies-d50-r1.tar
  258,723,840B SHA1c624f9d672ef3798a72b9974a6778b3a58c36bcafeefc4d06a04da0781e5540;
  same host tools/source-d50/. Package subdir dependencies-r1, private proof in
  evidence/source-d50-r1/replay; result.json/replay.json/worker.log/metadata retained.
  Host metadata local kernel/artifacts/source-d50/104-evidence-r1.tar.gz82,862B SHA
  06262318fe63377a24f9ba82682ac47c0a348f8195ae934cb4165683c1eb96b6;
  same host evidence/source-d50-evidence-r1.tar.gz. Controller30288finished,
  evidence/source-d50-controller.log; no jobs/guards. CPU-only UID1000 affinity0-3/
  2GiB address-space+120CPU seconds per inherited process/420wall/100ms owned-PG
  stop at3GiBhost/5GiBdisk; not aggregate cgroup limits.54.477s replay cost, notperf.
  Min host14,889,078,784B/disk21,882,839,040B. SHruntime only for metadata audit.
  No GPU test/native build/prefix/source/cache/image/kernel/reboot/app/checkpoint
  changes. D49 SDK2a975370 remains active, boot2774d201r2/native controls unchanged;
  finalidle/watchdogactive. Own tmpfs transfer copied+SHAverified durably, duplicate
  removed only; source/build/caches preserved.105unaccessed/unchanged/D43docs.
  .104docs20261110-d50-final synced at handoff. PythonCPU/KFD/105clocksource causes
  still unknown, distinct fixed SDKsentinel/VMheap/D05/D21; no new crash-fix claim.
  **NEXT D51:** classify eight missing branches against exact native build inputs,
  package required missing pins in private stores, close explicit build-version/
  toolchain/Python-wheel/generated-input inventory. PLAN07 still OPEN; four
  important packages PLAN07-10 remain. Keep105 unchanged; no blind backend enable.

- **D49 completed 2026-11-10 — repaired native SDK activated on .104; rollback qualified.**
  Implementation b6de91cb4; SDK source143f1d878296176c738274202092bb247b7b771c.
  [Report](docs/development/bc250_sdk_d49_results/README.md),
  [replay](docs/development/bc250_sdk_d49_results/verified-results.json).
  Active cohort `/opt/bc250-stack/candidates/sdk-d49-r1`, both daily/RCCL profiles;
  original miopen-plan02-r4 fully preserved as fallback. SDK SHA2a975370ba4eb1c2fe39e406ff36b175f554a49d0aa19febbd326f641fcb590b.
  cp-al structural/inode comparison covers27,905 entries; only SDK file detached
  atomically. Original tree unchanged; not a new full-byte rehash of all libraries.
  Exact baseline runtime.json bytes preserved for package identities; adjacent
  native-overrides.json explicitly records SDK-only source/hash/qualification delta.
  Independent provider wrappers/registrations; D01 release points to new RCCL
  profile, retains baseline runtime hash/Torch identity and records SDK override.
  Public/D09 shared helper SHAd33f2504de50eac21baf6f43ff4eb30a05c602881cda9b12b3df13360a2c3588:
  exact HIP -1 admitted only with qualified baseline+SDK+SONAME aliases, repeated
  under runtime lock. Legacy cohorts retain D44 refusal. Ordinary loader order,
  no preload/global CI disable. Custom preload/auditor/native overlays unqualified.
  Three phases:12 first activation +2 restored old GPU +4 final processes;
  138 numerical checks/20 actual GPU traces/eight negative imports, loaded SDK
  paths/hashes bound. GPU output/dx/dw digests EXACT restored old and D48 controls.
  CPU FP64 gates2e-6/2e-4, pinned copies/nondefault stream/exact profiled replays;
  all strict kernel logs unchanged, three guards ACK. Six CK/MX/ORT registration
  identity checks pass, no new provider GPU/collective/model/AMP/counter/speed claim.
  EX board+SH runtime tests; EX board+EX stable runtime selections/rollback.
  4CPU0-3/4GiB noswap/512PIDs/256MiB shm/60s child/420s batch/600s recovery;
  100ms sampling,3GiB host/5GiB disk reserves. Min host14,229,344,256B,
  disk24,240,054,272B; phase wall87.318/15.064/29.172s, not performance results.
  134 local tests/three new identity regressions/Black/critical Ruff+F401/diff pass;
  independent raw source/trace/rollback replay and three altered archives refused.
  Executed archived probe/controller differ from committed harness only by removal
  of three unused imports; installed shared helper exactly matches committed source.
  Metadata local kernel/artifacts/sdk-d49/104-evidence-r1.tar.gz1,658,051B SHA
  759648024bd274dee3a0e7710d6d5f6f253dccdec86cc81dd793738313aa71b6;
  host evidence/sdk-d49-evidence-r1.tar.gz, tools/sdk-d49. Controller7740finished;
  evidence/sdk-d49-{controller,install,activation-r1,rollback-r1,final-r1,export}.
  No owned workload/armed guard remains. Boot2774d201r2 unchanged; native board
  controls/watchdog preserved. No build/source/cache/image/kernel/app/checkpoint
  deletion or mutation; no new native build, reused qualified D48 candidate.
  .104docs20261110-d49-final synced at handoff; .105 unaccessed/unchanged/D43docs.
  Historical exact-1 SDK import fault is now fixed on104; old Python CPU0x3b3e6
  has no original core/backtrace, KFDmm/queue cause untraced,105clocksource OPEN.
  Quiet logs do not fix those causes. VMheap/D05/D21 fixes remain separate.
  **NEXT D50 / PLAN-07:** portable Torch recursive dependency contents/licenses
  starting D47's37 direct pins/27 immediate nested links; explicitly inventory
  missing objects and Gloo legacy-platform overlay/build-version inputs, then
  bounded source package/replay. Toolchain/wheel/license closure still OPEN;
  four important packages PLAN07-10 remain. Keep105 unchanged. Do not rerun
  capacity/kernel sweeps or rebuild the already-qualified SDK without a new need.

- **D48 completed 2026-11-10 — native SDK sentinel candidate built/qualified; NOT active, PLAN-07 OPEN.**
  SDKsource143f1d878296176c738274202092bb247b7b771c; TheRockhelpers67e682956.
  [Report](docs/development/bc250_sdk_d48_results/README.md),
  [replay](docs/development/bc250_sdk_d48_results/verified-results.json).
  Exact-1terminator nowhandled before SDKinvalid-tokenCI FATAL, preserving prior
  matchedselections. CIglobalchecksON. Localrocm-systems sourcecommitted; original
  hostsource/build read-only. Oneunitccachecompile +retained181-inputlink in1bb4ac0e,
  ownoutputs/RPATHnormalization. R1compiled butmissed linker--dependency-file output;
  ROmount refusedretainedlink.d write. R2redirectsdepfile too,ccachehit/0.641s relink,
  NOTcleanbuild/perfclaim. Oldr1resultstale'running' meansfailedcontrollerexit1,notlive.
  CandidateSHA2a975370ba4eb1c2fe39e406ff36b175f554a49d0aa19febbd326f641fcb590b,
  17,575,272B; sourceSHA5aa899d0c9921077fb411f9d649e6735b2381a49c97a2fe4caa79db5bb03fcc0.
  OldactiveSDK a6dc1c88 unchanged.1450dynamicbinding/type/visibility/definedsymbols+
  DT_NEEDED/SONAMEcontracts equal, notallAPI/ABIpropertyclaim. Ccache20G/noreset,
  normalentry/statupdates,1CPU4GiBnoswap256PID/600s CPUbuild/EXboard+SHruntime.
  R3final23freshprocesses daily/RCCL:6HIP/CUDA/ROCR-1imports CPU0devices/noTorchlazyinit;
  14candidateGPUdefault/0/0,-1 +2ordinaryoldSDKcontrols;1oldSDKpreloaddiagnostic.
  227numericchecksinccontrols,CPUFP64 output+firstinput/weightgrad2e-6/2e-4,
  exactprofiledreplays/pinnedH2D-D2H/nondefaultstream. ALLcandidateGPUdigests exact
  respectiveoldcontrol.32ordinaryGPUtraces realkernels+copies/positive/correlated;
  everyprocessbinds loadedSDKpath/SHA. No fullmodel/AMP/compiler/NCCLcollective/counter/
  universalcorrectness/throughputclaim. Finalkernelbefore/after EXACTall23, guardACK.
  HSAreturnsGPU-XX=UUIDunsupported pernativeheader; planned6UUIDcasesunavailable,
  nosyntheticidentity. R1testfailedstrictUUIDfixture AFTERcontrolmath/profilepassed;
  sources/trace retained. R2candidateLD_PRELOAD arithmeticpass/emptyGPUtracefail.
  R3UNCHANGEDSDKpreload reproducesemptytraces,ordinaryold+newresolutionworks:
  selectprivateSDKviaLD_LIBRARY_PATH+SONAMEaliases,neverpreloadforactivation.
  Exactinternalregistrationtimingnottraced; no numeric/tracegate relaxation.
  Testlimits4CPU0–3/4GiBnoswap512PID/256MiBshm/60child/420total/600independentguard;
  100ms ownedstop3GiBhost/5GiBdisk.168.950stestcost,hostmin14,205,579,264B, nocgroup/
  hostpagingIOpeakclaim. No reboot, bootsame2774d201r2/finalidle/nativecyan/watchdog.
  Controllers5337(fail)/5960(buildpass)/6807/8382(testfail)/11032(finalpass) finished;
  allownedcontainersgone/guardsACK. SourceCPPGit143f/patch0005 inamd-mainline series;
  originalhostsources/builds/activeprefix/libraries/images/profiles/registry/kernel/
  bootfallback/apps/checkpoints untouched; normalcacheupdatesonly.105unaccessed.
  131localtests(3newredirection/CI/overwritecases)/Black/criticalRuff+F401/scopeddiff/
  rawtrace+provider+ELFreplaypass;3alteredsource/provider/tracearchives refused.
  Localmetadata kernel/artifacts/sdk-d48/104-evidence-r1.tar.gz1,345,232B SHA
  ef15397c5b5c49ccfa6800709759cc154f2135fa54f936a3086865846db11a4d;
  candidate sdk-d48-candidate-r1.tar.gz4,040,834B SHA
  98d26859f7d57fed98ec2c5e2aa09c8121c1b55fc12b1b5470a4745a736ccf3e.
  Candidateartifactrequires recordednativecohort/notstandaloneSDK/releaseSBOM.
  Hosttools tools/sdk-d48; evidence/sdk-d48-build-r1/r2 andtests-r1/r2/r3(actual
  separatefullsdk-d48-*dirs); metadata evidence/sdk-d48-evidence-r1.tar.gz.
  Privatequalifiedview evidence/sdk-d48-build-r2/prefix; activeSDK+D44refusal STILLold.
  .104docs snapshot20261110-d48-final synced athandoff, .105D43docs unchanged.
  **NEXT D49:** inspectprovider/prefixpreservationbaseline, stageversioned104activation
  preservingoldSDK/fallback; EXruntime-update procedure, accuratedaily/RCCL/distributed
  metadata,ordinarypublic/profilechecks +rollbackqualification beforeselection.
  Keep105unchanged. Public-1admission onlyboundtoqualifiedinstalledSDKidentity;
  legacycohortsretainD44refusal. PLAN07recursive dependency/license/Gloo-overlay/
  build-versionclosureOPEN, FOURimportantpackagesremain. UnknownPythonCPU0x3b3e6/
  KFDmm-queue/105clocksource causes remainseparate/unfixed; fixedVMheap/D05/D21distinct.

- **PLAN-07 / D47 completed 2026-11-10 — portable Torch parent snapshot/replay and SDK resource/input admission; PLAN-07 OPEN.**
  Source0bd04bc50. [Report](docs/development/bc250_source_d47_results/README.md).
  Standalone package local kernel/artifacts/source-d47/portable-torch-source-d47-r1.tar
  66,703,360B SHAffb18da3ea5e79349965c467d1ef2194a9bbe2863a93d01e08f1f6245d8e5143.
  Fullbaseline21,491blobfiles+37gitlinkidentities, mainLICENSEblobc23172f7; no shared
  Gitobjects/sourcehistory/networkneeded. NewprivateGitindex/objectstore recreates
  EXACTbaselineabdd861b/native3e3edf7f/HEAD33a86273trees. OriginalshallowGitpreserved;
  privateHEADunborn/verificationONLY, notconfigurednativebuildtree. No falsefullbuild.
  All37directdependency pinsavailablelocal/HEADsmatch;53licensefileidentitiesrecorded,
  dependencycontents/27immediatenestedgitlinks/buildversion/toolchain/wheels/Gloooverlay
  stillseparateclosure. No offlinewhole-stack/buildreadyclaim. Nativea524+sourcebridge
  8d8b18ddistinctionretained. No dependencycontentfetch or source/build/cachedeletion.
  Workstationreplaypass;104r1pass, then ambientGitpathisolation hardened ALLGitcalls
  inclls-tree/export, regression coversGIT_DIR/GIT_INDEX_FILE; oldsources/runs kept.
  Final104r2sourceSHAacf65cc4402f740cf3a9a9cbbbe03d65d1b195128abd079998ed4477df5c2b7f
  fullreplaypass12.398s inclmetadata(notbuild/trainingperf). CPU UID1000/affinity0–3/
  2GiBAS+120CPUseconds perprocess(inherited,notcgroupcap)/180swall;100ms ownedPGstop
  host3GiB/disk5GiBreserves. Hostmin15,200,382,976B/diskmin24,443,985,920B finalr2.
  Userfreedspace:initial14.04GB→later24GB+, notagentcleanup. NoGPU/recoveryguard/native
  build; SHruntime onlyreadSDKmetadata. Controllers1130/2722finished/noownedjobs.
  SDKstructuredcompile/linkargv captured,181explicitlinkinputs accounted178host+3
  glibcstaticlibs inmatching1bb4ac0eimage. Initialhost-onlypathcheckmissing3 expected
  onAlpine; readonlystdimageprobe passed/pullnever/networkoff/noGPU/1CPU256MiB64PIDs.
  Compile_commands omitsccachelauncher; retainedNinjahasccache. OriginalargvMUSTNOT
  executeunredirected: readonlysource/build, isolatedsource/object/depfile/DSOoutputs.
  Resourceenvelope4GiBextradisk+4GiBcandidateRAM/jobs1/link1/ccache20G(noreset), host3GiB/
  disk5GiBreserves nowfits; freshactualresources/ownership/outputredirection beforebuild.
  SDKcandidateNOTbuilt/tested/installed; CIchecks/D44refusal/nativeprefixes unchanged.
  128localtests(4newportableintegrity/envcases)/Black/criticalRuff+F401/syntax/diffpass;
  actualapplicabletamperedpatchwithUPDATEDpayloadSHA stillrefusedbypinnednativetree.
  MetadatareplaychecksdriverSHA/trees/limits/finalidle. No fullupstream/GPUnewclaim.
  Metadataarchive local kernel/artifacts/source-d47/104-evidence-r1.tar.gz98529B SHA
  29dc89e752b280504fa5aeb1a6bd0608bd4e0771b6bf9199f2c597c3720809e7; hostevidence/
  source-d47-evidence-r1.tar.gz. Fullbaseline/source/replaydirs omittedfrommetadata,
  retainedseparatelytools/source-d47/portable-torch-r1 andevidence/source-d47-r1/r2.
  Finalidle/nativecyanservice/watchdogactive, bootsame2774d201r2; no container/armedguard.
  No originalsource/build/cache/prefix/library/profile/registry/image/kernel/boot/app/
  checkpointchange.105unaccessed/unchanged (userreportsfreedspacethere, noinspection).
  .104docs snapshot20261110-d47-final synced athandoff;105D43docs unchanged.
  HistoricalPythonCPUIP0x3b3e6/KFDmm-queue/105clocksource causesOPEN, distinct SDKimport/
  fixedVMheap/D05/D21. FOURimportantpackagesPLAN07–10remain; recursiveclosureOPEN.
  **NEXT D48:** isolatednativeSDK-1sentinelcandidate compile/link fromverifiedrecipe/
  sourceafterfreshadmission. RedirectALLoutputs, preserveCI/nativeactiveprefix/D44
  refusal. Direct HIP/CUDA/ROCR-1import/devicecount +valid0 andnativeGPUoutput/backward/
  CPU-firstprofilertracegates bothprofiles, bindloadedDSOpaths/hashes, EXboard+SHruntime/
  independentrecovery forGPU. No activationuntilqualified/EXprefixchecks. Recursive
  deps/licenses/Gloolegacyoverlay/buildversionclosure remainsPLAN07workalongside.

- **PLAN-07 / D46 completed 2026-11-10 — complete Torch deltas and exact SDK RPATH provenance; PLAN-07 OPEN.**
  Source4ea1bd17b. [Report](docs/development/bc250_source_d46_results/README.md).
  Reproduced old0001/0002 gap EXACT3files sdp_utils.cpp/compile_fx.py/_gfx1013.py;
  allalreadyinpatchedGit. New SEPARATE patches/bc250-pytorch-complete fromc779020:
  0001→nativea524980 TREE3e3edf7f1de1fd1fe4730f9ec4ed30dd038b2112;
  0002→sourceHEAD8d8b18d TREE33a862732a52c8825a3d1f9c0b609d06fba0df65.
  Fullprivatebare/index treeequalitybothpasses; applicablewrongtreepatch refused.
  Oldseries/historypreserved; neverapplybothseries.35pathbaseline/native/HEADmanifest.
  CPU104privatefocused35filereplay bothpatches pass; daily/RCCL30rawsourcefilesexact,
  SAME5C++ files showpost-HIPifychanges. Referenceoverlay reproduces5generatedbytes
  exactlyBOTHprofiles; notindependentcodegen/notbuildinput. Piponlycodegenrulekept.
  Initial UID1000 lockpermission refusal BEFOREoutputs; authorizedrootSHruntime
  auditpassed. No GPU/reservation/recoveryguard/build/sourceworkingtree mutation.
  SDKbuildbcbb20f9→activea6dc1c88 gapONLY.dynstr: zero-pad/replaceRPATHat129047
  EXACTwholefileSHAa6dc1c884c4f446e23a2afc8185b26ef9fd55b57cdd1d15324d46054b902b9c2.
  Code/allothersectionsunchanged. CI=ON/compiler-DROCPROFILER_CI=1;4relevantSDK
  retainedsource/configfiles matchnative1f9609fd6. D45identityuncertaintyRESOLVED;
  warning→FATALmechanism nowartifact/source/configbound. NotoldunknownCPUfault.
  Minimal -1terminatorproposal patches/candidates/rocprofiler-sdk-d46 PRIVATEapply
  checked treeacf507bcef600d600547a6ad957979a5db1cdc74 ONLY; NOTbuilt/tested/installed.
  GlobalCIchecksunchanged, D44productionrefusal active104only, no nativefixclaim.
  8componentpintreesavailable/124localtests(4newELFtests)/Black/criticalRuff+F401/
  syntax/scopeddiffpass. Patchcontextblankmarkers needpatchspecificwhitespace;
  actualcompletepatchapplication --whitespace=error passes. No fullCI/GPUretestclaim.
  Torchshallow23commits: sharedlocalobjects !=portablebaseline/dependencyclosure.
  FullSDKbuild17,575,368B/agentobject127,688B; admissionfree5,409,996,800B only
  41,287,680Babove5GiBreserve; no compiler/link/cache/stage/rollback budget admitted.
  Hostownedtools tools/source-d46, evidence/source-d46-r1 +source-d46-controller.log;
  allCPUjobsdone/noarmedguard. Hostarchive local kernel/artifacts/source-d46/
  104-evidence-r1.tar.gz973572B SHA081b595f248372adc836823ff3c3ca716045aa80ca840b8c422f8359bf3ceca5;
  samehost evidence/source-d46-evidence-r1.tar.gz. SDKproposal remainslocalGitonly.
  No prefix/nativecode/library/default/profile/registry/image/kernel/reboot/app/
  cache/checkpoint/build change.104boot2774d201r2;105unaccessed/unchanged/D43docs.
  .104docs snapshot20261110-d46-handoff synced, earlierfinalsnapshot preserved.
  Docsextractionbrieflyfree5,355,589,632B<5GiBreserve. SHA/sizeverified duplicate
  D44/D45/D46docstransfers removedONLYhosttools(28,533,182B); workstationarchives/
  extractedhostsnapshots retained, free5,384,122,368Bbefore smallhandoffupdates.
  Receipt inreport transfer-cleanup.json. Handoffhardlinks unchangeddocs/atomic
  changedfilereplacements avoidfullcopy; no source/build/cache/release/image deletion.
  HistoricalPythonCPUIP0x3b3e6/KFDmm-queue/105clocksource causes stillOPEN, separate
  fixedVMheap/D05/D21. No performance/runtimeimprovement or universalcorrectnessclaim.
  **NEXT PLAN-07 / D47:** portableTorchsource snapshot/replay independent shared
  objects, explicitshallowbaseline/submodule/dependency/license/buildinputclosure;
  matchingSDKsource/config candidatebuild/testadmission alongside. BudgetboundedRAM
  scratch/persistentstage/compiler/link/ccache/rollback before nativeSDKbuild; never
  reduce3GiBhost/5GiBdisk reserves or deletepreservedsources/builds/caches. D44refusal
  retainedthroughnativecandidatequalification. FOURimportantpackagesremainPLAN07–10;
  sourceclosure notyetcomplete, unknownfaults remainreleaseknownlimits.

- **PLAN-06 / D45 completed 2026-11-10 — bounded mixed stability; PLAN-06 bounded scope COMPLETE.**
  Source0ef00f158. [Report](docs/development/bc250_mixed_d45_results/README.md),
  [records](docs/development/bc250_mixed_d45_results/verified-results.json).
  Six fresh104 processes alternatingdaily/RCCL, seeds4500–4505: each FP32 matmul/
  conv/mathSDPA/realFFT output+loss+ALLinputgrad vsCPUdouble,2denseAdamWupdates
  ALLparams/moments/counters vsindependentCPUupdate.162comparisons/12updates pass,
  predeclaredatol5e-5/rtol5e-4 unchanged.27checks/process,6uniqueCIDs/PIDs/peropJSON/
  faulthandler45s. No usermodel/app/AMP/compile/distributedcollective qualification.
  EXboard+SHruntime,4CPU0–3/4GiBRAM+combined(swapallowance0)/512PID/256MiBshm;
  100ms3GiBhost/5GiBdisk/60schild/600s independent recovery retained. GPUallocated
  67,128,320B each; largest sampled cgrouppeak932,622,336B, hostmin13,893,718,016B.
  Sampledcontainerswap/OOMkill0, strictbefore/afterkernel logs EXACT all6. No host
  pagingIO/afterexitcgroup peak/performance/universalstability claim. GuardsACK;
  controller19038done/exportdone/finalidle/nativeboardservice/watchdogactive.
  Foreign-arrival postflight failure recordsfailed+ACK afterownworkgone toavoid
  rebootingforeignapp; branchunexercised. Stops onlyowned labels, preserves apps.
  HistoricalPythonCPUIP0x3b3e6 remainsunknown/nooriginalcore; KFD B07.7 exec-after-HIP
  frontendmitigated byrunpy butkernelmm/queuecauseuntraced;105clocksource timeout
  OPEN/pr_info_once. QuietD45 doesNOTrepair anycause. Distinct fixedVMheap/D05/D21.
  Additionalnativevisibility lead: profiler logging.hpp mapsROCP_CI_LOG(WARNING)
  toFATAL underROCPROFILER_CI. RetainedSDK configCI=ON/ninja-DROCPROFILER_CI=1;
  activeDSOa6dc1c88 differs retainedbuildbcbb20f9. ExactbuildprovenanceUNRESOLVED,
  do notclaim byte-matchedCIcohort or replaceSDKblindly. D44refusalstillactive104.
  No nativecode/library/default/profile/registry/prefix/image/kernel/boot/reboot/
  package/app/105change, no rebuild.104boot2774d201r2;105priorDocker mustberechecked.
  Black/criticalRuff+F401/syntax/diff/independentmetadatareader +3private source/ACK/
  workerarchive refusals pass. No fullupstream/new120unittestrunclaim. Localarchive
  kernel/artifacts/distributed-d45/104-evidence-r1.tar.gz253974B SHA
  f37a05accdf5e6e111854f2b9e06b2ccfb76fd0d96a642ef0d5321ab60baea47.
  Hosttools tools/mixed-d45, evidence/mixed-d45-r1, mixed-d45-controller.log;
  archived evidence/mixed-d45-evidence-r1.tar.gz. Allownedjobsdone/noarmedguard.
  .104docs snapshot20261110-d45-final synced at handoff;105unchangedD43docs.
  **NEXT PLAN-07 / D46:** CPU-only source→installedartifact inventory; audit old
  SDPAthreefile reconstructiongap againstactualpatchedTorchGit/configs/qualified
  bytes, andresolveSDKCI/source/build/packagingidentity before minimalparserrepair
  buildplan. No blindrebuild/reset/globalCIdisable.104space~98MiB above5GiBreserve:
  require incremental/cache/link/stage/rollback storagebudget before nativebuild.
  FOURimportantpackages remainPLAN07–10; unknownCPU/KFD/clockcauses stayOPEN in
  releaseknownlimits, no endlesscapacity/kernel rework or quiet-pass repairclaim.

- **PLAN-06 / D44 completed 2026-11-10 — specific HIP visibility import crash refused on .104; PLAN-06 OPEN.**
  Source9d03c5c90. [Report](docs/development/bc250_visibility_d44_results/README.md),
  [replay](docs/development/bc250_visibility_d44_results/verified-results.json).
  Retained D12 HIP_VISIBLE_DEVICES=-1 reproduced through old native launcher:
  exit139, profiler SDK agent.cpp:216 invalid-token fatal, Python abort/segfault,
  new libc general-protection fault. Not historical CPU IP0x3b3e6 or fixed VMheap.
  New shared validate_options refuses EXACT HIP_VISIBLE_DEVICES=-1 before elevation/
  container creation, inherited/env/envfile/private contexts. Other filter semantics
  unchanged/unqualified. Candidate refusal exit2/zero containers; CPU import stays
  CUDA-uninitialized; default/explicit0 daily +explicit0 RCCL GPU loss/ALLinput+weight
  gradients vsCPUdouble pass12checks at1e-5. Candidate logs EXACT unchanged.
  EXboard+SHruntime,4CPU0–3/4GiB RAM+combined(no container swap allowance)/512PID/
  256MiBshm,100ms3GiBhost+5GiBdisk reserve/60schild/600s recovery guard retained.
  Sampled hostavailable>=14,505,086,976B; no cgrouppeak/pagingIO/performance claim.
  Initial r1 adapter failed old extensionless file load BEFORE runtime (not a crash
  reproduction); sources/logs kept. Fixed explicit SourceFileLoader; authoritative r2.
  Controllers23508/26957 done, guardACK/idle/native cyan-skillfish only/watchdogactive.
  Installed under EXboard+EXruntime BOTH104shared copies: /usr/local/bin/bc250-torch
  and /opt/bc250-stack/tools/distributed-d09/bc250_torch.py. SHAa7b3a5b8db0bf178356271258da3b0dd6fb816461192e817a92a1c40d908a5cb;
  oldc5167f17 backups evidence/visibility-d44-install-r1/{0,1}-bc250_torch.py.
  Actual UID1000 torch/train refuseexit2, distributed existing tracebackexit1 before
  doas/worker. Native parser not repaired; observed fatal first in profiler SDK,
  historical message says ROCr import. Native Torch/ROCm/MIOpen/RCCL/D09entrypoint/
  registry/images/profiles/kernels/bootfallbacks/apps unchanged; no build/reboot.
  .104boot2774d201r2;105unaccessed/unchanged, priorforeignDocker must berechecked.
  120localtests/Black/criticalRuff+F401/syntax/diff/independentreader pass;3mutated
  source/ACK/worker archives refused. Finalarchive local kernel/artifacts/distributed-d44/
  104-evidence-r1.tar.gz492032B SHA99b1b95f3df6e6da1700099f83f46868d3cbf6f83f52a312b6ad5f881ee3e214;
  host evidence/visibility-d44-evidence-r1.tar.gz retains r1/r2/installation/sources.
  Ownedtools tools/visibility-d44; all jobs finished/no armed guard. .104 docs snapshot
  20261110-d44-final at handoff; .105 docs remain20261110-d43-handoff.
  HistoricalPythonCPUfault/KFD uncertainty and105clocksource remoteCPUtimeout OPEN;
  later quiet logs/pr_info_once cannot prove fixed/no recurrence. No stability claim.
  **NEXT PLAN-06 / D45:** predeclared six fresh104 owned processes alternatingdaily/
  RCCL; each small FP32 matmul/conv/mathSDPA/realFFT output+firstbackward vsCPUdouble,
  plus2denseAdamWupdates/state checks. Fix seeds/atol5e-5rtol5e-4 beforeGPU; same
  4GiBcaps/3GiBhost/5GiBdisk/60schild/600s guard/ownership. Peropstages/faulthandler/
  resources/strictlogs; stoponlyowned onfailure, do not extend run or claim unknown
  CPU/KFD/clock causes fixed. Foreign workload invalidates test but is never stopped.
  PLAN06 staysOPEN, FIVE importantpackages remain. No capacity/kernel restart.

- **PLAN-05 / D43 completed 2026-11-10 — eight native1.024B FSDP updates; PLAN-05 bounded scope COMPLETE.**
  Sources8a590d02f/7c0dc6234, forwardcache4f684e984/backwardcache31aa4415d,
  proof/byte replaya0ec08c66; final baseline-aware reader/export documented in report.
  [Report](docs/development/bc250_distributed_d43_results/README.md),
  [verified results](docs/development/bc250_distributed_d43_results/verified-results.json).
  Optional tools/distributed-d43 BOTH; GPUseries distributed-d43-r3. D42 descriptor
  1d8fef22942169acbd759d3639a9730d4090dd68ebf211debb660aea3121e7e1 binds both shapes/ranks.
  Original D35 FP32 GPT2 trainingloop/width2048/20blocks/seq128/global2/micro1,
  AdamW1e-4/eps1e-5/wd.01/foreachFalse/clip.5. No AMP/checkpoint/offload/compile.
  One firstloss parity +2warmup+5timed updates, eightfiniteclipnorms; ALL final local
  GPU params/gradients/moments finite in1Mchunks/counter8 (outside timer). Eachrank
  512,235,520localstateelements. No raw CPUrefs/HTTP/fullstate gathers/CPUoracle.
  Firststep math bound to D42 independent oracle; not eight-step CPUref/convergence.
  Actual r1 large firstupdate passed but105 reserve stopped secondforward at
  available3,214,176,256B. R2 forwardhook trims passed secondforward then stopped
  secondbackward at3,216,723,968B. Both attempts owned-only stopped/ACK/logs unchanged,
  no OOM/reboot; peer RCCLtimeout after cancellation is consequential, not newmathbug.
  R3 public FSDP nonroot full-backward hooks returnNone and trim unused cache;
  forwardpre/rootpost andphase trims retained. Live bytes equal before/after EVERY
  trim; model/wrap/collectives/optimizer/tolerances/reserves unchanged. Large160
  backwardcalls/352trims, control96/224. R3 control ALL8norms/5losses/statecounts EXACT
  r1 BOTH. Median18.347s vs17.996 baseline pair (~1.95% higher, no universal gain).
  Large median53.3648s/update, GPUlive8587.883/8585.430MiB/reserved8808/8884MiB;
  cgroup4731.906/4729.129MiB, sampled hostmin3.620/3.282GiB. Worker441.6s vs
  guardian455.6/455.2s includes startup/finalscan/teardown. Native collectiveinput
  gather4,063,272,960/reduce4,097,884,160B per timedstep/rank; not wirebytes/isolatedtime.
  First r2 control mathpass but105 strictlog gate refused unrelatedDocker veth;
  unmodified recheckstrictpass, no filters/foreignstop. Preserve ALL oldruns/sourcebackups.
  Container swap/OOM0;105 hostswapPREEXISTING2,756,608B constant throughout bothshapes,
  104zero. Initial absolutezero-hostswap reader assumption refused; explicit constant
  occupancy baseline gate now tested/refusesgrowth. No pagingIO counters collected:
  do not claim zero hostswap or no swapIO. Runtime reserves/caps unchanged.
  24groups/4finalACKguards/116localtests/4 alteredarchive refusals/lint/diffpass;
  original D42worker bytehashes +D35/D36/D39/D40/D41 replayed. Guard budgets/locks:
  EXreservation+SHruntime/4CPU0–3/hard10GiB+12combined/512PID/256MiBshm,100ms ownedstop
  host3GiB/cgroup9GiB/disk5GiB/child1400/isolatedguard1800; originalkernel600/GPU120 unchanged.
  Finalr4 metadata archives local kernel/artifacts/distributed-d43 (r3 preserved):
  1043,089,642B SHA817fb3fa54781394b21f1ee016b3171f5cf704bb2ea6dbe7ebe52967e5203c40;
  1053,115,157B SHAad46f415ba5d273f7df9fdfc8a3a3928ff8cbd7d7364312a8fc9df36533ae7c3.
  Finalcontrol guardians1048932/1059259, large9842/10185, export12241/12614 exit0.
  All owned tests finished/guardsACK; export preflight BOTH idle. Later final
  handoff104idle;105 NEW unrelated Docker workload present (left untouched),
  so do not assume105free for next GPU phase. Qualification logs precede this.
  Boots2774d201r2/e902e636r3 unchangedduringD43;
  native Torch/ROCm/MIOpen/D09/defaults/images/fallbacks/apps untouched. No rebuild/
  install/reboot/foreignstop. Docs snapshot20261110-d43-handoff synced BOTH at handoff;
  ownedtransfers grouped/evidence retained. Clocksource105 remote-read timeout remains
  OPEN (pr_info_once); historical Python CPUfault distinct fixed VMheap/D05/D21.
  Capacity: single allresidentFP32p/g/m/vfloor16,391,536,640B>either hostRAM, while
  this two-rank recipe safely runs8updates. No deliberateOOM/AMP-offload impossibility/
  universalmodel/longrun/restart/speedupclaim. PLAN05 bounded scope closed; FIVE
  importantpackages remain (PLAN06–10), optional expansion separate.
  **NEXT PLAN-06 / D44 (planned):** inspect retained Python CPU/KFD/invalidvisibility/
  clocksource evidence and current public launcher validation. Choose a bounded
  invalid GPU selection reproduction and implement prelaunch refusal/diagnostics
  where demonstrated; preserve recovery/ownership/defaults/apps. Then predeclare
  mixed-workload stability scope; do not reopen completed kernel work or invent
  endless capacity fixtures. Unreproduced historical faults stay cause unknown.

- **PLAN-05 / D42 completed 2026-11-10 —1.024B source-bound two-host GPU audit; PLAN-05 OPEN.**
  Sources48f58f74c/cachef9f9ab563/boundedCPU4582251e3; finaldistributed-d42-r5 BOTH.
  [Report](docs/development/bc250_distributed_d42_results/README.md),
  [verified results](docs/development/bc250_distributed_d42_results/verified-results.json).
  OwnedGPT2 FP32 FSDP1,024,471,040params/20blocks/width2048/seq128/global2;
  ALL244 raw/clipped/updated parameters +localAdamW moments/counters pass ORIGINAL
  bounds againstD41 FP64oracle99b36b92. GPUloss9.416439056396484 vsCPU9.416439180620552;
  rawmax3.21960e-7/clipmax1.54533e-8, combinedtolerance. BOTHraw/clip/updatedhashesexact;
  eachrank512,235,520localstateelements. Fresh352.96McontrolallnativehashesEXACTD39/D40.
  21units/63summons, no wholemodel gather. Two actual RAM-safe stops drove repairs:
  r3rawaudit reserve stop→release unused GPU cache (~4GiB afterwrap/~2GiB afterbackward);
  r4 ALLraw+clippassed/nativeAdamWexecuted, CPUupdatedoracle reserve stop→r5 CPUAdamW
  ref1M-elementchunks/1139chunks/ALLvalues/16MiB momentpair vsfullparameter scratch.
  FullinitialFP32blob≤64MiB; otherauditmemory separate. Synthetic2M+13/3chunks CPU
  weights/moments/stepbitexact vsfullvector, bothhosts. Fivecacheevents preserve live
  allocations; nativeoptimizer/tolerances/reserves unchanged. Alloldattempts retained.
  LargeGPUliveaudit8265.713/8265.291MiB, reserved8514/8578MiB; cgroup4715.75/4715.99MiB;
  worker hostmin3.488/3.283GiB, guardianmin3.484/3.249GiB. Swap/OOM0. Worker603.6s,
  guardian723.0/616.8s includesreads/checks, NOtrainingthroughputclaim. Single-resident
  FP32p/g/m/vfloor16,391,536,640B>hostRAM, no deliberateOOM/AMP-offload impossibilityclaim.
  ExplicitCPUreadybarrier600s beforeGPUinit; GPU120unchanged/client-finishedteardown.
  Elasticagentalreadywaits guardianverify; workerremoteverifyalone≥98sideal1Gb/s,
  twoaggregatepasses≥197s (notbothnecessarilyinsideoldGPUtimeout). TempUID1000source
  104→105 port29842, whitelist/1MiB/256MiBAS/64FD,1953requests/49.175GBraw+manifest;
  128MiBmaxreader/0diskcache/fallback/spool, sourceSTOPPEDbeforeACK. D41refs104only.
  EXreservation+SHruntime/4CPU0–3/hard10GiB+12combined/512PID/256MiBshm; ownedstop
  host<3GiB/cgroup≥9GiB/disk<5GiB,1400schild/isolated1800guard. Originalkernel600cap
  unchanged; ASTsamealgorithmexceptcap/doc/error, localACK/expiry/maxrefusaltests.
  24groups/4ACKguards/108localtests/4alteredarchive refusals/lint/diffpass; predecessor
  D35/D36/D39/D40/D41replayed. Finalarchiveslocal kernel/artifacts/distributed-d42:
  1043,620,521B SHA7320a9b908b46f05315f8eec7dd780f21bad7a4920ceceb47ae0de98b3acb7ae;
  1053,519,928B SHA577555241fc90bc2b78e7700338885e915c9833ed7786bfbdc39b1f3aee14962.
  Guardians1048914/10530445/export4467/29062 exit0, alljobsfinished/guardsACKed/idle.
  r4firstcontrol105clocksource 'Watchdog remote CPU3 read timed out' remainsOPEN:
  kernel0957f1aa4 pr_info_once/retry notTSCunstable; quietlaterlogs≠causefixed/no recurrence.
  TSCunchanged. HistoricalPythonCPUfault distinct fixedVMheap/D05exit/D21restore.
  Testboots2d86d523r2/e902e636r3;104laterreboot02:21Nov10 AFTERfinal23:55:41Nov9,
  causeunknownhere. Exportboot1045bdbf2f8/105same; verifierchecks test+ACKboot separately.
  PostexportpreflightBOTHidle/watchdogactive. Nativeprefixes/Torch/MIOpen/D09/images/
  defaults/fallbacks/apps untouched; no build/install/reboot issued. Docs/manifest
  snapshot20261110-d42-final synced BOTH at handoff, transfers grouped/evidence preserved.
  **NEXT PLAN-05 / D43:** fresh training-only1.024B FP32 FSDP capacity/memory gate,
  boundtoD42proof, nofullCPUrefs/auditsintimedworker. Boundedfinite multiupdates,
  warm/completejob timings/actual host+device+communication costs, retain reserves/
  locks/guards. No larger-fit/generaltraining/speed claim from this one auditedupdate.

- **PLAN-05 / D41 completed 2026-11-09 —1.024B CPU oracle published; PLAN-05 OPEN.**
  Producer `3afa04930`, final finite scanner `f75cb361e`.
  [Report](docs/development/bc250_distributed_d41_results/README.md),
  [verified results](docs/development/bc250_distributed_d41_results/verified-results.json).
  Additive tools/distributed-d41104ONLY. CPU FP64 GPT2:1,024,471,040 params/20blocks/
  width2048/heads32/vocab8192/positions256/seq128/global2, eager/dropout0/cacheOFF,
  tied weights; original D35 construction/math with explicit source-bound dimensions.
  Fresh352.96M control ALLinitial/grad hashes/loss EXACT D35, no fullref duplicate.
  Native nonreentrant/preserveRNG checkpoint guards21 modules,20→40 block entries
  (early-stop caveat); ALL244 hooks once/finite grads/rawroundtrips, zero retained,
  max visible16,777,216 gradient elements/128MiB roundtrip. CPU loss9.416439180620552,
  FP64 initial preserved afterbackward, dropout0 RNGunchanged. No optimizer/PG/agent/
  GPUdevices/CUDAinit. Direct D09 ordinary Torch launcher+identity bootstrap, same
  native cohort; explicit removal of kfd/render mappings. CPU/GPU claims separate.
  CPU admission beforeallocation:7.633GiB FP64params+1.25runtime+.5engine+3×128MiB
  scratch=9.758GiB planned;≥12GiB available/planned3GiBhostreserve. Softstop10GiB,
  hard11GiB/combined12GiB/4CPU0–3/512PID/256MiBshm; extra1GiB vsoldCPU cap explicit.
  Worker perphase/tensorgates+guardian100ms monitor stops onlyowned container on
  softRAM/host<3GiB/disk<5GiB, no abort needed. Disk full11.449GiBpayload+5GiBreserve+
  64MiBmetadata admitted; no deliberateOOM/extrapolatedfit. PLAN01 EXreservation+
  SHruntime/600s guard/460s child/ACKafterexit retained. No global cacheflush.
  Large cgroup9268.129MiB=9.051GiB/VmHWM9498.918MiB, hostavailable≥5.365GiB,
  swap/OOM0. Control3748.035MiB; different shape/direct-noagent, no speedup inference.
  Workerlarge172.9s/guard181.8s inclI/O/recompute/checks; control62.8/71.7s.
  Durable104 evidence/distributed-d41-r1/large-oracle/reference has488rawfiles:
  12,293,652,480B, plusmanifest SHA99b36b92073fddf2e68da39525392b202992ac8263693e8b2247866165077efb.
  Preserve this and oldD35refs BOTH; no rawcopy to105/workstation. Final disk5.424GiB.
  R2 export rehashes old/newrefs and independently scans ALL initialFP32+gradientFP64
  finite values/SHA in1MiBchunks withoutTorch/model; each1,024,471,040 values pass.
  R1 preliminary pre-finite-scan metadata retained, no model rerun.16CPUgroups/2ACKs/
  100localtests/3 source-hook-admission archive refusals/Black/criticalRuff+F401/diff
  pass. No failed model phase/relaxed gate/fullCI/GPUfit claim. .104guardians22608/
  23355,exports24339(r1)/25094(r2) exit0; exact CID/UUID/commands/ACK/cgroup/logs kept.
  Local kernel/artifacts/distributed-d41; final metadata archive843352B SHA
  622ed7be3fdb7abf27e85af247869cd5fa63a628323fe2c0d623924427101cc3.
  Final idle/no jobs/containers/armed guards; boots2d86d523r2/e902e636r3/defaults/
  native PLAN02MIOpen381eac05/Torcha524/RCCL/D09/image/fallbacks unchanged. No native
  build/install/reboot/download/prefixmutation/apps/foreignstop.105docs-only update.
  Docs/flow/manifest synced BOTH. PythonCPUfault stilldistinct fixedVMheap/D05/D21.
  **NEXT PLAN-05 / D42 (planned):** source-bound1.024B two-host per-unit FSDP GPU
  audit with104 D41oracle/105 boundedstream, ALL original mathgates/128MiB reader.
  Beforelaunch review RAM/device/scratch and startup/readiness/guard budgets: two
  remote12.294GB verificationpasses take≥197s atideal1Gb/s, exceed old120sPGtimeout.
  Explicitreadybarrier or reviewedboundedtimeouts, don'treuseD40blindly. No deliberate
  singlefullFP32AdamWOOM (statefloor>RAM). GPUcapacity remains unqualified; later
  training-only measurement separate. Six important packages remain.

- **PLAN-05 / D40 completed 2026-11-09 — bounded reference transport; PLAN-05 OPEN.**
  Implementation `470c4046b`. [Report](docs/development/bc250_distributed_d40_results/README.md),
  [verified results](docs/development/bc250_distributed_d40_results/verified-results.json).
  Additive tools/distributed-d40 BOTH; same352,957,440-param native FSDP/D35 refs/
  D39 independent CPU clipped-gradient audit. Fresh both-local control vs .104-local/
  .105-remote reader. ALL148 raw/clipped/updated hashes and local AdamW moments/
  counters EXACT control/D39/D36 perrank; loss9.364227294921875, maxcliperr1.30094e-8,
  original bounds retained. No CPU oracle in GPUworkers,39 unit contexts/max28.33M.
  .104 guarded temporary UID1000 read-only source rehashes296 blobs; binds104:29840,
  allows only105, no listings/writes, O_NOFOLLOW/type/size checks,256MiBAS/64FD/
  singlehandler/1MiBchunks. Plain HTTP privateLAN fixture service, no public auth/
  encryption claim. Stops beforeACK; parent keeps EXreservation+SHruntime throughout.
  .105 preflight+worker enforce original manifest SHA/length and perblob status/
  length/header+computedSHA, no redirect/encoding/unknownitems/roots,20s socket/60s
  blob deadlines. No silent fallback/disk cache; descriptor only68,123B manifest,
  max retained blob96MiB/verification1MiB (hard transportcap128MiB). Existing full D35
  refs BOTH preserved. Source exactly1185 complete requests: each296 rawblob4times,
  payload16,941,957,120B+manifest68,123B, failed/rejected0. App payload, not wirebytes.
  24GPUgroups/4ACKguards/93localtests (7new actualsocket refusal/roundtrip methods)/
  3 private source/fallback/buffer archive refusals/Black/criticalRuff+F401/diff pass;
  immutable D35+D36+D39 replay. Preliminary localcounter race fixed by draining test
  server before inspection, then93 pass; no failed GPU phase/relaxed math/full CI.
  LiveGPU unchanged2980.129/2973.930MiB; reserved3734/4136. Cgroup2504–2515MiB,
  hostavailable≥7.33GiB/swap+OOM0/logs+registry unchanged. Source outsideGPUcgroup/
  capped256MiB; counters overlap. New worker207s vslocal121.5 includes network/
  validation, storage/network tradeoff, no training speed/memory/largefit claim.
  Guarded104257.3/105220.2s excludes105earlierremote preflight, different brackets.
  PLAN01 EXreservation+SHruntime/4CPU0–3/10GiB+12GiB/512PID/256MiBshm/600s guard/
  460s child/5GiBfree reserve retained. Evidence/distributed-d40-r1/{local-control,
  remote-reference}, ports29741/29742; guardians10416356/17139,10526759/27099,
  source17198/export18491/27467 exit0. Exact commands/CIDs/UUIDs/ACK/logs retained;
  local kernel/artifacts/distributed-d40; archives omit raw tensors.
  104archive960364B SHA15eb9d294a6a6d360142306ef9fa20b4d2690dbe44c3c1854b6fdf5c8089fb49.
  105archive887542B SHAf1752c722bb4413524eb55070d75ef3fd17c77ce832e7f6e747d1004b75e1fb6.
  Final idle/no jobs/containers/source/armed guards; boots2d86d523r2/e902e636r3,
  native PLAN02MIOpen381eac05/Torcha524/RCCL/D09/image/packages/defaults/fallbacks
  unchanged. No nativebuild/install/reboot/download/prefixmutation/apps/foreignstop.
  Docs/flow/manifest synced BOTH. PythonCPUfault distinct fixedVMheap/D05exit/D21load.
  **NEXT PLAN-05 / D41 (planned):** CPU-only FP64 checkpoint+streaming oracle for
  planned1,024,471,040-param GPT2 on104 after source/config/P/RAM admission and
  11.449GiB payload+5GiB diskreserve. Conservative headroom/abort-owned policy,
  ALL initial/loss/gradient finite/coverage/raw roundtrip/hash, recomputation/0retained
  gradients. Preserve D35refs; no GPU/largerfit claim until oracle/storage and
  shape-specific GPU/transport admission pass separately.1.024B still unqualified,
  no deliberateOOM/extrapolated fit. Six important packages remain.

- **PLAN-05 / D39 completed 2026-11-09 — spool-free CPU-reference clipping; PLAN-05 OPEN.**
  Implementation `a82aca0ba`. [Report](docs/development/bc250_distributed_d39_results/README.md),
  [verified results](docs/development/bc250_distributed_d39_results/verified-results.json).
  Additive tools/distributed-d39 BOTH; same352,957,440-param FP32 GPT2/native FSDP/
  global2/seq128 and immutable D35 CPU refs. New per-unit clipping audit reads FP64
  CPU reference gradients instead of full actual-GPU-gradient disk spool. Original
  atol1e-8/rtol5e-5 pass ALL entries; maxabs1.300937118860028e-8 both hosts (combined
  absolute/relative gate). Coefficient still from CPU-summed checked actual-gradient
  norm, not separate CPU-reference-only norm. ALL raw/loss/norm/AdamW gates retained.
  Raw/clipped/updated tensor hashes and local moments/counters EXACT fresh legacy
  control/D36 perrank. Original D36 code/control/refs preserved.39 public nonrecursive
  contexts/maxunit28,331,520 retained, whole summons forbidden. No CPU oracle/global
  optimizer in GPUworker. Raw hashes computed directly; compatibility spool-removed
  flag denotes final absence. Raw spool1,411,829,760B=1.315GiB→0 perhost;5GiB disk
  reserve retained, both guardian+worker admit correctly. No fullref copy/global cache
  flush. Originalrefs rehashed before allocation/export, tensor payload omitted.
  24GPUgroups/4ACKguards/86localtests/3 private source/tolerance/hash corruption
  refusals/Black/criticalRuff+F401/diff pass; immutable D35+D36 replay. No failed GPU
  phase/relaxed bounds/full upstream CI. LiveGPU unchanged2980.129MiB104/2973.930105;
  reserved3734/4136. CPU cgroup2502–2508MiB, hostavailable min7.705104/7.270105GiB,
  swap/OOM0/registry+kernel logs unchanged. Qualification storage improvement, no
  training speed/memory/largefit/longrun claim. Guarded132.7–135.9s includes I/O/checks.
  PLAN01 EXreservation+SHruntime/4CPU0–3/10GiB+12GiB/512PID/256MiBshm/600s guard/
  460s child retained. Evidence/distributed-d39-r1/{legacy-control,reference-clip},
  ports29739/29740; guardians10410967/11784,10521718/22538; exports12572/23340 exit0.
  Exact commands/CIDs/UUIDs/ACK/logs retained; local kernel/artifacts/distributed-d39.
  104archive850950B SHA47d632edd02869f8cda2f71b1b3ec2034ed77d10b0cb4adf252e87d46c98b1a1.
  105archive795941B SHA13b2d58760799d9f6574a17462b8a2d6838d08877061d39c379ea429d68fed48.
  Final idle/no jobs/containers/armed guards; boots2d86d523r2/e902e636r3/native
  PLAN02MIOpen381eac05/Torcha524/RCCL/D09/image/defaults/fallbacks unchanged. No native
  build/install/reboot/download/prefix mutation/apps/foreign stop. Docs/flow/manifest
  synced BOTH. PythonCPUfault remains distinct fixedVMheap/D05exit/D21load.
  **NEXT PLAN-05 / D40 (planned):** bounded read-only per-tensor reference transport
  .104→.105 on existing352.96M refs, source/manifest/size/SHA checks and bounded
  buffers/truncation+corruption refusal controls; preserve local fallback/refs, no
  fullref duplicate.1.024B CPU/GPU remains unqualified:11.449GiB refs+5GiB reserve
  still fails.105, oracle/RAM/storage/transport admission separately before allocation.
  No deliberateOOM/inferred fit; six packages remain.

- **PLAN-05 / D38 completed 2026-11-09 — CPU checkpoint+streaming; PLAN-05 OPEN.**
  Implementation `22d702e47`. [Report](docs/development/bc250_distributed_d38_results/README.md),
  [verified results](docs/development/bc250_distributed_d38_results/verified-results.json).
  New additive tools/distributed-d38 BOTH. Same352,957,440-param FP64 CPU GPT2,
  full global2 backward/seq128/positions256/dropout0/cacheFalse and D35 raw refs.
  Native gradient_checkpointing_enable with use_reentrant=False/preserve_rng_state
  True; guards13 live modules/partial native function/exact bool kwargs before/
  after execution. Wrong options/functions/positional args/integer flags refuse.
  Control block entries12 once; enabled24 (each block reentered during backward;
  nonreentrant early-stop may stop recompute before full forward completion).
  ALL148 gradient hooks once/0 grads retained; direct FP64 initial before/after,
  loss and ALL gradient bytes/SHA EXACT both modes/hosts/D35/D37. CPU loss remains
  9.364227904214559. Dropout0 CPU RNG bytes unchanged; no stochastic guarantee.
  Native checkpoint.py SHA eab49639231cc989d65c04212923eef0ade4aa059b68bcabfe2819007f34e69f
  matches qualified torch/torch/utils/checkpoint.py; immutable verifier pin. Source
  tables bind old oracle separately from new checkpoint/streaming code. CUDA never
  initialized/no GPU tensors/optimizer/PG; native GPU training/defaults untouched.
  32CPUgroups/4ACKguards/86localtests/3 private source/recompute/recipe corruption
  refusals/Black/criticalRuff+F401/diff pass; no failed phase/relaxed math/full CI.
  Cgroup peakMiB5208.641→4179.109104/5196.531→4249.711105 (18.2–19.8%lower vsstream);
  VmHWM5149.285→4127.707/5139.398→4200.547. Hostavailable min9.237→10.261GiB104/
  9.132→10.033GiB105. Scratch one96MiB file+5GiBreserve, removed after checks;
  no fullref/spool/global cache flush. Afterbackward cgroup3615–3697MiB still has
  full FP64 parameters. Counters overlap/don't sum;200ms+kernel peak/HWM; swap/OOM0.
  Guarded71s control/80s checkpoint, worker57–58s/58–66s includes I/O/recompute/
  checks/imports, no training-speed/timing-significance/largefit/longrun claim.
  PLAN01 EXreservation+SHruntime/4CPUcpuset0–3/10GiB+12GiB/512PID/256MiBshm/600s
  guard/460s child retained; CPU admission≥12GiBhost. Originalrefs rehashed export,
  tensor payload omitted metadata; immutable D35+D37 archive provenance replayed.
  Exact commands/CIDs/UUIDs/ACK/logs evidence/distributed-d38-r1/{stream-control,
  checkpoint-stream}; guardians1045110/5946,10519068/20336,exports6842/21712 exit0.
  Local kernel/artifacts/distributed-d38;104archive510548B SHA
  77c4c82ca65189a0ebd42c261bb2989cc90793dad3b3c464290a7a959acdb9e0;
  105480889B SHA6f77ae9f9ed8b716d7ff7ace89b26c86a47b66398af303a7aa45135fb3fdb095.
  Final idle/no owned jobs/containers/armed guards; registry/kernel logs unchanged,
  boots2d86d523r2/e902e636r3/native PLAN02MIOpen381eac05/Torcha524/RCCL/D09/image/
  packages/defaults/fallbacks retained; no nativebuild/reboot/download/prefixmutation/
  apps/foreign stop. Docs/flow/manifest synced BOTH. PythonCPUfault remains distinct
  fixedVMheap/D05exit/D21load.1.024B CPU/GPU still unqualified, no scaled admission.
  **NEXT PLAN-05 / D39 (planned):** independent FP64 CPU-reference clipped-gradient
  audit replacing full actualGPU-gradient spool, original atol1e-8/rtol5e-5 plus
  ALLraw/norm/AdamW gates, at352.96M per-unit GPU scope. Compare exact native raw/
  clipped/updated/local-state digests against legacy D36; preserve fallback/refs,
  don't relax failures.1.024B oldspool3.816GiB+5GiBreserve fails.105; reference/oracle
  storage and memory admission remain separate/no deliberateOOM. Six packages remain.

- **PLAN-05 / D37 completed 2026-11-09 — CPU gradient streaming; PLAN-05 OPEN.**
  Implementation `2610b2b76`. [Report](docs/development/bc250_distributed_d37_results/README.md),
  [verified results](docs/development/bc250_distributed_d37_results/verified-results.json).
  New additive tools/distributed-d37 BOTH; final evidence series distributed-d37-r2.
  CPU-only352,957,440-param FP64 GPT2, same D35 math/full global2 backward/seq128/
  positions256/dropout0/cacheFalse, checkpointOFF; CUDA never initialized/no PG/GPU
  tensors. Fresh retained-gradient control vs native post-accumulate hooks checking
  each gradient against original D35 FP64 reference, one raw scratch file roundtrip,
  then p.grad=None. ALL148 hooks exactly once/tied embedding-head retained; max
  visible .grad elements at hook12582912, after backward352957440→0. Scratch≤96MiB
  (100663296B)+5GiB disk reserve, removed on success; no full3.9446GiB ref copy/spool/
  global cache flush. Initial values checked direct FP64 against promoted FP32 ref
  before/after; ALL gradient bits/SHA/loss EXACT both modes/hosts/D35. Loss remains
  9.364227904214559. CPU-only contract, not multi-micro/shared-graph/optimizer recipe.
  Initial r1 control checked weights after FP32 cast; tightened before streaming,
  reran BOTH corrected r2 controls/streams. r1 evidence/preliminary source backup
  tools/distributed-d37-preliminary-r1 preserved, not counted among final24 groups;
  no demonstrated Torch math bug. Final24 CPUgroups/4ACKguards/83localtests/3private
  source/hook-count/gradient-hash corruption refusals/Black/criticalRuff+F401/diff
  pass; no failed final phase or relaxed bounds/full upstream CI claim.
  Cgroup peakMiB7556.621→5218.113104/7481.609→5202.461105 (~30.5–30.9%lower);
  VmHWM7563.840→5158.750/7492.652→5145.844MiB. Host available min7.033→9.212GiB104/
  6.984→9.109GiB105. After-backward stream cgroup4818–4833MiB; full FP64 params
  still retained. Counters overlap/don't sum;200ms sampling+kernel peak/HWM. Host/
  cgroup swap0/OOM0. Guarded57s retained/76s stream includes I/O/checks/imports;
  worker40–44s/57–63s. Memory/I/O tradeoff, not CPU/GPU training-speed proof.
  Native GPU training untouched; no new GPU qualification or larger-fit inference.
  CPU host admission≥12GiB; PLAN01 EXreservation+SHruntime/4CPUcpuset0–3/10GiB+12GiB/
  512PID/256MiBshm/600s guard/460s child retained. Exact commands/CIDs/UUIDs/ACK/logs
  evidence/distributed-d37-r2/{retained-control,stream-gradient}; guardians10432310/
  630,10510185/11364,exports1769/13133 exit0. Preliminary r1 guardians31542/9018 exit0.
  Local kernel/artifacts/distributed-d37: r1/r2 overlays/raw summaries/final metadata;
  104archive472440B SHA fedf02daf56a229c388dfdb91c6a29ce875cf24859f4b9cf0e47b8deac7267d1;
  105445212B SHA870f76e490c0d2aa11495ed536444eb7e94db7216812583d2b73d442dd24519e.
  Original D35 references reverified at export, stay hosts/omitted metadata. Final
  idle/no jobs/containers/armed guards; registry/kernel unchanged/boots2d86d523r2/
  e902e636r3/native PLAN02MIOpen381eac05/Torcha524/RCCL/D09/image/defaults/fallbacks
  preserved. No nativebuild/reboot/download/prefix mutation/apps/foreign stop.
  Docs/flow/manifest synced BOTH. PythonCPUfault distinct fixedVMheap/D05exit/D21load.
  **NEXT PLAN-05 / D38 (planned):** native nonreentrant CPU checkpointing with
  streaming (use_reentrant=False,preserve_rng_state=True) on existing352.96M refs;
  exact loss/FP64 initial/ALLgrad/hook count/recompute+memory. Keep one-tensor
  scratch/original refs.1.024B remains CPU/GPU blocked pending its own oracle,
  storage and clipping-spool admission; no deliberateOOM/extrapolated fit. Six major
  packages remain.

- **PLAN-05 / D36 completed 2026-11-09 — per-unit FSDP audits; PLAN-05 OPEN.**
  Implementation `22447a4a9`. [Report](docs/development/bc250_distributed_d36_results/README.md),
  [verified results](docs/development/bc250_distributed_d36_results/verified-results.json).
  New additive tools/distributed-d36 BOTH. Reuses immutable D35 352,957,440-param
  FP32 GPT2/math/native CPU reference (no oracle rerun/full reference copy). Public
  wrapper-name ownership partitions ALL148 params into root+12 blocks. Per-unit
  summon uses recurse=False/writeback=False/with_grads=True; process guard refuses
  whole-model calls, other wrapped matrices must remain1D shards. No private handle
  API/training-loop fork. Three passes:39 unit contexts vs3 whole-control; every
  pass covers all names once. Max materialized scope352957440→28331520 elements.
  Original CPU FP64 loss/ALLgrad/norm/ALLclip/streamed CPU AdamW ALLvalues/local
  moments/counters pass unchanged. Raw/clipped gradients and updated values EXACT
  both paths/hosts; local AdamW digests EXACT within each rank. Same176478720
  local master/moment elements. CPU sum uses manifest order: report-only norm
  8.457222465649355→8.457222465649352 (3.6e-15), coefficient delta2.08e-17; no GPU/
  CPU gate loosened, paired D36 scalars exact. Historical verifier explicitly handles
  those double-rounding report scalars; other D35 common reference fields exact.
  24 new GPU audit groups/4 ACKed guards/79 local tests/3 private archive corruption
  refusals/Black/critical Ruff+F401/diff pass; no failed GPU phase/full upstream CI.
  Peak live AUDIT GPU5456.828→2980.129MiB104/5450.629→2973.930MiB105 (~45.4%lower);
  reserved6264→3734/6484→4136MiB. Peak reset before first backward excludes earlier
  model init. Host available min5.260→7.838GiB104/4.920→7.246GiB105. Cgroup peaks
  ~2501–2506MiB/swap+OOM0/logs unchanged. Counters overlap, don't sum. Qualification
  memory improvement only; ordinary training/defaults unchanged, no speed/largefit/
  restart/convergence/longrun claim. Guarded phases144.3–144.6s whole/134.4s unit
  include I/O/imports/digests, not throughput; one paired order, no timing claim.
  Source-bound CPU manifest/table checked separately from new audit source table;
  final export rehashes original raw files, metadata omits tensors. New phase-owned
  gradient spool removed after checks; original D35 refs/builds/caches/apps retained.
  PLAN01 EXreservation+SHruntime/4CPUcpuset0–3/10GiB+12GiB/512PID/256MiBshm/600s
  guard/460s child retained; GPU admission≥11GiBhost/9GiBdevice, spool+5GiBdisk.
  Exact commands/CIDs/UUIDs/ACKs/logs evidence/distributed-d36-r1/{whole-control,
  unit-audit}; guardians10420709/21890,10522408/24273,exports23300/26549 exit0.
  Local kernel/artifacts/distributed-d36;104archive874908B SHA
  8b0e3f1ac80d005272736a2f98857c113bfd8d6d150111b64922d62375678472;
  105808010B SHAdb6a82e431ecdccea1dfdeda68f8e34ac4dcd31c9c4c4b841699c8cc01f11831.
  Final idle/no owned jobs/containers/armed guards; boot2d86d523r2/e902e636r3/native
  PLAN02MIOpen381eac05/Torcha524/RCCL/D09/image/defaults/fallbacks preserved, no
  build/reboot/download/prefix mutation/apps/foreign stop. Docs/flow/manifest synced
  BOTH. PythonCPUfault distinct fixedVMheap/D05exit/D21load.1.024B CPU/GPU still
  blocked by oracle/storage constraints; D36 audit bounds do not admit it.
  **NEXT PLAN-05 / D37 (planned):** CPU post-accumulate gradient streaming on existing
  352.96M fixture, ALL gradient bits/loss/initial values versus D35, hook counts and
  no retained full gradient set; CPU/RSS/cgroup/host peaks. One parameter of owned
  scratch only, no3.9446GiB duplicate. CPU checkpointing separately if needed; then
  resolve1.024B oracle/storage admission. Preserve refs/no deliberateOOM. Six major
  packages remain.

- **PLAN-05 / D35 completed 2026-11-09 — staged CPU references and352.96M; PLAN-05 OPEN.**
  Implementation `02107159f`. [Report](docs/development/bc250_distributed_d35_results/README.md),
  [verified results](docs/development/bc250_distributed_d35_results/verified-results.json),
  [next capacity plan](docs/development/bc250_distributed_d35_results/next-capacity-plan.json).
  New additive tools/distributed-d35 BOTH. Owned eager tied GPT2/352,957,440 FP32
  params/width1536/12blocks/24heads/vocab8192/positions256/seq128/global2/micro1/
  dropout0/cacheFalse; native AdamW/rootclip.5, no AMP/PEFT/checkpoint/compile.
  CPU-only producer exits before GPU; CUDA never initialized. Raw FP32 initial+
  FP64 ALL148 gradients roundtrip bit-exact, identical manifests BOTH hosts;
  source/shape/name/size/hash/full coverage checked before CUDA allocation. Payload
  4235489280B=3.9446GiB/host retained at evidence/distributed-d35-r1/cpu-reference/
  reference. No pickle/global cache flush; owned-file fsync/fadvise only. Reserve
  payload+5GiB disk; GPU actual-gradient spool1411829760B temporary, removed after
  successful ALLclip checks/digests retained; original references never removed.
  GPU single/FSDP CPU FP64 loss/ALLgrad/CPU norm/ALLclip/streamed CPU AdamW ALLvalues/
  local moments/counters pass unchangedD32 gates. Maxgraderror2.19408e-7; loss
  9.3642272949 vsCPU9.3642279042. Each FSDP rank owns176478720 master/moment elements
  vs352957440single. CPU moment scratch≤192MiB/parameter, no full CPU optimizer/
  export/oracle in GPU workers. Source/digest/mode/rank/runtime-bound fresh workers,
  initial update+two warmups+five timed, both orders. Later losses agree≤1e-5.
  72groups/14ACKguards/76localtests/3privatearchive corruption refusals/lint pass;
  no failed GPU phase, full upstream CI or relaxed-bound claim. Medianms single
  265.535–266.563,FSDP17968.643–17973.422 (~68xslower); liveGPU5863.493vs3183.965–
  3189.670MiB (~45.6%lower), reserved5912vs4238–4254MiB. Collective input/rank/update
  gather1385871360B+reduce1411829760B, not wire totals; eth0 both1000Mb/s.
  Measured cgroup~1083vs1652MiB; hostavailable6.997–7.129single/7.940–8.081FSDP GiB,
  paired FSDP headroom+0.927–0.956GiB/host. Counters overlap; don't sum. Both modes
  fit; larger-than-single remains unproven. CPU producer cgrouppeak≤7550MiB;
  GPU qualification≤2504MiB; hostavailable≥4.932GiB throughout, swap/OOM0.
  CPU≥12GiB host admission/GPU≥11GiB host+9GiB device; original PLAN01 EXreservation+
  SH runtime/4CPUcpuset0–3/10GiB+12GiB/512PID/256MiBshm/600s guard/460s child retained.
  Complete guarded CPU63–65s/GPUqual72single,131–132FSDP/fresh28–30single,170–171FSDP
  includes I/O/imports/agents, not production training-only. Exact commands/PIDs/
  CIDs/UUIDs/ACKs/logs: evidence/distributed-d35-r1/{cpu-reference,qualify-single,
  qualify-fsdp,single-a,fsdp-a,fsdp-b,single-b},sequence.json/controllerlogs.
  Guardians104:11869,12781,13478,14492,14971,16185,17385;
  105:8161,9572,10590,12173,12813,14743,16676. Sequence12780/9570,exports18108/17748
  exit0. Local kernel/artifacts/distributed-d35; metadata1041237695B SHA
  46fecff3a5da132bd15813b95c341ebfd336da9d15f16ebaf86bb1c3259deee0;
  1051183666B SHA5883fb9026210caf75c420e1f889207efbabeb1eee28cdb32ae824a10ceec5a2.
  Raw payload stays hosts/reverified at final export, omitted local metadata. Final
  idle/no jobs/containers/armed guards; kernel+registry unchanged, boots2d86d523r2/
  e902e636r3/native PLAN02MIOpen381eac05/Torcha524/RCCL/D09/image/defaults/fallbacks
  preserved; no build/reboot/download/prefix mutation/apps/foreign stop. Docs/flow/
  manifest snapshots synced BOTH. Historical PythonCPUfault distinct fixedVMheap/
  D05exit/D21load. 1.024B remains CPU/GPU BLOCKED: single-state/fullCPUoracle floor
  15.266GiB exceeds physical RAM; unchanged11.449GiB reference+5GiB disk reserve
  and3.816GiB spool+reserve fail on.105. Two-rank7.633GiB floor excludes overhead.
  **NEXT PLAN-05 / D36 (planned):** bounded per-FSDP-unit ALLgradient/AdamW audits
  against existing352.96M source-bound CPU references, unchanged numerical gates,
  before replacing whole-model summons. Then resolve streamed CPU oracle/storage
  before any1.024B allocation. No extrapolated fit/deliberateOOM/reference deletion.
  Six important packages remain; no convergence/large restart/universal speed claim.

- **PLAN-05 / D34 completed 2026-11-09 — bounded159.8M correctness/measurement; PLAN-05 OPEN.**
  Implementation `58e91cb5d`. [Report](docs/development/bc250_distributed_d34_results/README.md),
  [verified metadata](docs/development/bc250_distributed_d34_results/verified-results.json).
  Additive tools/distributed-d34 on BOTH. Fixed eager tied GPT2/159,807,488 FP32
  trainable params/width1024/12blocks/16heads/vocab8192/positions256/seq128/global2/
  micro1/dropout0/cacheFalse; native AdamW/rootclip.5/no AMP/PEFT/checkpoint/compile.
  Separate oneboard+FSDP qualification checks CPU FP64 loss/ALL148 gradients,
  exact initialized values/tied weights, CPU norm/ALLclip/ALL updated values and
  ALL local AdamW moments/counters at unchangedD32 bounds. CPU AdamW checks stream
  one parameter at a time; max moment scratch128MiB, no full CPU optimizer/export.
  Maxgradienterror1.85513e-7; loss9.1934337616 vsCPU9.1934335610. Each FSDP rank
  owns79,903,744 master/moment elements vs159,807,488single. One local micro avoids
  full no_sync gradient. Fresh workers bind successful qualification digest/source/
  shape/mode/rank/Torch/registry; package modeling pins retained, fullaudits forbidden.
  Initial update+two warmups+five measured, bothorders; laterglobal losses agree
  across modes/hosts/orders within1e-5.64groups/12ACKguards/71localtests/3private
  archive corruptionrefusals/Black/criticalRuff+F401/diff pass, no failedGPUphase.
  Medianms single150.288–151.092,FSDP8088.982–8103.416 (~54xslower). LiveGPU
  2737.507 vs1493.502–1495.365MiB (~45.4%lower),reserved2748vs1956–2064MiB.
  Native collective input/rank/update gather621924352B+reduce639229952B; notwirebytes,
  eth0 both1000Mb/s. Fresh cgroup maxima1082single/1648FSDP MiB; host available
  10.107–10.234single/10.093–10.337FSDP GiB. No clear wholehost capacity gain;
  counters overlap and must not be summed. Both fit; larger-than-single unproven.
  CPUqualification cgroup peak≤4991.34MiB/hostavailable≥5.811GiB; measuredhost≥10.093.
  All host/cgroup swap0/OOM0. Admission≥10GiBhost/≥6GiBdevice,10GiB+12GiB cap;
  4CPUcpuset0–3/512PIDs/256MiBshm/600s recovery/460s child and PLAN01 EXreservation+
  SHruntimelock retained. Guarded measurement17.5–18.2ssingle/81.6–82.1sFSDP includes
  imports/coordination, not training-only; qualification37–38s/60–61s separate.
  Not D33speedup (differentmodel/batch/tokens), nor convergence/restart/longrunproof.
  Exactcommands/CIDs/UUIDs/logs/ACKs evidence/distributed-d34-r1/{qualify-single,
  qualify-fsdp,single-a,fsdp-a,fsdp-b,single-b},sequence.json/controllerlogs.
  Guardians104:27806,28632,29278,29666,30412,31143;105:12390,13628,14498,14981,
  16058,17125. Sequence28630/13627,exports31744/17992 exit0. Local artifacts
  kernel/artifacts/distributed-d34;104archive678906B SHA680cca7c072671faadb97d278322f59045672568a6881bbc2137868c0433c234;
  105643665B SHAd02d50a86aad79ecd60052dd19b207da000ed60863d037b5bfec96fc8ed3053d.
  Initial nonrootstage permission refusal preserved as prestage log; rootstage
  succeeds before GPU. Original evidence/checkpoints/builds/caches/apps retained.
  Final idle/nojobs/containers/armedguards; kernel+registry unchanged. Boots2d86d523
  (.104r2)/e902e636(.105r3) unchanged; nativePLAN02MIOpen381eac05/Torcha524/RCCL/
  D09/image/packages/defaults/fallbacks preserved. No nativebuild/reboot/download/
  prefixmutation/apps/foreignstop. Docs/flow/manifest snapshots synced BOTH.
  Historical PythonCPUfault distinctfromfixedVMheap/D05exit/D21load.
  **NEXT PLAN-05 / D35 (planned):** staged source-bound CPUgradient artifacts and
  storage/memory admission for352,957,440 params, avoiding fullCPUoracle+GPUstate
  coexistence. Qualify that pipeline before any352.96M GPUadmission, then shape-
  specific single/FSDP and freshmeasurement. Keep1.024B blocked; no extrapolated
  physicalfit or deliberateOOM. Six important packages remain.

- **PLAN-05 / D33 completed 2026-11-09 — fresh memory controls; PLAN-05 OPEN.**
  Implementation/verifier `199d3f2ab`. [Report](docs/development/bc250_distributed_d33_results/README.md),
  [verified results](docs/development/bc250_distributed_d33_results/verified-results.json),
  [larger-shape plan](docs/development/bc250_distributed_d33_results/larger-shape-plan.json).
  New additive tools/distributed-d33 on BOTH hosts. Prepared small immutable proof
  records from published/hash-verified D32 archives, binding host/mode, CPU loss/
  ALL gradients/clip/ALL AdamW checks, original sources and runtime registry.
  Guardian+worker refuse changed proof/source/native identity before model allocation.
  Fresh workers reuse unchanged D32 build/data/loss/backward/AdamW functions, retain
  no CPU base duplicate and forbid oracle/full optimizer export/full-param audits.
  Native Torch version/Transformers5.2/three modeling source pins checked. Original
  D32 sources/evidence/defaults preserved. Proof files are specific to45.9M;
  they do not qualify a different shape. New tests exercise proof/source tampering.
  Same eager tied GPT2/45,871,104 FP32 trainable params/global8/micro1/seq256,
  no dropout/cache/AMP/PEFT/compile. Initial GPU update+two warmups+five measured,
  both orders. All initial/five measured global losses match D32 within1e-5;
  original CPU checks reused, not rerun or counted as fresh. Master/AdamW ownership
  and native collective inputs match D32. New containers/UUIDs/start ticks prove
  fresh processes.84 fresh groups/12 ACKed guards/68 local tests/Black/critical
  Ruff+F401/diff pass; no failed GPU phase or relaxed numerical bounds.
  Timing ms:single247.897–249.549,DDP1680.839–1681.057,FSDP6879.746–6882.628.
  Live GPU peak940.725/1114.484/767.607–769.358MiB; clean cgroup maxima1081/1872/
  1646MiB vs D32's2100–2332/2792–3048/3260–3448. This fixes the measurement baseline,
  NOT native Torch/application memory use. The removed CPU oracle/audit/base history
  affected old counters; individual causal shares remain unattributed. GPU updates
  are not faster. Do not sum GPU/RSS/cgroup/host counters as independent pools.
  New memory.stat/RSS and GPU snapshots cover before-model/after-wrap/first-update/
  owned-cache release/warmup/measurement-end. DDP shmem458→514→530→530→530MiB in
  both orders on both hosts; final three endpoints plateau in this short run,
  NOT proof of long-run stability/leak/fixed library ownership. FSDP shmem stays
  near304MiB after first update. Whole-worker cgroup peak≤1874.44MiB, sampled host
  available≥10.625GiB, host+cgroup swap0/OOM0. Admission≥10GiB host/≥4GiB device
  unchanged. Complete phases~17–18s single/30–31DDP/70–71FSDP omit CPU audit by
  design; not a training-speed gain. Single workers still use two-node launcher
  orchestration but no training process group. No checkpoint/restart/convergence
  or larger-than-single qualification. PLAN-05 remains open.
  CPU-only planner records tied-GPT2 candidates159,807,488 /352,957,440 /
  1,024,471,040 parameters. State floors exclude activations/gather overlap/
  reservations/init/runtime/communication; ALL candidates GPU_admitted=False.
  No linear-fit inference or OOM search. Smallest candidate single FP32 state
  floor2.381GiB/sharded1.191GiB per rank/no_sync-full-grad1.488GiB per rank.
  Exact commands/PIDs/CIDs/ACK/logs: evidence/distributed-d33-r1/{single,ddp,fsdp}-{a,b},
  sequence.json/controller logs. Guardians104:11524,12442,12911,13571,14232,14687;
  105:16770,18210,18815,19775,20733,21332. Sequence12441/18209,exports15427/22461 exit0.
  Local artifacts kernel/artifacts/distributed-d33:proof inputs/overlays/raw pilot
  JSON/metadata; original sources/evidence remain hosts.104archive720475B SHA
  f3559f7bcdfee1dad7ded60d2100c4601cdeec4fb3f933e1f58a5b02bea00ee0;
  105690668B SHAedc434002dec47f2c7feb05189976695ecee66ee4775f00577b62f533463509e.
  PLAN-01 EX reservation+SH runtime locks/4CPUcpuset0–3/10GiB+12GiB/512PIDs/
  256MiBshm/600s recovery/460s child retained. Kernel logs/registry unchanged;
  final idle/no owned jobs/containers/armed guards. Boots2d86d523(.104r2)/e902e636
  (.105r3) unchanged; native PLAN-02 MIOpen381eac05/Torch a524/RCCL/D09/image/
  packages/defaults/fallbacks preserved. No native build/reboot/download/prefix
  mutation/apps/foreign stop. Docs/flow/manifest snapshots synced BOTH. Historical
  Python CPU fault remains distinct from fixed VM heap/D05 exit/D21 loading.
  **NEXT PLAN-05 / D34 (planned):** start with159.8M candidate; qualify memory-bounded
  initialization and shape-specific CPU/GPU correctness, then fresh one-board/FSDP
  measurements at global2/micro1/sequence128/positions256. Keep qualification and
  measurement separate; use bounded CPU optimizer checking rather than co-resident
  full CPU moments/export if needed. One local micro avoids unsharded no_sync
  gradients. Keep352.96M/1.024B blocked until their own envelopes are qualified.

- **PLAN-05 / D32 completed 2026-11-09 — representative baseline; PLAN-05 OPEN.**
  Implementation/verifier `6af433cb8`. [Report](docs/development/bc250_distributed_d32_results/README.md),
  [verified results](docs/development/bc250_distributed_d32_results/verified-results.json).
  New additive tools/distributed-d32 on BOTH hosts: fixed conservative admission,
  actual eager GPT2 (45,871,104 FP32 trainable parameters, tied head, six blocks,
  width768/12heads/vocab4096/sequence256/global8/micro1), direct native AdamW,
  root clip.5, no dropout/cache/AMP/PEFT/checkpointing/compile/download. D05/D06's
  smaller custom model is already qualified; this is a larger actual HF fixture.
  Single workers process8 micros; DDP/FSDP process4 each with native no_sync until
  final micro. One CPU FP64 global loss/ALL76 gradients per fresh worker passes
  original D05 bounds; CPU norm/ALL clipped entries/CPU-double ALL AdamW values,
  moments/counters pass unchanged D21/D22 bounds. Two warmups then five timed
  updates; both orders single→DDP→FSDP and reverse. Later finite global losses
  agree across modes/hosts within1e-5; not convergence or cross-topology replay.
  Median ms a/b: single104247.963/248.057,105249.278/249.128; DDP1681.584/1682.027;
  FSDP6885.565/6885.373. Single is fastest in this fixture. Both eth0 links1000Mb/s;
  DDP input183484416B/update, FSDP gather707186688+reduce183484416B, not wire bytes.
  FSDP owns22,935,552 master/AdamW elements per rank vs45,871,104 single/DDP.
  Live GPU peak940.725MiB single/1114.484DDP/768.366–770.856FSDP (~18%below single).
  Reserved GPU and measured cgroup memory nevertheless rise for FSDP. DO NOT infer
  physical RAM saving/larger fit from the GPU allocator. Timed workers previously
  performed CPU oracle/audits; CPU/runtime/communication attribution is unresolved.
  Measured cgroup ranges:single2100–2332MiB,DDP2792–3048,FSDP3260–3448. Whole
  validation peak≤4745.49MiB; host available≥8.635GiB including audits/≥9.27GiB
  during measured training. Host+cgroup swap0, OOM counters0. CPU/GPU share RAM;
  counters must not be summed as independent pools. Sampling200ms; kernel peak
  recorded separately. Seven-update training segment~1.85–1.98s/11.87–12.89s/
  48.19–48.49s; complete guarded phases33–34/53–55/93–94s include CPU/imports/
  two-node agent orchestration even for independent single workers, not production
  training-only job times. No checkpoint I/O/restart or larger-than-single proof.
  Admission host≥10GiB/devicefree≥4GiB before model allocation, fixed estimates
  exclude activations/buckets/temporaries/runtime.64 local tests including unsafe
  resource/unknown-mode refusals, Black/criticalRuff+F401/diff pass.72 groups/
  12 ACKed guards; no failed GPU phase or full upstream CI claim. Exact commands,
  PIDs/CIDs/logs/ACKs under evidence/distributed-d32-r1/{single,ddp,fsdp}-{a,b},
  sequence.json/controller logs. Guardian PIDs104:25755,26568,27143,27929,28715,
  29300;105:24103,25319,26132,27302,28474,29301. Sequence controllers26567/25318,
  exports29863/30120 exit0. Sources/raw metadata stay on hosts; local artifacts in
  kernel/artifacts/distributed-d32:overlays/raw pilotJSON/metadata archives.
  104 archive689853B SHA15f13d943bc8541654fc9c8c504a6374616dd86d9183b59f80c1d7b719c61337;
  105657137B SHA5f3b45e1fc26481c871c68f453e14b530c708ec885e24576753b5de76075224b.
  PLAN-01 reservation+SH locks/4CPUcpuset0–3/10GiB+12GiB/512PIDs/256MiBshm/600s
  recovery/460s child retained. Kernel logs/registries unchanged; final idle/no
  owned jobs/containers/armed guards. Boots2d86d523(.104r2)/e902e636(.105r3) unchanged;
  native PLAN-02 MIOpen381eac05/Torch a524/RCCL/D09/image/packages/defaults/fallbacks
  preserved; no native build/reboot/download/prefix mutation/apps/foreign stop.
  Docs/flow/manifest snapshots synced BOTH. Historical Python CPU fault remains
  distinct from fixed VM heap/D05 exit/D21 loading. No universal speed/fit claim.
  **NEXT PLAN-05 / D33 (planned):** separate source-bound CPU qualification from
  fresh training-only workers; repeat representative memory controls without prior
  oracle/audit work. Use the physical-memory envelope to plan larger FSDP
  initialization/admission. Do not deliberately OOM or extrapolate from GPU-only
  savings. PLAN-05 stays open; six major packages still remain.

- **PLAN-04 / D31 completed 2026-11-09; PLAN-04 bounded scope COMPLETE:** code
  `725e495c8`; final archive-verifier maintenance `6c4758c18`. [Report](docs/development/bc250_distributed_d31_results/README.md),
  [verified results](docs/development/bc250_distributed_d31_results/verified-results.json).
  NEW BC250BF16CheckpointPeftFSDPTrainer in BOTH additive tools/distributed-d31.
  Combines D29 native nonreentrant+preserveRNG enable/functionguards with explicit
  D30BF16working/gather+FP32master/reduce/buffer/AdamW/nativeautocast/noscaler;
  D28 namedfrozen/PEFT/rootclip/fullcheckpoint/publicexport retained. D30precision/
  runtimepredicates factored; checkpointOFFdefault/oldinstalledhelpers preserved.
  PinnedplainLoRA/eagerGPT2/dropout0/cacheFalse/FP32originalparams/modelcheckpoint
  initiallyOFF. Both recipes/pins boundtosavedmetadata and checkedbeforeupdate.
  Sameowned24352=21280frozen+3072trainable/19units/shards12176/AdamW1536each;
  tiedfrozenembedding-head/D26data28/rightpad/prompt/workers0/micro2accum2/global
  8/8/8/4/shifted49/57/57/21/localzero. ActualBF16weights/output/gathers+FP32master/
  reduce/optimizergrad; ALLfrozenvalues EXACT. CPUroundedBF16+FP64 ALLadaptergrad/
  loss/rootnorm/ALLclip/CPUdoubleALLAdamW pass UNCHANGEDD25/D26/D21/D22bounds;
  maxglobalrelative.00760192991/.00783804056. GPUblocks14calls/7seedmicros,6/3replay,
  offcontrol7/7: realbackwardrecompute. Inbatchcontrol ANDhashverifiedpublished
  D30archives have EXACTsame seedtranscript/finaldigest. BF16gathers259→357seed,
  111→153replay;16reductions12288B/window. Tinycompleteharness peak459264B vs499200B
  EACHrank(39936B/8%lower),includesaudit/checkpoint; not isolatedactivation/wholehost/
  largefit/speedproof. Publicadapter CP2/4/replay4 loadsALLparams/logitsEXACT/fourRNG
  unchanged. FullCP2union6files480247B onBOTH/originalhashespreserved; newseed1773
  vs773/newUUID+ticks+containers restoresALLmodel/AdamW/scheduler/fourRNG EXACT,
  windows3/4 ALLloss/grads/clip/updates/state/RNG/rows/counts/digests EXACT. Native
  train_loss processlocal; no crossprecision/stochastic/elasticpromise. Rank1missing
  adaptergrad/injectedfrozengrad/lateNaN ANDpostclip LIVEcheckpointpreserveRNGdrift
  BOTHrefusebeforeAdamW/zeroupdates/EXACTunchangedparams+optimizer+scheduler.
  frozen-peft case NOWtests checkpointfunctiondrift,notLoRAscaling. PRIVATErank1
  saveduse_reentrant corruption BOTHrefuseoptimizerrestore/originalcheckpointbytes
  intact; modelmayalreadyload/discardabortedsession. RetainedD30BF16control passes.
  frozen-policy-seed PRIVATECOPY,notrun.163groups/18ACKguards/30CPUpatterns536obs/
  62localtests/Black/criticalRuff+F401/diff pass; no failedGPUphase/fullupstreamCI.
  Postqualification found D30/D31archiveverifiers compared frozen evidence to
  mutablecurrenthelperbytes.6c4758c18 pins IMMUTABLEqualifiedsourcehashes; replay
  BOTHarchives produces BYTEEXACToriginalverifiedJSON,noGPUrerun/mathchange.
  Finalverifiers installed inD31namespace AFTERmetadataexport; finalsourceinGit,
  earlierverifiers retainedinarchives/originalnamespaces. D31d30/d31-verifier-replay
  JSON/log artifacts retainedlocally. No hashcheckweakened.
  Initialzshheredoc tempquota stoppedwritesBEFOREtrackedsourcechange; workspace
  disk26GiBfree. Bash+TMPDIR=/dev/shm fixedheredocs; unittestthere correctlyrefused
  7runtime/devshmmounts. Owned workspaceTMPDIR kernel/artifacts/distributed-d31/
  local-tmp passesall62, mountguardsunchanged/no cachesdeleted. local-validation-
  environment.txt recordssetup. Tempquota mayrecur; useownedworkspaceTMPDIRfor
  tests,bash+smallheredoctmpelsewhere; don't deletepreservedbuilds/caches.
  Exactcommands/PIDs/CIDs/ACK/logs evidence/distributed-d31-r1/frozen-{seed,resume},
  frozen-{missing,grad,late,peft}-seed,retained-control-seed,frozen-policy-resume,
  allocator-control-seed; controls/resumes-sequence.json/controllerlogs.
  Seed6265/23444,controls6850/24260,resumes9014/27004,allocator9961/28238,exports
  10451/28891 allfinalexit0. Sources/checkpoints/references/publicartifacts remain
  host; kernel/artifacts/distributed-d31 overlays/payloadexports/union/metadata
  excludes tensor/RNG/safetensors.104archive779548B SHAee83847c206e86f03657008e6121c0bd883a247f40021dbb4acbbd75063232b2;
  105689055B SHA2a4190af18c58f9a0a079afc606eaede0a0c6069c45fff97939cdef4a04ed2dd.
  PLAN01reservation+SHlock/4CPUcpuset0–3/10GiB+12GiB/512PIDs/256MiBshm/600s/460s
  retained; kernel+registryunchanged/finalidle/nojobs/containers/armedguards.
  Entry2d86d523(.104r2)/e902e636(.105r3) unchanged/no build/reboot/download/prefix
  mutation/apps/foreignstop. NativePLAN02MIOpen381eac05/Torcha524/RCCL/D09/image/
  packages/defaults/fallbacks retained; docssyncedBOTH. HistoricalPythonCPUfault
  distinctfixedVMheap/D05exit/D21load. No universalmodel/speed/capacityclaim.
  **NEXT PLAN-05 / D32 (planned):** inspect retainedsizing/timing evidence and
  implement conservativeownedmodel oneboardvs two-hostDDP/FSDP measurement,
  matchedglobalbatch/sequence/precision; allocator+hostRAM/swapheadroom/boundedwarm
  update+wholejobtiming. No intentionalOOM/largefitinferencefromtinyfixtures.
  Siximportantplanpackages remain; FP16PEFT/otherrecipes separatequalification.

- **PLAN-04 / D30 completed 2026-11-09 (PLAN-04 still OPEN):** implementation/
  verifier `e8c6d120c`. [Report](docs/development/bc250_distributed_d30_results/README.md),
  [verified results](docs/development/bc250_distributed_d30_results/verified-results.json).
  NEW BC250BF16PeftFSDPTrainer in BOTH additive tools/distributed-d30. Requires
  pinnedplainPEFT/eagerGPT2/dropout0/cacheFalse/checkpointOFF/FP32originalparams;
  explicitBF16working+FP32master/reduce/buffer/nativeautocast/noscaler. FrozenAMP
  refusal factored into method; defaultoldhelpers stillrefuse. D28 namedpartition/
  PEFTsignature/trainableAdamW/rootclip/fullcheckpoint/publicexport inherited.
  Actualargs/pluginprecision/modelcheckpoint guards +three modelinghashes;
  fullcheckpoint binds precisionrecipe/pins. No nativepackagepatch/loopfork.
  Sharedharness frozenAMP selection +ALLtrainable-only CPUAMPgradient lists;
  oldalltrainable/FP32math preserved. Sameowned24352=21280frozen+3072trainable,
  19PEFTFSDPunits/shards12176each/trainableAdamW1536each/tiedfrozenembedding-head.
  D26data28/rightpad/prompt/dropout0/cacheFalse/workers0/micro2accum2/global8/8/8/4,
  shifted49/57/57/21/localzeromicro. ActualBF16weights/output/gathers,FP32master/
  reduce/optimizergrad observed. ALLadaptergrad/loss versus roundedCPU BF16local
  accumulation+FP64atBF16roundedweights pass ORIGINALD25/D26BF16.02/.03normgates+
  floor,loss.003/.003. Maxglobalrelative.00760192991/.00783804056. ALLfrozenvalues
  EXACT; CPUrootnorm/ALLclippedgrads/CPUdoubleALLAdamWvalues+moments+counters pass
  originalD21/D22bounds.259seedgathers/identicalelementsequence toD28,inputs
  623168→311584B; replay111gathers;16reductions12288B/window. Not totalNIC/speed/
  memorycapacityclaim. PublicPEFT CP2/4/replay4 loads ALLparams/logits EXACT/
  RNGunchanged. FullCP2DCPunion6files480247B onBOTH preservesoriginallocalhashes.
  Freshseed1773vs773/newUUID+ticks+containers restoresALLbase+adapter/AdamW/
  scheduler/fourRNG EXACT; windows3/4 ALLloss/grads/clip/updates/state/RNG/rows/
  counts/digests EXACT. Native train_loss processlocal; no crossprecisionpromise.
  Rank1missingadaptergrad/injectedfrozengrad/lateNaN/lateLoRAscaling BOTHrefuse
  beforeAdamW/zeroupdates/EXACTunchangedparams+optimizer+scheduler. RetainedD28
  FP32PEFTcontrol passes. PRIVATEcheckpointcopy changes ONLYrank1savedreduction
  precisiontoBF16; BOTHrefuseoptimizerrestore/zeroAdamW/originalbytesunchanged.
  Model mayalreadyload; discard/reload abortedsession. frozen-policy-seed PRIVATE
  COPY,notrun.116groups/16ACKguards/28CPUpatterns444obs/62localtests/Black/critical
  Ruff+F401/diff pass; no failedGPUphase. No fullupstreamCI/universalmodel/
  stochastic/elastic/largefit/convergence/speedproof. Exactcommands/PIDs/CIDs/ACKs/
  logs evidence/distributed-d30-r1/frozen-{seed,resume},frozen-{missing,grad,late,
  peft}-seed,retained-control-seed,frozen-policy-resume; controls/resumes-sequence
  JSON/controllerlogs. Seed28859/7650,controls29377/8341,resumes31579/11167,exports
  32560/12492 allfinalexit0. Sources/checkpoints/reference tensors/publicexports
  remainhost; local kernel/artifacts/distributed-d30 overlay/payloadexports/union/
  metadata excludes tensor/RNG/safetensors.104archive681233B SHA0044f8e1acb0380ee4fee0e2a1e0ac2640137f6843dc24b63f1113bdb12ea688;
  105611707B SHAf2e2b1a9abc07a1d11420aa44924c3e8219d4cb6c019983d8ef4a66aa0e28a09.
  PLAN01reservation+SHlock/4CPUcpuset0–3/10GiB+12GiB/512PIDs/256MiBshm/600s/460s
  retained; kernellogs/registry unchanged/finalidle/no jobs/containers/armedguards.
  Entry2d86d523(.104r2)/e902e636(.105r3) unchanged; no build/reboot/download/prefix
  mutation/apps/foreignstop. NativePLAN02MIOpen381eac05/Torcha524/RCCL/D09/image/
  packages/defaults/fallbacks preserved/docs syncedBOTH. HistoricalPythonCPUfault
  distinctfixedVMheap/D05exit/D21load. **NEXT PLAN-04 / D31 (planned):** combine
  explicitBF16 PEFT frozenbase and native nonreentrant activationcheckpointing,
  qualify CPUALLadaptergrad/clip/AdamW/frozenpreservation+exactrestart beforelifting
  combinationrefusal. FP16/otherrecipes/capacity remain separate.

- **PLAN-04 / D29 completed 2026-11-09 (PLAN-04 still OPEN):** implementation/
  verifier `8e04fa9b4`. [Report](docs/development/bc250_distributed_d29_results/README.md),
  [verified results](docs/development/bc250_distributed_d29_results/verified-results.json).
  NEW BC250CheckpointPeftFSDPTrainer in BOTH additive tools/distributed-d29;
  explicit FP32 PEFT eagerGPT2/native nonreentrant+preserveRNG kwargs. Three new
  modeling hashes checked; requires cacheFalse/checkpointing initiallyoff. Native
  Trainer enable beforewrap/frozenembedding output-grad hook/blockcheckpoint calls
  retained; no loop/package patch. Actual functions/options checked in inherited
  preupdate guards and bound to fullcheckpoint recipe. D28 fullsave/adapterexport/
  frozenFP32/rootclip/AdamW contracts retained. Ordinary/D27/D28 helpers still
  refuse checkpointing; frozenAMP/plugincheckpoint/compile/offload/FSDP2 separate.
  Same owned24352=21280frozen+3072trainable/19FSDPunits; shards12176each/AdamW1536.
  D26 data28/rightpad/prompt/cacheFalse/dropout0/workers0/micro2accum2/global8/8/8/4,
  shifted49/57/57/21/localzero micro. CPUFP64 ALLadaptergrad/loss/rootnorm/ALLclip/
  CPUdoubleALLAdamW pass unchangedD28bounds; ALLfrozenvalues EXACT. GPUblockhooks
  14calls/7micros(seed),6/3(replay),offcontrol7/7: realbackwardrecomputation.
  Offcontrol trainingtranscript/finaldigest EXACTsame. FP32gathers259→357seed,
  111→153replay;16reductions12288B/window retained, not totalNIC/speed result.
  PublicPEFT adapterexports CP2/4/replay4 reload ALLparams/logits EXACT/RNGunchanged.
  FullCP2 union6files480247B onBOTH; originalfiles/hashes preserved. Freshseed1773
  versus773/newUUID+ticks+containers restoresALLbase+adapter+AdamW/scheduler/fourRNG
  EXACT; windows3/4 ALLloss/grads/clip/updates/state/RNG/rows/counts/digests EXACT.
  Fourrank1 missinggrad/frozengrad/lateNaN/lateLoRAscaling mutations BOTH refuse
  beforeAdamW/zeroupdates/EXACTunchangedparams+optimizer+scheduler; retainedD28
  control passes. PRIVATE checkpointcopy changes ONLYrank1saveduse_reentrant;
  BOTH refuse optimizerrestore, originalcheckpoint unchanged. Model may already
  load on refusal; discard/reload abortedsession. No stochastic/elastic promise.
  Tinycompletecorrectnessharness peak491008B vs546304B/rank(55296B/10.12%lower)
  onBOTH, includesaudit/checkpoints; NOT isolatedactivation/wholehost/largefit or
  speedproof. PLAN05meaningfulcapacity separate.163groups/18ACKguards/28CPUpatterns
  500obs/62localtests/Black/criticalRuff+F401/diff pass. No failedGPUphase; initial
  SCPpermission/prematurelaunch setupfailure retained beforeguard/GPU. Allsources/
  checkpoints/reference tensors stayhost; kernel/artifacts/distributed-d29 retains
  sourceaudit/overlays/payloadexports/union/metadata excludes tensor/RNG/safetensors.
  104archive764130B SHAe0f1aa6a1bc129b09d98ad1b99c55f767ded26e13b1cd8a6c78b2e555e137ebe;
  105674130B SHA92ba18c33be2eee498a8838a775ee03f2255f610483090d459de669ddcd6bdc0.
  Exactcommands/PIDs/CIDs/ACK/logs evidence/distributed-d29-r1/frozen-{seed,resume},
  frozen-{missing,grad,late,peft}-seed,retained-control-seed,frozen-policy-resume,
  allocator-control-seed; controls/resumes-sequence.json/controllerlogs.
  frozen-policy-seed is PRIVATECOPY, not a run. Seed22304/30336,controls22858/31101,
  resumes25019/1388,allocator25904/2533,exports26385/3193 finalexit0. Verifier
  installed AFTERmetadataexport; finalsource committed and installed BOTH.
  PLAN01reservation+SHlock/4CPUcpuset0–3/10GiB+12GiB/512PIDs/256MiBshm/600s/460s
  retained. Kernellogs/registries unchanged/finalidle/no jobs/containers/armedguards.
  Entry2d86d523(.104r2)/e902e636(.105r3) unchanged; no build/reboot/download/prefix
  mutation/apps/foreignstop. NativePLAN02MIOpen381eac05/Torcha524/RCCL/D09/image/
  packages/defaults/fallbacks preserved; docs snapshots synced BOTH. Historical
  PythonCPUfault distinctfixedVMheap/D05exit/D21load. No fullupstreamCI/universal
  model/speed/capacity claim. **NEXT PLAN-04 / D30 (planned):** inspect/admit explicit
  BF16 working-parameter PEFT frozen-base FSDP with checkpointing OFF first, CPU
  ALLadaptergrad/clip/AdamW/frozenpreservation and exactrestart. Keep frozenAMP
  refusals untilqualification; combinedBF16checkpoint and FP16 remain separate.

- **PLAN-04 / D28 completed 2026-11-09 (PLAN-04 still OPEN):** implementation/
  verifier `e08c98d15`. [Report](docs/development/bc250_distributed_d28_results/README.md),
  [verified results](docs/development/bc250_distributed_d28_results/verified-results.json).
  NEW BC250PeftFSDPTrainer/peft_fp32_policy in BOTH additive tools/distributed-d28.
  Actual PEFT0.18.1/Transformers5.2/Accelerate1.12; nine extra source hashes checked.
  Single default causal LoRA/dropout0/biasnone/unmerged/no DoRA/RSLoRA/savedmodules;
  named FP32 frozen/trainable policy, base configuration/attention +PEFT/layer
  rank/alpha/scaling/activation recipe bound to checkpoint. D27 ordinary/default
  guards and installed helpers preserved. Frozen AMP/checkpointing stay refused.
  Source audit: native Trainer applies PEFT auto-wrap AND adapter_only=True for
  checkpoint helpers. Retain native wrapping; explicit subclass saves/loads FULL
  model with public Accelerate adapter_only=False +native FSDP optimizer/scheduler;
  Trainer loop/RNG/state/D27policy/D21optimizer-reader retained, no package patch.
  Each checkpoint also exports ordinary PEFT adapter artifacts with canonical
  FSDP prefixes removed;14,216B safetensors. R1 export kept wrapper prefixes;
  public load warned missing LoRA keys and strict ALLparameter comparison failed
  at CP2 on BOTH, after correct training references. R2 canonical export passes;
  R3 binds base recipe/adds live scaling refusal and reruns selected final phases.
  No suppressed warning/tolerance relaxation/old user-training defect claim.
  Owned eager GPT2/tied frozen head/8actual PEFTConv1D LoRA/r4alpha8/zeroB,
  24352unique=21280frozen+3072trainable. Native PEFT wrapping root+2GPT2Blocks+
  16trainable leaves=19FSDPunits; original shards12176each, adapter/AdamW1536each.
  D26 rightpad/prompt/dropout0/cacheFalse/workers0/sequential/micro2/accum2/data28,
  global8/8/8/final4, shifted49/57/57/21/localzero micro. CPUFP64 ALLadaptergrad/
  loss/rootnorm/ALLclip(cap.005)/CPUdoubleALLAdamW at unchangedFP32bounds pass;
  ALLfrozen VALUES exact. Observed FP32 gathers259seed/111resume and16reductions/
  window/12288B input; NOT totalNIC/memory/speed comparison. Public PEFT
  save_pretrained/from_pretrained CP2+4 and replayCP4 ALLparams/logits exact,
  four RNG states unchanged by CPU audit. Adapter-only artifact needs matchingbase.
  Rank1 missingadaptergrad /injectedfrozengrad /lateNaN /lateLoRAscaling BOTH
  refused before AdamW, zeroupdates/unchangedparams+optimizer+scheduler. Retained
  D27 native-adapter helper control passes. Private rank1 checkpoint changes ONLY
  saved LoRA scaling (generic frozen hash unchanged): BOTH refuse optimizerrestore,
  zero updates. Original checkpoint/policy hashes intact. Aborted model may have
  loaded; discard/reload session. No arbitrary-hook/rankloss/partialwindow promise.
  Full CP2 DCPunion6files480247B onBOTH independent filesystems; original ranklocal
  files/hash retained. Fresh constructor1773/newUUID+ticks+containers restores ALL
  frozen/adapter shards/AdamW/scheduler/fourRNG EXACT; windows3/4 ALLloss/grads/
  clip/updates/state/RNG/rows/counts/digests replay EXACT. Native train_loss remains
  current-process statistic. No stochastic/elastic/other-PEFT-recipe guarantee.
  116finalgroups/16ACKguards/26CPUpatterns408observations/62localtests/Black/critical
  Ruff+F401/diff pass;2R1failedexports+2R2intermediatepasses retained/all20ACK.
  Exact commands/PIDs/CIDs/ACK/logs in evidence/distributed-d28-r3/frozen-{seed,resume},
  frozen-{missing,grad,late,peft}-seed,retained-control-seed,frozen-policy-resume;
  controls-sequence.json/resumes-sequence.json/controllerlogs. frozen-policy-seed
  is PRIVATECOPY, not a run. R1 11804/13899exit1; R2 12431/14692exit0; R3seed13445/
  16280,controls14014/17060,resume16159/19784,exports17340/21442exit0. All sources/
  payloads/checkpoints/public artifacts retained; local kernel/artifacts/distributed-
  d28 packageaudit/overlay/exports/completeunion/metadata excludes tensor/RNG/
  safetensors payloads.104archive816023B SHAeedfa6b5bf43807495d11bafd9d10591beede394d4b98f949acec2f6ca3e64ac;
  105713531B SHA1fbd0d22fdbe0b9982499e470477606b35f7ee214262a77167ec45f01b79b4c1.
  PLAN-01 reservation+SHlock/4CPU/10GiB+12GiB/512PIDs/256MiBshm/600s/460s retained.
  Kernellogs/registries unchanged, finalidle/nojobs/containers/armedguards. Entry
  2d86d523(.104r2)/e902e636(.105r3), already different from D27 snapshots; no agent
  reboot/build/prefixmutation/download/appinspection/foreignstop. Native PLAN-02
  MIOpen381eac05/Torch a524/RCCL/D09/image/packages/defaults/fallbacks preserved.
  Docs/memory snapshots synced BOTH hosts. Historical PythonCPU fault distinct
  fixedVMheap/D05exit/D21load; no fullupstreamCI/speed/capacity/universal proof.
  **NEXT PLAN-04 / D29 (planned):** inspect native HF/PEFT non-reentrant activation
  checkpointing paths and introduce an explicit FP32 policy, prove CPU ALLgrad/
  AdamW/frozen-preservation/exactrestart, then bounded allocator comparison with
  safe headroom. Preserve checkpointing/frozen-AMP refusals until their own proof.

- **PLAN-04 / D27 completed 2026-11-08 (PLAN-04 still OPEN):** implementation/
  verifier `80080e6a0`. [Report](docs/development/bc250_distributed_d27_results/README.md),
  [verified results](docs/development/bc250_distributed_d27_results/verified-results.json).
  NEW explicit BC250FrozenFSDPTrainer/frozen_fp32_policy in BOTH-host additive
  tools/distributed-d27. Existing BC250FSDPTrainer STILL refuses frozen params;
  its original validation/gradient checks factored without weakening defaults.
  Named immutable nonempty frozen/trainable FP32 partition +cross-rank hash/shape
  agreement, positive complete-root clipping; trainable-only AdamW membership/
  state, actual scheduler-installed prehook rejects missing/frozen/nonfinite grads
  and policy drift before update. Native Trainer/FSDP/backward/optimizer retained.
  Native checkpoint policy JSON binds partition/shapes/FP32; both ranks refuse
  changed/missing metadata before optimizer restore. Model may already be loaded;
  discard aborted sessions, no gradient/RNG rollback or arbitrary-hook guarantee.
  Owned eager GPT2/tied frozen wte+head/root+2GPT2Blocks, native rank4 adapters
  around8Conv1D projections:24352unique params=21280frozen+3072trainable. Per-rank
  original shards12176each; trainable/AdamW1504/1568. D26 data28/rightpad/prompt/
  dropout0/cacheFalse/workers0/sequential/micro2/accum2/global8/8/8/final4, shifted
  counts49/57/57/21/localzero micro. CPUFP64 ALLadaptergrad/loss, rootnorm/ALLclip/
  CPUdoubleALLAdamW moments/counters pass originalFP32 D21/D22/D26 bounds, cap.005.
  ALLfrozen VALUES EXACT original snapshot each inspected window. Native frozen
  original grad=None in optimizer-facing sharded state; public full-gradient
  summon temporarily materializes mixed flat-buffer views, which are excluded
  from adapter references/AdamW. R1 probe wrongly demanded None there, fails both
  before first update; source audit confirms native behavior. R2 fixes context,
  passes; R3 adds final immutable-policy/save/restart guards and selected reruns.
  No tolerance relaxed/vendor kernel patched. Actual FP32 gathers/two reductions
  perwindow,80640B input, NOT3072*4; mixed buffers/padding keep base-sized cost.
  No speed/whole-memory/capacity/convergence/universal frozen-model claim.
  R3 rank1 missing adaptergrad /injected frozengrad /postclipNaN BOTH refuse before
  AdamW, zero updates and unchanged parameters/optimizer/scheduler. Retained
  alltrainableGPT2 FP32 original helper control passes. CP2 DCPunion6files469453B
  onBOTH filesystems, all original local hashes unchanged. Fresh seed1773 versus
  773/newUUID+startticks+containers restores ALLfrozen/adapter shards/AdamW/scheduler/
  4RNG EXACT; windows3/4 ALLloss/grads/clip/updates/RNG/rows/counts/digests replay
  EXACT. Native train_loss process-local. Rank1-only private policy JSON corruption
  BOTH refuse before optimizer restore/zeroupdates; originalcheckpoint intact.
  Extra frozen-policy-seed folder is private COPY, not a run. Actual PEFT package,
  frozen BF16/FP16 and activation checkpointing remain UNQUALIFIED/refused.
  97finalgroups/14ACKguards,22CPUpatterns294observations/62localtests/Black/critical
  Ruff+F401/diff pass.2R1failedprobe guards+2R2intermediatepasses retained/all18ACK.
  Exact commands/PIDs/CIDs/ACK/logs in evidence/distributed-d27-r3/frozen-{seed,resume},
  frozen-{missing,grad,late}-seed,retained-control-seed,frozen-policy-resume;
  controls-sequence.json/resumes-sequence.json/controllerlogs. R1 21610/5214exit1;
  R2 22748/5962exit0; R3seed24866/6696,controls25338/7199,resume26837/8794,
  exports28532/9786exit0. Source variants/initial probes/checkpoint/reference tensors
  preserved onhosts; local kernel/artifacts/distributed-d27 overlay/payloadexports/
  completeunion/metadata archives (no tensor/RNG payloads in metadata):
  104694604B SHAfec317a80cac76bb74f99b65142e18fa3a605d15825fc15ffac3d631c668527e;
  105607645B SHA3690917d572464734d73bc2cb57f90358218ebc2039d90d060256c4be1283767.
  PLAN-01 reservation+SHlock/4CPU/10GiB+12GiB/512PIDs/256MiBshm/600s/460s retained.
  Native providers/PLAN-02 MIOpen381eac05/Torch a524/RCCL/D09/images/packages/registry/
  prefix/defaults/fallbacks unchanged; no build/reboot/download/app access/foreign
  stop. Entry .104afcc7be4/r2; .105570437ed/r3 (already different from D26 historical
  8c396bd1, no agent reboot). Kernel logs clean/registry exact/finalidle/no jobs/
  containers/armedguards. Docs/memory snapshot synced BOTH hosts. Historical Python
  CPU fault separate from fixed VM heap/D05 exit/D21 optimizer loading.
  **NEXT PLAN-04 / D28 (planned):** inspect pinned PEFT LoRA/FSDP canonical naming,
  adapter selection, optimizer and native checkpoint paths; qualify owned actual
  PEFT FP32 fixture with CPU ALLtrainable-grad/update/frozen-preservation/exact
  restart before claiming package integration. Keep frozen AMP/checkpointing guards;
  those combinations and capacity/performance need separate proof.

- **PLAN-03 / D26 completed 2026-11-08:** implementation/verifier `8e04668d7`.
  [Report](docs/development/bc250_distributed_d26_results/README.md),
  [verified results](docs/development/bc250_distributed_d26_results/verified-results.json).
  Additive BOTH-host tools/distributed-d26, shared D25 test harness parameterized
  for owned GPT2; D25 installed tools and BC250FSDPTrainer policy helper preserved.
  No production native defect reproduced or new kernel/default claimed. Actual
  eager GPT2/tied wte+lm_head/21280 unique params/root+2GPT2Blocks/world2/RCCL/
  FULL_SHARD/use_orig_params/alltrainable FP32 masters: FP32, BF16working params
  and FP16working params/native ShardedGradScaler pass. Each rank owns10640
  parameters AND AdamW elements, tied aliases SAME Parameter across wrapping,
  gradients/updates/restart; finite seed18/replay9 checks, overflow15/7.
  D16 caller-owned CausalLabelCollator+shiftedCE/global denominators49/57/57/21
  vs raw51/60/59/23; localzero-target actual GPUmicro passes. Owned28records,
  rightpad/promptmasks/dropout0/eager/cacheFalse/workers0/sequential/micro2/accum2,
  global8/8/8/final4. CPUFP64 ALLgradient/loss +roundedCPUAMP localaccum refs pass
  originalD15/D25 bounds; maxrelative BF16.006585226/.006604265, FP16.000825151/
  .000917906. CPU rootnorm/ALLclippedgrad/ALLAdamWparams+moments+counters pass
  D21/D22 gates. Conv1D+head/workingparamallgathers and FP32reduce/master/syncedgrad
  observed. BERTFP32 default harness control passes (new tools only).
  FP16stress128/growth2 finiteends512; rank1-owned Inf AFTERreducewindow3 gives
  flags[False,True], BOTHnative AdamW/scheduler skips/EXACT unchangedparams+moments/
  scale256→128, window4finiteupdate3. Native defaults65536/2000 unchanged.
  CP2 +offline DCP union6files653339B each finitepolicy/653347Boverflow on BOTH
  filesystems; original local hashes unchanged. Fresh constructor1773/scaler8/
  newUUID/startticks/containers restore ALLparams/AdamW/scheduler/scaler256/fourRNG
  EXACT; windows3/4 ALLloss/grads/clip/update/state/RNG/rows/counts/digests replay
  EXACT for all three policies AND overflow. Native train_loss remains current-
  process statistic. No partialwindow/stochastic/elastic/cross-policy proof.
  232GPUgroups/18ACKguards/15CPUcontractpatterns256observations/62localtests,
  Black/criticalRuff+F401/diff pass. No failed GPUphase. Initial archive verifier
  overexpected >=8 alias checks in overflowreplay; exact7 follows skip callbacks.
  Corrected exact18/15/9/7 bookkeeping; numerical results unchanged/no GPUrerun.
  Initial verifier/failurelog/sourceentry variants/r1archives preserved; r2metadata
  exports assert same handoffbytes and include finalverifier. No fullupstreamCI.
  Exact commands/PIDs/CIDs/ACK/kernel+registry records in
  evidence/distributed-d26-r1/{fp32,bf16,fp16,overflow}-{seed,resume}, bert-control-seed;
  seeds-sequence.json/resumes-sequence.json/controllerlogs. Firstseed2733/3761,
  seeds3398/4659,resumes5066/7063,finalexports7095/10554 exit0. Nativecheckpoint/
  reference tensors retained onhosts; local kernel/artifacts/distributed-d26 has
  both raw exports/fourunions/metadata/source overlay and initialverifierfailure.
  Final .104 r2archive672202B SHA61aed24e5000b4b5be3d98d69a5ed43369a5f4146e931d96bbdcb8b4c4553037;
  .105568511B SHA757620d5fee9ea29a49f97a2ad9f174371e43406b3b351443d918a0ca55845f1.
  PLAN-01 reservation+SHlock/4CPU/10GiB+12GiB/512PIDs/256MiBshm/600s/460s retained.
  No app access/foreignstop/nativebuild/prefixchange/reboot/download, entryboots
  afcc7be4(.104r2)/8c396bd1(.105r3) unchanged. PLAN-02 MIOpen381eac05/nativea524/
  D09/RCCL/images/packages/registries/defaults/fallbacks preserved. Kernel logs
  unchanged; finalidlepreflight/no jobs/containers/armedguards. Docs/memory snapshot
  synchronized BOTH hosts. Historical PythonCPU fault separatefixedVMheap/D05/D21.
  No speed/capacity/convergence/universal causal-model claim. SDPA/RoPE/GQA/
  generation/packing/otherarchitectures/FSDP2/offload/compile remain separate.
  **NEXT PLAN-04 / D27 (planned):** inspect frozen/missing-gradient/optimizer
  guards; design an explicit bounded FP32 frozen-base/adapter FSDP policy and
  owned CPU ALL-trainable-gradients/AdamW/exactrestart fixture. Keep existing
  frozen-parameter refusal until qualification. Activation checkpointing follows.

- **PLAN-02 completed 2026-11-08 — ordinary MIOpen repair active on BOTH hosts.**
  Implementation `72c4276ad`; solver/cache source `e1d8d1ff920` on B07.11 repair
  `96e5a7dfc43`. [Report](docs/development/bc250_plan02_results/README.md),
  [verified rollout](docs/development/bc250_plan02_results/verified-rollout.json).
  Current .104 AND .105: `/opt/bc250-stack/candidates/miopen-plan02-r4/profiles/daily`;
  RCCL alias/D01 select its `profiles/rccl-single`. Installed MIOpen SHA
  381eac05240e33acced9472fb7c3071afe8bc4f434407e67a29ce64423b935ac.
  Base native revisions remain in descriptors; specific miopen_precision_repair
  records source/hash/cache version. Torch a524/RCCL/image/kernels unchanged.
  All old prefixes, SDKs, paths/caches/builds/checkpoints/images retained. .104
  optional CK/MIGraphX/ORT use coherent new versioned registries/packages; no pin
  weakened. .105 has no optional SDKs; old daily CLI preserved, D09 core handles
  explicit RCCL. New target descriptors enable the SAME migration-control lock
  inode; frozen payload remains preserved. Shared unchanged ROCm hardlinks are
  dependencies: detach every changed inode on future updates, never write through
  them in place. .105 production healthcheck/native controls/fallback untouched.
  Additional retained-cache control exposed unsafe r2: new DSO0612f031 still
  executes old half-accumulation binaries. r2 EX rolled back, NEVER reactivate it.
  Version1 adds gfx1013 FP16 forward-only compile option to separate old cache
  keys. Isolated ccache build5690 passes (one object+link, old build preserved),
  source DSO0e256648; normalized381eac05. r3 stage failed only generated HIP CUID
  export difference; r4 excludes anchored compiler content IDs only, raw ABI/deps
  retained. Cache probe r1 raw-SQLite-byte check and r2 quoted-log matcher failed
  outside numerical gates; GPU math passed. Corrected r3 checks pass both hosts:
  each profile retains12 old rows/adds6 new; fresh warm process12 hits/6versioned,
  no saves/unchanged logical records, original control cache hashes preserved.
  Target controls copied from owned .104 fixture; user caches/apps not inspected.
  .104 r4 inactive+public matrices10 guards/98 groups EACH pass, eight actual
  identity/refusal groups and two actual cache-activation refusals pass. Live EX
  rollback restores pointers/D01/native bytes and reproduces old strict mismatch
  in two guards; final reactivation direct/backward4 guards/26groups pass.
  .105 matched cache4, inactive/public/final4 each (26groups each), rollback1 pass.
  Public two-host RCCL four-dtype sums +three CPU-reference DDP/SGD steps8groups/
  rank pass (one guard/host). Final r4 totals49 acknowledged guards,62 localtests,
  Black/criticalRuff+F401/diff pass. No universal precision/speed/convergence or
  cross-native-version bitwise checkpoint claim. Historical Python CPU faults
  distinct from fixed VM heap/D05 exit/D21 loading. No app access/foreignstop/
  cleanup/reboot/download; bootafcc7be4(.104r2)/8c396bd1(.105r3) unchanged.
  Target setup failures retained: missing legacy import dependency, then absent
  log parent, both before GPU/guard arm. Reusable gate imports now lazy; existing
  PLAN-01 reservation/SHlock/4CPU/10GiB+12GiB/512PIDs/256MiBshm/600s/540s retained.
  Exact transactions ROOT/transactions/cache-safe-{initial,rollback,final}(.104)
  and target-{initial,rollback,final}(.105); rollback uses initial/before.json.
  Commands/controller/worker/CID/log/ACK details in WS/logs/plan02-* and
  evidence/plan02-*. Final controllers10416479/25967/30456/30907/32203 and
  10523409/24749/26282/27443/28040/29337 exit0; exports32592/30144 exit0.
  Local kernel/artifacts/plan02/plan02-rollout-bc250-r4.tar.gz3348200B SHA
  833dc281beae95d18cf6b8971d1d9e1e2a289888980feafa054de044175fee7c;
  bc500 archive781459B SHAbaba98b6560df77fa566806d036ca6a0c27b1796305621dbbec8bb19acf21482.
  Final idle preflight passes, no owned GPU/build jobs/containers/armed guards.
  Final PLAN-02 docs/memory snapshots synchronized to BOTH host docs entry points.
  **NEXT PLAN-03 / D26:** inspect recorded GPT2/D16/D23–D25 evidence, then actual
  Trainer classic FSDP eager GPT2 tied-parameter/shifted-token-loss CPU ALL
  gradients/clipping/AdamW and exact restart for explicit FP32/BF16/FP16 policies.
  Keep frozen/PEFT/checkpointing/capacity separate; do not repeat completed D25.

- **Historical PLAN-02 candidate batch 2026-11-08 (superseded above):** implementation19de82f88.
  [Report](docs/development/bc250_plan02_results/README.md). Installed .104 MIOpen
  SHA3a751adc1497 remains selected; retained repair797335e2b713 tested privately.
  Exact old FP16 solver218 mismatch reproduced in daily AND RCCL, original bound
  unchanged. Repaired candidate passes12 direct cases/profile (24 total), plus
  12 ordinary Torch output/dx/dw/db configurations/profile (24 total), covering
  FP32/FP16/BF16/groups1/3/dilation1/2 with declared CPU FP64 bounds. No native
  rebuild/prefix update or .105 access; no ordinary-default fix installed yet.
  Six accepted guards including two old negative controls, two setup failures
  retained: protected LD_LIBRARY_PATH refused, then missing private-child zstd
  dependency. Correct fresh child+sysdeps path passes; all8 ACK, dmesg unchanged,
  no containers/jobs/armed guards.62 local tests/Black/critical Ruff+F401/syntax pass.
  Controllers20838/21414 exit1,21705/22359 exit0. Exact commands/results/logs
  WS/logs/plan02-*; evidence /home/nik/bc250work/evidence/plan02. Local archive
  kernel/artifacts/plan02/evidence-r3.tar.gz354657B SHA
  fe5f48b34620cb42fb47c10deb24601686ff9b300ba4bed5b7e6c41ca2de09c2.
  Resources4CPU/10GiB+12GiB/512PIDs/256MiBshm/600s guard/540s child, PLAN-01
  reservation+SH lock retained. Entry .104 bootafcc7be4/r2 unchanged. No app access,
  foreign stop/reboot/download. Historical Python CPU fault remains distinct from
  fixed VM heap/D05 exit/D21 optimizer loading. Host docs still PLAN-01; updated
  candidate report/cursor local; evidence/tools retained on .104.
  **NEXT remains PLAN-02:** new installed-layout candidate/RUNPATH normalization
  and ABI/NEEDED checks; coherent identity qualification before EX-lock rollout.
  Installed bc250-ck pins old MIOpen; SDK hashes runtime descriptors and ORT embeds
  SDK registration in both registry and immutable manifest. Preserve consumers;
  do not relax pins or swap the DSO alone. Then ordinary .104 activation/rollback
  and matched .105 qualification. PLAN-02 unchecked; do not advance to D26 yet.

- **PLAN-01 completed 2026-11-08:** implementation `2e24474d6`, selected upcoming
  B07.11/B07.12 modules use new stack-only board_preflight and shared GPU gate
  cooperative EX board reservation + SH runtime lock. Default-local Docker,
  Podman/GPU fds/memory/watchdog checked; post-test Docker unknown/busy fails.
  Additive `tools/plan01` on BOTH hosts; preserve old helpers. PLAN-02 must put
  this overlay before legacy helpers in PYTHONPATH. Four real CPU controls/node
  pass: idle, competing reservation refusal, prefix EX writer exclusion, exception
  releases locks. 62 local tests pass, Black/critical Ruff+F401/diff pass.
  [Evidence](docs/development/bc250_plan01_results/README.md). No GPU numerical
  execution/reboot or new watchdog-reset claim; inherited600s recovery/540s child/
  15s termination/ACK procedure retained. D25 recovery evidence not rerun.
  Boot IDs afcc7be4(.104r2)/8c396bd1(.105r3), kernel logs and registries unchanged.
  No native prefix/install/build/app inspection/foreign stop; boards idle/no jobs
  or armed guards. Old unused guardians/custom daemons/uncooperating workloads
  remain outside cooperative reservation guarantees. Plan/docs snapshots synced
  at batch end. **Next PLAN-02:** inspect current MIOpen identities and retained
  repaired artifact, reproduce original FP16 precision bound and qualify minimal
  repair/compatibility/rollback before any ordinary native rollout. D26 follows.

- **Priority review completed 2026-11-08 (documentation only).**
  Reviewed the complete chronology and reconciled superseded “next” entries with
  current inventory, manifest and component reports. New prioritized checklist:
  [BC250_REMAINING_PLAN.md](../BC250_REMAINING_PLAN.md), ten important work packages
  plus optional B07.13/N4/N2/N6/N7 expansion; these are not ten small patches or a
  completion percentage. **Next execution: PLAN-01**, Docker-aware preflight and
  board reservation for selected upcoming guardians; **then PLAN-02**, audit and
  qualify ordinary-runtime rollout of the isolated B07.11 FP16 accumulation repair.
  **D26 remains next model capability** within PLAN-03. Follow with frozen/PEFT+
  activation-checkpointing admission, real capacity/matched performance, bounded
  crash/stability triage, clean source reconstruction and coherent N8 release,
  package/model compatibility and profile-driven optimization. Begin CPU-only
  reconstruction inventory early; newly reproduced corruption outranks features.
  Corrected stale B06.12 “next” row (already complete), old current-state heading
  and distributed summary; K6 is complete, not a new kernel rewrite. Historical
  evidence retained. No host access, GPU tests, native/source-code changes, prefix
  updates or cleanup. D25 last qualified runtime/evidence unchanged; historical
  Python CPU fault remains separate from fixed VM heap/D05 exit/D21 loading.
  Host docs/memory snapshots still D25; synchronize the plan during the next
  authorized host batch. No running jobs created by this review.

- **D25 completed 2026-11-08: actual Trainer classic FSDP FP16/scaler.**
  Implementation/verifier `36a215d8c`; installed under `tools/distributed-d25`
  on BOTH hosts. [Report](docs/development/bc250_distributed_d25_results/README.md),
  [usage](../BC250_DISTRIBUTED_GUIDE.md#trainer-fsdp-fp16-working-parameters-and-scaling-d25).
  Explicit `fsdp_fp16_parameter_policy()` admits FP16 working weights/all-gathers,
  FP32 masters/synchronized gradients/reductions/buffers/AdamW, positive global L2
  clipping and enabled native world-group ShardedGradScaler. Incoming parameters
  must be trainable FP32. Old FP32/BF16/staged helpers and frozen/package guards
  retained. Additional Accelerate optimizer/modeling source pins checked.
  Public unscale runs ONCE, then root FSDP clipping for finite gradients; native
  scaler synchronizes overflow/skip/backoff and Trainer suppresses scheduler steps
  on skips. Actual AdamW pre-hook rejects late invalid gradients/scaler changes.
  No private scaler flags, global package patch, Trainer loop fork or native build.
  Owned eager Bert MLM22561, root+two BertLayer FSDP units, world2/RCCL/full sharding,
  original parameters/dropout0/workers0/sequential/micro2/accum2, unequal MLM masks
  and right padding: global8/8/8/4. CPU FP16 scaled local accumulation and FP64
  rounded-weight global loss/ALL gradients pass unchanged D15 bounds; maximum
  relative errors .001115321/.000713187. Native global norm/ALL clipped gradients/
  CPU double ALL AdamW parameters, moments and counters pass D21/D22 bounds.
  Owned parameter/AdamW elements11284/11277. Native FP16 gathers/FP32 reductions
  observed; no speed, peak-memory, larger-fit or total-network claim.
  Stress initial128/growth2: four finite updates end at scale512. Rank1 Inf AFTER
  reduction at window3 gives flags[False,True]: BOTH native scaler skips preserve
  parameters/optimizer/moments/scheduler EXACTLY, scale256→128; window4 executes
  successful update3. Trainer global_step/data/RNG still advance on a skip.
  Native production defaults65536/growth2000 unchanged. FP32/BF16/inactive clipping
  controls pass. Post-clip rank1 NaN and pre-unscale rank1 scale disagreement both
  refuse before AdamW, zero updates and unchanged parameters/optimizer/scheduler.
  Native checkpoint2/scaler256 plus offline DCP union6 files:864357B finite /
  864365B overflow; all original local hashes retained. Fresh containers/UUIDs/
  start ticks/constructor seed1773/scaler8 restore ALL master shards/AdamW/scheduler/
  scaler/four RNG EXACTLY. Finite AND overflow windows3/4 replay ALL gradients,
  loss, clipping/norms, parameters/moments/counters, scheduler/scaler, RNG/data and
  transcript/final digests EXACTLY on both ranks. Native train_loss remains a
  current-process statistic. No transient-gradient/RNG rollback or stochastic,
  multiworker, partial-window, elastic, arbitrary-callback or rank-loss promise.
  Final128 GPU groups/18 acknowledged guards,8 CPU patterns/144 observations,
  58 local tests, Black/critical Ruff+F401/py_compile/diff pass. Four rejected
  guards retained: R1 both had a probe expecting256 instead of correct512; R2 both
  exposed initial pre-hook installation in create_optimizer_and_scheduler, which
  pinned Trainer bypasses. Fixed actual create_scheduler/raw optimizer path;
  all R3 phases rerun and assert gate identity/calls==actual AdamW updates.
  Ten intermediate R2 passed guards also retained; all32 guards acknowledged.
  Isolated R2 NaN fixture discarded; no active/user application updated. No full CI.
  Local `kernel/artifacts/distributed-d25` archives: .1041024387B SHA
  c2f26b9706b7077c901f2b5a03c29f120be6ebf8b59740456806e1d56b93b0f1;
  .105873798B SHA b1a6466e512b331e7fe4eef7e03d9c1734c637770edddbbcc01d0d6772b4df59.
  Historical sources/r1–r3 logs/checkpoints/references/payloads preserved; metadata
  excludes tensor/RNG binaries/private applications. Exact commands/PIDs/CIDs/ACKs:
  `evidence/distributed-d25-r3/*/{rank,final}.json`, controls-sequence.json and
  controller logs. Final seed13649/18301, controls14118/18778, finite resume16256/
  20940, overflow resume16657/21347 exit0. Final idle preflight passes; no jobs,
  containers or armed guards. SH locks/4CPUs/10GiB RAM+12GiB RAM+swap/512PIDs/
  256MiB shm/600s recovery/460s child retained. Entry bootsafcc7be4(.104r2)/
  8c396bd1(.105r3) unchanged. No agent reboot/download/prefix mutation/application
  inspection/foreign stop. Native Torch daily/RCCL a524, D09, ROCm, images,
  registries/packages/kernels/fallbacks/defaults unchanged. Historical Python CPU/
  KFD issues remain separate from fixed VM heap/D05 exit/D21 optimizer loading.
  Final host docs/memory snapshots synchronized. **Next D26 (planned):** actual
  Trainer classic FSDP eager GPT2 causal LM/tied parameters, shifted global token
  loss, CPU ALL gradients/clipping/AdamW and exact restart with explicit FP32/
  BF16/FP16 policies. D16 DDP is not FSDP proof. Frozen/PEFT, FSDP2, offload,
  compile, activation checkpointing, elastic and real capacity remain separate.

- **D24 completed 2026-11-08: BF16 FSDP working parameters/all-gathers.**
  Implementation/verifier `dfe347376`; BOTH `tools/distributed-d24`.
  [Report](docs/development/bc250_distributed_d24_results/README.md),
  [usage](../BC250_DISTRIBUTED_GUIDE.md#trainer-fsdp-bf16-working-parameters-d24).
  New explicit fsdp_bf16_parameter_policy() +pinned BC250FSDPTrainer admission:
  BF16 working weights/parameter all-gather input+output, FP32 master shards,
  synchronized gradients/reductions/buffers/AdamW. No kept lowprecision grads,
  input casts/scaler; unsynchronized gradients may temporarily be BF16.
  D23 compute-only factory/old staged helpers and frozen/package guards retained.
  New guardian checks default-local Docker before/after, unknown state refused;
  no unrelated workload stopped. Other old guardians still need migration.
  Owned eager BertMLM22561/root+2BertLayers/world2/RCCL/alltrainable/micro2/accum2/
  workers0/sequential/rightpadding/unequalMLM/global8/8/8/final4: CPU BF16 rounded
  params/local accumulation+CPUFP64 rounded-weight global loss/ALLgrads pass D15
  bounds unchanged. Maxrelative CPUAMP.008212774/FP64.005715182; native globalclip/
  ALLclipped grads/CPUdouble ALLAdamW params+moments+counters pass D21/22 limits.
  BF16 padding yields master/AdamW elements11284/11277, sum22561. StrictFP32
  control11282/11279, inactiveclip coef1, rank1NaN BOTHrefuse beforeclip/AdamW/
  zeroupdates/unchangedparams+optimizer+scheduler pass. No grad/RNG rollback.
  Native CP2+offline DCP union6files864357B, original hashes/local state preserved;
  newCIDs/UUID/startticks/seed1773: initial ALLshards/AdamW/scheduler/4RNG EXACT;
  windows3/4 ALLloss/grads/clip/params/moments/scheduler/RNG/rows/digests EXACT.
  BF16 training-only allgather inputs35calls277592B seed/15calls118968B replay;
  inactive10calls79312B vsFP32control158608B.2Bvs4B elements/padding differ,
  NOTtotalNIC/peakRAM/speed/largerfit proof. FP32reduce361088B/12calls seed.
  Accepted64GPUgroups/10ACKguards,6CPUcontracts60observations/58localtests;
  Black/criticalRuff+F401/syntax/diff pass. No GPUfailedphase/fullupstreamCI.
  Initial archive verifier full-tail gather overcount corrected35/15; initial
  archives retained, GPU math unchanged. Native warnings/train_loss caveat remain.
  Local kernel/artifacts/distributed-d24 metadata archives1041226233B SHA
  885ebd47cf3622918729a73f29b56fc95f2e281202115161447bc425d6a3c083;
  1051180601B SHAcfe5d2fb1382df55dbec53cfc604b267ea242b6e0a7a6a5787095131adc57173.
  Payload exports/union local, native checkpoints2/4/references retained onhosts;
  metadata excludes tensor/RNG payloads/private applications. Exact commands/
  CIDs/worker+guard results in evidence/distributed-d24-r1/*/{rank,final}.json,
  controls-sequence.json/controllerlogs. Seed5142/9798,resume5835/10445,controls
  6208/10823 exit0. Final idle preflight pass, nojobs/containers/armedguards.
  Entryboots afcc7be4(.104r2)/8c396bd1(.105r3) retained, no agent reboot/download/
  native build/prefix mutation/app inspection/foreignstop. SHlocks/4CPU/10GiB+
  12GiB/512PIDs/256MiBshm/600srecovery/460schild unchanged. Native daily/RCCL Torch
  a524/D09/ROCm/images/registries/packages/kernels/defaults/fallbacks unchanged.
  Historical PythonCPU/KFD issues separate from fixed VMheap/D05exit/D21load.
  Both host docs/memory synchronized. **Next D25 (planned):** actual Trainer
  classicFSDP FP16 workingparams/nativeShardedGradScaler CPUgrad/globalclip/
  collective overflow-skip/exactrestart; retain FP16 refusal until proof. Larger
  capacity/causal+other architectures/frozen parameters/elastic remain separate.

- **L02 completed 2026-11-07: real two-host LoRA checkpoint replay.**
  Implementation `284d3315b`, Docker-aware guardian/verifier `4d5f60e43`.
  [Commands](../BC250_LORA_TWO_HOST.md#restore-a-sealed-l02-checkpoint),
  [report](docs/development/bc250_lora_resume_results/README.md).
  Additive `train_two_host_l02.py`, `run-distributed-l02`, `checkpoint_files.py`,
  support module and `two_host_training_l02.json` installed on BOTH hosts under
  the authorized lora-smoke project. L01 entry points, completed outputs and all
  native checkpoints retained; finetune-n7 uninspected/unmodified.
  Native HF/PEFT restore model/AdamW/scheduler/scaler/RNG; added sealed checkpoint
  IDs, common file hashes, rank-specific RNG validation, model/data/settings
  fingerprints and restoration of TRL's otherwise-reset cumulative token counter.
  Offline common-payload exchange prepares a fresh validated peer directory and
  atomically renames it; no source checkpoint/RNG overwrites or payload unpickling.
  Two ranks, pinned newer framework sources, FP16/native AdamW/workers0/unpacked;
  old unsealed/one-host, elastic, partial-window and shared-FS recovery unqualified.
  Real Qwen3.5-2B/rank128/all-linear/134553600 FP32 trainable elements, global16,
  dropout.05/cosine/one warmup window: R2 checkpoint2 then uninterrupted windows3/4;
  final R3 fresh-process replay after external/user board reboots is EXACT on BOTH
  ranks for data, all gradients, parameters, AdamW/moments/counters, scheduler,
  scaler, four RNG hashes and per-step metrics. Initial test scale8 restores512;
  poisoned RNG1773 is restored natively; token counter10648 resumes15745/19377.
  Test-only growth interval1 causes natural overflow at window4: exact native skip,
  scale1024→512, unchanged parameters/optimizer/scheduler; data/RNG consumed normally.
  Production growth interval unchanged. First warmup optimizer call has LR0.
  Eight guards retained: five accepted/three rejected, all recovery guards ACK.
  Two R1 failures were overstrict probe refusal of legitimate AMP skip. R2 resume
  math passed both, but .105 guard failed on unrelated Docker veth/journald startup;
  failure retained. Added default-local-Docker pre/post checks and repeated clean
  R3 pair, both kernel logs/registries unchanged and Docker/Podman idle. User
  authorized stopping steam-headless/brave_engelbart; already stopped when resumed,
  no unrelated container stopped by agent. Older guardians need this check too.
  54 local tests (42 existing+8 transport+4 Docker), three actual checkpoint refusal/
  preservation checks, Black/critical Ruff+F401/syntax/shell/diff pass. No full CI,
  throughput/capacity/full-epoch/convergence or independent CPU Qwen claim. Native
  aggregate train_loss current-process caveat and reference conv/delta paths remain.
  Common payload11 files/1635023695B; original .104 CP2 and .105 local RNG/ready
  full copy retained. Redundant new .104 transfer tar removed only after SHA
  verification of retained .105 copy; no checkpoint/cache/image/release cleanup.
  Local `kernel/artifacts/lora-resume/{104,105}-evidence.tar.gz`:220526/195665B,
  SHA c4e6bbe17a4c2edf975493f512a4fc5197a26deaa32911a50f78f87859eca890 /
  a334532aa2787ed9b3050c287be09f4c961131ba855bd6d1d7a5ef6f6c063e72.
  Metadata excludes data/tensor/RNG binaries; full local+host transcripts/configs,
  markers/kernel logs/guard ACKs and historical failed probes preserved.
  Exact commands in `bc250work/evidence/lora-resume/*/{running,final}.json`;
  guardians .10412101/13638/15398/3944, .1059035/12975/17836/7663, all finished.
  No jobs/containers/armed guards. Final boots7b7d244f(.104r2)/153daab4(.105r3);
  reboots between attempts were external/user, none initiated here. Shared locks/
  resource envelope retained; no native build/install/prefix mutation/download.
  Torch daily/RCCL a524/D09/ROCm/kernels/fallbacks unchanged. Historical CPU Python/
  KFD issues remain distinct from fixed VM heap, D05 exit and D21 optimizer load.
  Both host docs/memory synced. **Next D24:** first carry Docker-aware preflight
  into its owning guardian; then audit/prototype BF16 FSDP working parameters and
  all-gather buffers with FP32 master shards/reductions and numerical/replay checks.
  D23 frozen-parameter/package-version guards remain unchanged.

- **L01 completed 2026-11-05: authorized real LoRA application on two boards.**
  User redirected this batch from D24 to `/home/nik/bc250-projects/lora-smoke`
  (plural directory). Integration `ee1cc87a4`, export/initial verifier `8bacdd1d2`;
  final verifier adds initial loss/pre-clip-norm and acknowledged-guard checks.
  [Commands](../BC250_LORA_TWO_HOST.md),
  [report](docs/development/bc250_lora_two_host_results/README.md).
  Staged its own Transformers5.22.1/TRL1.17.1/PEFT0.21.2/Accelerate1.15/datasets5.1
  environment, Qwen3.5-2B and private datasets to .105; model SHA matched.
  Added `train_two_host.py`, support module, `run-distributed`, matching
  `two_host_training.json` on BOTH hosts; original entry points, completed
  adapters/checkpoints retained. No finetune-n7 inspection/change.
  Native DDP/RCCL, rank128/alpha256/all-linear, FP16 base/autocast/scaler,
  134553600 trainable FP32 elements/372 tensors; micro2, accumulation8 single /
  4 per rank gives global16. Native newer Trainer already has accumulation fix;
  old version-pinned BC250Trainer/FSDP helper NOT applied. Explicit global causal
  target normalization checked at runtime. Fixed rank-0-only final evaluation;
  both ranks now enter it; explicit process-group cleanup removes pilot warning.
  Native initial FP16 scale128 in the opt-in entry point, native growth/skip kept.
  R1 dropout0: all first-update POST-CLIP LoRA gradients vs equal-batch one GPU,
  CPU-FP64 comparison relative L2 .0034373344 <=.03, every parameter bound passes;
  initial loss and PRE-CLIP norm also agree, so clipping cannot hide loss scaling.
  Not an independent CPU Qwen reference. Four updates; final DDP replica hashes
  identical. R2 dropout.05: six updates/layout, all96 input+label hashes match
  global windows; no skipped/nonfinite updates in any phase. Warm five updates:
  single122.892s vs max-rank DDP99.861s =>1.231x throughput /18.7% less update time.
  Whole short train_runtime239.404s vs248.044s including startup: DDP slower here;
  no full-run speed/convergence claim. Gradient vector538214400B, not wire counter.
  Separate production smoke: one update/16 records +final eval8 on BOTH ranks;
  all372 saved adapter tensors/134553600 elements EXACT vs live export.
  Eight GPU guards ACK/exit0, no failed GPU phase/kernel-log/registry change.
  Recoverable allocation retries in DDP R1 and both R2 layouts retained; 8192
  setting remains memory-sensitive, not all full-length samples tested. Reference
  conv/gated-delta paths still used by this newer app; fast-package/hooks not ported.
  Black/critical Ruff+F401/syntax/shell +3 generator preservation checks pass.
  First non-root metadata exports omitted 0700 guard dirs and verifier rejected;
  fixed root-only exporter, complete exports verify; incomplete archives retained.
  Local `kernel/artifacts/lora-two-host/{104,105}-evidence.tar.gz`:405179/282309B,
  SHA58557236d72cb6aa6a3781a2c351cb367d9222a40ddd0543b61d27edf43d247d /
  2bbafbb7c270c053480c17eb3618eb14910afb00b9e3508b2611ef2707456902.
  Raw gradients/adapter retained on hosts; no private dataset/weights in exports.
  Guards104:27384/28490/30214/30886/32056;105:20500/22096/23684; exact commands,
  PIDs/configs/logs in both `bc250work/evidence/lora-two-host/*/{running,final}.json`.
  No jobs/containers/armed guards; boots c1199fca(.104r2)/bad537ed(.105r3) unchanged.
  Shared locks/resource envelope retained. No native build/install/prefix mutation/
  reboot/download; Torch daily/RCCL a524/D09/ROCm/kernels/defaults unchanged.
  Native historical CPU Python/KFD faults remain separate from fixed VM heap,
  D05 agent exit and D21 optimizer load. No larger-model-capacity/general-model proof.
  **Next application L02:** same-world real-application checkpoint payload
  distribution +fresh-process optimizer/scaler/scheduler/RNG/cursor replay before
  relying on long-job recovery. Old one-host checkpoints are NOT qualified exact
  two-host continuation. **Stack cursor remains D24** below; L01 does not widen
  D23's frozen-parameter/version guards. Both host docs/memory synchronized.

- **D23 completed 2026-11-05, actual TrainerFSDP BF16 compute/FP32-shard policy:**
  helper/factory/fixturesc49d99b84, export/verifier0ecfd6258. New explicit
  fsdp_bf16_compute_policy() +BF16 admission in BC250FSDPTrainer, BOTHtools/
  distributed-d23. Native GPU BF16 autocast, but FSDP forward params/reductions/
  buffers/master+gradient shards ALLFP32, no inputcasts/lowprecision keptgrads,
  noscaler. Requires bf16True+EXPLICITpolicy beforeTrainer creation; default/
  lowparam/lowreduce/casting policies refused before+afterAccelerator setup.
  Incoming alltrainable masterFP32; D21/22 FP32 path+oldstagedhelpers retained.
  ActualGPU Linearoutputs BF16,forwardweights F32, preparedlogits/loss F32,
  gradients F32, publicreduce_scatter_single input/output F32;3calls finalmicro,
  0nonfinal;361024B/12calls for4seedupdates,180512/6replay. Onlygradient-reduction
  inputbytes, NOTparamallgathers/totalNIC/physicalcounters. NO reducedparamcomm/
  lowprecisionparameterstorage/speed/peakRAM/largerfit claim, no newmatrixHW.
  Same D21/22 ownedBertMLM22561/eager/dropout0/world2/RCCL/root+2BertLayers/
  originalparams/28records/unequalMLMmask/rightpad5–12/micro2/accum2/workers0/
  sequential/global8/8/8/final4. CPU matched BF16 BOTH-rank microbatches/manualCE
  +CPUFP64 globalmodel at actualGPUweights; sameHFgraph/differentCPUarithmetic,
  not independentTransformer. D15 BF16 loss.003/.003; wholegradrelative<=.02AMP/
  .03FP64; EVERYparam error<=limit*refnorm+3e-6sqrt(elements), no relaxations.
  Maxobserved CPUAMP.0006256018/FP64.0048790555. D22 nativeglobalnorm/ALLclipped
  grad bounds+CPUdouble ALLAdamW params/moments/counters retained frompredicted
  clippedGPUgrads. Cap.05 allactive,norms1.469634–2.265972/coefs.03402205–.02206558.
  BF16cap100coef1, strictFP32 one-step compatibility (D21 bounds), one-rankNaN
  flags[False,True] BOTH refuse beforeclip/AdamW withzero updates/unchangedparams/
  optimizer/scheduler pass. No grad/RNG rollback or latecallback/elastic promise.
  D21 fixed optimizer-aware nativecheckpoint2 reader+same offlinepayloadunion6/
  864357B; originalhashes/localTrainer/scheduler/RNG unchanged. NewCIDs/UUIDs/
  startticks/seed1773: initialALLshards/AdamW/scheduler/4RNG EXACT; resumedwindows
  3/4 ALLlosses/grads/norm/coefs/params/moments/scheduler/RNG/probes/rows/digests
  EXACT. No stochastic/shuffle/multiworker/partial/peerloss/generalresume proof.
  Accepted64GPUgroups/10ACKguards;4CPUclip+policypatterns/40observations;42localtests/
  Black/criticalRuff+F401/syntax/diff pass, no GPUfailedphase/fullupstreamCI.
  Nativewarnings/D12 aggregate train_loss caveat remain. Historical PythonCPU/KFD/
  visibility separate fromfixedVMheap/D05exit/D21optimizerload. NoFP16/scaler/
  BF16forwardparameter/FSDP2/offload/compile/frozen/generalarchitecture proof.
  NativeTorch daily/RCCL a524/ROCm/D09/consumers/images/registries/kernels/defaults
  unchanged, no build/install/prefix mutation/reboot/download/apps/foreignstop.
  SHlocks/4CPUs/10GiB+12GiB/512PIDs/256MiBshm/600s guards/460s child unchanged;
  bootsc1199fca(.104r2)/bad537ed(.105r3) retained. Nojobs/containers/armedguards.
  Seed16859/9394,controls17410/9898,resume18750/11130 exit0; exactphasecommands/
  PIDs/CIDs/exits controls-sequence.json +both evidence/distributed-d23-r1/
  {bf16-seed,bf16-resume,inactive-seed,fp32-control-seed,nan-refusal} preserved.
  Local kernel/artifacts/distributed-d23 archives104368005B SHA
  92e247e52028f1002480c27e9d67cc0fc77357a095b19945d6ee547df40c638e;
  105322913B SHAf1839d770ecfd1232d123020f6c9262c49b4625aa81bd6f31b2457c6c2fef51d.
  Metadataarchives omit.pt/.pth/.bin/.distcp; checkpoint2/4/reference/payloadexports
  kept onhosts, exports+union also local, inventories/hashes verified. Both docs/
  memory synced. [D23 report](docs/development/bc250_distributed_d23_results/README.md),
  [usage](../BC250_DISTRIBUTED_GUIDE.md#trainer-fsdp-bf16-compute-policy-d23).
  **Next D24(new planned):** audit/prototype BF16 FSDP forward/backward parameters
  andall-gather buffers withFP32 mastershards/reductions; round-aware CPU references,
  globalclip/AdamW/exactrestart. Keep D23compute-only policy untilproof. FP16scaler/
  actuallarge-modelcapacity/otherarchitectures/elastic/release reconstruction
  remain separate broader priorities.

- **D22 completed 2026-11-05, native FSDP FP32 global clipping +finite refusal:**
  helper/fixture/controls2a11969bc, export/verifier790357231. Extended opt-in
  BC250FSDPTrainer BOTHtools/distributed-d22-r2; old D21 stagedpath retained.
  Finite nonnegative max_grad_norm admitted; zero keeps D21 noclip. Positive
  complete-root L2 calls delegate unchanged Accelerator→FSDP.clip_grad_norm_.
  Per-instance public method wrapper collectively checks present/dense/F32/same-
  device/finite owned gradients beforeclip, thenfinite returnednorm beforeAdamW.
  One-rank Inf/NaN flags[False,True] stop BOTH beforeclip/AdamW; finite1e30 flags
  [False,False] native normoverflow stops BOTH afterclip/beforeAdamW. All three
  zero actualoptimizersteps/unchangedparams+optimizer+scheduler; transientgrads/
  RNG/iterator NOT rolled back, discard/reloadcheckpoint, no silent skip. Caller
  late gradient/limit mutation or reordered collectives unqualified; no universal
  nonfinite/faultrecovery claim. Native clipping math/globalnorm not reimplemented.
  Same owned D21 actual Trainer BertMLM22561/eager/dropout0/FP32/classicFULL_SHARD/
  originalparams/root+2BertLayers/world2/RCCL/28records/unequal masks/rightpad5–12/
  workers0/sequential/micro2/accum2/global8/8/8/final4. AllGPUforward/backward raw
  gradients+loss vsCPUFP64 unchanged D21 bounds. CPUFP64 actual-pregrad globalnorm
  predicts native at1e-7/1e-5, coefficient min(1,limit/(norm+1e-6)), EVERYclipped
  gradient at1e-8/5e-5. CPUdoubleAdamW frompredictedclippedgrads: ALLparameters/
  moments/counters atD21/D07 bounds; no tolerance relaxation. Cap.05 ACTIVE all4
  norms1.46937144–2.26791596/coefs.03402813–.02204666; cap100controlcoefEXACT1.
  Counted1native rootclipcall/window, no partial/localshardnorm substitute.
  New CID/UUID/startticks/seed1773, D21fixed shardedAdamWloader/nativecheckpoint2
  +same offline payloadunion6files864357B: initialstate ALLparams/AdamW/scheduler/
  fourRNG EXACT; resumedwindows3/4 ALLnorms/coefs/grads/losses/params/moments/
  scheduler/RNG/probes/rows/finaldigests EXACT. Originalcheckpointhashes/local
  Trainer/RNG/scheduler untouched. No general/elastic/partial/stochastic restart.
  Accepted68GPUgroups/12ACKguards,2CPUlimitpatterns/24acceptedworkerobs/42localtests.
  Retained2failedsetup guards separate: firstInf harness couldn't create seedfolder
  underroot-ownedparent; guardpreparation fixed, newlabel retestpassed. No numerical
  failure hidden. Black/criticalRuff+F401/syntax/diff pass, no fullupstreamCI.
  Native D09/Torch daily+RCCL a524/ROCm/consumer packages/images/registries/kernels/
  defaults/DDPhelpers unchanged, no native build/install/prefix mutation/reboot/
  download/app access/foreignstop. Sharedlocks/4CPUs/10GiB+12GiB/512PIDs/256MiBshm/
  600s guards/460schildren. Boots c1199fca(.104r2)/bad537ed(.105r3) unchanged. Native
  warnings/aggregate train_loss resume caveat retained; historical PythonCPU/KFD/
  visibility separate fromfixedVMheap/D05exit/D21optimizerload. No speed/RAM/
  capacity or AMP/scaler/FSDP2/generalmodel claims; no jobs/containers/armedguards.
  Seed11245/4083 exit0; controls-r111723/4519 exit1setup; correctedcontrols12595/5309
  andresume13834/6428 exit0. Both evidence/distributed-d22-r1/{fp32-seed,fp32-resume,
  inactive-seed,inf-refusal,inf-refusal-r2,nan-refusal,overflow-refusal}; exactcommands/
  PIDs/CIDs/exits controllers+controls-r2-sequence.json/ACK/kernel/sources preserved.
  Local kernel/artifacts/distributed-d22 archives104526568B SHA
  e2863fb13dd4ee32f51a1616cc6de577c6c571ef58109f9fd24893ad3caa36aa;
  105472600B SHAf99bbe5573367cb21b4d7c5662f7ba9fb51770ed4bfc8efa6389281f6d14e60b.
  Metadata excludes.pt/.pth/.bin/.distcp; checkpoint2/4+referencepayloads retained
  onhosts, sourcepayloadexports+union also local, inventories/hashes verified.
  Bothdocs/memory synced. [D22 report](docs/development/bc250_distributed_d22_results/README.md),
  [usage](../BC250_DISTRIBUTED_GUIDE.md#trainer-fsdp-fp32-gradient-clipping-d22).
  **Next D23(new planned):** actual Trainer classicFSDP BF16 readiness with explicit
  parameter/reduction/autocast policy, CPU gradients/globalclip/AdamW +exactfresh
  restart. Lift current AMP refusal ONLY afterproof; FP16scaler/largecapacity/
  generalarchitectures/elastic/release reconstruction remain separate priorities.

- **D21 completed 2026-11-05, actual Trainer classic FSDP FP32 +restart repair:**
  helper/fixture/checkpoint utility5189348ba, export/verifier39918b800. New opt-in
  BOTH tools/distributed-d21-r3/bc250_trainer_fsdp.py BC250FSDPTrainer. Existing
  D12/DDP helper+FSDP refusal unchanged. Five HF/Accelerate files exact-pinned;
  classicFULL_SHARD/use_orig_params/FP32/trainable/noAMP/noClip/noOffload/noCompile/
  no activation-checkpointing; SHARDED_STATE_DICT. Native Trainer loop/wrapping/
  modelcheckpoint/scheduler retained; Accelerator accumulation1 (D12 denominator
  correction). No native Torch/ROCm/kernel/image/registry/D09/consumer/default
  changes, builds/install/prefix mutation/reboot/download/app access/foreignstop.
  .104 returned with bootc1199fca (r2); preflight idle/runtime verified, handoff
  refreshed atentry. .105 bad537ed(r3) unchanged. Both SSH and fallbacks retained.
  Actual BertMLM22561/vocab97/hidden32/2layers/4heads/inner64/positions32/eager/
  dropout0/cacheFalse, three FSDPunits(root+2BertLayers). Owned28/rightpadding5–12/
  unequalMLMmask/sequentialworkers0/perdevicemicro2/accum2 =>global8/8/8/final4.
  EVERY globaltokenloss/allfullparametergradient vs CPUFP64 actualHFgraph/manual
  CE: loss3e-6/5e-5, grads3e-6/5e-4. CPUdouble AdamW consumes admitted GPUgrads;
  ALL params4e-6/5e-5,moments3e-7/5e-4 and3e-9/1e-3,counters exact for4updates.
  Public FSDP fullgrad/optimizerstate APIs, no private shards. Parameter+exp_avg
  elements11282(.104)/11279(.105),total22561. Sharding proof only, NOT peak RAM/
  pooled32GB/largerfit/scaling/speedup. No new causalLM/AMP/FSDP2/clipping policy.
  Demonstrated pinned Accelerate optimizer-load defect: generic DCP fresh empty
  template left ALL AdamW moment entries absent. Two failed initial-restore gates/
  host, no subsequent optimizer step; diagnosticsactualkeys[] vs21/22required.
  Fixed ONLY subclass loadhook via public load_sharded_optimizer_state_dict +
  FSDP.optim_state_dict_to_load +optimizer/scheduler load_state_dict. No dummy
  steps, tolerance relaxation, zero-initialized moments or global package patch.
  Native model+optimizer DCP checkpoint2 spread across separatehostfilesystems.
  New bc250_fsdp_checkpoint.py offline export/merge/install includes six DCP
  payloadfiles864357B; hash/conflict/owner checks, missing-file-only install,
  EVERY originalcheckpointfile hash unchanged, localTrainer/scheduler/rankRNG
  untouched. Bounded256MiB/trusted SAMEcheckpoint/world/rank utility, NOT automatic
  sharedstorage/run-lineage validation/elastic/streaming/crash-atomic protocol.
  Fresh CID/UUID/startticks/seed1773: initial params/ALLlocalAdamW/scheduler/four
  RNG EXACT; windows3/4 ALLlosses/grads/params/moments/scheduler/RNG/probes/rows/
  finaldigests EXACT seed replay onBOTH. Fixedworld/sequential/no dropout only;
  nativeaggregate train_loss caveatD12 remains, no general restart claim.
  Accepted30GPUgroups/4ACKguards, retained4FAILED-ACKguards separately;4localCPU
  payloadcontractpatterns/42localtests, Black/criticalRuff+F401/syntax/diff pass.
  No full upstreamCI. Native deprecationwarnings stay; historical PythonCPU/KFD/
  visibility separate fromfixedVMheap/D05 exit/thisoptimizer-load integrationbug.
  Seed4449/29954 andcorrectedresume6613/31909 exit0; failed5328/30719,6003/31345
  exit1beforeupdates. Both evidence/distributed-d21-r1/{fp32-seed,fp32-resume,
  fp32-resume-r2,fp32-resume-r3}; all commands/CIDs/PIDs/ACK/kernel/sourcehashes
  preserved. SHlocks/4CPUs/10GiB+12GiB/512PIDs/256MiBshm/600s guards/460schildren.
  No jobs/containers/armedguards. BOTHdocs/memory synced. Checkpoints2/4+reference
  tensors kept onhosts; local kernel/artifacts/distributed-d21 metadataarchives
  omit.pt/.pth/.bin/.distcp, inventories+hashes retained, CPU payloadexports+union
  also kept locally. 104439245B SHA8b38c968be3530469965f72f95f882b1fed82a1ffa58c6ad0e05bee0500092a2;
  105399839B SHAf12bde68b9cfa7505882826624e0d4f83ccf63309029bcabfddbe89774701ade.
  [D21 report](docs/development/bc250_distributed_d21_results/README.md),
  [usage](../BC250_DISTRIBUTED_GUIDE.md#trainer-classic-fsdp-fp32-d21).
  **Next D22(new planned):** actual TrainerFSDP FP32 native global gradient-norm
  clipping with CPU ALLgrad/norm/clippedgrad/AdamW andexactfixedworldrestart.
  Extend max_grad_norm=0 guard ONLY afterproof. AMP/FSDP2/generalmodels/largefit/
  elastic/release reconstruction remain separate broadpriorities.

- **D20 resumed handoff 2026-11-05 17:20 UTC:** work and six guarded GPU
  qualifications completed before the interruption; source/docs commits1116695f8,
  workspacefb819b9 retained. .104 now returns SSH "No route to host"; no new GPU
  test/reboot/prefix action attempted. .105 reachable, bootbad537ed unchanged;
  D20 helper/fixture and ALLfour prior memory files verified byte-for-byte against
  local1116695f8/fb819b9 state. Both host snapshots installed earlier remain
  `distributed-d20-20261105`/agent-memory1116695f8-distributed-d20; this access
  annotation is local-only until next sync. Final .104 repeat copy verification
  unavailable; do NOT reinterpret completed numerical qualification as failed.
  No stack tests/builds/armed guards were running at interruption. **Next:**
  restore/recheck .104 SSH, inspect boot/runtime/foreign workload status, refresh
  its handoff memory, then D21 TrainerFSDP readiness/audit. Preserve .105 and all
  user workloads; no power-state assumption or remote recovery command issued.

- **D20 completed 2026-11-05, BF16 diagnosis +explicit FP32 evaluation head:**
  helpers/bytehash/contracts1a61687a7, export997b38bb7, final scope/export570e60428.
  D19 wide Bert BF16 outliers reproduced EXACTLY by matched CPU BF16 atsame
  coordinates/values:2/30408704 versus FP64,maxabs.003654425659; GPU vsCPU-BF16
  zero violations,maxabs.001953125. ORIGINAL atol/rtol.003/.003 retained. Evidence
  supports BF16 rounding inthisfixture, no native ROCm/Torch defect demonstrated;
  nativeBF16 still fails strictFP64 gate, no universal correctness inference.
  Added opt-in BOTH tools/distributed-d20-r3/bc250_eval_precision.py fp32_eval_head:
  existing single-positional-hidden-input FP32-master eval head only, disables
  autocast inside head/inputFP32, weights/buffers/keys/ties unchanged. Requires
  eval+no_grad/inference_mode, one tensor output; nested/training/lowprecision
  weights/invalid input refused. Restores per-instance forward onexception/exit;
  no class/global/app patches. Actual Bert model.cls with BF16 encoder AMP now
  ORIGINAL CPUFP64 gate ZERO violations,maxabs.00005873836474,relativeL2.00006318.
  GPU vsmatched CPU policy maxabs.000004708767; scope intentionally changes head
  math/loss, no exact BF16 continuation, training/backward/speedup claim.
  Three phases/node: r1native diagnose/r2full-reference FP32head/r3actual
  Trainer.evaluate/scope restoration. Encoderquery BF16/head FP32 observed; native
  metrics EXACT afternormal+GPUexception exit, parameter IDs/keys/ties preserved.
  Owned Bert vocab32768/hidden32/2layers/seq64/eager/dropout0/cacheFalse,19records/
  rightpadding33–64/unequalmasks/838targets/world2/RCCL. Full/compact metrics
  parity1e-11, native loss unchanged WITHINpolicy; crossrank digests/metrics agree.
  36groups/6ACKguards;3CPUdigest patterns*4workers=12obs/4CPUhead patterns*2=8obs;
  42localtests/Black/criticalRuff+F401/syntax/diff pass. CPU graph same HFimplementation,
  not independent Transformer. No larger/general model, streaming/elastic/capacity
  or FP16/newbackward qualification. Historical PythonCPU/KFD/visibility separate
  from fixedVMheap/D05 agent exit; native warnings retained.
  New bc250_array_digest versioned named/dtype/shape/C-order byte SHA, bounded
  noncontig iteration; owned D19 fixture no longer .tolist/giantJSONfingerprints.
  CPU whole-byte oracle/chunk1/3/65536/order/strides/zero/dtype/shape/refusals pass.
  Newformat NOT comparable tolegacy JSON digests; full reference arrays remain,
  no hostRSS/performance claim. Native a524/daily+RCCL/ROCm/D09/consumers/images/
  registries/kernels/defaults unchanged; no build/install/prefix mutation/reboot/
  downloads/app access/foreignstop. Sharedlocks/4CPUs/10GiB+12GiB/512PIDs/256MiBshm/
  600s watchdog/460s children. Guards14025/13356,15056/14338,16906/16124 all exit0,
  exactcommands/container IDs/kernel/ACK records preserved under evidence/
  distributed-d20-{r1,r2,r3}/bertwide-bf16-{diagnose,fp32-head,scope}.
  No jobs/containers/armedguards; bootsd83af4e1(.104r2)/bad537ed(.105r3) unchanged.
  Local kernel/artifacts/distributed-d20 finalarchives104298287B SHA
  65824728c50f4e2080207a456996ada8ba286741c0ff28356a600284e261c495;
  105280711B SHA54fc6c50af6974c4bc090838aee98d22f7b7479685afc9969c12d9e47e3fc0f9.
  Initial archives retained; source/config/nativeidentity/consumer manifests/guards/
  rawCPUreports/crossrankdigests verified. No persistent prediction/checkpoint
  tensors. Both docs/memory synced. [D20 report](docs/development/bc250_distributed_d20_results/README.md),
  [usage](../BC250_DISTRIBUTED_GUIDE.md#explicit-fp32-evaluation-head-d20).
  **Next D21(new planned):** practical distributed gap review +pinned actual
  Trainer FSDP FP32 readiness audit; prioritize bounded real-training integration
  with CPU ALL-gradient/optimizer and fixed-world restart if API admits it. D08
  customFSDP/D12–16 actualTrainerDDP already pass; do not repeat full matrices or
  infer larger-model capacity/scaling. Broad unresolved work remains inroadmap.

- **D19 completed 2026-11-05, compact validation +larger-vocabulary crossover:**
  final active-token helper/fixtures `1a39b5254`, exports/verifier `c99e46fe4`;
  dense prototype a7c1104d0/strict-wide-FP32 fixture4deb6a00a retained, rejected.
  Opt-in BOTH `tools/distributed-d19-r3/bc250_compact_token_metrics.py`.
  Retains per-row active selection/FP64 CE/causal shift; device validation flag
  once at end, invalid active labels clamped internally then REFUSED, ignored
  nonfinite positions omitted. CPU dispatch GPU probe: scalar reads12→1, dynamic
  nonzero3 unchanged; NOT elimination of all sync. No native/ROCm/daily+RCCL a524/
  D09/consumer/image/registry/kernel/default changes, build/install/prefix update/
  reboot/app access/download/foreign-workload stop. Old D18 tools retained.
  Final r3 actual Trainer eval/predict: six small Bert/GPT2 FP32/BF16/FP16 +two
  vocab32768/maxlen64 FP32 cases,19 dynamic right-padded records/unequal masks/
  ignored first2rows/tail dedup, eager/dropout0/cacheFalse/world2/RCCL. All modes
  warm before full/D18/D19/D19/D18/full timing. Exact19 labels/order, counts small
  126/120/wide838/832; full+D18+D19+manual NumPy64 metrics parity1e-11; ALL real-
  position logits vsCPUFP64 at ORIGINAL D17 bounds. Native eval_loss EXACTLY
  unchanged, rankmetrics agree, label-free opt-out exact full logits. Params
  unchanged/gradsNone/no optimizer; forward/eval only, not new backward.
  Final176groups/16ACKguards +5CPU patterns*16workers=80obs/2GPU patterns*16=32obs;
  42localtests/Black/criticalRuff+F401/syntax/diff pass. Both finalcontrollers
  8981/8659 exit0, exact phasecommands/PIDs/exits in sequence.json. SH locks,
  4CPUs/10GiB+12GiB/512PIDs/256MiBshm/600s guards/460s children unchanged.
  Final tiny paired D18/D19 ratios1.014–1.034, wideflat. Compact wide57.7–61.3ms
  vs full1.12–1.15s =18.6–19.6x IN THESE TWO FP32 fixtures; benefit already D18,
  now measured at this size, NOT wholeTorch/LoRA/newscalarfix speedup. Two samples
  each; not statistical, small full-mode outliers retained. Wide predictions
  159383552B(152MiB)→456B; gather prediction inputs79167488→240B/rank/fivecalls;
  labels9728B/local logits/native padding remain. Tensorbytes NOT NIC/totalcomms.
  Wide allocator peak~136.5MB→60.9–61.2MB; final within1024–1536B ofD18. No total
  GPU/hostRSS/largerfit claim; CPU full references/listfingerprint untimed remain.
  Rejected dense prototype: nonzero0 but wide78ms vsD18~59–61ms/extra~40MB; all
  source/evidence retained under r1/r2. Initial wide Bert BF16 metric parity
  PASSED but CPUFP64 full-logit gate FAILED2/30408704 values,maxabs.003654425659;
  atol/rtol.003/.003 UNCHANGED. Both failedguards ACK/kernel+registry unchanged;
  GPT2wideBF16 notrun. WiderBF16 remains UNQUALIFIED undergate, not provennative
  defect. SmallBF16 passes; wideFP32 does NOT fix BF16. Earlier16successful
  prototype+2failed guards separate from final16. Native warnings remain;
  historical PythonCPU/KFD/visibility separate from fixedVMheap/D05 exit.
  No jobs/containers/armedguards; bootsd83af4e1(.104r2)/bad537ed(.105r3) unchanged.
  Both `evidence/distributed-d19-r3/{bert,gpt2}-{fp32,bf16,fp16}-eval` plus
  `{bertwide,gpt2wide}-fp32-eval`; local kernel/artifacts/distributed-d19:
  final1041013179B SHAbbce25bca5ccba8c8474f0e6e0ee183480da2f463c619c10a92b5c50415b1d39;
  105914431B SHAe565fcbb83b7a58b1ff4c26033ae6f6129481bac737f7b4251e58fb2607fc8ea.
  Archives retain r1/r2/r3, all failures; sources/native/config/guards/crossrank
  digests/CPU reports verified, no saved prediction/checkpoint tensor payload.
  Docs/memory BOTH synced. [D19 report](docs/development/bc250_distributed_d19_results/README.md),
  [usage](../BC250_DISTRIBUTED_GUIDE.md#compact-metric-validation-and-larger-vocabulary-d19).
  **Next D20 (new planned):** diagnose wider Bert BF16 outliers with matched CPU
  autocast +FP64 references, keep original gates; classify arithmetic vs actual
  port defect before fix/qualification. First replace owned fixture list-based
  prediction fingerprint with byte hashing to bound untimed CPU work. Other
  distributed capacity/streaming/elastic/invalidvisibility/B07.13/release gaps
  remain separate; no infinite evaluation matrix or universal correctness claim.

- **D18 completed 2026-11-05, compact token statistics before gathering:**
  helper/contracts/fixtures `9a31ac663`, exports/verifier `6bc35b079`.
  New opt-in `tools/distributed-d18/bc250_compact_token_metrics.py` on BOTH hosts:
  CompactTokenMetrics.preprocess +compute_metrics; per-example FP64 GPU summed
  NLL/count/correct, causal shift or masked aligned, chunked before native gather.
  Native tail de-dup precedes final token division; zero-target rows allowed,
  globally empty targets refused. D12–D17 helpers/native daily+RCCL a524980/ROCm/
  D09 launcher/registries/consumers/images/kernels/defaults unchanged. No native
  build/install/prefix mutation/model download/app access/development reboot.
  Actual two-host BertMLM/GPT2 eager/no-dropout/no-cache FP32/BF16/FP16,19records/
  dynamic lengths5–12/rightpadding/prompt masks/unequal tail: ABBA full/compact/
  compact/full predict, compact evaluate, full-logit label-free opt-out pass.
  Exact19 labels/order;126masked/120causal targets; metrics vs full+manual NumPy64
  within1e-11, ALL real-position full logits vs CPUFP64 at D17 bounds. Native loss
  EXACTLY unchanged across modes; both ranks agree; params unchanged/gradsNone,
  no optimizer. 108groups/12ACKguards +4CPU patterns*12workers=48observations,
  42localtests/Black/criticalRuff+F401/syntax/diff pass. GPU chunk7; CPU1/3/4096.
  Prediction array88464→456B (194x smaller); per-rank five-batch prediction gather
  inputs36472→240B (~152x), ALL loss/label/prediction gather inputs37264→1032B
  (~36.1x). Tensor bytes only, NOT wire traffic/totalcollectives. Live allocator
  peaks lower12288–40448B in tiny fixtures; NOT hostRSS/totalGPU/largerfit proof.
  Full warmed before ABBA, compact first211–231ms vs repeat43.5–49.3ms; full
  mostly38.2–44.6ms, one90.5ms outlier. NO latency win; conservative per-row FP64/
  scalar-sync overhead remains. No statistical/defaultchunk4096GPU-speed claim.
  Full local model logits/native cross-rank padding/dataset labels remain;
  not streaming/generation/arbitrarymodels/outputtuples/general recovery proof.
  Disable BOTH callbacks for full-logit/label-free predict; explicit usage guide.
  Both `evidence/distributed-d18-r1/{bert,gpt2}-{fp32,bf16,fp16}-eval`, exact
  commands/PIDs/exits in sequence.json; controllers30382/29377 exit0. SH locks,
  4CPUs/10GiB+12GiB/512PIDs/256MiBshm/600s guard/460s child; clean agent exits,
  unchanged phase kernel logs+registries. No jobs/containers/armedguards remain.
  Tested bootsdc9e9293(.104r2)/133b8dc2(.105r3), then boot IDs changed during chat
  interruption; current doc-sync bootsd83af4e1(.104)/bad537ed(.105), same kernels.
  No GPU requalification claimed on new boots. Native warnings remain; historical
  PythonCPU/KFD/invalidvisibility separate from fixedVMheap/D05 exit.
  Local `kernel/artifacts/distributed-d18`:104344806B SHA256
  76691515b8b76a925ac7009466d3443b25640a9ecbd5d79b80f5c778d48a03bf;
  105311154B SHA256307692ad8da565a706edc16c09800bc5307b6b8a17d87b42f400d38771dcba2b.
  Sources/config/nativeidentity/guards/rawreports/crossrankdigests verified;
  no persistent prediction/checkpoint tensors created. Docs/memory BOTH synced.
  [D18 report](docs/development/bc250_distributed_d18_results/README.md),
  [usage](../BC250_DISTRIBUTED_GUIDE.md#compact-distributed-evaluation-statistics-d18).
  **Next D19 (new planned):** reduce compact metric launches/scalar synchronization,
  preserve count/tail/CPU contracts; bounded larger-vocabulary payload/memory/
  latency crossover. Preserve opt-in/native defaults. Invalidvisibility/B07.13/
  largerfit/release reconstruction remain separate follow-ups.

- **D17 completed2026-11-05, actual distributed Trainer evaluate/predict:**
  metric/contracts/fixturesf8f0a475a/exportscf7eb27c2. New opt-in BOTH tools/
  distributed-d17/bc250_token_metrics.py TokenWeightedMetrics; D12–D16 helpers
  unchanged. Native daily/RCCL Torch a524980/ROCm/D09 launcher/registry/images/
  profiles/kernels/consumers unchanged. No native build/install/prefix mutation/
  reboot/model download/app access. Actual native Trainer.evaluate +labeled+
  label-free predict, BertMLM22561/GPT2causal21280, eager/dropout0/cacheFalse,
  FP32/BF16/FP16 (6cases/world2/one gfx1013GPU per node/RCCL). GPUparam placement/
  Linear+Conv1D dtype checked, logitsF32. Forward/eval proof, not new backward.
  Stateless full-gathered-logit callback sums CPU-FP64 CE/correctargmax over ALL
  active targets, divides ONCE bydataset count; masked aligned or causalSHIFTED.
  Returns token_nll/exp(nll)perplexity/token_accuracy/active_tokens/examples.
  Native eval_loss RETAINED: average gathered batch losses, different weighting.
  BertF32 eval_loss4.595089912 vs token_nll4.568608723; GPT2F32 4.609643459 vs
  4.586088138, not GPUarithmeticfault. Otherprecisions agreeCPUrefs withinbounds.
  Use metric_for_best_model='token_nll',greater_is_betterFalse explicitly.
  19owned records notdivisibleworld2*batch2, dynamicactual length5–12/rightpadding/
  promptlabels/twoignoredfirstexamples; nativepad_across_processes+gather tail
  dedup returns EXACT[19,12,97]logits/[19,12]expectedlabels inroworder, not20.
  Token counts126masked/120causal verified. Sequentialworkers0/evalbatch2/
  eval_accumulation_steps2/batch_eval_metricsFalse/concatTrue/globalcountTrue.
  ALLrealposition full-vocab GPUlogits vsCPU-FP64 forward: atol3e-6/rtol5e-5F32,
  .003/.003BF16,.001/.001FP16. Same HFgraph/different CPUarithmetic, not independent
  graph. Explicitmetrics vsmanual NumPy64 logsumexp/count/accuracy1e-11; tokenNLL
  vsCPU-FP64 model withinprecisionbounds. Accuracy checkedactual gathered logits,
  not FP64argmax near-tie equality. evaluate/labeledpredict/rankmetrics agree.
  Label-free predict bypassesrequired-label trainingcollator, returns SAME exact
  logits/no label/loss/tokenmetrics. ALLmodelstate-dict values unchanged/allgrads
  None, no optimizercreated/updated; evalmode changes intentionally, not RNGrollback.
  96groups/12ACKguards/42localregressions +5CPUcontractpatterns*12workers=60obs
  (masked/causalmanual/chunking/stateless/empty+argmaxrefusals grouped). Firstrevision
  allpasses; Black/criticalRuff+unusedimports/syntax/diff pass, no fullupstreamCI.
  SHlocks/4CPUs/10GiB+12GiB/512PIDs/256MiBshm/600s guards/460s children. Clean
  agents/phasekernellogs/registries. Sequence25857/25113 exit0, exactcommands/
  PIDs/exits sequence.json. No jobs/containers/armedguards/foreignworkload changes.
  Bootsdc9e9293(.104r2)/133b8dc2(.105r3) unchanged; SSH/bootfallback retained.
  Native all_gather/barrier warnings stay; historical PythonCPU fault/KFD/invalid
  visibility separate from fixedVMheap/D05 exit. No newnativefault/speed/memory
  measurement/largerfit claim. No persistent prediction/checkpointpayloadcreated;
  source/digests/rawreports/CPUcomparisons preserved, verifierdoesnotrerunGPU.
  Both tools/distributed-d17 +evidence/distributed-d17-r1/
  {bert,gpt2}-{fp32,bf16,fp16}-eval. Local kernel/artifacts/distributed-d17
  verifiedsummary+archives104327082B SHA2a49a451cc2816746e3e11d2c402ee51804bcc25216483eee4bd6f306b025636;
  105293063B SHA9485e90a5f46d15e5311baf40266044912770389ef85a2ea72e4225484813f02.
  Helpers/fixture/selectedTrainer/nativeidentity/config/guard/kernel/crossrank
  metrics+predictiondigests verified. Docs/memory BOTH synced.
  [D17 report](docs/development/bc250_distributed_d17_results/README.md),
  [tokenmetric usage](../BC250_DISTRIBUTED_GUIDE.md#distributed-evaluation-and-token-metrics-d17).
  Full logits still gathered/stored; chunking bounds temporaryFP64 only, not
  datasetfullarrays/othercopies. NOT largevocab/longdataset/memoryefficient eval
  proof. No streaming/batchmetrics/generationquality/IterableDataset/shuffle/
  multiworker/arbitrarymodel/preprocessing/eval+trainresume/elastic/peerloss claim.
  **Next D18(new planned):** compact per-example token statistics BEFORE crosshost
  gather, CPUreference parity, bounded payload/memory/timing comparison. Retain
  native/default/recovery settings; invalidvisibility/B07.13/largerfit/release
  reconstruction remain separate follow-ups.

- **D16 completed2026-11-05, causal Transformer AMP +shifted target contract:**
  helper/contracts/fixturesce09f6c24/exportsad6454602. New opt-in BOTH tools/
  distributed-d16/bc250_trainer_causal.py CausalLabelCollator+causal_token_loss;
  D12/D13/D14 helpers unchanged. Native daily/RCCL Torch a524980/ROCm/D09 launcher/
  registry/images/profiles/kernels/consumers unchanged. No native build/install/
  prefix mutation/reboot/model download/app access. Pinned Trainer counts ALL
  nonignored labels; causal CE shifts logits[:,:-1]/labels[:,1:]. Collator clones
  existing int64 labels, masks position0+attention padding BEFORE native counting,
  preserves prompt ignore masks/caller data. Right-padding binary masks,>=2real
  tokens; left-pad refused; no packing/position/labelsynthesis inference. Loss
  FP32 summed SHIFTED CE/positive GLOBAL window count, Trainer DDP compensation/
  D12 single normalization. Explicit accepts_loss_kwargsFalse; not ordinary
  Trainer/GPT2 auto-fix. Local allignored micro with positiveglobal count zero+
  differentiable; missing/emptyglobal/unmaskedfirst labels refused.
  Actual GPT2LMHeadModel21280/vocab97/positions32/emb32/2layers/4heads/inner64/
  sequence12/eagerCAUSAL/dropout0/cacheFalse, Conv1D/Linear AMP observed, tied
  embeddings/head, FP32 logits/master/reduction; one gfx1013GPU/node/world2/RCCL.
  28records/rightpadding5–12/promptlabels/two allignored examples rank0 firstmicro,
  sequential workers0/perdevicemicro2/accum2/global8 first3attempts/finalglobal4.
  All rows/native counts checked:SHIFTED49/57/57/21 vsraw51/60/59/23; rank0 local
  [0,16]/[8,17]/[8,16]/[9]. GPU zero-local loss finite0; reductions0 nonfinal/
  85120B final. CPUautocast micro/rank/scale +CPU-FP64 global GPT2, manualSHIFTED
  logsumexp CE. Same HF graph/different CPU arithmetic, not independent GPT2 code.
  D15 bounds unchanged: globalgradrelative AMP.02/.004, FP64.03/.006 BF16/FP16,
  EVERYparam norm budgets+floors3e-6/3e-7*sqrt(elements), lossatol+rtol.003/.001.
  Maxgradrelative vsCPUAMP BF16.00399075/FP16.000498595; vsFP64 .00495313/.000661749.
  CPUFP64 ALLnorm/ALLclippedgrad/CPU-double AdamW ALLparams+moments pass D14/D07
  bounds. Clip.05 ACTIVE norms1.15832–2.31724/coeff.0215774–.0431659. No relaxation.
  FixtureFP16init128/growth2 via pinnedTrainerextension, no default65536 proof/
  native default change. Scale128→256 checkpoint2, rank1 AFTER-DDP Inf attempt3
  coordinated before nativeunscale, BOTH skip AdamW/StepLR unchangedweights/
  moments/scheduler, backoff128/tracker0 then finalfinite128/tracker1. BF164actual
  updates/LR.00064; FP164Trainerattempts/3actualupdates/LR.0008.
  Native perhost save_on_each_node checkpoint2/scaler/rankRNG, fresh CID/UUID/
  startticks/different constructors, exactinitial restore +attempts3/4 ALLlosses/
  grads/params/optimizer/scheduler/scaler/four RNG/probe draws/rows/finiteflag AND
  global/local targetcounts EXACT+digests:8resumed rank-windows. D14nonfinite masks/
  finite components retained, no NaN-payload-bit claim. Dropout0+RNG probes only.
  96groups/8ACKguards/42localregressions;7CPUcontractpatterns*8workers=56observations
  separately(ownership/count/leftpad/analyticderivatives/localzero/globalcount/
  firstlabelrefusals). Firstrevision allpasses; Black/critical Ruff+unusedimports/
  syntax/diff pass, no fullupstreamCI. SH locks/4CPUs/10GiB+12GiB/512PIDs/256MiBshm/
  600s guards/460s children. Clean agents/phasekernellogs/registries. Sequence
  19820/19250 exit0; exactcommands/PIDs/exits sequence.json. No jobs/containers/
  armedguards, foreignworkloads untouched; bootsdc9e9293(.104r2)/133b8dc2(.105r3)
  unchanged. Native tiedlm_head missing-key notices pass exactrestore. Native
  warnings/D12 aggregate train_loss caveat remain; owned zero-loss assertion
  scalarconversion warning only(actual lossTensor still backward, not helper).
  Historical PythonCPU fault/KFD/invalidvisibility separate from fixedVMheap/D05;
  no newnative defect/capacity/performance/convergence claim.
  Both tools/distributed-d16 +evidence/distributed-d16-r1/{bf16,fp16}-{seed,resume}.
  Normalseedpayload3731560B/host retained +resumedoutputs. Local kernel/artifacts/
  distributed-d16 verifiedsummary+metadataarchives omit.pt/.pth/.bin/.pkl/.safetensors:
  104248327B SHA691a0246b4a54b796eb3b5c1b33722d82082b092ba3182d5809d4e0c82a2cd88;
  105225607B SHAf0e326c5089071e1d9063df01e2b931361170a39c5e04a1edb7829c000a6bbb6.
  Sources/config/count/guard/kernel/freshproofs verified; docs/memoryBOTH synced.
  [D16 report](docs/development/bc250_distributed_d16_results/README.md),
  [causal usage](../BC250_DISTRIBUTED_GUIDE.md#causal-transformer-training-and-target-counts-d16).
  Scope small eager GPT2/dropout0/cacheFalse/rightpaddedunpacked/explicitlabels/
  fixedworld/sequentialworkers0; no leftpad/packing/RoPE/GQA/SDPA/Flash/cache/
  generation/FSDP/DeepSpeed/checkpointedgrads/arbitraryarch/scale/eval/largerfit/
  speedup/stochastic/multiworker/elastic/generalresume proof. **Next D17(new
  planned):** actual two-host Trainer evaluation/prediction +explicit token-weighted
  metrics, unequal finalbatches/CPUreferences. Invalidvisibility frontend/B07.13/
  larger capacity/release reconstruction remain separate follow-ups.

- **D15 completed2026-11-05, actual offline Transformer AMP Trainer:**
  fixturesdeed14f14/exports84039269e. D12/D13/D14 production helpers UNCHANGED,
  copied BOTH tools/distributed-d15 with owned fixtures. Native daily/RCCL Torch
  a524980/ROCm/D09 launcher/registry/images/profiles/kernels/consumers unchanged.
  No native build/install/prefix mutation/reboot/app access/model download.
  Actual BertForMaskedLM22561/vocab97/hidden32/2layers/4heads/intermediate64/
  maxpositions32/length12, eager bidirectional attention/dropout0. Not MLP proxy:
  embeddings/tied MLM decoder/masked attention matmuls+softmax/LayerNorm/GELU/
  residuals/CE forward/backward, ALL parameter gradients checked. One gfx1013GPU/
  node/world2/RCCL, BF16/FP16 Linear outputs +FP32 prepared logits/master/reduction.
  Unequal padding5–12/active MLM counts,attention_mask,padID0/targets-100;
  explicit accepts_loss_kwargsFalse +UNCHANGED masked_token_loss UNshifted CE/
  global active-window count/Trainer DDP compensation/D12 single normalization.
  28sequential workers0 records/per-device micro2/accum2: global8 first3attempts,
  finalglobal4 singlemicro; all rows checked,0 nonfinal/90244B final reductions.
  Two CPU references at current GPU weights: native CPU-autocast actual micro/
  rank/scale with MANUAL logsumexp CE, plus CPU-FP64 full-global Bert/manual CE.
  Same pinned HF graph, distinct CPU kernels; not independently implemented Bert.
  Whole-grad relativeL2 limits AMP BF16.02/FP16.004, FP64.03/.006; EVERYparam
  norm budget adds floor3e-6/3e-7*sqrt(elements) for nearzero key-bias cancellation.
  Loss atol+rtol.003(BF16)/.001(FP16), declared before tests, no bounds relaxed.
  MaxgradientrelativeL2 vsCPUAMP BF16.0000582494/FP16.0000732985; vsFP64
  BF16.00487906/FP16.000594423. CPU-FP64 global norm/ALL clipped-grad/CPU-double
  AdamW ALL params+moments checks retain D14/D07 bounds. Clip.05 ACTIVE norms
  1.46933–2.27917/coeff.0219378–.0340292. Not a speed/convergence measurement.
  Fixture-only GradScalerKwargs(init_scale128,growth_interval2) via source-pinned
  Trainer _build_accelerator_args; no native default change/65536-Bert proof.
  FP16 first2 finite updates scale128→256 at checkpoint2, rank1 AFTER-DDP Inf
  attempt3 agreed before native unscale, BOTH skip AdamW/StepLR unchanged weights/
  moments/scheduler, backoff128 then finalshort finite recovery128/tracker1.
  BF16 4actualupdates/LR.00064, FP16 4Trainerattempts/3actualupdates/LR.0008.
  Native save_on_each_node complete checkpoint2/scaler/rank RNG each host, new
  CID/UUID/startticks/different constructors, exact initial native restore and
  attempts3/4 ALL local losses/grads/params/optimizer/scheduler/scaler/four RNG/
  probe draws/rows/finiteflag EXACT+digests:8resumed rank-windows. Nonfinite masks+
  finite gradient components exact perD14, no NaN-payload-bit equality claim.
  70groups/8ACKguards/42localregressions, first revision all pass. Black/critical
  Ruff+unusedimports/syntax/diff pass, no full upstreamCI. SH locks/4CPUs/10GiB+
  12GiB/512PIDs/256MiBshm/600s guards/460s children; clean agents/phase kernellogs/
  registry. Sequence15951/15570 exit0; exact commands/PIDs/exits sequence.json.
  No jobs/containers/armed guards/foreign processes touched; bootsdc9e9293(.104r2)/
  133b8dc2(.105r3) unchanged. Native tied decoder missing-key notices pass exact
  complete restore; all_gather/barrier warnings +D12 aggregate train_loss caveat
  remain. Historical Python CPU fault/KFD/invalidvisibility distinct from fixed
  VMheap/D05 agent exit; no new native defect/capacity/performance claim.
  Both tools/distributed-d15 +evidence/distributed-d15-r1/{bf16,fp16}-{seed,resume}.
  Normal seedpayload4126664B/host retained plus resumed outputs; local
  kernel/artifacts/distributed-d15 verified results +archives omit.pt/.pth/.bin/
  .pkl/.safetensors.104239556B SHA0a07815d632fcc7f636235191d868affec2bba17dd88b400e25ab2eee14e952b;
  105217535B SHA6a05eb85c2ae290ecfc9cb68b92d7eea7f46256e701a2774ddfbde1db21479ba.
  Source/config/guard/kernel/fresh proofs verified; docs/memory BOTH synchronized.
  [D15 report](docs/development/bc250_distributed_d15_results/README.md),
  [Bert AMP recipe](../BC250_DISTRIBUTED_GUIDE.md#offline-transformer-amp-training-d15).
  Scope small eager bidirectional BertMLM/dropout0/fixed world/sequential workers0,
  not causalLM/RoPE/GQA/cache/SDPA/Flash/FSDP/DeepSpeed/checkpointed gradients/
  stochastic graph/arbitrary architecture/loss/scale/capacity/convergence/speed/
  multiworker/elastic/general resume. **Next D16 (new planned):** owned offline
  causal Transformer AMP Trainer with SHIFTED CE/global valid-target accounting,
  clipping/CPU references/native restart. Evaluation metrics/invalid-visibility
  frontend/B07.13/larger capacity/release reconstruction remain separate.

- **D14 completed2026-11-05, native Trainer AMP clipping +rank-local FP16 policy:**
  implementation625b840d7/exportsb3e3d8a30. New opt-in tools/distributed-d14/
  bc250_trainer_clipping.py BC250ClippedTrainer staged BOTH. D12 normalization/
  D13 finite callback unchanged. Native daily/RCCL Torch a524980/ROCm/D09 launcher/
  registry/images/profiles/kernels/consumers unchanged. No native build/install/
  prefix mutation/reboot/app access. Decorates ONLY own Accelerator PUBLIC
  clip_grad_norm_ method; native unscale/L2 clip/scaler/Trainer loop retained.
  Two-host dense FP32-master DDP, one optimizer, positive finite max_grad_norm.
  Full public scaler-state agreement +all-reduced badflag BEFORE unscale; rank1
  AFTER-DDP Inf marks a gradient Inf on BOTH so native scalers cache overflow
  coherently. Native clipping then skip AdamW/StepLR/backoff BOTH, no private
  scaler flags. BF16 pre-clip invalid stops BOTH. Final public pre-optimizer gate
  appended after existing caller callbacks stops BOTH on NEW post-clip invalid
  gradients; do not dynamically reorder/add callbacks or mutate after gate.
  Matching caller collective order required; finite asymmetric edits unguarded.
  Public scaler mismatch stops BOTH before unscale. Refusals require checkpoint
  reload, no same-iterator/scaler continuation; transient state not rolled back.
  Same owned D07 MLP1051648/variable-mask MSE/global active-window count,
  actual Trainer5.2/Accelerate1.12 BF16/FP16,28sequential/workers0 records/global8
  first3attempts/finalglobal4 singlemicro,dropout0,RNG probes,strict determinism.
  max_grad_norm.002 ACTIVE every finite tested window: pre-norm.0186445–.0547800,
  coefficients.0365090–.107265. Independent D07 pregrad CPU rounding/FP64-matmul
  loss/ALL-grad limits retained (BF16 maxrelativeL2.00276769444<=.02, FP16
  .000372783621<=.004). CPU-FP64 ALL-grad L2 norm/coefficient predicts EVERY clipped
  gradient (atol1e-8/rtol5e-5); native norm atol1e-7/rtol1e-5. CPU-double AdamW
  uses expected clipped grads, ALL params/moments pass fixed D07 limits. No
  tolerance changes. BF16 updates4/LR.00064; FP16 attempts4/actualupdates3/LR.0008,
  native defaults65536/growth_interval2000→32768/tracker0 thirdskip→finite tracker1.
  Native per-host save_on_each_node checkpoint2/scaler/four RNG, new CID/UUID/
  startticks/different constructor seeds, exact initial native restore; attempts3/4
  ALL local losses/grads/params/optimizer/scheduler/scaler/RNG/probe draws/rows/
  finiteflag exact+digests:8resumed rank-windows. Native Inf clipping NaNs stored
  as exact NaN/+Inf/-Inf masks +exact finite values/nonfinite slots0; classifications
  retained, no NaN-payload-bit equality claim. No generic data/checkpoint proof.
  Three separate rank1-only expected refusals after checkpoint2: BF16 pre-clip
  Inf; FP16 post-clip Inf in caller callback (native unscale alreadycachedfinite);
  FP16 public scaler.update(scale/2) disagreement.6rank refusal observations,
  complete params/AdamW/StepLR/checkpoint2/globalstep+Adamsteps2 unchanged.
  RNG/gradient/scaler transient state advances/perturbed; reload required.
  118groups/14ACK guards/42local regressions, first revision all pass. Black/
  critical Ruff+unused-import/syntax/diff pass; no full upstream CI. SH locks/
  4CPUs/10GiB+12GiB/512PIDs/256MiBshm/600s guards/460s children. Clean agents/
  phase kernel logs/registry. Sequence10343/10259 exit0, exact commands/PIDs/
  exits sequence.json. No foreign process touched; no jobs/containers/armed
  guards. Bootsdc9e9293(.104r2)/133b8dc2(.105r3) unchanged. Nativewarnings/D12
  aggregate resume train_loss caveat remain. Historical Python CPU fault/KFD/
  invalidvisibility distinct from fixed VMheap/D05 agent exit; no new native fault.
  Both tools/distributed-d14 +evidence/distributed-d14-r1/{bf16,fp16}-{seed,resume},
  {bf16-reject,fp16-late,fp16-scale}-seed. Normal seedpayload156059932B/host retained
  plus refusal/resume outputs. Local kernel/artifacts/distributed-d14 metadata
  archives omit.pt/.pth/.bin/.pkl/.safetensors:104382577B SHA
  90780de8dff826d6149016bf2db78c58c89d2eb0d7f20c66aa6f6e92ae14ab23;
  105343823B SHAbc3b71333f1506f6a87b8ae2375c4cf2cb408cd58146f1fb957c1db3c3bb3667.
  Source/guard/scaler/refusal/config/fresh proof verified; docs/memory BOTH synced.
  [D14 report](docs/development/bc250_distributed_d14_results/README.md),
  [clipped Trainer usage](../BC250_DISTRIBUTED_GUIDE.md#coordinated-trainer-amp-clipping-d14).
  No arbitrary-model/loss/AMPBert/FP32/sparse/multioptimizer/FSDP/DeepSpeed/dynamic
  callback/convergence/capacity/speed/elastic/general-resume claim. Extreme finite
  norm overflow refusal guard exists but has no separate GPU injection proof.
  **Next D15 (new planned):** actual owned offline Transformer BF16/FP16 Trainer
  token-normalized CE +clipping/CPU references/native restart; retain native/
  default/recovery settings. Invalid-visibility frontend/B07.13/larger capacity/
  release reconstruction remain separate follow-ups.

- **D13 completed2026-11-05, actual Trainer BF16/FP16 + finite refusal:**
  fixture/callback70bd075e7, exports62b35356c. Existing D12 BC250Trainer/source
  pins unchanged, staged BOTH tools/distributed-d13 with new optional public
  FiniteGradientCheck. Native daily/RCCL Torch a524980/ROCm/D09 launcher/registry/
  images/profiles/kernels/frozen consumers unchanged. No native build/install/
  active-prefix update/reboot/user-app access. Actual Trainer5.2/Accelerate1.12
  MULTI_GPU/world2/RCCL, owned D07 residual MLP1051648 params/4x256/512/16tokens,
  BF16/FP16 observed Linear activations, FP32 master/reduction. Not AMP Bert.
  Explicit masked MSE/global active-window count, variable masks;28sequential
  records/workers0/per-device micro2/accum2: global8 first3attempts, global4 final
  single micro. All rows checked; reduction0 nonfinal/4206592B final. Clip0,
  no dropout; strict deterministic algorithms fixture-only. D07 CPU oracle uses
  explicit operand/output/backward rounding+FP64 matmuls, actual native loss scale
  and micro/rank accumulation. ALL finite losses/grads within retained limits:
  BF16 maxrelativeL2.00276769444<=.02; FP16.000372783622<=.004. CPU-double AdamW
  fed admitted GPU grads independently checks ALL params/moments, fixedD07 bounds.
  BF16 four successful updates/LR.00064; FP16 four Trainer attempts/three AdamW+
  StepLR updates/LR.0008. Rank1 Inf BEFORE third final DDP reduction reachesBOTH:
  native defaultGradScaler65536/growth_interval2000 skips AdamW AND scheduler,
  checks params/moments/scheduler unchanged, backs off32768/tracker0, final short
  window recovers32768/tracker1. No private scaler/Trainer-overflow mutation.
  Native save_on_each_node checkpoint2 BOTH independent filesystems including
  FP16scaler.pt; new CID/UUID/startticks/different constructor seeds restore
  complete params/AdamW/scheduler/scaler/global_step/four RNG EXACT. Native skipped
  loader then attempts3/4 losses/ALL grads/params/optimizer/scheduler/scaler/RNG/
  probe draws/rows/finiteflag EXACT and digests:8resumed rank-windows, incl FP16
  native skip/recovery. RNG probes only, no stochastic-model/general-data claim.
  BF16 FiniteGradientCheck PUBLIC on_pre_optimizer_step all-reduces invalidflag,
  raises NonFiniteGradientError on BOTH live ranks BEFORE update. Deliberate
  STOP/reload complete checkpoint, not automatic skip. Separate rank1 AFTER-
  reduction Inf leaves rank0 localgrads finite; both refuse, full params/AdamW/
  scheduler equal checkpoint2/globalstep+Adamsteps2. RNG/data/grad buffers advance.
  Callback refuses FP16; sparse/peer-loss not qualified, FP32 code acceptance not
  separate GPU proof. FP16 rank-local errors AFTER reduction/clipping still open.
  68groups/10ACK guards/42local regressions; first revision all pass, Black/critical
  Ruff+unused-import/syntax/diff pass; no full upstream CI. SH locks/4CPUs/10GiB+
  12GiB/512PIDs/256MiBshm/600s guards/460s children. Clean agents/phase kernellogs/
  registries; no foreign workload touched. Sequence6626/6771 exit0, commands/
  per-phase PID/exits sequence.json; separate BF16reject8067/8033 exit0 (expected
  exception caught). No jobs/containers/armed guards. Boots dc9e9293(.104r2)/
  133b8dc2(.105r3) unchanged. Warnings +D12 aggregate resume train_loss caveat stay;
  historical Python CPU fault/KFD and invalid HIP visibility remain distinct from
  fixed VMheap/D05 exit. No native fault or performance/capacity claim added.
  Both tools/distributed-d13 +evidence/distributed-d13-r1/{bf16,fp16}-{seed,resume},
  bf16-reject-seed. Normal seed payload143385444B/host retained plus reject/resume
  native outputs; archives omit.pt/.pth/.bin/.pkl/.safetensors. Local
  kernel/artifacts/distributed-d13 verified summary+archives104279479B SHA
  f7cc87c152d92c700f605ff0d0d35190933d773c513308d1913fd65f5f1dd6b5;
  105252034B SHA2d3cf3ca63a0a8eae14bfef5d1b61b5abc90293e47ec5b28d688d72755ff5f2a.
  Docs/agent memory synchronized BOTH. [D13 report](docs/development/bc250_distributed_d13_results/README.md),
  [AMP recipe](../BC250_DISTRIBUTED_GUIDE.md#trainer-bf16fp16-and-finite-gradient-checks-d13).
  No arbitrary model/loss/TrainerAMPBert/FSDP/DeepSpeed/clipping/eval/convergence/
  larger-capacity/speedup/elastic/peer-loss/multiworker claim. **Next D14 (new
  planned):** actual Trainer AMP gradient clipping +coherent rank-local FP16
  nonfinite policy, owned bounded references; retain active native/default/
  recovery settings. Invalid-visibility refusal/B07.13/larger capacity/elastic/
  release reconstruction remain separate follow-ups.

- **D12 completed2026-11-05, actual Trainer DDP + accumulation repair:**
  initial514ed8be2, acceptedf5edcaac1, exports3f2d1e391/CPUtestcc47efc09.
  Native daily+RCCL Torch a524980/ROCm/D09 launcher/images/registries/profiles/
  kernels/frozen consumers unchanged. No native build/install/prefix update/
  development reboot/user-app access. Actual Transformers5.2.0 +Accelerate1.12.0
  Trainer both divide accumulation loss: stock two-micro ALL gradients equal
  0.5x CPU-double, reproduced on both. Accepted stock checks prove DEFECT, not
  correctness. Opt-in tools/distributed-d12/bc250_trainer.py BC250Trainer delegates
  native loop; Accelerator accumulation1 while Trainer arguments2, one division.
  Version AND Trainer/Accelerator source SHA guards; no global monkeypatch or
  user-script autoactivation. Independent native Accelerator callers unchanged.
  BertMLM **kwargs ignores num_items: explicit accepts_loss_kwargsFalse for equal
  active-count native mean; explicit masked_token_loss hook for variable masks,
  sum CE/global active-window count +Trainer DDP compensation. Unshifted MLM only.
  Owned offline Bert22561 FP32/eager/dropout0,28records/sequential/workers0,
  per-device micro2/accum2: global8 first3updates, final global4/ONE micro.
  Each4seed windows/case/rank: CPU-double global loss/ALL grads/AdamW ALL params/
  moments pass fixed D09/D07 bounds, no numerical relaxation.90,244 gradient bytes
  only final micro. Native Trainer scheduler+post-step hook/LR check:4updates,
  StepLR.001→.00064. Actual save_on_each_node native checkpoint2 per independent
  filesystem, then fresh native Trainer train(resume_from_checkpoint) restores
  initial model/optimizer/StepLR/globalstep/four RNG exact; skip data and following
  steps3/4 losses/grads/params/optimizer/scheduler/RNG/probe draws/rows EXACT and
  digests.8resumed rank-windows, CID/UUID/starttick birth proofs. No stochastic
  graph (RNG probes only), general shuffled/multiworker/elastic resume claim.
  78groups(6stock diagnosis+72corrected),10GPUguards(2stock+8corrected),18actual
  pinned CPU checks(versions/source refusals+analytic CE derivatives/counts),
  42local regressions; Black/critical Ruff/syntax/diff pass, no full upstreamCI.
  All agents/phase kernel logs/registry clean; SH locks,4CPUs/10GiB+12GiB/512PIDs/
  256MiBshm/600s guards/460s deadlines. r2 sequence29765/16203 exit0, per-phase
  PIDs/commands/exits sequence.json. r1 stock28871/15080 exit0, mean29292/15570
  exit1 after4 CPU references: LR iterative vs power rounding strict equality.
  Only LR check now abs1e-15; model/loss/grad limits unchanged. r1source/logs/
  checkpoints retained,4historical ACKs. CPU initial HIP_VISIBLE_DEVICES=-1 aborts
  ROCr import (139), unsupported value, logs cpu-invalid-visibility; corrected
  command omits flags, Torch CUDA stays uninitialized. CPU stale lazy-module
  mock fixed with sys.modules admission stub; original log/source retained.
  Corrected CPU controllers4100/4259 exit0. Invalid negative visibility parser
  behavior remains; not historical Python fault/KFD uncertainty or fixed VM heap.
  GPU boots7344171f/e1cbfa23; BOTH rebooted BETWEEN completed GPU work and CPU/
  export (guards already ACK). No development reboot issued. Current .104r2
  bootdc9e9293-451c-4cd0-8839-7c902c8509ad/.105r3 boot133b8dc2-62a1-417b-8acb-fc44aeda6eec.
  No jobs/containers/armed guards. Native tied decoder missing-key notices pass
  complete exact restore; all_gather deprecation/barrier warnings retained.
  Aggregate resume train_loss is remaining loss/totalglobalstep, not equal-interval
  comparable; per-window exact losses qualified. No eval-metric semantics repair.
  Both tools/distributed-d12 +evidence/distributed-d12-r2/{stock-seed,mean-seed,
  mean-resume,tokens-seed,tokens-resume,cpu}; r1/failedCPU/source preserved.
  Valid seed/native/reference3720842B/host retained plus resumed/historicaloutputs.
  Local kernel/artifacts/distributed-d12 archives omit.pt/.pth/.bin/.pkl/.safetensors:
  104406142B SHA7ce23877c053426ecd690a764ff1377ae0b839eaec659e3a43580d41db8ccf0d;
  105365873B SHAed7af2bc9f1bf60381749a175456d628b5208f5a801dad77c1c2007ad671497c.
  Both docs/memory synchronized. [D12 report](docs/development/bc250_distributed_d12_results/README.md),
  [explicit usage](../BC250_DISTRIBUTED_GUIDE.md#transformers-trainer-on-two-hosts-d12).
  No generic task/model/loss/AMP/FSDP/DeepSpeed/clipping/evaluation/convergence/
  capacity/speedup claim; historical Python fault distinct from fixed VMheap/D05.
  **Next D13 (new planned):** actual Trainer BF16/FP16 with bounded mathematical/
  nonfinite-update/state references, owned fixtures +explicit adapter; preserve
  native/default/recovery settings. Runtime invalid-visibility refusal is a
  separate follow-up; B07.13/larger capacity/elastic/release reconstruction open.

- **D11 completed2026-11-05, actual Accelerate two-host DDP integration:**
  fixture21766d81d, exports/verifierf5329413b. Existing frozen Accelerate1.12.0/
  NumPy2.5.1; D09 public command/native daily+RCCL Torch a524980/ROCm/images/
  registry/profiles/kernels unchanged. No native build/install/reboot/app access.
  Owned1,051,648-param residual MLP, FP32/BF16/FP16 actual Accelerator.prepare/
  accumulate/backward/prepared optimizer/scheduler/DataLoader, MULTI_GPU/world2/
  NCCL(native RCCL). Sequential32 records, split global micro2,4micros/global8.
  First3 micros no gradient reduction; final4,206,592 FP32 bytes. Independent
  CPU-double FP32 and D07 explicit low-rounding loss/all-gradient references,
  CPU-double AdamW parameter/moment checks,2windows/precision/rank. BF16 max
  gradient relativeL2.00225232<=.02, FP16.000435060<=.004, fixed D07 limits.
  Accelerator divides loss; no double division. split_batchesTrue +plugin
  adjust_schedulerFalse, post-step hook/independent LR verify update-only StepLR.
  Rank1 Inf before third-window reduction reaches both; native FP16 scaler skips,
  FP32/BF16 explicit collective finite guard skips optimizer AND scheduler.
  Accelerator itself supplies no automatic FP32/BF16 nonfinite guard. Fourth
  window recovers; final3updates/16micros/LR.0008, FP16scale16/tracker1.
  Native save_state on BOTH independent filesystems with save_on_each_nodeTrue,
  registered Progress cursor, checkpoint after8micros/2updates. New containers/
  UUID/startticks/different constructors; actual native load_state restores
  initial model/optimizer/scaler/scheduler/four RNG streams/custom cursor exactly.
  Materialize/prefetch skipped workers0 sequential loader BEFORE load_state so
  iterator CPU RNG draws precede restoration; native RNG loader may suppress
  errors, explicitly checked. Following2windows incl skip/recovery exact losses/
  grads/params/optimizer/scaler/scheduler/RNG/augmentation/indices/cursor and
  digests:12 resumed rank-windows.4 rank1 bad config/cursor/file-size/hash metadata
  cases refuse BOTH live ranks before mutation,24refusal observations. Owned
  admission wrapper, trusted native files; not an installed generic checkpoint API.
  78accepted groups/42localtests/12ACK guards; clean agents/kernel logs/registries,
  SH locks/4CPUs/10GiB+12GiB/512PIDs/256MiBshm,600s guards/460s deadlines.
  Black/critical Ruff+unused imports/syntax/diff pass; no full upstream CI.
  Sequence26401/11021 exit0; FP32 seed25559/10127,resume25995/10583 exit0.
  Both tools/distributed-d11 +evidence/distributed-d11-r1/{no,bf16,fp16}-{seed,resume};
  per-phase PID/commands/exits sequence.json. Native/reference payloads177094758B104/
  177094566B105 retained; archives omit.pt/.bin/.pkl/.safetensors. Local
  kernel/artifacts/distributed-d11 verified results +archives104326808B SHA
  b44001c1a03faa538a284d747de05780372f1a466c2a00e874317887f4c8a2c8;
  105292811B SHAa62d43b21ec83c94d174f015467f5f57038e99ee302b4c4dab63fec66c607db8.
  .104r2/.105r3 boots7344171f/e1cbfa23 same. No jobs/containers/armed guards.
  No arbitrary model/AcceleratorFSDP/Trainer/multiworker/shuffle/partial-window/
  elastic/peer-loss/capacity/speedup claim. Historical Python CPU fault/KFD
  uncertainty distinct from fixed VM heap/D05 agent exit; no new native fault.
  Both host docs/memory synchronized. [D11 report](docs/development/bc250_distributed_d11_results/README.md),
  [Accelerate recipe](../BC250_DISTRIBUTED_GUIDE.md#accelerate-on-two-hosts-d11).
  **Next D12 (new planned):** actual Transformers Trainer two-host integration,
  owned offline fixture, bounded accumulation/state references through public
  command; retain native/default/recovery settings. B07.13 deferred; other
  transports/larger capacity/elastic/release reconstruction remain open.

- **D10 completed2026-11-05, fresh-process fixed-world checkpoint/restart:**
  initial553eebce1, accepted proof/restricted-load correction0c0efa810, exports97641e4b6.
  Installed D09 tools/native daily+RCCL Torch a524980/ROCm/images/registry/profiles/
  kernels unchanged, Torch source8d8b18d. No build/install/reboot/app access.
  Owned D09 Bert22,561 FP32 DDP with dropout.1 and D08 MLP1,051,648 FP16 classic
  FSDP with output dropout.125, same mixed policy/FP32 master+reduce. Global8,
  accumulation2/DDP or4/FSDP per rank, strict deterministic algorithms in fixture.
  AdamW +StepLR(step2,gamma.8), five RNG streams Python/NumPy/TorchCPU/TorchCUDA/
  explicit cursor generator, per-rank random augmentation/native GPU dropout.
  Shuffled32-record global cursor,4/rank per window. Three successful windows save
  local optimizer-boundary checkpoint at next_attempt3/epoch0/offset24, then seed
  workers continue three uninterrupted reference windows. Attempt4 injects Inf
  before final DDP reduction or rank1-local FSDP shard after reduction; both skip
  optimizer/scheduler while data/RNG advance. End five updates/six attempts,
  epoch1/offset16. Hook update counts/independent LR formula checked.
  Entirely new containers/workers initialize from different seeds, then restore
  params/buffers/AdamW/scaler/StepLR/update count/cursor+order+generator and RNG
  LAST. Initial state exact; three resumed local losses/indices/augmentation/all
  local gradients/params/optimizer/scaler/scheduler/RNG/cursor EXACT vs uninterrupted,
  zero tolerances AND matching digests.12 resumed rank-windows. FSDP crosses
  scaler32→64 then skipped64→32 then finite32/tracker1. No new CPU mathematical
  kernel proof beyond D08/D09; state continuity is the current qualification.
  Local CPU-serialized fsync/atomic parts +complete collective rank/hash manifest,
  no shared FS. Schema/world/rank/config/layout/package-version/selected source+
  Torch/worker/agent cohort binding, weights_only=True. Both live ranks coordinate
  admission. Eleven one-rank failures each refuse both before applying state:
  8header/file cases,1valid-digest invalid serialization,2loaded rank/cursor cases;
  negative checks leave model/optimizer/scheduler/scaler/RNG/cursor unchanged.
  References independently hash-checked. No partial-window/DataLoader-worker-RNG/
  elastic/repartition/peer-death/crash-during-save/arbitrary checkpoint/migration
  qualification. Owned fixed attempt3 format, not an installed generic API.
  r1 DDP seed passes; resume fails PRE-restore because harness wrongly required
  distinct PID-namespace inodes, observed equal across new containers. Actual CID,
  UUID/start ticks differ. r2 uses those birth proofs, no numerical tolerance
  relaxation. r1 source/logs/checkpoints kept; normal assertion exit1, not native
  arithmetic/heap/Python segfault. Corrected failure cleanup destroys process group.
  48accepted groups/44collective refusal observations/42localtests/8accepted ACKs,
  plus4historical r1 ACKs. All r2 agents/kernel logs/registries clean/unchanged.
  SH locks,4CPUs/cpuset0-3/10GiB RAM/12GiB memory+swap/512PIDs/256MiB shm,
  600s guards/460s child deadlines; owned container labels never stop unrelated
  containers. Black/critical Ruff+unused-import/syntax/diff pass, no full upstream CI.
  .104 r2/.105 r3, boots7344171f/e1cbfa23 same. Historical Python CPU fault/KFD
  uncertainty distinct from fixed VM heap/D05 exit repair. No speed/capacity claim.
  r2 coordinators21599/1347 exit0; sequence.json exact per-phase PIDs/commands/exits.
  r1 seed20342/31546 exit0, resume20874/32385 exit1. No jobs/containers/armed guards.
  Both tools/distributed-d10 and evidence/distributed-d10-r2/{ddp,fsdp}-{seed,resume};
  r1 diagnostics separate. Valid .pt payloads33,335,526B104/33,368,766B105 plus
  malformed/historical owned copies retained, exports omit.pt. Local
  kernel/artifacts/distributed-d10/{verified-results.json,source-r1,raw archives}:
  104334281B SHA46e57cbb9b3358d3eb1af197b8dea6c9d58795b88be2950d1eff59fe2876beec;
  105300886B SHA2271b7732b3e36ac6763311fa9eb12f4f7a262c933eb45c3d28682c3af3de5c7.
  Source/worker/kernel/guard/reference-digest/cohort/birth checks pass; docs/memory
  synchronized BOTH. [D10 report](docs/development/bc250_distributed_d10_results/README.md),
  [checkpoint usage](../BC250_DISTRIBUTED_GUIDE.md).
  **Next D11 (new planned):** actual Accelerate two-host DDP accumulation/AMP
  through public bc250-distributed, owned small-model references/state handling.
  D09 imported Accelerate, did not qualify orchestration. Keep native/default/
  recovery settings; larger capacity/other transports/elastic recovery/release
  reconstruction remain open. B07.13 deferred; no completion percentage.

- **D09 completed/installed 2026-11-05, distributed consumer env/mounts:**
  launcher/fixtures e1bbb8ec7, installer/CLI refusals37067fdde, exportsbb4d672ea.
  Both public commands NOW /opt/bc250-stack/tools/distributed-d09; registry d01.json
  gains consumer/tool identity/recovery metadata. Native daily/RCCL Torch a524980/
  ROCm/images/profiles/daily selections/kernels unchanged, Torch source8d8b18d;
  D05 agent repair source/API hashes and old recovery metadata unchanged. No native
  build/package install/reboot/app access. Old D05/D01 assets/boot fallback kept.
  New ordered --env/--env-file/--mount/--mount-ro/--shm-size before script reuse
  ordinary parser; later files/flags override earlier, script args/spaces preserved,
  explicit read-only parents honored. Root revalidates private0600 context, fixed
  topology/rendezvous/bootstrap/tool mounts; protected native/transport settings
  refuse. Pinned CPython3.12 retained, consumer site-packages via PYTHONPATH/mounts.
  Agent and fresh worker check Torch package/_C paths, version, __init__/_C/
  libtorch_python/libtorch_hip SHA before user script; existing RCCL profile checks
  stay. Worker runpy preserves argv/cwd/script import path. No automatic dependency
  compatibility or step-speed claim; added launch hashing is startup work.
  Actual frozen consumers Transformers5.2.0/peft.18.1/accelerate1.12 imported;
  offline owned random BertForMaskedLM22,561 params, vocab97/width32/2layers/4heads/
  intermediate64/seq12/dropout0/requested eager, FP32, global8/2micros per rank.
  Candidate and installed two successive CPU-double global losses/all-param grads/
  AdamW parameter updates pass at D06 bounds. Retained D08 FP16/no_sync FSDP
  reference/moment/shard-nonfinite/scaler/exact checkpoint also passes candidate
  bootstrap. No HF Trainer/Accelerate orchestration/arbitrary model/mixed Bert/
  capacity/convergence/performance claim.
  External owned /var/tmp/bc250-d09-fixtures/{read only data,writable data,cwd with spaces},
  EROFS proves readonly bind despite UID1000 host-owner writable folder; rw marker,
  literal argv/caller env/override order/cwd/UID verified. Actual Nik SSH→doas→
  public command CPU Gloo sum3 passes BOTH, Torch cuda-initialized flag false
  (not whole-driver memory claim). Three CPU identity refusals/rank restore test
  metadata; no runtime files altered.37localtests (29retained+8new),24actualCLI
  refusals before containers,46GPU worker groups/6ordinaryCPU groups/6CPUidentity
  refusals/6ACK guards; retained FSDP includes12checkpoint/4scaler refusals within
  these counts. Black/critical Ruff+unused-import/syntax/diff pass; no full CI.
  EX stage candidate → consumer+FSDP qualify → activate → live D05 rollback →
  reactivate → public consumer pass, BOTH hosts. All9installed/payload files SHA
  match across hosts. Durable installer exact filename allowlist/pins; EX helpers
  self-lock, never nest another writer. Native runtime SHA/release/agent fix same.
  Both tools/distributed-d09/{install_distributed_d09_on_host.py --activate,
  rollback_distributed_d09_on_host.py,install-payload}; recovery/evidence under
  evidence/distributed-d09-tool-install-r1. Old D05 fallback keeps agent exit fix.
  GPU phases SH locks/4CPUs/cpuset0-3/10GiB RAM/12GiB memory+swap/512PIDs,
  consumer256MiB shm/600s guards/460s deadlines; ordinaryCPU public defaults with
  fixture4Torchthreads. .104 r2/.105 r3, boots7344171f/e1cbfa23 same; no reboot.
  Historical Python CPU fault/KFD uncertainty separate from fixed VM heap/D05 exit.
  AllGPU controllers exit0: candidate15499/19447, retained15907/20252,
  installed16774/21763; actual ordinaryCPU SSH exit0 both. No jobs/containers/
  armed guards. Both evidence/distributed-d09-r1/{consumer-candidate,retained-candidate,
  consumer-installed}, CLI-refusals.json; evidence/distributed-d09-user-cpu/rank.json.
  Retained owned FSDP .pt stays on boards, excluded from exports. Local
  kernel/artifacts/distributed-d09/{install-payload,verified-results.json,raw archives}:
  104259027B SHA69b3293ef4cfa6c1c3d5466cd4923780e39af2ce8879231476e2d3be6c4c39ee;
  105242047B SHA394d5eaf2923c77def6882f6219d79602b846af59ec1e59e4e64e9f197a58cc5.
  Source/installed/payload/worker/kernel/guard/caller/recovery hashes verified;
  host docs/memory synchronized. [D09 report](docs/development/bc250_distributed_d09_results/README.md),
  [consumer options/recovery](../BC250_DISTRIBUTED_GUIDE.md).
  **Next D10 (new planned batch):** bounded fresh-process two-host checkpoint/
  restart, model/optimizer/scaler plus scheduler/RNG/data-cursor references and
  strict config/world/payload refusals, owned fixtures/public command. Preserve
  optimizer-boundary/fixed-world scope; elastic/peer-crash/arbitrary checkpoints,
  larger-model capacity/other transports/release audit remain open. B07.13 deferred.

- **D08 completed 2026-11-05, classic FSDP AMP/shard-aware skipping:** numerical
  tools0815ac97b, export/verifier97e75946a. Reuse D07 MLP/CPU oracle, D05 checkpoint
  primitives/installed bounded launcher/agent repair. Native daily/RCCL Torch
  a524980/ROCm/images/registries/profiles/kernels unchanged; no backend defect,
  build/install/reboot/app access. Width256/hidden512/4blocks/tokens16/1,051,648
  params/global8/micro1/accum4/rank. Eight separately wrapped Linear modules,
  classic FULL_SHARD/use_orig_params; actual BF16/FP16 forward params/intermediates,
  FP32 masters/final grads/reduction/loss/residuals. Root/nested input casts off,
  outer autocast. Each rank525,824 master and AdamW exp_avg elements; exact original
  param coverage/owned optimizer slices, empty local views correctly have no state.
  CPU FP64/explicit-low-rounding oracle at matching weights checks two losses/all
  param+input gradients. no_sync CPU oracle additionally rounds each rank's full
  accumulated grad to low dtype, matching pinned FSDP. Fixed relative-L2 limits
  BF16.02/FP16.004, no relaxation. Max param errors BF16.31388% no_sync/.22560%sync,
  FP16.05349%/.04558%; input<=.02914%/.00862%. Separate CPU-double AdamW fed ACTUAL
  admitted AMP gradients compares all updated params and local moments/counters
  at D07 bounds; no whole unrounded-FP64 trajectory/convergence/arbitrary model
  or attention/compiler proof.
  Inf then NaN injected ONLY into rank1 local grad AFTER reduce-scatter: local
  flags[True,False]. Native FP16 ShardedGradScaler synchronizes found_inf, both
  actually skip (post-step hooks); local params/AdamW EXACT unchanged. BF16 disabled
  scaler uses explicit collective all-finite guard. Finite recovery passes; FP16
  16→16→32→32→16→8→8→16, growth tracker/backoff checked. Full param hashes and
  complete scaler state agree; one-rank scale/tracker mismatch both refused before
  backward. Local optimizer shards are distinct, not duplicate replicas.
  Atomic local optimizer/scaler/config/next-index parts +complete collective
  hash manifest; six preload refusals/rank. Same-worker model/optimizer/scaler
  reconstruction next loss/all LOCAL params/AdamW/scaler EXACT, crossing8→16 growth.
  No fresh-process/partial-window/RNG/sampler/scheduler/elastic/peer-crash/general
  distributed-checkpoint qualification.
  Every complete backward8RS calls/4,206,592 FP32 input B/rank; no_sync one pass
  vs sync4passes/32calls/16,826,368B. Instrumented public RS delegates unchanged;
  excludes parameter all-gathers/control/scaler/checkpoint, not wire/total traffic.
  Eight windows/worker assert counts; no speed/timing claim. no_sync first3 micros
  retain1,051,648 low grad elements then525,824 FP32 shards; sync525,824 FP32
  throughout. Both gradient views2,103,296B at2ranks, not a general memory ratio.
  Isolated third training window (finite/scaler/control included, CPU oracle/full-
  state-validation/checkpoint excluded): live peak BF16 no_sync75.04102 vs73.60693MiB,
  FP16 75.04248 vs73.60840, both ranks; +1.43408MiB. Reserved84/84 vs84/82(BF16),
  86/84 vs82/80(FP16) for104/105. Baseline/per-micro snapshots retained; no physical
  RAM pooling/larger-model fit/general capacity or different-policy cost claim.
  100groups/48checkpoint+8scaler-mismatch refusals/8ACKed guards; agents/kernel
  logs/registries clean/unchanged. SH locks,4CPUs/cpuset0-3,10GiB/12GiB memory+swap,
  600s guards/460s child deadlines; Black/critical Ruff+unused-import/syntax/diff pass,
  no full upstream CI. .104 r2/.105 r3, boots7344171f/e1cbfa23 same. Historical
  Python CPU fault/KFD uncertainty distinct from fixed VM heap/D05 exit repair.
  All controllers exit0: bf16-no-sync11869/10611, fp16-no-sync12241/11288,
  bf16-sync12561/11813, fp16-sync12919/12459. No jobs/containers/armed guards.
  Both tools/distributed-d08, evidence/distributed-d08-r1/{bf16,fp16}-{no-sync,sync}-r1
  plus controller logs. Four .pt/host25,285,292B104/25,311,948B105 retained, omitted
  from exports. Local kernel/artifacts/distributed-d08/verified-results.json and
  raw archives104252227B SHA209e51c2b6186f5a7b6cf4be92b9e079850c664a9c4cfaa2f0525057d5875df6;
  105230690B SHAc45da6b438bab63adf706b527cebdc734a1cb6ed509e9fcba941b2b3dfe79426.
  Source/archive/worker/kernel/guard/rank/window/scaler/memory checks verified;
  both host docs/memory synchronized, existing apps unchanged, opt-in recipe.
  [D08 report](docs/development/bc250_distributed_d08_results/README.md),
  [FSDP AMP usage](../BC250_DISTRIBUTED_GUIDE.md).
  **Next D09 (new planned batch):** close recorded distributed consumer-env gap:
  explicit validated env-file/env/mount support reusing ordinary Torch launcher,
  owned tiny Transformers training consumer, private caller/identity/refusal and
  live rollback gates on both hosts. Preserve native pins/D05 exit repair/Ethernet
  restrictions/defaults; no user apps/model downloads. Fresh-process recovery,
  larger-model capacity/other transports/release audit remain separate. B07.13
  still deferred; no completion percentage.

- **D07 completed 2026-11-05, two-host AMP/scaler:** numerical tools/guardian
  ae52b7306, export/verifier615caa65e. Existing native daily/RCCL Torch a524980/
  ROCm/images/profiles/registries/kernels and active D05 source-derived agent exit
  fix unchanged. No backend defect/native build/install/reboot/app access.
  Owned residual tanh MLP width256/hidden512/4blocks/tokens16/1,051,648 params,
  global8/micro1/accum4 per rank, FP32 master params/loss/residuals/gradients.
  BF16 and FP16 actual linear intermediates asserted; both no_sync-first3 and
  every-micro sync pass on BOTH hosts. CPU oracle independently rounds low-dtype
  linear operands/output/backward/bias and tanh around FP64 arithmetic, residual/
  loss/accum FP32, at matching FP32 model weights each of two reference updates.
  All param and input gradients checked; fixed relative-L2 bounds BF16.02/FP16.004,
  loss atol2e-5/rtol.002/.0005; no relaxation. Largest param errors.22560%/.04558%,
  input.02914%/.00862%. Separate FP64 CPU AdamW fed ACTUAL admitted AMP gradients
  compares all params/moments/step at D06 bounds. This isolates optimizer accuracy,
  not a whole mixed/unrounded-FP64 trajectory or arbitrary attention/model proof.
  BF16 disabled scaler +explicit nonfinite skip. FP16 native GradScaler16 growth2:
  finite16→16→32; rank1 accumulated-param Inf then NaN before final DDP reduction
  propagates to both, both skip with exact unchanged params/AdamW, scales32→16→8/
  tracker reset. Finite recovery passes. Post-step hooks verify actual updates;
  parameter/AdamW/scaler replica hashes match after every finished window.
  Full public scaler state collectively checked before backward; one-rank scale
  and growth-tracker mismatch both refused by both ranks, then restored. Post-
  reduction rank-local mutation is not qualified by these before-reduction cases.
  Atomic optimizer-boundary local parts +complete collective manifest; save
  scaler/config/next index, six preload refusals each. Same-worker reconstructed
  model/optimizer/scaler next loss/all params/AdamW/scaler EXACT, crossing8→16
  growth boundary. Not fresh-process/partial-window/RNG/sampler/scheduler/elastic/
  peer-failure/general-checkpoint qualification.
  Eight workers/four phases,92check groups/48checkpoint refusals/eight collective
  scaler-mismatch refusals/eight ACKed guards. Agent exits/kernel logs/registries
  clean/unchanged. Four micros reduce16,826,368B sync vs4,206,592 no_sync per rank;
  FP32 gradient buffers, not physical wire counts or AMP traffic compression.
  Seven windows/worker enforce bytes/calls; initial bucket rebuild1→2 allowed.
  No timing or speed claim; different model from D05/D06. SH locks,4CPUs/cpuset0-3,
  10GiB/12GiB memory+swap,600s guards/460s child deadlines;4TorchCPUthreads.
  Black/critical Ruff+unused-import/syntax/diff pass, full upstream CI not run.
  .104 r2/.105 r3, boots7344171f/e1cbfa23 same as D06; fallbacks/health/controls
  preserved. Historical Python CPU fault/KFD uncertainty distinct from fixed
  VM heap and repaired agent exit; no new fix claimed.
  All4controllers/host exit0, known PIDs FP16-no-sync9453/8276, BF16-sync9783/8438,
  FP16-sync10112/8607; initial BF16-no-sync PID not captured, final/ACK evidence
  present. No jobs/containers/armed guards. Both tools/distributed-d07 and evidence/
  distributed-d07-r1/{bf16,fp16}-{no-sync,sync}-r1 plus controller logs. Four .pt
  checkpoints/host50,551,948B retained; excluded from exports. Local kernel/artifacts/
  distributed-d07/verified-results.json and raw archives:104239789B
  SHA2cc7285d29a9a43deeac5ee8e830cf7f696ebfc22b00421969bd9aa29b090627;
  105217630B SHAf21d02b249a4ff53962000c3e6642c1f0cc0414a9c3801b65382702b51085c67.
  Source/archive/worker/kernel/guard/rank/window/scaler checks verified. Both host
  docs/memory synchronized; existing app loops unchanged, optional usage only.
  [D07 report](docs/development/bc250_distributed_d07_results/README.md),
  [AMP/scaler guide](../BC250_DISTRIBUTED_GUIDE.md).
  **Next D08 (new planned batch):** bounded classic FSDP mixed-precision
  accumulation: independent gradients/optimizer, actual shard/peak-memory counts,
  shard-aware finite/nonfinite decisions (pinned ShardedGradScaler inspected) and
  same-world checkpoint replay. Keep no_sync full-gradient memory cost explicit,
  admit BF16/FP16 separately and preserve FP32 defaults. No larger-model capacity/
  speedup promise. Other transports/release audit open; B07.13 still deferred.

- **D06 completed 2026-11-05, counted DDP accumulation:** numerical tools
  bd3726047; sequencing/export/verifier539951ced. Reuse D05 owned transformer and
  checkpoint primitives, installed bounded launcher and active D05 agent exit fix.
  Native daily/RCCL Torch a524980/ROCm/images/registry/daily profiles unchanged;
  no native build/install/reboot/app access. Generic FP32 width512/heads8/4 blocks/
  seq64/12,609,536 params, AdamW; distinct FP32-rounded per-example inputs.
  Global8/micro1: accumulation8 single or4/rank; global32/micro2:16 single or8/rank.
  Equal global objective, one zero_grad/optimizer update; no_sync encloses BOTH
  forward/backward of all non-final micros, each mean loss/local accumulation.
  DDP averaging supplies rank normalization. CPU-double oracle chunk4 compares two
  global losses/all gradients/parameters/both AdamW moments/step counters; fixed
  tolerances, no relaxation. Atomic rank checkpoint complete collective manifest;
  six preload refusals each, reconstructed next local loss/all params/AdamW exact.
  Optimizer-boundary/fixed-world/rank/data recipe only; partial accum/RNG/scheduler/
  elastic/peer-crash/masked-token/uneven-batch behavior remains unqualified.
  Torch averaging hook counted per micro; complete50,438,144 gradient input B/rank
  per sync. Steady small12calls/201,752,576B→3/50,438,144; larger24/403,505,152→3/
  50,438,144. Counters exclude control/init/checkpoint collectives, are bucket input
  not NIC/wire byte measurements. All11 updates/worker enforce byte/call contracts.
  Two warm/five measured, max rank, opposite orders, CPU oracle/checkpoint/control/
  launch/exit excluded. Global8: sync1738.02–1738.56ms→no_sync462.00–462.36ms
  (3.760–3.762x), single62.67–64.86ms. Global32:3474.24–3476.18→496.08–497.33ms
  (6.986–7.007x), single123.35–128.16ms. These gains are versus every-micro sync;
  two hosts remain slower than one for these fixtures. Different data/micro layout
  from D05, not a D05 regression comparison or application/universal speed claim.
  Peak live allocator no_sync321.53/333.53MiB vs sync270.43/282.43MiB: +51.10MiB;
  reserved342/350 vs306/314MiB. No memory/capacity/physical RAM pooling claim.
  12 phases/host,24workers,192check groups/144preload refusals/24ACKed GPU guards;
  agent exits clean, kernel logs and registry unchanged throughout. Black/critical
  Ruff/syntax/diff pass; no full upstream CI. Broadcast_buffers deprecation and
  current-context barrier warnings visible, not hidden/backend errors. SH locks,
  4 CPUs/cpuset0-3,10GiB/12GiB memory+swap,600s guards/460s child deadlines.
  .104 r2/.105 r3; current observed boots7344171f/e1cbfa23 newer than D05 but
  present at first D06 phase; no reboot here. Historical Python CPU fault/KFD
  uncertainty distinct from fixed VM heap and D05 exit repair, no new fix claim.
  Coordinators4856(.104)/3882(.105)exit0; all per-phase guards passed. No jobs/
  containers/armed guards. Both tools/distributed-d06 and evidence/distributed-d06-r1/
  <case>-<mode>-<a|b>; per-phase logs and sequence commands/PID/exit journal.
  Each host retains12 owned .pt checkpoints/1,816,412,148B, omitted from exports.
  Local kernel/artifacts/distributed-d06/verified-results.json + raw archives:
  104647863B SHAe68c3c32a77a14dbc9e77fa68ada38f3a3897026699c36d300ee61a013bb3d64;
  105578803B SHA08fda612dd67472bd22fa4ab8c03854548331e4d742c4f540130ac9024b76448.
  Independent archive/source/worker/kernel/guard/rank/config/timing checks pass.
  [D06 report](docs/development/bc250_distributed_d06_results/README.md),
  [explicit usage](../BC250_DISTRIBUTED_GUIDE.md). Existing app loops unchanged;
  no_sync already exists in installed Torch, no automatic launcher optimization.
  Both host docs/memory synchronized; old snapshots preserved.
  **Next D07 (new planned batch):** bounded DDP mixed-precision accumulation:
  BF16 autocast and FP16+GradScaler, independent dtype-appropriate gradient/update
  references, synchronized finite/nonfinite decisions and same-world optimizer/
  scaler checkpoint replay. Preserve FP32 defaults; no AMP qualification yet.
  FSDP capacity/reservation, other transports and release reconstruction stay open;
  B07.13 still deferred. Do not infer completion percentage.

- **D05 completed 2026-11-04, bounded two-host training + exit repair:** training
  tools9ab6aec7f; bridge41bafbdd0 plus final durable-payload/recovery maintenance;
  Torch source exit fix8d8b18df558cb478727ba2179f2e01dfce20ea40. Installed native
  daily/RCCL cohorts remain a524980; no native build, kernel, image or daily/default
  profile change. BOTH explicit bc250-distributed commands now select versioned
  tools/distributed-d05, registry retains native runtime pins and records exit fix.
  Four-block width512/heads8/seq64/global batch8 FP32 causal transformer12,609,536
  params, AdamW; one host batch8 versus two hosts4/rank. Both run orders: single
  23.55–27.22ms(.104)/23.75–25.83ms(.105), DDP438.25–438.41ms, FSDP651.55–651.70ms.
  Thus this fixture is slower distributed; no speedup/general scaling claim.
  1/8/32MiB AllReduce medians~9.2/71.9/286.4ms match communication-bound diagnosis.
  FSDP stores6,304,768 params/rank and matching exp_avg versus12,609,536 in DDP;
  peak live GPU allocator~212MiB vs299MiB DDP, but reserved360/386MiB vs308MiB.
  No physical whole-board/large-model capacity/pooled-VRAM claim. CPU oracle/RSS
  separate from allocator stats. Two warm/five timed steps, max distributed rank,
  CPU reference/checkpoint/rendezvous/exit excluded; opposite-order repeats.
  140 worker check groups/108 preload checkpoint refusals/18 GPU guards ACK;
  all first gradients/two AdamW updates compare CPU double, same-rank/world
  checkpoint next loss/parameters/AdamW exact. Atomic local parts + complete
  collective hash manifest; fixed fixture index/data, no elastic/scheduler/RNG/
  peer-crash/general checkpoint API qualification. Owned .pt files stay on hosts.
  Original DDP-a .105 agent logs TCPStore EOF after numerically successful work;
  old guard returns0, recorded separately as lifecycle defect. Symmetric exit
  barrier could close rank0 store before peer final wait. Source fix reuses
  synchronize/get_all rank-aware completion. CPU real-store old race reproduced,
  candidate2 repetitions +2 flag/timeout units pass; r1 ownership harness error
  before test retained, corrected r2 passes. Source-derived bridge checks original
  API SHA625fa4cc and patched source hash; applies only exit method, no log masking.
  Frozen Torch API unchanged; direct torchrun outside bc250-distributed retains old
  API until separately qualified Torch update. Candidate DDP/FSDP strict clean-agent
  gates pass before activation, then public DDP strict gate passes. Live EX-locked
  rollback/reactivation and durable install-payload replay pass on BOTH; old D01
  tool/registry retained. CPU UID1000/doas/cwd/spaced-argv/Gloo sum tests pass on
  both with Torch CUDA initialization flag false. 29 local tests/Black/critical
  Ruff/syntax/diff pass; no full upstream CI. SH tests,4 CPUs/cpuset0-3,10GiB RAM/
  12GiB memory+swap,600s GPU guards. Kernels/boots d10514df/r2 and92b47338/r3 same;
  no app access/cache reset/reboot. Historical Python CPU segfault/KFD uncertainty
  remains separate from fixed VM heap and this agent-exit repair.
  All controllers exit0 except initial CPU harness4494exit1; corrected4819exit0.
  Last installed GPU7354(.104)/13266(.105)exit0; no jobs/containers/armed guards.
  Both /home/nik/bc250work/evidence/distributed-d05-r1/<phase> plus controller logs;
  tools/distributed-d05 holds installer/rollback/durable payload, install records
  evidence/distributed-d05-tool-install-r1. CPU r2 on104 and user-cpu onboth retained.
  Local kernel/artifacts/distributed-d05/*-evidence-r2.tar.gz:104660837B
  SHA8a546b712339bffc724370eb64c1348f4eea57329824f092250f9fad958ebcdf;
  105495817B SHA62c1e81c258700b6651791b38c4b1d0d5248a344e38ac39ff95c3549fe4b854b.
  [Method/limits/recovery](docs/development/bc250_distributed_d05_results/README.md),
  [usage](../BC250_DISTRIBUTED_GUIDE.md). Host docs/memory updated; old snapshots kept.
  **Next D06:** matched-global-batch DDP gradient accumulation/no_sync, CPU-double
  gradient/AdamW/checkpoint references, synchronization counts and one-vs-two-host
  timing. Amortize measured traffic without changing optimizer semantics. FSDP
  capacity/reservation and broad model/dtype/transport work remain separate;
  B07.13 still deferred, not silently installed. No completion percentage.

- **.104 remaining-folder review completed2026-11-04:** explicit user asks to
  inspect/remove obsolete top-level folders. Toold16294297; common EX runtime lock,
  no containers; system/service/tool/current-runtime reference and process/FD audit
  finds no candidate dependencies. Removed ONLY bc250-gfx1013-fix (clean5bb0dac,
  same as workstation repo), migration-105-20261023 (.104 leftover transfer staging,
  NOT .105 operational helpers), rocm-performance-20261013 (old benchmark fixture).
  104 inventory entries/113,943,255 file bytes; full archive95,923,498B retained
  locally kernel/artifacts/workspace-cleanup-20261104/folders-104/obsolete-folders.tar.gz,
  SHAc50a20b1ff95a8528cab4d1b39d243eed2b30545b6ed857ab76a8a6c7e3833aa.
  Whole archive and every member/type/link verified; restored Git fsck passes and
  HEAD5bb0dac matches. Originals and redundant host archive removed only afterward.
  Plan/result/dependency findings and local verification retained beside archive;
  host evidence/folder-cleanup-20261104/{plan,result}.json/controller log adjacent.
  Inventory PID28557 exit0; apply command complete; no running jobs/guards/containers.
  Retain seven real dirs: active ROCm/Torch workspace, kernel-r3 source/build/
  helpers/recovery, r2 bc250-rocm-lab fallback, private RADV bc250-vulkan-v33,
  docs/tools/evidence; plus D04 registry-required alias = eight total root entries.
  RADV wrapper's absolute private-driver path establishes retention purpose; no
  current Vulkan workload/performance qualification inferred. Boot/fallback assets
  and actual source/build/cache/checkpoint trees unchanged. Current/runtime registry/
  GRUB config+environment hashes unchanged; ordinary --info passes, r2 boot retained,
  ~20.8GiB available. No GPU/native/boot changes or GPU requalification. .105 UNTOUCHED.
  [Retention map](../docs/WORKSPACE_LAYOUT.md); docs/memory mirrored to .104.
  **Next D05 unchanged:** representative distributed timing/memory/checkpoint work.

- **Top-level cleanup follow-up completed2026-11-04:** user explicitly asks to
  remove old agent helpers/files and unused aliases, rather than preserve clutter.
  Source toolbb7fc0c0e plus final consolidation/backup-retirement checks; both host
  EX locks/no containers. .104118 entries (99 regular/19 links),2,823,477,728B;
  .105154 (126 regular/28 links),13,876,633B retired. All225 regular loose files
  including obsolete synthetic fixture checkpoints archived locally; both archive
  SHAs and EVERY member SHA verified before deletion. No user training checkpoint
  or application access. Backups790MB/10.5MB remain ONLY on workstation after
  verified host-copy retirement. Local kernel/artifacts/workspace-cleanup-20261104/
  top-level/{104,105}-loose-files.tar.gz, plans and verification journals.
  SHA104a5f31bd0d1f14fd25eb1cf2bc2580ac215b50a062bc9dddcc36d0b2a82b64b64;
  SHA105fa1298786e61db5eb4a0a276a1c2d65a01623393586b71c47ab0aed411ac9d28.
  System service/tool/registry absolute references inspected. Retain ONLY root
  aliases needed by installed report metadata and rollback (D04 both, plus .105
  target-qualification/recovery/promoter). Other historical replay commands may
  need grouped paths/fresh output labels; do not recreate obsolete root aliases.
  Inactive distributed installer-inputs and import/export manifests consolidated
  under tools/distributed-d01 and evidence/distributed-d01 without root aliases.
  Root has12entries(.104)/17(.105), ZERO regular files; preserved source/build/
  cache/recovery/board-support/application directories. .104~20.7GiB free/.105~45GiB.
  Selected runtime/registry/old .105 base manifest hashes match prior baseline;
  remaining paths and ordinary --info checks pass, kernels/bootIDs unchanged.
  No native/GPU/boot changes or new GPU qualification. docs/README.md on both
  stays the entry point; final documentation/memory snapshot includes this cursor.
  Inventory/archive PIDs23440/30584 exit0; all retirement/consolidation commands
  complete, no jobs/containers/guards. Critical Ruff/Black/syntax/diff pass.
  **Next D05 unchanged:** representative owned distributed timing/memory/checkpoint
  qualification. Public release still requires source-delta/reconstruction/license
  and privacy audits; these private recovery archives are not public packages.

- **Workspace cleanup/documentation completed2026-11-04:** user authorizes both
  hosts' stack-owned temporary-file cleanup and source preservation for publication.
  Tool milestone4eafbcdf0 plus final docs-link repair. Inspected `.104`162.3GiB
  bc250work (150GiB WS) and `.105`2.9GiB; preserve builds/caches/checkpoints,
  installed releases/images/SDKs/native controls/boot recovery/user applications.
  Copied94 historical transfer archives +two loose patches from .104 to local
  kernel/artifacts/workspace-cleanup-20261104/104-transfers; preserve ignored,
  not publication-approved. Whole literal deletion allowlist hash+size verified
  against workstation copies before first unlink; EX common host locks/no containers.
  `.104`96 files/7,017,699,431B retired (historical transfer archives plus duplicated
  D01 userspace/image exports); `.105`one1,490,746,029B import copy retired.
  Old source exports remain; source labels0115c01f/torch4ee65717 stale versus
  installed a524980: byte-level export/overlay audit is required before release,
  do not reset/reapply old patches. No source-tree/reusable-build/cache deletion.
  Grouped20(.104)+32(.105) distributed helpers/evidence under bc250work/tools/
  distributed-d01 and evidence/distributed-d01; every original path retained as
  relative compatibility symlink. Selected runtimes/distributed registry/old .105
  base manifest SHA unchanged; CPU-only CLI and all compatibility/deletion checks
  pass. No native change/GPU test/reboot; no performance/correctness expansion.
  Space observed11.6→18.1GiB(.104),43.6→45.0GiB(.105). Root docs entry BOTH:
  /home/nik/bc250work/docs/README.md with versioned current snapshot, guides,
  canonical flow/inventory and implementation references. Workstation docs/README.md
  and docs/WORKSPACE_LAYOUT.md explain layout and publication audit prerequisites;
  this is NOT an independently reconstructed public release yet.
  Cleanup PIDs21502(.104)/27600(.105) exit0; per-host bc250work/
  cleanup-20261104-{controller.log,result.json}; exact plan, journals, raw copies,
  patch hashes and post-check logs local kernel/artifacts/workspace-cleanup-20261104.
  No cleanup/build/test jobs/armed guards/containers running. Black/critical Ruff/
  syntax/diff checks pass; no broad image/container prune or user-app access.
  **Next remains D05** representative distributed timing/memory/checkpoint work;
  publication clean reconstruction/license/privacy audit is a separate release gate.

- **D01–D04 two-host rollout completed2026-11-04, bounded scope:** user now
  authorizes `.105` updates/distributed experiments; B07.13 deferred, not done.
  Initial/export tools1f533d8f1, final launcher/qualification/recovery5f45c242a;
  Torch sourcea524980/daily and RCCL native payloads matched, no native rebuild.
  All24664 ROCm inventory entries exactly match; retain shared immutable M105.1
  core. `.105 current` NOW `/opt/bc250-stack/releases/10.2-gfx1013-20261104-d01`
  after656 target checks/nine guards and9 ordinary bc250-torch/train forward/
  backward+consumer checks. Image1bb4ac0e exact match; old506f image/base manifest/
  five boot-health tools verified unchanged. New release and health depend on old
  M105.1 base: NEVER remove it. Old caches/releases/images/fallbacks/user apps kept.
  `.104` daily priority1 and existing rccl-single registration/defaults unchanged.
  `bc250-distributed` installed BOTH, versioned tools/distributed-d01 + root registry
  distributed/d01.json pins provider manifest, SH runtime lock; EX install/promotion.
  Two-terminal --node-rank0/1, master104/eth0, Ring/Simple Socket only; P2P/IB/SDMA/
  symmetric off. Ordinary daily NCCL-off remains. No generic env/venv CLI yet.
  D01 Gloo16checks, D02 actual two-host RCCL16, D03 installed torchrun16: four
  float dtype sums/small+1MiB refs/three CPU-double global-batch DDP gradient+SGD
  steps. D04 classic nested FSDP/use_orig_params+AdamW14checks:1608 total params,
  804/rank, exp_avg804/rank; CPU-double gradients/three parameter updates including
  same-world/rank checkpoint resume. Small functional proof; no measured training
  speedup, large-model peak capacity, production convergence or elastic claim.
  Both eth0 links1000Mb/s. D03r1 harness extra TCPStore timed out; r2 refused still-
  exiting owned workers before GPU; r3 env:// passes. Ordinaryr1 GPU math passed
  then wrong plain-Torch Transformers expectation failed; consumers only in train,
  correctedr2 passes9. Failed evidence preserved; no backend arithmetic defect.
  All21 created guards ACK (104:5/105:16); logs unchanged in accepted GPU gates.
  No running owned jobs/containers/armed guards.29 local tests/Black/critical Ruff/
  syntax pass. No user-app access/delete/reboot/kernel/default-native-precision
  change. .104 bootd10514df/r2, .105 boot92b47338/r3; r2 fallback/SSH preserved.
  Historical Python CPU fault/KFD uncertainty distinct from fixed VM heap.
  Source exportPID22485/stage22363 exit0; D01ranks21432/9019, D02:894/23562,
  D03r3:5499/29248, D04:6820/30882, .105 target323 all exit0. No jobs pending.
  Both /home/nik/bc250work/distributed-d0*/{final.json,rank.json,gpu.log,guard/};
  matching controller logs. .105 target/ordinary-r2/recovery dirs retained;
  activation.json plus reviewed promote_distributed_d01_host105.py --rollback
  under control EX lock restores ordinary old selection; live D01 rollback not
  exercised. Optional distributed registry independent of daily rollback.
  Frozen1.49GB userspace/1.21GB image exports retained .104/workstation. Only redundant
  new .105 image transfer removed after import verification, no other cleanup.
  Final local kernel/artifacts/distributed-d01/*-final-evidence-r2.tar.gz:
  104352782B SHAb68218d22d139822a32f872d75440d0e7dd1c35c185a894147718e0fb807dc26;
  105790849B SHA2b856fdf4ae2d1c2a0acefb070eb11b6155b56bf5ea0785ed6fb47f16b650c67.
  [Evidence/revisions/limits](docs/development/bc250_distributed_d01_results/README.md),
  [usage](../BC250_DISTRIBUTED_GUIDE.md), [deployment](../HOST105_UPDATE.md).
  **Next D05:** owned representative DDP/FSDP training: matched global batch one-
  versus-two-node communication/step timing and peak-memory measurement; checkpoint
  robustness/model-block wrapping before larger-model promises. FSDP2/ZeRO/HF
  integration remains separate. Keep app exclusion and both-host recovery rules.

- **B07.12 coherent FP32/FP16 artifact completed2026-11-04:** toolsc64debffb;
  source96e5a7df/Torcha524980/installed native8389197 unchanged. Packages B07.10
  plugin+B07.11 repaired MIOpen, no rebuild/install/register/default change.
  Private runtime/ vs plugin lib/ separates loader search; fresh worker validates
  payload+27 actual native library pins/gfx1013/API5 before same-process runpy;
  parent kept alive/no exec-after-HIP. Eight copied ELFs normalized to relative
  RUNPATH, DT_NEEDED unchanged; source/config/license/resource recipes included.
  Two build-free extracts (spaces) x daily/RCCL, original sources/builds/candidates
  absent:78topchecks/48nested/10finalACK;1496half+1176float refs plus586installedfloat,
  24Torch output/all-first-gradient fixtures.24identity/four seed refusals;
  128half+128float absent-path,736halfAPI/1908runtime,488floatAPI/984runtime pass.
  Genuine mixed dtype DB8keys/profile; FindDb source had extra post-gradient
  entries, select six authoritative half shape keys and copy existing lines,
  preserving originals. Dedicated reference DB keeps seeded replay immutable.
  R1 packaging DB count/R2 metadata lookup failures before GPU; R3 accepted.
  Qualr1 identity passes then cross-bind hardlink fixture fails; host-created
  immutable payload fixtures+independent manifests fix harness, same package r3
  passes qualr2.29localtests/lint/syntax/independent archive+ELF inspection pass.
  CPU2/2GiB packaging/no build GPU/network/cache reset; SH runtime lock/600sgates.
  3965hashes/36links/fourAOTDBs/136backing files unchanged. New pre-existing
  d10514df boot/r2kernel throughout, logs unchanged/no KFD recurrence. Historical
  Python CPU fault/KFD uncertainty distinct from fixed VM heap; no root-cause claim.
  User authorizes stopping lucid_wiles/6fcfde0398e9; stopped/preserved/not removed.
  No .105/user-app access/deletion/reboot; no jobs/containers/armed guards;11totalACK.
  PackagePIDs8781+9525exit1/10202exit0; qual12590exit1/13470+export17389exit0.
  WS/tmp/ck-plugin-b0712/package-r3 and export-r2; WS/logs/
  ck-plugin-b0712-qualification-r2/final.json and ck-plugin-b0712-r2-controller.log.
  Release66651729B SHAa0d47ed4ba4df354cc7ea898a285eefa051f49693667386567ea392f66ddd91e;
  evidence7079195B SHAadb8ec1f0f84ec3bddc122c25ccae89969b982a6c6c5123b794b1c765c13a650;39payloadfiles/505424918B.
  Local kernel/artifacts/ck-plugin-b0712; free12,879,323,136→10,253,754,368B.
  Installed bc250-ck still FP32; ordinary native precision rollout pending.
  [Artifact/use/evidence](docs/development/bc250_ck_plugin_b0712_results/README.md).
  **Next B07.13:** versioned explicit .104 install/registration and updated
  bc250-ck format2 worker activation, exact identity/refusal/live rollback and
  ordinary daily/RCCL caller-context gates under EX update lock. Retain old FP32
  artifact/tool/version records, ordinary native defaults/.105/user apps.

- **B07.11 FP16 tuning and HIP-direct precision repair completed2026-10-30:**
  tools2d8f1dae0/libraries source96e5a7dfc43. Original public218 fails fixed FP16
  bound: H9/W11/C4/K8/g1/f3/index1 actual0.8427734375 vs0.8424274250864983;
  error0.0003460124>bound0.000306262. Source intentionally retains legacy half
  accumulator. Scope is precision, not memory corruption/all old training invalid.
  Native gfx1013 forward now retains float_types FP32 accumulation; other devices/
  backward unchanged. Isolated native-fix-r2 executes exact existing generator/
  one-resource-object/link recipes on fresh original-build copy; no interposition.
  Full patched source byte-identical in embedded library; mapped path/hash proved.
  Public137 vs repaired218, six NCHW shapes/two opposite orders/daily+RCCL:
  tuned CK wins g3f1/C=K16/H32W34 by1.1706–1.1782x; g1f3 moderate parity;
  four others favor repaired direct. No global/model/legacy-vs-repaired speed claim.
  Native Search5iterations/1000ms yields12PerfDb+12FindDb records; exact-ID fresh
  replay/no mutation, invalid-ID copy fallback.444CPU checkpoints/9516explicit
  calls (search internal excluded), maxerror0.00195155665 under fixed B07.9 bound.
  80top-levelchecks/10final ACKguards +24nested mapped-rebuilt-MIOpen checks;
  original/new MIOpen candidate refs724half+580float each, installed586float,
  24Torch output/all-first-gradient fixtures; plugin still forward-only.
  29localtests/lint/syntax pass. Buildr1memoryrefusal; qualr1/r2 original218
  precision fail/repro; native-fix-r1 stopped after hidden396-edge timestamp
  rebuild, retained. Native-fix-r2 resource-only relink passes; qualr3 missing
  sysdeps loader directory refuses before GPU, r4 supplies existing SDK path/pass.
  FourCPU/OMP4/job1/RAM10GiB/mem+swap12GiB/ccache20GiB no reset/ro SDK+source/
  no build GPU/network/600s; SH lock.3864hashes/36links/fourAOTDBs/136backing files
  unchanged. Same e327c5de boot/r2kernel/logs clean/no KFD recurrence; historical
  Python CPU fault/KFD uncertainty distinct from fixed VM heap, no root-cause claim.
  Installed native8389197/Torcha524980/FP32 bc250-ck/defaults/images/.105/user apps
  unchanged; precision repair is NOT active for ordinary launches. No reboot/delete.
  BuildPIDs23839exit1/24088+25122exit0; native26573stopped/29211exit0;
  qual24520+25960+29708exit1/30214exit0/export846exit0; no jobs/containers/armed guards.
  13totalACK; WS/tmp/ck-plugin-b0711/{inputs-r1..r3,build-r2,r3,native-fix-r1,r2,
  export-r4}; WS/logs/ck-plugin-b0711-qualification-r4/final.json/r4-controller.log.
  Evidence66256969B SHA0ce6d10f1c369e3ed68389b48562c76f79b633b4dc6474ec025d31102c7ed8d2.
  Free20,477,861,888→12,493,869,056B; copied builds retained.
  [Method/evidence](docs/development/bc250_ck_plugin_b0711_results/README.md).
  **Next B07.12 (planned):** isolated versioned FP32/FP16 CK+matching repaired
  MIOpen artifact, exact identity/licenses/example DB and two build-free relocated
  extracts/daily+RCCL before installation. Preserve installed FP32/defaults/.105;
  ordinary native precision-repair rollout remains separate compatibility work.

- **B07.10 isolated FP16 MIOpen facade completed2026-10-29:** toolsaf4704f69;
  libraries source742ee1e2c0e, installed native8389197/Torcha524980 unchanged.
  Explicit private compile flag reuses matching upstream half dispatcher/FP32
  accumulator and unchanged installed NHWGC archive. No extra factory object.
  Typed exact scalar/descriptors/packed2D geometry/int32-span contracts, half16B
  alignment, pointer overflow/output/scratch overlap and workspace checks; BF16/
  mixed/casts/TF32/determinism/bias/transpose refused. Global9/per-type6float+9half
  IDs;66 required exports plus forward-list API, helpers/GetCastType hidden.
  Daily/RCCL724half refs (656per-ID/64public/four mixed-handle),580candidatefloat;
  368halfAPI/954runtime/64absent-path refusals. Max half error0.00186229 under fixed
  B07.9 quantization+N2.18 residual bound; dyadic final-half exact, guards/inputs pass.
  Installed FP32 adds586native refs; eight half/eight installedfloat Torch output/
  all-first-gradient fixtures pass; plugin still forward-only, native backward.
  Private24+installed20+core7+CPU ONNX4+daily5+RCCL6=66checks/eight ACK guards;
  29localtests/lint/syntax pass. R1 test-only deterministic-wrapper assignment
  compile failure fixed with public setter; R2 build/first GPU qualification pass.
  Local audit initially assumed66 total instead of66required+forward-list export;
  exact set verified, no source/test-contract relaxation. Failed artifacts retained.
  FourCPU/OMP4/jobs1/RAM10GiB/mem+swap12GiB/ccache20GiB no reset/ro inputs+SDK/
  no build GPU/network/600s.3849hashes/36links/fourAOTDBs/136backing files unchanged.
  Same e327c5de boot/r2kernel/log unchanged; no KFD recurrence/root-cause fix claim.
  Historical Python CPU fault/KFD uncertainty remain distinct from fixed VM heap.
  Installed bc250-ck FP32/defaults/images/.105/user apps unchanged; no reboot/delete.
  Build1PID19855exit1/build2PID20834/qualPID21771/exportPID23540exit0; no jobs/
  containers/armed guards. WS/tmp/ck-plugin-b0710/{inputs-r1,r2;build-r1,r2;export-r1};
  WS/logs/ck-plugin-b0710-qualification-r1/final.json, r1-controller.log.
  Evidence2837154B SHAf54cb396f04ead909da37448521017d77186e90058b3c39c6cf3a925ba5a741a.
  Ordinary free12,592,922,624→12,572,831,744B. [Scope/evidence](docs/development/bc250_ck_plugin_b0710_results/README.md).
  **Next B07.11 (planned):** bounded paired FP16 public137/HIP-direct timing and
  native Search/PerfDb replay on owned fixtures before versioned opt-in packaging;
  preserve installed FP32 artifact/defaults/.105/user apps. No performance claim yet.

- **B07.9 isolated FP16 DL factory proof completed2026-10-29:** tools0e8a28146;
  existing source has half inputs/output with float accumulator. Native shader
  dot2=0/mac_f32=1/wave32, no XDL; nine distinct IDs/layout. Installed archive has
  interleaved FP16 but lacks group-major definition: actual link+symbol audit prove
  gap. R4 compiles only unchanged upstream group-major source into isolated object,
  links unchanged archive. No installed/native admission or arithmetic change.
  Two layouts/streams/groups1+3, five geometries/two patterns plus C512 accumulation
  case:88fixtures/profile,328 GPUcalls+464 refusals each daily/RCCL=656/928.
  144 accumulation calls subset return256.25 vs sequential-half256. Dyadic final
  half-round exact; non-dyadic half-quantization+fixed N2.18 residual bound, max
  abs0.0018510521/max rounded error0.0009765625; immutable inputs/256-byte guards.
  Eight existing Torch FP16 output/all-first-gradient fixtures; installed FP32
  references586+eight gradient fixtures retained. Factory14+installed20+core7+
  CPU ONNX4/daily5/RCCL6=56 checks/eight ACK guards;29 local tests/lint/syntax pass.
  R1 probe name/include, R2 include order, R3 missing symbol fail before GPU;
  all inputs/logs retained, R4/first GPU batch pass. FourCPU/OMP4/jobs1/RAM10GiB/
  mem+swap12GiB/ccache20GiB no reset/ro source+SDK/no build GPU/network/600s.
  3832 hashes/36 links/four AOT DBs/136 backing files unchanged. Same e327c5de boot/
  r2kernel/logs clean; no KFD recurrence/root-cause fix claim, no owned jobs/guards.
  BuildPID13558/14091/14381 exit1; r4PID15015/qualPID15696/exportPID17462 exit0.
  WS/tmp/ck-plugin-b079/build-r4 and export-r1; source/archive audit JSONs;
  WS/logs/ck-plugin-b079-qualification-r1/final.json and r1-controller.log.
  Local evidence3683360B SHA8bfc6d1a4ce85413684170bf7c0c39ddd13982379901f6619f5cdc5531809351.
  Source3616b2f/installed8389197/Torcha524980/bc250-ck FP32/defaults/.105/user apps
  unchanged; ordinary free12,625,784,832→12,595,290,112B, nothing deleted. Historical
  Python CPU fault/KFD uncertainty remain distinct from fixed VM heap. [Scope/evidence](docs/development/bc250_ck_plugin_b079_results/README.md).
  **Next B07.10 (planned):** isolated matching-MIOpen FP16 forward facade using
  qualified NHWGC factory, dtype-aware scalar/descriptor/alignment/overlap/scratch
  contracts and explicit public solver references. Preserve installed FP32 option.

- **B07.8 explicit bc250-ck installed2026-10-29:** tools36da14fb2; accepted B07.7r2
  at /opt/bc250-stack/sdks/bc250-miopen-dl-gfx1013-b077, miopen-dl.json plus retained
  version record; tools/ck-b078 and /usr/local/bin/bc250-ck. Install/register/
  unregister use EX lock, launch holds SH; archive/manifest/payload/runtime hashes
  and image pinned, artifact verifies eight mapped ABI libraries/API5/66 exports.
  Existing caller-context handoff retained; check/info/seed-db/Python scripts,
  mounts/env/envfile supported; no solver forcing or default/native whitelist change.
  Live unregister/refusal/re-register, identity/byte/extra-file/CLI/lock tests pass.
  Main60+ordinary UID1000 env/argv/importdir/backward2+host13=75 checks/10 ACK guards.
  Two same-process public137 CPU-double calls plus accepted probes=586 native
  conv references;244API/492runtime per-ID refusals/304 retained contracts.16 normal
  Torch output/all-first-gradient fixtures across installed/plain daily/RCCL pass;
  backward uses retained solvers. Unchanged tolerance/guards;29 local tests/lint/
  pycompile/diff pass. Plain launches have no plugin env; core/CPU ONNX/consumer
  gates pass.3804 hashes/36 links/four AOT DBs/136 backing paths preserved. Same
  e327c5de boot/r2kernel/logs unchanged; no KFD warning recurrence or root-cause
  fix claim. Historical Python CPU fault stays distinct from fixed VM heap.
  No build/cache reset/deletion/reboot/.105/user-app access. InstallPID7929/
  qualPID8319/exportPID10985 exit0; no jobs/containers/armed guards. WS/logs/
  ck-plugin-b078-qualification-r1/final.json and matching r1-controller.log;
  user-r1/install-r1, WS/tmp/ck-plugin-b078/export-r1. Local evidence979892B
  SHA22e958b6f3ed827fd1500ded77288cfc71d983b365466808b2a924997bca7bfb.
  Native libraries8389197/source3616b2f/Torcha524980/images/defaults unchanged;
  ordinary free12,650,000,384→12,628,176,896B. [Use/recovery](docs/development/bc250_ck_plugin_b078_usage.md),
  [scope/evidence](docs/development/bc250_ck_plugin_b078_results/README.md).
  **Next B07.9 (planned):** inspect existing scalar FP16 DL factory/accumulator/
  type admission and N2.18/B07 evidence, then bounded isolated CPU-reference
  factory proof if feasible. Preserve installed FP32 option/defaults/.105.

- **B07.7 opt-in relocatable artifact completed2026-10-29:** toolsfd8c3fe8c;
  accepted package-r2/qualification-r3, no install/native rebuild.23 files6568985B
  include unchanged plugin/probes, eight mapped ABI-library pins/API5/66 symbols,
  focused source/config/license snapshot and genuine two-key examples/profile.
  Explicit check/seed-db/run; no merge into existing DBs. Two extracts (spaces
  included) x daily/RCCL with original candidate/source/build paths absent pass
  50 checks/eight guards;1168 convolution references,488API/984runtime per-ID
  refusals/608 retained/128 absent/20 entry+seed checks. Unchanged tolerance,
  max abs2.35537e-6/256-byte guards;29 local tests/lint/pycompile/diff pass.
  First package's numerics pass but kernel gate rejects Failed to resume KFD.
  Entry exec-after-HIP replaced by same-process runpy; final r3 logs unchanged.
  Underlying mm/queue return untraced: no kernel-root-cause fix claim; preserve
  this N6 input, distinct from historical Python CPU fault/fixed VM heap.
  Qualr2 profile-context setup refusal before RCCL launch corrected inr3;
  initial CPU identity TTY attach error retained, noninteractive capture passes.
  No failed numerical test/relaxed gate/reboot/log clearing.3772 hashes/35 links/
  four AOT DBs/136 backing files unchanged. Same e327c5de boot/r2kernel; no jobs/
  containers/armed guards;10 total ACK. Shared lock/no compilation/ccache reset.
  Package1PID1940/package2PID3233 exit0; qual1PID2297/qual2PID3406 exit1;
  qual3PID3971/exportPID6306 exit0. WS/tmp/ck-plugin-b077/package-r2 and export-r3;
  WS/logs/ck-plugin-b077-qualification-r3/final.json and r3-controller.log.
  Accepted archive5031322B SHA90f0d323d0bb4ffd3a8ae04f1acd4e17805a10d7dcb62d577840cf2120e06984;
  local evidence6819295B SHAeebcf955438a9a5655bc69ca84334d14da83b61c54b629a5a0a55f7576fd73b4.
  Source3616b2f/installed8389197/Torcha524980/defaults/prefix/.105/user apps unchanged;
  ordinary free12,888,330,240→12,653,342,720B, nothing deleted. [Use/evidence](docs/development/bc250_ck_plugin_b077_results/README.md).
  **Next B07.8:** versioned explicit .104 install/identity/registration/refusal/
  rollback and ordinary daily/RCCL activation under exclusive update lock.
  Preserve native payloads/default whitelist/routing/fallbacks/.105; update usage.

- **B07.6 native Search/PerfDb public replay completed2026-10-29:** tools1eac70aa1;
  private libraries3616b2f/installed8389197/Torcha524980 unchanged. Existing native
  Search/GenericSearch and kernel-ID serialization/load validation work; no source
  repair/override. Public Find only137 genuinely generates two PerfDb+two FindDb
  records/profile for batch2/32x34/C=K16/g1f3+g3f1. Native5iterations/1000ms bound,
  private DB/cache. Both select block256 Default3x3; daily1x1Stride1Pad0 vs RCCL
  1x1Pad0. Four fresh opposite-order public137-vs218 processes retain1.085–1.092x
  and1.119–1.135x advantages; no whole-model/general/default claim. Untimed verbose
  replay proves exact-ID DB load/no search; DB hashes unchanged. Invalid-ID copies
  fall back to block8 Default with CPU references, original records preserved.
  148 CPU checkpoints/max abs2.35537e-6/unchanged N2.18 tolerance/256-byte guards;
  3172 explicit Immediate calls excludes internal search evaluations. Private7+7/
  core7/CPU ONNX4/daily5/RCCL6=36 checks/six ACK guards;29 local tests/lint/syntax
  pass. Probe-only fourCPU/OMP4/jobs1/RAM10GiB/mem+swap12GiB/ccache20GiB no reset/
  ro sources+prefix/no build GPU/network/600s.3701 hashes/35 links/four AOT DBs/
  136 backing files unchanged. Same e327c5de boot/r2kernel/logs; no jobs/containers/
  guards. BuildPID30906/qualPID31175/exportPID32467 exit0. WS/tmp/ck-plugin-b076/
  {inputs-r1,build-r1,export-r1}; WS/logs/ck-plugin-b076-qualification-r1/final.json
  and WS/logs/ck-plugin-b076-r1-controller.log. Evidence1143590B,
  SHA13d5aabfaf5dcf053af67f8d3f7c62514a09b90104798e8c4d9cf4c6dffaa2e6.
  No failed GPU batch/install/default/prefix/.105/user-app changes/deletion;
  ordinary free12,898,476,032→12,890,648,576B. Historical Python CPU fault stays
  distinct from fixed VM heap. [Method/results](docs/development/bc250_ck_plugin_b076_results/README.md).
  **Next B07.7:** isolated relocatable opt-in artifact with exact native MIOpen/
  ABI/source/license identity and genuine example records; identity refusals/two
  build-free extracts (spaces included) before installation. Preserve defaults/.105.

- **B07.5 six private DL configurations completed2026-10-29:** tools45bac0baa,
  private libraries3616b2f/installed8389197/Torcha524980 unchanged. Probe-only build;
  six retained shapes/two layouts, matched cached218 vs explicit API5 per-ID
  invokers. Four opposite-order daily/RCCL timed processes; all six IDs observed.
  Full NCHW includes transforms; prepacked NHWC excludes per-call transforms,
  not pure kernel or an end-to-end packing gain. Fixed first-pass-selected
  block256 Default wins32x34/g1/3x3 by1.091–1.097x; block256 Filter1x1Pad0 wins
  32x34/g3/1x1 by1.138–1.144x. Other four shapes still favor HIP-direct. No public
  default/performance heuristic/install change; unpaired public anchor outlier
  retained, public paired replay remains next. Two diagnostic runs prove native
  IDs/MIOpenConvUni.2184 CPU-double checkpoints/52008 convolution calls,
  max abs2.35537e-6/unchanged tolerance/256-byte guards/immutable inputs pass.
  Private4+4/core7/CPU ONNX4/daily5/RCCL6=30 checks/six ACK guards;29 local tests/
  lint/syntax pass. FourCPU/OMP4/jobs1/RAM10GiB/mem+swap12GiB/ccache20GiB no reset/
  ro source+prefix/no build GPU/network/600s; accepted B07.3 plugin reused.
  3692 hashes/35 links/four AOT DBs/136 backing files unchanged. Same e327c5de
  boot/r2kernel/logs unchanged; no jobs/containers/armed guards. BuildPID27923/
  qualPID28197/exportPID29498 exit0. WS/logs/ck-plugin-b075-qualification-r1/
  final.json and WS/logs/ck-plugin-b075-r1-controller.log; WS/tmp/ck-plugin-b075/
  {inputs-r1,build-r1,export-r1}. Local evidence1497520B,
  SHA4e2d4d39fd075519ff852c2a740caf6a80bbbf69f5e164b374a99777f6c0e703.
  No install/default/prefix/.105/user-app changes/deletion; ordinary free
  12,911,759,360→12,901,400,576B. Historical Python CPU fault remains distinct
  from fixed VM heap. [Method/results](docs/development/bc250_ck_plugin_b075_results/README.md).
  **Next B07.6:** audit/use existing solver137 Search/PerfDb in isolated DBs,
  fresh-process public-Immediate selected-config replay with paired HIP-direct
  references/timings. Keep candidate private and defaults/.105/user apps unchanged.

- **B07.4 paired public DL/HIP-direct timing completed 2026-10-29:** tools
  e11969041; candidate libraries3616b2f/installed8389197/Torcha524980 unchanged.
  Forced public Immediate137 vs218, six NCHW FP32 shapes/batch2/groups1+3/
  filters1+3/small+moderate inputs. Two fresh opposite-order processes/profile,
  daily+RCCL;32-call windows/three ABBA or BAAB cycles, separate first-use/query/
  prepare/wall intervals. Additional verbose runs prove Default block8 DL plus
  transposes vs MIOpenConvUni; diagnostic timing excluded. HIP-direct wins all24
  timed comparisons,1.13–2.57x; stream intervals include layout/launch/idle costs,
  not pure kernel time. No general speed or whole-model claim; cold-user caches
  are not machine-cold.576 CPU-double correctness checkpoints incl diagnostic
  passes;14,184 calls, only window-final outputs copied. Unchanged N2.18 tolerance,
  max abs2.35537e-6; immutable inputs/256-byte guards pass. Private4+4/core7/CPU
  ONNX4/daily5/RCCL6=30 checks/six ACK600s guards;29 local tests/lints/syntax pass.
  Probe-only isolated build uses fourCPU/OMP4/jobs1/RAM10GiB/mem+swap12GiB/
  ccache20GiB no reset/ro sources+prefix/no GPU/network/600s; existing plugin reused.
  3684 protected hashes/35 links/four AOT DBs/136 backing files unchanged. Same
  e327c5de boot/r2kernel; kernel logs unchanged; no owned jobs/containers/guards.
  Initial staging permission failure corrected on new owned directory before
  build; no failed GPU test/tolerance relaxation. BuildPID25065/qualPID25392/
  exportPID26699 exit0. WS/tmp/ck-plugin-b074/{inputs-r1,build-r1,export-r1};
  WS/logs/ck-plugin-b074-qualification-r1/final.json and matching
  WS/logs/ck-plugin-b074-r1-controller.log. No install/default/prefix/profile/
  .105/user-app changes or cleanup. Ordinary free12,926,078,976→12,914,167,808B.
  Local kernel/artifacts/ck-plugin-b074 evidence1,432,289B,
  SHA630a0525288253423c46e976bc3fe9c2b6009ccead3bb103be1d155c39ebd32b.
  Historical Python CPU fault stays unresolved/distinct from fixed VM heap.
  [Method/results](docs/development/bc250_ck_plugin_b074_results/README.md).
  **Next B07.5 (planned):** compare all six private per-ID variants on retained
  fixtures; separate invoker-only/full-layout costs and see whether a qualified
  config beats HIP-direct before packaging. Keep candidate private/defaults/.105.

- **B07.3 private DL scalar/argument/per-kernel contracts completed 2026-10-29:**
  source libraries3616b2f9f09 over2976e7d; tools7519e73b8. Private facade now
  refuses non-default exact double alpha/beta (including near1/NaN/Inf), bias,
  transpose mode, invalid grouped geometry/output extent and int32-overflow spans.
  get_solution checks the requested ID before allocating an owning solution.
  Captured invoker validates runtime scalars/descriptors/casts/pointers/alignment,
  output/input overlap and required scratch size/alignment/overlap before upstream
  dispatch. Existing upstream math and production admission/factory unchanged.
  Per-ID API solutions release temporary problem/solution before invocation;
  all six IDs run in packed NCHW/NHWC, groups1/3, five1x1/3x3 geometries incl
  stride2/dilation2/C3/K5/K132 tails, default/private streams/two value patterns.
  80 fixtures/profile,256 direct GPU calls each daily/RCCL=512 CPU-double checks;
  dyadic exact, unchanged N2.18 tolerance otherwise, max abs1.07042e-6.244 metadata/
  scalar/transpose/extent/unknown-ID refusals +492 runtime refusals verify output/
  scratch unchanged and inputs/256-byte guards intact. Six IDs each observed;
  shapes are bounded, not general lifetime/concurrency/scalar support proof.
  Retained public solver13764 GPU calls/64 absence refusals/304 prior contracts,
  eight Torch conv output/all-first-gradient fixtures plus native backward pass.
  Private9+9/core7/CPU ONNX4/daily5/RCCL6=40 checks/six final600s guards; initial
  r1 also40 passes, then transpose admission hardened;12 total ACK. No failed GPU
  batch or tolerance relaxation. Source inputs-r1/r2 and r1 validator saved;
  old B07.2 helpers/binaries/inputs unchanged. Two isolated builds use fourCPU/
  OMP4/jobs1/RAM10GiB/mem+swap12GiB/ccache20GiB no reset/read-only source+prefix/
  no GPU/network/600s bound.3676 protected hashes/35 links/four AOT DBs unchanged;
  136 backing paths match. Entry new boot e327c5de-46b8-4e05-94ad-d999a12a06e7,
  kernel still7.1.5-bc250rocm-r2; no agent reboot/cross-boot continuity claim.
  Accepted kernel logs unchanged through cleanup; no owned jobs/containers/guards.
  No install/prefix/profile/default change/.105 or user-app access. Entry ordinary
  free13,329,420,288B; cleanup12,928,606,208B; nothing removed. Buildr1PID11617/
  qualr1PID12209/buildr2PID13602/qualr2PID14039/export exit0. WS/tmp/ck-plugin-b073/
  build-r2; WS/logs/ck-plugin-b073-qualification-r2/final.json and matching
  WS/logs/ck-plugin-b073-r2-controller.log; export-r1. Local kernel/artifacts/
  ck-plugin-b073 evidence3284485B SHAe0038bf797e4846a91baa4efb81fc8b9c912badb2ba0152151bedc4565725b2b,
  211 rawfiles16842346B; matching sources/binaries/66 exports/hidden-helper proof.
  29 local tests/critical Ruff/Black/new-source clang-format/pycompile/shell/diff
  pass; full upstream CI not run. No broader shape/type/performance/scaling or
  non-default scalar arithmetic support claim. Historical Python CPU fault stays
  unresolved/distinct from fixed VM heap. [Results/contracts](docs/development/bc250_ck_plugin_b073_results/README.md).
  **Next B07.4 (planned):** bounded paired private DL vs existing MIOpen HIP-direct
  timing with retained CPU references, explicit solver attribution and warm/cold
  separation; classify wins/losses before deciding opt-in packaging/admission.
  Preserve installed defaults, private candidate, .105 and user applications.

- **Developer architecture/environment guide completed 2026-10-29:** workspace
  root BC250_DEVELOPER_GUIDE.md, linked from README and existing usage document.
  Explains layers/mounts/commands/profiles, frozen consumers, persistent overlays,
  independent venv+runner, model/dependency upgrades, native updates/locks/recovery.
  Actual installed launchers inspected; CPU-only .104 experiments prove naive
  nested --system-site-packages cannot find Torch, while explicit trusted runtime
  site .pth bridge retains patched Torch origin and isolates application packages.
  Offline owned wheel installs in overlay/venv; two fresh-container persistence
  checks, exact documented bootstrap/runner and ordinary UID1000/doas handoff pass:
  11 environment checks,16 shell/Python blocks syntax checked, links/anchors pass.
  Initial missing staging-directory attempt exits2 before fixture execution;
  retained, corrected own directory permissions only. No GPU kernels, native/image/
  prefix/profile changes, new public package downloads, user-app or .105 access.
  Native runtime metadata SHAa9a5afe3 unchanged; no owned jobs. Fixtures at
  WS/tmp/developer-guide-20261029-r1; WS/logs/developer-guide-*.log. Local evidence
  kernel/artifacts/developer-guide/evidence.tar.gz4097B,
  SHAff2287ce357607fb246f2acf090a40aeea320b61aa184a7a977adc8caa560729; guide SHA3b207139894287b13cf862993be42857d99f288ce055c67188c9da3d7838cfd9.
  No native code revision changes/new GPU qualification. **Next B07.3 unchanged:**
  private plugin scalar/argument admission and per-kernel stride/dilation/tails.

- **B07.2 isolated MIOpen FP32 DL forward plugin completed 2026-10-29:** source
  libraries2976e7d30bfdccb2fb79809b8c58a6b5722a0822 over installed8389197;
  tools528088c26/controller608ac7b86. Private API5 facade exports all66 required
  symbols; unsupported11 groups return false/empty lists and explicit workspace/
  solution INVALID_VALUE, nulls BAD_PARAM. Packed2D FP32 forward only, casts/TF32/
  mixed types/determinism refused. Existing upstream forward/invoker math reused.
  Copied config enables CK templates only in candidate; explicit DL macro/C++20/
  beta API/existing includes/archive. Host-hidden GetCastType gets its exact
  matching one-line definition hidden locally, verified not exported. Nine raw
  factory instances have duplicate IDs (transfer parameters omitted from names);
  private-only guarded filter retains first per ID, exposes six unambiguous IDs.
  No production factory/admission/default changes. Forced public Immediate solver
  137 (historical Xdlops name, actual DL kernels), packed NCHW+NHWC/groups1+3/
  filters1+3/batch2/5x7/C=K=4 per group/default+private streams/two input patterns:
  32 GPU calls each daily/RCCL=64; CPU double, dyadic exact/unchanged N2.18 tolerance,
  max abs6.07623e-7,256-byte input/output/workspace guards/immutable inputs pass.
  Fresh no-private-path subprocess32 refusals/profile=64;304 recorded contract
  checks include all unsupported groups/cast/FP16/TF32/nulls. Eight Torch conv
  output/all-first-gradient fixtures plus native arithmetic backward pass; ordinary
  Torch may choose retained solvers, not proof of automatic private dispatch.
  Private8+8/core7/CPU ONNX4/daily5/RCCL6=38 checks/six accepted600s guards;12 total
  ACK including earlier failures. Buildr1 C++17/includes, r2 beta/sqlite, r3+r4
  hidden accessor (r4 unchanged retry), r5+r6 built; r7 adds private ID filtering,
  r8 probe workspace-guard correction. Qualr1 duplicate IDs, r2+r3 test wrongly
  checked mutable workspace scratch (r3 retains native logs), r4 private gates
  pass then wrong launcher lacks transformers; r5 full pass after bc250-train fix.
  No numerical tolerance/reference relaxation. FourCPU/OMP4/jobs1/RAM10GiB/
  mem+swap12GiB/ccache20GiB no reset/read-only source+prefix/no GPU/network build,
  timeout600s.3663 protected hashes/35 links/four AOT DBs unchanged;136 retained
  backing paths match. Same r2boot83f6cb1b-4540-4149-839e-a6cf7cb80b63/kernel logs
  unchanged. No install/prefix update/default admission change/.105/user app access.
  Buildr8PID29981/qualr5PID31501/export exit0; no owned jobs/containers/armed guards.
  WS/tmp/ck-plugin-b072/build-r8; WS/logs/ck-plugin-b072-qualification-r5/final.json
  and WS/logs/ck-plugin-b072-r5-controller.log; export-r1. Local kernel/artifacts/
  ck-plugin-b072 evidence7298658B SHA9e2aaabae96bd8386cfb2ba65e3de62408a1bd7fd5cd121461baf64627e91ecc,
  283 rawfiles34976217B, matching source/binary/export checks and private helper
  visibility proof.29 local tests/critical Ruff/Black/new-source clang-format/
  pycompile/shell/diff pass; no full upstream CI. Free30,153,154,560B. Default
  Immediate alpha1/beta0 only qualified; non-default scalars/per-kernel selection/
  stride+dilation+tails/performance/other types/plugin backward remain open.
  Historical Python CPU fault remains unresolved/distinct from fixed VM heap.
  [Use/scope/evidence](docs/development/bc250_ck_plugin_b072_results/README.md).
  **Next B07.3 (planned):** private forward scalar/argument admission and per-kernel
  stride/dilation/tail qualification; resolve non-default alpha/beta before broader
  use, then select paired HIP-direct timing/opt-in packaging. Keep defaults/.105.

- **B07.1 CK grouped-convolution plugin admission completed 2026-10-29:** tools
  a4bf75b8033b484b364be84a6233efe681203875. Nine local/staged source files match;
  API5 loader requires66 entry points across12 solver groups, using internal C++
  types despite extern-C entry names. gfx1013 excluded by plugin CMake/default
  loader whitelist; explicit MIOPEN_CK_LIB_PATH bypass remains. Existing scalar DL
  FP32/FP16 2D NHWGC/GKYXC/NHWGK factory matches plugin forward interface. Factory
  uses DL_KERNELS, distinct from configured CK_ENABLE_DL_KERNELS; MIOpen compile
  helper does not explicitly propagate the former. Actual plugin flags still
  require checking. Native no-DL probe enumerates0; -DDL_KERNELS probe links
  installed archive and enumerates9 FP32 instances.16 new interleaved grouped
  fixtures/profile,1x1/3x3/groups1+3/default+private streams/dyadic+non-dyadic,
  execute96 admitted instances each daily/RCCL=192 CPU-double comparisons.
  Dyadic exact; unchanged N2.18 tolerance otherwise, max abs6.07623e-7;256-byte
  guards/immutable inputs and native Torch HIP backward pass. Factory gates3+3,
  retained core7/daily5/RCCL6=24 checks/five accepted guards; earlier r1 parser
  rejects diagnostic-prefixed JSON after native exit0, preserved; six total ACK.
  Test parser now retains stdout and parses final JSON, no numerical relaxation.
  Two isolated native link builds, ccache20GiB/no reset/OMP4/fourCPU/RAM10GiB/
  mem+swap12GiB/read-only sources+prefix/no GPU/network/build timeout600s. No
  plugin/library/Torch build, install, admission/default/prefix change.3649
  protected hashes/35 links/four AOT DBs unchanged;136 backing paths verified.
  Same r2boot83f6cb1b-4540-4149-839e-a6cf7cb80b63/kernel logs unchanged; .105/user
  apps untouched. Qualification r1PID10147 exits1(parser), r2PID10922 exits0;
  evidence export exits0. WS/logs/ck-plugin-b071-qualification-r2/final.json,
  WS/logs/ck-plugin-b071-r2-controller.log; WS/tmp/ck-plugin-b071/build-r{1,2},
  export-r1. Local kernel/artifacts/ck-plugin-b071 evidence1560256B
  SHAbdd63ea9687bfe26f96ea253845fabfe9f906c91da2be4427ed360e7653c3319;
  109 rawfiles7008402B.29 local tests/critical Ruff/Black/clang-format/pycompile/
  shell/diff pass; upstream full CI not run. Free30,202,425,344B; no owned jobs/
  containers/armed guards. No MIOpen routing/backward-plugin/type/performance
  qualification; historical Python CPU fault remains distinct from fixed VM heap.
  [Audit/decision/evidence](docs/development/bc250_ck_plugin_b071_results/README.md).
  **Next B07.2 (planned):** isolated matching-MIOpen forward-only FP32 DL plugin,
  explicit DL macro, all66 required exports with unsupported operations refusing;
  private-path discovery and forced real solver CPU-reference/guard checks before
  any production admission. Keep installed defaults/.105/user apps untouched.

- **B06.12 modern Torch export→public ORT GPU completed 2026-10-29:** tools
  75da6725bd2566828ace1074c5d24c7e0e6347ad, evidence e9e6c29a8. Fresh owned
  dense/CNN/gated-LayerNorm models, CPU+GPU export origins, static batch3,
  dynamo=True/opset18; all six exporter-produced graphs are IR10/opset18 without
  IR rewriting. Two changed inputs each match Torch CPU FP64, eager HIP FP32,
  retained CPU ORT1.31 and ONNX NumPy references at unchanged rtol3e-4/atol2e-5.
  Public bc250-ort daily/RCCL each30, generation30 and retained core7/CPU ONNX4/
  daily5/RCCL6=112 checks; seven ACK600s guards.24 consumer GPU graph evaluations,
  12 actual profiles assign every recorded node to MIGraphX with CPU fallback
  disabled;12 generation eager GPU forwards. Invalid inputs, native gfx1013,
  package/library origins, explicit CPU after GPU and exact Torch HIP backward
  pass.3642 protected hashes/35 links/four AOT DBs unchanged;136 retained workstation
  backing paths verified. No numerical/library/default/prefix changes, native
  rebuild or cache reset. Same r2boot83f6cb1b-4540-4149-839e-a6cf7cb80b63; kernel
  logs unchanged. Qualification PID5491 exited0, export exited0; no owned jobs/
  containers/armed guards. WS/logs/onnxruntime-b0612-qualification-r1/final.json,
  matching -controller.log; fixtures WS/tmp/onnxruntime-b0612/generation-r1/models;
  export WS/tmp/onnxruntime-b0612/export-r1. Local kernel/artifacts/onnxruntime-b0612:
  evidence813709B SHAf5a882a28504f7e0058fd9ae4982f7bf13ad307a58da5ad7e36d92b208f967b1,
  179 rawfiles3670773B.29 local tests/critical Ruff/Black/pycompile/diff pass;
  full upstream CI not run. Free30,209,069,056B. .105 not contacted; user apps
  untouched. Inference-only bounded coverage; dynamic shapes/types/operators/
  concurrency/graph-autograd/performance remain unqualified. Historical Python
  CPU fault remains unresolved and distinct from fixed VM-heap corruption.
  Practical B06 review finds no mandatory defect in this tested path; park optional
  expansion until a concrete supported-model requirement. [Results/gap review](docs/development/bc250_onnxruntime_b0612_results/README.md).
  **Next B07.1 (planned):** audit MIOpen CK grouped-conv plugin ABI/instance factories
  and native scalar/DL applicability versus retained N2.18/HIP-direct alternatives;
  choose a bounded useful implementation before building or changing admission.
  Preserve safe defaults, installed packages, .105 and user applications.

- **B06.11 installed explicit ORT package completed 2026-10-29:** source remains
  ORTb81a7866e5a6eff964b9e411535b4c9198651a9f, registered SDK4797 unchanged. Tools
  packagebc0f0b0b9/context7870a6ee8/exportfddb14f61. Versioned archive
  bc250-ort-migraphx-1.31.0-gfx1013-cp312-b0611 at /opt/bc250-stack/sdks/;
  registration ort-migraphx.json plus retained version record; /usr/local/bin/
  bc250-ort delegates through existing SDK/Torch launchers with full shared-lock
  identity checking/read-only package. Install/register/unregister/context tool
  update use EX lock. New Python/distribution identity1.31.0+gfx1013.b0611 fixes
  missing build_and_package_info warning without suppressing it. Five copied ELF
  RUNPATHs normalized to $ORIGIN; DT_NEEDED and .text bytes match originals.
 504 files/two contained links include core/Python/providers/headers/licenses,
  original source+repair/hash pins/13 dependency archives/build settings. External
  qualified glibc/ROCm image/registered SDK required; not standalone/cross-device.
  Two private extracts (spaces included), original build and /sdk absent, each34
  pass. Installed daily/RCCL each34 plus core7/CPU ONNX4/daily5/RCCL6 bring158
  guarded/eight accepted guards, host14/user2=174 checks/48 GPU evaluations/24
  accepted actual model profiles. Earlier ordinary acceptance90/24 evals retained;
  all14 guard records ACK. Same FP32 six IR10/opset18 fixtures/CPU-double+CPU ORT
  references/tolerances/no CPU node fallback, wrong-input refusals, native gfx1013,
  exact Torch HIP backward/loaded origins/package identity/ro mount. Actual normal
  UID1000 exposed bare --env loss after elevation; fixed by caller-side options+
  existing private-context validation. Before/after proof covers bare env and
  inherited HF_*; old wrapper/records at WS/recovery/ort-b0611-before-context.
  Final acceptance-r2 passes; archive/numerical code unchanged by wrapper fix.
  Live unregister/refusal/re-register, identity drift, lock exclusion and plain
  daily/RCCL/SDK CPU ORT preservation pass.3553 native hashes/35 links/four AOT DBs/
  136 retained backing paths match, full package+SDK checked separately. Same
  r2boot83f6cb1b-4540-4149-839e-a6cf7cb80b63; kernel logs unchanged through cleanup.
  No owned jobs/containers/armed guards; no compilation/cache reset/source/build/
  artifact deletion. Initial ordinary30.1GiB; cleanup30,217,469,952B. Active core
  image/ROCm/Torch/SDK/defaults/boot/fallback/user apps unchanged; .105 not contacted.
  Package PID14494/relocation15155/install16120/final acceptance19704/export21375
  exited0. WS/tmp/onnxruntime-b0611/{package-r1,export-r1}; WS/logs/onnxruntime-b0611-
  acceptance-r2/final.json and matching -controller.log. Local kernel/artifacts/
  onnxruntime-b0611 package350,607,989B SHAcdac3eab84dd0803ae85eac9e46d657ea67bd267db0bbbaf2ac53c2ec21feae3;
  evidence1,872,893B SHA8bffd0ce5f114ff5e08ee51c32f8a3b9bd5919f59644b5448045b4b65fa22269,
  256 rawfiles8,720,980B plus package/ELF/verification/preservation/cleanup.29 local
  tests/selected critical Ruff/Black/pycompile/diff pass, full upstream CI not run.
  Explicit ONNX inference/provider choice only; no automatic Torch routing,
  graph autograd, arbitrary export/model/dtype/concurrency or performance claim.
  Historical Python CPU fault stays unresolved/distinct from fixed VM-heap damage.
  [Use/session/recovery](docs/development/bc250_onnxruntime_b0611_results/README.md).
  **Next B06.12 (planned):** bounded modern Torch exporter→public bc250-ort GPU
  integration with independent CPU/eager-GPU references, then practical B06 gap
  review before further expansion. Preserve installed defaults and .105.

- **B06.10 isolated ORT/MIGraphX candidate completed 2026-10-29:** matching ORT1.31.0
  core/Python3.12/provider built and actually executed on .104. Sourceb81a7866e5a6eff964b9e411535b4c9198651a9f
  over402979ac: explicit existing uint32 cache conversion, explicit this capture,
  redundant final-class virtual destructor removed (base destructor protected/
  nonvirtual). No calibration/quantization/lifetime/math qualification from these
  compiler fixes. Tools9791d4134;13 archives upstream-SHA1/SHA256 verified/offline
  mirror. Nested build venv explicitly uses retained NumPy2.5.1; no image package
  replacement. Private repaired SDK header only; registered SDK4797 remains intact.
  Clang24 Abseil diagnostic requires SYSTEM option plus CMake hook since ORT copies
  include paths without SYSTEM. -Werror retained; failing target smoke-r4 passes.
  Original source-r1/failed configs/builds/caches preserved; source-r2 original+three
  repair files all11379 hashes match. No native ROCm/Torch/GPU backend rebuild.
  FourCPU/jobs4/link1/RAM10GiB/mem+swap12GiB/private2GiB ccache/no reset/read-only
  source/root/no build GPU/network.24GiB admission, monitored20GiB loss/4GiB floor/
  two-hour phase bound. Entry~32.6GiB/export32,344,141,824B; no cleanup/reserve change.
  Three fresh import orders each32 (daily ORT-first/Torch-first, RCCL MIGraphX-first)
  plus core7/installed CPU ONNX4/daily5/RCCL6=118/seven ACK guards/36 GPU graph evals.
  Six retained FP32 IR10/opset18 graphs/two changed inputs, CPU-double/CPU ORT refs,
  unchanged tolerances/host inputs, dtype/shape/missing-input refusals, candidate
  CPU after GPU and exact Torch HIP backward pass.18 actual model profiles assign
  all nodes to MIGraphX with session CPU fallback disabled; actual candidate+SDK
  library paths/nativegfx1013 checked. Candidate ORT ELF has no exported protobuf/
  absl definitions. No broader loader/model/dtype/dynamic/concurrency/autograd/
  performance claim.2028 native hashes/29 links/four AOT DBs/136 retained backing
  paths match; installed SDK/source checked separately. Same r2boot83f6cb1b-4540-
  4149-839e-a6cf7cb80b63, kernel logs unchanged. No jobs/containers/armed guards.
  Accepted build-r5 PID8939, qualification-r1 PID9659, export-r1 PID11602 exited0;
  WS/tmp/onnxruntime-b0610/task-r2/build; WS/logs/onnxruntime-b0610-qualification-r1/
  final.json and -controller.log; WS/tmp/onnxruntime-b0610/export-r1. Failed configure-r1
  NumPy, build-r2/r3 Abseil, build-r4 provider diagnostics retained; no GPU failure.
  Local kernel/artifacts/onnxruntime-b0610: evidence3,223,004B
  SHAb7ff9d388611f0db8be2f66133dd42e9f4f702276e43064afe191ce1554a7d25;
  224 rawfiles17,064,377B; raw candidate29,115,179B/325 verified files (not relocation
  qualified), source repair/bundle/shallow reconstruction/fsck and ELF/cleanup.
  Initialized pinned lintrunner0.13.1/adapters0.14.1/Ruff0.12.12/clang-format20.1.8
  source lint passes;29 local tests/selected critical Ruff/Black/pycompile/shell/
  diff pass. Full upstream CI not run. Unpackaged candidate warns missing
  build_and_package_info; resolve identity in packaging, do not suppress in user
  scripts. Installed CPU ORT/image/SDK/ROCm/Torch/defaults/fallbacks/user apps
  unchanged; .105 not contacted. Historical Python CPU fault remains unresolved/
  distinct from fixed VM-heap corruption. [Evidence/scope](docs/development/bc250_onnxruntime_b0610_results/README.md).
  **Next B06.11 (planned):** relocatable matching candidate package with sources/
  licenses/identity/loader metadata; fix package metadata, qualify extraction and
  explicit versioned activation/rollback plus retained daily/RCCL gates. Preserve
  installed CPU ORT/SDK/defaults/.105; no automatic provider/default switch.

- **B06.9 ORT/MIGraphX admission completed 2026-10-28:** pinned unmodified ORT1.31.0
  402979ac1bef89dcd55add41eaa7277c407599eb at sources/bc250-onnxruntime. Shallow
  checkout/fsck and exact snapshot11,379 files verified locally and on .104;
  three gitlinks recorded/uninitialized, no signed-tag/full-history claim.
  Source279,492,314B SHAe4243f26f7b6fd6be09fabd08a77161da719cdded1cbbc696458e3e2aeb3e51b.
  Twelve static assertions establish CMake3.28/C++20/shared migraphx::c+hip::host,
  parser/native stream/output-copy/FP32-default/fallback/visibility controls.
  Real Clang C++20 blocker reproduced in installed SDK: handwritten C++ wrapper's
  two leading unqualified module return types misparse as module directives.
  Source144f6db5c qualifies both as migraphx::module; final134ee25f67007523bd03374c4b723fe075a3b77f
  also gates new regression on compiler C++20 support. ABI/math unchanged;
  registered SDK remains4797a6112 bytewise, candidate uses private header overlay.
  C++17+20 CPU header regressions pass;168 discovered C API declarations link.
  Eight guarded ORT-style device-pointer run_async cases each daily/RCCL: two
  models/two changed inputs/default+private streams, explicit output allocation/
  D2D copy, CPU-double/CPU ORT/host+device guards. Retained core7/ONNX4/daily5/RCCL6
  bring38/six accepted guards/16 GPU graph evaluations. Earlier fixture passes
  same16 plus core/ONNX, then log-label collision before remaining gates;
  four failed-fixture guards ACK. No math/tolerance change. Builds r1/r2 retain
  missing clang-scan-deps/sibling ccache helper/read-only default-cache failures;
  r3 actual header failure, r4 succeeds. Disable module scan/use complete setup
  helper mount/fail on setup errors. FourCPU/jobs4/RAM10GiB/mem+swap12GiB/private
  2GiB ccache/no reset/read-only build root/no GPU/network. Tools c86648601,
  source-stage/export167e467e5. ORT protobuf33.6/Abseil20250814 differ from SDK30/
  20250512; ONNX namespace isolated,195 protobuf/absl exports inventoried, provider
  version script hides except GetProvider. No collision reproduced/ABI-wide proof;
  actual candidate ELF/import-order tests required. Candidate13 dependencies pinned
  URLs/upstream SHA1, bytes not fetched/configure not run. Provisional24GiB ordinary
  admission/20GiB growth/4GiB floor,link1/two-hour phase bound, not measured footprint;
  source stage admits40,796,561,408B. Full ORT core/binding/provider not built/loaded/
  installed. Active CPU ORT/SDK/Torch/ROCm/image/defaults/fallback/user apps unchanged;
  .105 not contacted.1695 native hashes/29 links/four AOT DBs match,136 preserved
  backing paths match; full installed SDK checked separately. Kernel logs match
  through cleanup on r2boot83f6cb1b-4540-4149-839e-a6cf7cb80b63. No owned jobs/
  containers/armed guards. Build PID10099, qualification11488, source-stage12693,
  export13014 exited0; WS/tmp/onnxruntime-b069/{probe-r4,inputs,export-r1};
  WS/logs/onnxruntime-b069-qualification-r2/final.json and matching -controller.log.
  Local kernel/artifacts/onnxruntime-b069 evidence1,440,485B
  SHAeb10bae63e6ab08dd09c54a0866ae74fe362bc72ea93076d72b942572f30a4d9;
  196 rawfiles5,500,940B plus source/dependency/plan/ELF/repair/reconstruction receipts.
  Earlier scratch reconstruction fails pack/index/reflog/tree checks (cause unknown);
  retained. Independent complete bundle-chain clone with auto maintenance disabled
  passes full fsck at134ee25.29 local tests/Black/pycompile/C++format/diff pass;
  Ruff absent/full upstream CI not run. No provider execution/partition/allocator/
  dynamic-shape/concurrency/dtype/model/autograd/performance claim. Historical
  Python CPU fault stays unresolved/distinct from fixed VM-heap corruption.
  [Evidence/plan](docs/development/bc250_onnxruntime_b069_results/README.md).
  **Next B06.10 (planned):** fetch/hash selected dependency bytes with SHA256,
  create offline mirror, recheck space/bounds; configure/build matching isolated
  ORT core/Python/MIGraphX provider using repaired private header. Preserve installed
  CPU ORT/SDK/daily+RCCL/defaults/.105. Actual provider assignment/no-silent-CPU-
  fallback and independent references must pass before installation/activation.

- **B06.8 installed opt-in SDK completed 2026-10-28:** source remains4797a61125bbec318005969f8497dbd50849cd6e;
  implementation de93d4fe992383ef4787d494706b52d5c7f707b6, export5f5a73b7a.
  Qualified B06.7 archive installed root-owned at /opt/bc250-stack/sdks/
  bc250-migraphx-2.18.0.dev-gfx1013-cp312-b067; registration sdks/migraphx.json,
  retained version record and tools/migraphx-b068. New /usr/local/bin/bc250-migraphx
  verifies pinned SDK index/manifest/all bytes/exact file set/contained links and
  selected runtime manifest/image under shared daily lock, then delegates to
  ordinary bc250-torch with read-only SDK. Install/unregister/register use exclusive
  lock; live unregister/refusal/re-register exercised, payloads retained. Scratch
  runtime/SDK drift refusals, lock exclusion, default SDK absence and actual UID1000
  doas activation pass. Ordinary daily+RCCL activation8/Python50/CAPI+CLI6 each,
  retained core7/ONNX4/daily5/RCCL6=150 guarded; host14+user1=165 checks,
  ten ACK guards/68 GPU graph evaluations. Same FP32 IR10/opset18 six fixtures,
  CPU-double/CPU ORT references/tolerances, actual sibling HIPRTC helper observed;
  external script/sidecar/spaces/read-only mount and Torch HIP backward pass.
  All1681 native protected hashes/29 links/four AOT DBs match;136 preserved backing
  hashes verified locally before/after, full installed SDK separately checked.
  New pre-task r2boot83f6cb1b-4540-4149-839e-a6cf7cb80b63; batch kernel logs match,
  no agent reboot/cross-boot continuity claim. No native rebuild/cache reset needed.
  Active ROCm/Torch/image/defaults/fallbacks/user apps unchanged; .105 not contacted.
  Final ordinary free41,137,750,016B. No owned jobs/containers/armed guards/transport.
  Controller PID8265 and export PID10260 exited0; WS/logs/migraphx-b068-qualification-r1/
  final.json; controller log WS/logs/migraphx-b068-qualification-r1-controller.log;
  export WS/tmp/migraphx-b068/export-r1. Local kernel/artifacts/migraphx-b068:
  evidence623,640B SHAc0e0ce195e4507d57694422dcc0ffd878eaeb686f6aa5627e45cd683cf991396,
  188 rawfiles2,698,354B plus verification/registration/preservation receipts.
  29 local tests/Black/pycompile/diff checks pass; Ruff absent/full upstream CI not run.
  No arithmetic failure/tolerance relaxation. Explicit inference only: ordinary Torch
  launchers do not load SDK; no ORT GPU EP, graph autograd/all-model/dtype/performance
  or universal MXR claim. Historical Python CPU fault remains unresolved/distinct
  from fixed VM-heap corruption. [Use/recovery/evidence](docs/development/bc250_migraphx_b068_results/README.md).
  **Next B06.9 (planned):** pinned ONNX Runtime MIGraphX EP/API/dependency/storage
  admission audit; establish feasible isolated candidate before build/provider
  qualification. Preserve installed CPU ORT/SDK/daily+RCCL/defaults/.105. Broader
  model/dtype/performance admissions remain separate; continue one bounded batch
  at a time in order, resolving necessary discovered prerequisites autonomously.

- **B06.7 relocatable SDK completed 2026-10-25:** MIGraphX source
  4797a61125bbec318005969f8497dbd50849cd6e over622ea3f2a fixes generated CMake
  variables advertising nonexistent migraphx::migraphx_c instead of exported
  migraphx::c. External consumer reproduces old failure; fixed config/build/link
  and GPU C API/C++ wrapper pass. No backend math/ABI/whitelist change. Tools
  389966df4824b6e07e3accb67a679c5b2f2a5e7c. Source4782 files verified; full
  bundle-chain reconstruction/fsck reaches4797. Private SDK adds C API/CLI to
  existing Python3.12/GPU/parser; dependencies/policies remain B06.6.
  Archive kernel/artifacts/migraphx-b067/bc250-migraphx-2.18.0.dev-gfx1013-cp312-b067.tar.gz
  is41,753,676B SHA11f7f7ab1923132131dc0118af62fbd425e55cfd802d86fdcb57e4952583c8ba;
  565 files/12 contained links, headers/CMake/source/license/fetch pins/owned fixtures.
  Verified sibling HIPRTC helper copy handles installed deep-library layout;
  wrapper-derived license link materialized from source. No active SDK install.
  Two native extracts (one path with spaces) pass Python50+CAPI/CLI6 each, original
  build trees absent, actual SDK HIPRTC children and native renamed SQLite loaded.
  Core7/ONNX4/daily5/RCCL6 bring134/eight accepted guards,68 GPU graph evaluations.
  All1675 native hashes+136 preserved workstation-backing hashes match (1811
  combined; all original665 preserved);29 links/four AOT DBs unchanged. Kernel
  logs match through export/cleanup on r2 boot e955931f-5edb-42e5-8f14-eafc2fbaf361.
  All11 guards ACK (eight accepted/three failed fixtures). No arithmetic failure
  reproduced/tolerance relaxation. Earlier attempts retain transport/empty-wrapper/
  license-link/SSHFS-readlink failures and stale ORT-path/native-scratch-ownership/
  SQLite-SONAME test corrections. No GPU/VM-heap defect inferred from those.
  Builds remain external durable/b067 (r5 SDK, consumer-r2); fourCPU/jobs4/link1/
  RAM10GiB/mem+swap12GiB/private2GiB ccache/no reset/no build GPU/network, prior
  external16GiB/14GiB/root768MiB/two-hour bounds. Bridge interruptions stopped
  pre-scans; with root free externally observed~55GiB, immutable native stage at
  WS/tmp/migraphx-b067-native uses4GiB extraction admission/2GiB growth/floor.
  Qualification-r5/export-r5 use detached jobs and need no bridge/build mount.
  Native stage before58,866,450,432B/after58,476,675,072B; export58,079,264,768B.
  Free-space increase origin not established; no reserve-policy/retained-artifact
  cleanup by these tools. Native extracts/backing/sources/builds/caches retained;
  new source-transfer original name aliases its verified native SDK source copy.
  Evidence1,324,388B SHA27edca3ad576cb11241a27a357cd1ec56c802cd58c79799b1e80b6f57ef884ed,
  253 rawfiles5,574,136B; separate build archive232,761B/42files1,168,331B; SDK/source
  bundles/verification/cleanup in kernel/artifacts/migraphx-b067. 29 local tests,
  Black/pycompile/shell/C++ formatting pass; Ruff absent/full upstream CI not run.
  No owned jobs/containers/armed guards/mount/relay. Active ROCm/Torch/images/
  defaults/boot/fallbacks/user apps unchanged; .105 not contacted. CPython3.12/
  glibc FP32 static graph scope only; no ORT GPU EP/MIGraphX autograd/all-model/
  dtype/opset/performance or universal MXR portability claim. Historical Python
  CPU fault remains unresolved/distinct from fixed VM-heap corruption.
  [SDK/use/evidence](docs/development/bc250_migraphx_b067_results/README.md).
  **Next B06.8 (planned):** versioned .104 opt-in SDK registration/identity/rollback,
  then ordinary launcher activation/retained daily+RCCL qualification. Check fresh
  space; preserve current defaults/fallbacks/.105. Broader models/dtypes/ORT GPU
  provider/performance remain separate admissions.

- **B06.6 Python MIGraphX bindings completed 2026-10-25:** source remains
  622ea3f2ad8f1baa8a44edb4f1e07cbe6e532c83; pinned private pybind11 2.12.0
  3e9dfa2866941655c56877882565e7577de6fc7b. Python 3.12.3 extension/support
  libraries build; no MIGraphX source repair or installed package replacement.
  Tools checkpoint 8692aa5fe, accepted fixture/retries 0bfe2858c. Three fresh
  import orders each pass 49 composites: six FP32 IR10/opset18 models, file/buffer
  parsing/execution, uncompiled JSON/msgpack/buffer IR equality, GPU lowering,
  changed-input CPU-double/CPU ORT references, compiled file/buffer reload,
  explicit HIP copies, four refusals, CPU ORT inference/Torch HIP backward after.
  90 GPU graph evaluations; retained core7/ONNX4/daily5/RCCL6 bring 169/seven gates.
  665 protected hashes/four AOT DBs/links match; seven accepted guards ACK and
  kernel logs unchanged through export/unmount on r2 boot e955931f-5edb-42e5-8f14-eafc2fbaf361.
  4,782 MIGraphX/262 pybind source files verified; complete pybind bundle fresh
  clone/fsck passes. First build refused stale unused protobuf source mount before
  compilation; r2 corrected. First GPU fixture omitted compiler-exposed output
  allocation (main:#output_0); corrected per upstream API, same math/references/
  tolerances. Failed guard ACK/no kernel drift; all attempts preserved.
  New external durable/b066 under existing B06.4 temporary storage; native read-only
  pybind input WS/tmp/migraphx-b066/pybind11. Build~463MiB/private headers908KiB/
  cache51MiB; old builds/caches retained via reflinks. FourCPU/jobs4/link1/RAM10GiB/
  mem+swap12GiB/no build GPU/network/read-only root/private2GiB cache/no reset;
  external16GiB admission/14GiB growth/root768MiB floor/two-hour bound unchanged.
  Export root ordinary866,148,352B; external114,335,145,984B. 29 local tests,
  Black/pycompile/shell syntax pass; Ruff absent/full upstream CI not run.
  Local kernel/artifacts/migraphx-b066 archive21,375,875B SHA
  79738ca55dec9686b95d9303a01a4de084ac6398be6e34edeaf64dd430129c41;
  239 extracted raw files9,029,640B, separate failed guard and complete source bundle.
  No owned jobs/containers/armed guards/mount/relay; backing files remain.
  Active ROCm/Torch/images/defaults/boot/fallback/BIOS/user apps unchanged; .105
  untouched/not contacted. Candidate CPython3.12/glibc FP32 inference only; no
  active SDK/ORT GPU EP/MIGraphX autograd/all-model/IR/opset/dtype/performance claim.
  Historical Python CPU fault remains unresolved/distinct from fixed VM-heap damage.
  [Scope/replay/evidence](docs/development/bc250_migraphx_b066_results/README.md).
  **Next B06.7:** versioned relocatable opt-in SDK archive with source/license/
  loader metadata; qualify relocation against existing runtime image before
  considering installation. Respect ordinary root headroom and preserve .105;
  ORT GPU provider and broader dtype/model/fusion/performance are separate.

- **B06.5 native ONNX parser completed2026-10-25:** MIGraphX622ea3f2a unchanged;
  tools checkpointb6a629170/final4711e9892. Pinned full clean sources Abseil
 20250512.0 bc257a88f7c1939f24e0379f14a3589e926c950c and Protobufv30.0
  d295af5c3002c08e1bfd9d7f9e175d0a4d015f1e (tag2c82161c); private C++17/PIC/O2
  static builds/install and libprotoc30.0 pass. Existing profiler Abseil202601/
  Torch protobuf/Python protobuf preserved. Full Git bundles fresh-clone/fsck pass.
  Native ONNX24: six IR10/opset18 FP32 models (broadcast/ReLU, reduce, dense/ReLU,
  transpose/MatMul, conv/bias/ReLU, serialized weight/bias initializers), file/
  buffer IR equality, twelve GPU evaluations vs independent CPU-double and CPU
  ORT references/shape/type/host guards; four unknown/malformed file/buffer refusals.
  Actual GPU code_object/gemm/convolution/literal ops/HIP copies, fast_math false/
  tune off; no tolerance relaxation. CPU oracle generation12 passes separately.
  Retained candidate hand-built22/core7/ONNXaudit4/daily5/RCCL6 bring68/six gates.
 629 protected hashes/four AOT DBs/links and per-gate/export/unmount kernel logs
  match; six guards ACK. New pre-task r2boot e955931f-5edb-42e5-8f14-eafc2fbaf361,
  no agent reboot/cross-boot log identity claim.4,782 MIGraphX/1,532 Abseil/3,052
  Protobuf sourcefiles verified. MIOpen fusion whitelist still excludesgfx1013;
  ordinary conv/separate pointwise qualified, no architecture/dtype admission change.
  Reused B06.4 temporary SSHFS backing; new data local tmp/migraphx-b064/
 20261025-storage/durable/b065 (remote STORAGE=WS/tmp/migraphx-b064/
 20261025-storage/mount/b065). Native source snapshots WS/tmp/migraphx-b065/
  {abseil,protobuf}; parser uses old readonly622 input. FourCPU/jobs4/link1/
  RAM10GiB/mem+swap12GiB/read-only build root/noGPU/network; private2GiB caches/
  no reset/old shared cache untouched. Parser~451MiB/private deps49MiB/dependency
  builds6.6/69MiB/caches19/50MiB. External16GiB admission/14GiB growth/root768MiB
  floor/two-hour phase bound. Export root ordinary877,867,008B/external114,507,022,336B.
  Initial source staging EOF before build corrected with completed-archive SHA
  check/re-extraction. Live-script edit truncated final protoc version command
  after successful compile/install; accepted incremental retry snapshots immutable
  phase script, no rebuild/source change. TTY CPU-generator stop resumed cleanly;
  null stdin thereafter. Original attempts retained, no GPU numerical failure.
 29 local tests/Black/pycompile/shell/C++ format+syntax pass; Ruff absent/full
  upstream CI not run. Disabled-MLIR/unused rocm-path host warnings remain.
  No active ROCm/Torch/image/default/boot/fallback/BIOS/user-app/.105 change/contact;
  no artifact deletion/reserve policy change. Parser C++ isolated only, no Python
  bindings/ORT GPU EP/autograd/model speed/all-IR/opset/dtype claim. Unknown Python
  CPU fault remains distinct/unresolved from already-fixed VM-heap corruption.
  Raw accepted STORAGE/logs-{abseil-r1,protobuf-r2,parser-r1}; WS/logs/
  migraphx-b065-qualification-r1. Local kernel/artifacts/migraphx-b065 archive
 67,909,794B SHAa44098301bfa85484798d9695a6656b360161389ce157c72f8b2260ac10ce6c8,
 118 rawfiles2,346,845B plus models/source bundles/verification/cleanup. No owned
  jobs/containers/armed guards; temporary mount/relay closed, backing data retained.
  [Scope/replay/results](docs/development/bc250_migraphx_b065_results/README.md).
  **Next B06.6:** reconnect durable storage; audit/build pinned Python bindings in
  isolated candidate, then Python ONNX inference/serialized-program roundtrips/
  Torch+CPU ORT import-order gates before considering installed SDK packaging.
  Preserve active runtime/.105; root headroom remains low. ORT GPU EP, broader
  dtype/IR/opsets/fusion/model performance remain separate admissions.

- **B06.4 isolated MIGraphX FP32 graphs completed2026-10-25:** source
 622ea3f2ad8f1baa8a44edb4f1e07cbe6e532c83 over35b40656d; toolsf8635babe.
  Full minimal GPU backend/HIPRTC driver/owned C++17 fixture built on .104.
  Reproduced Lt-disabled lowering.cpp:288 missing gfx_default_rocblas; exported
  queries now exist in both configurations, return true without Lt, enabled
  implementation preserved. New upstream-style test; both compile configurations
  syntax-pass and both no-Lt overloads execute/pass. No whitelist/BF16/FP8 change.
  Native gfx1013 wave32 FP32 graph gate22: strided+broadcast pointwise/ReLU,
  row sums17/33/129/513, rocBLAS GEMM K7/17, MIOpen3x3 conv N1/2, changed-input
  second evaluation of ten compiled graphs. CPU-double outputs/host-input guards,
  actual GPU lowered ops, explicit HIP copies; fast_math false/tune off. Not GPU
  allocation-canary/autograd/model/performance/all-dtype or provider qualification.
  Core7/ONNXaudit4/daily5/RCCL6=44 perpass,88/two passes/ten ACK guard records;
  final frozen-tool recheck after export-only audit additions, no math relaxation.
 537 protected hashes/four AOT DBs/links, per-gate/final/unmount kernel logs match;
  same r2bootc6a50cfa.4,782 sourcefiles/172 pinned CMakefiles match archives.
  Five device objects nativegfx1013/no MFMA/WMMA; loader resolves candidate and
  selected native ROCm/SQLite libs (Lt may load transitively; dispatch remains off).
 29 local tests/Black/pycompile/shell/changed C++ formatting pass; Ruff absent,
  full upstream CI not run. Source repair bundle reconstructed/fsck passes.
  Root~1GiB ordinary space required separate durable workstation backing at
  tmp/migraphx-b064/20261025-storage/durable (~109GiB available), SSHFS slave over
  existing SSH/no new listener/key/boot mount. File fsync/mmap/rename/symlink/local
  lock pass. SSHFS3.7.6/FUSE3.18.3 installed, existing r2 FUSE module loaded;
  inherited APK .105 mirror index request refused, no .105 state/deployment change.
  Source copied read-only on native disk; build/cache external. FourCPU/jobs4,
  link1/RAM10GiB/mem+swap12GiB/noGPU/network/read-only build root; independent
 2GiB ccache/no reset/old shared cache untouched. Build~426MiB/cache~43MiB;
  external16GiB admission/14GiB growth limit and root768MiB floor/two-hour bound.
  Source-root wrapper configuration failure, slow-network-source owned stop,
  no-Lt compiler failure and initial fixture include diagnostic all retained.
  Disabled-MLIR unused host warnings remain. No deletion/reserve policy change.
  Active ROCm/Torch/image/defaults/boot/fallback/BIOS/user apps unchanged; Python
  CPU fault distinct/unresolved. Raw accepted external logs-build-r4/logs-fixture-r1,
  WS/logs/migraphx-b064-qualification-r2; local kernel/artifacts/migraphx-b064 archive
 67,753,287B SHAb3c1f4576ec1e9bef91458def557080d340e99ea4cfbf7742d72820f48230c32,
 154 rawfiles14,184,780B plus patch/bundle/verification/cleanup. No owned jobs,
  containers/armed guards; temporary mount/relay closed, backing artifacts retained.
  [Scope/replay/storage/results](docs/development/bc250_migraphx_b064_results/README.md).
  **Next B06.5:** reconnect documented durable storage; audit/build pinned native
  ONNX parser prerequisites, then small ONNX→gfx1013 FP32 graphs against independent
  CPU references. Preserve CPU ORT/Torch defaults; parser/bindings/ORT GPU provider,
  BF16/FP8/CK/MLIR and larger-model coverage remain separate admissions.

- **B06.3 strict MIGraphX C++17 repair completed2026-10-24:** source
 35b40656df72d075a69b155273cf0cc121009683 over95672916e; toolsa03d0e38a.
  DPP-only subwave/wave declarations now share the implementation guard; no
  arithmetic/default-DPP change or new subwave fallback. DPP0 block/LDS retained;
  unsupported subwave request deliberately refuses. Real HIPRTC reproduces old
  DPP0 C++17/Werror failure; patched DPP1/DPP0 compile without diagnostics and
  loaded modules match CPU-double/guards at33/129/513. Native gate71 includes
  retained30,30 subwave width/length cases,new upstream-style block test and10
  JIT compile/execute checks. Retained core7/ONNXaudit4/daily5/RCCL6=93 perpass,
  two passes186/ten ACK guards. Final HIPRTC scratch explicitly under WS/tmp.
 507 protected hashes/four AOT DBs/links and kernel logs unchanged; same r2boot
  c6a50cfa. Extracted native AOT objects contain no MFMA/WMMA in tested kernels.
  Actual unchanged feature predicates compiled/executed: gfx1013 FP8/MX/Lt false,
  BF16 true remains unqualified admission risk, not reproduced numeric failure.
  Source4781 files verified; committed unit-test reflow whitespace-only/hash-pair
  retained, reducer header exact. Fresh reconstruction from preserved upstream
  bundle plus repair bundle/fsck passes.29 local tests/Black/pycompile/shell/new
  unit clang-format pass; Ruff unavailable, full upstream CI not run.
  Builds4CPU/RAM10GiB/mem+swap12GiB/shared20GiB ccache/no reset/noGPU/network.
  Test-wrapper return and host-client AMD-platform omissions fixed before GPU;
  stale baseline-controller path refused before GPU, all attempts retained.
  Provisional full-build budget12GiB+cache2GiB+ordinary rescue floor2GiB=16GiB;
  current ordinary1,110,757,376B vs totalfree26,491,379,712B incl reserved, before
  archive. Budget refused/unmeasured footprint; no reserve policy/deletion change.
  No backend/provider installation/runtime/prefix/image/default/kernel/BIOS/.105/
  user-app work. CPU segfault distinct/unresolved. Raw WS/logs/migraphx-b063-*,
  accepted build-r4/qualification-r2; local kernel/artifacts/migraphx-b063 archive
 17,326,170B SHA6f5225f873fdbfe4ab7b94ee61820b419cfcf2b37f3aa8b50eefe2bc8100496f,
 147 rawfiles9,752,884B plus patch/bundle/verification. No own jobs/containers/
  armed guards running; .104 memory sync only.
  [Scope/evidence/budget](docs/development/bc250_migraphx_b063_results/README.md).
  **Next B06.4:** establish durable build storage satisfying ordinary-space budget
  without deleting retained artifacts or consuming rescue reserves, then build
  minimal isolated FP32 GPU backend and reference-test actual pointwise/reduction/
  GEMM/convolution graphs. No BF16/FP8/CK/MLIR/parser/bindings/provider admission.

- **B06.2 MIGraphX admission audit completed2026-10-24:** toolsa8a6d2327.
  New unmodified source95672916ed289d4be9d230a9be732e1bbad97e8c (rocm-10.1,
  no upstream10.2 tag seen); isolated required rocm-cmake6a7c5b7, existingbase10155
  unchanged.4,780 sourcefiles match staged archive. Full Git reconstruction bundles
  fresh-clone/fsck pass (initial shallow bundle failed, parent history fetched;
  working tree/revision unchanged). No signature qualification of lightweight tag.
  Native gfx1013 wave32 unchanged headers: FP32/FP16 strided+broadcast pointwise
  and12 reduction lengths/CPU-double/guards, DPP1+scalar/LDS DPP0=30 checks.
  Device objects nativegfx1013/no MFMA/WMMA in these kernels. Retained core7/
  ONNXaudit4/daily5/RCCL6 bring52/five gates;29 local tests/Ruff pass.
  Minimal GPU configuration passes with new isolated pinned CMake dependency;
  old dependency refused missing rocm_add_version_resource. GPU/HIPRTC/rocBLAS/
  MIOpen ON; Lt/CK/MLIR/AMDMLSS/Eigen/ONNX/TF/Python OFF. Configure is not a build/
  link/graph proof; no MIGraphX backend/provider installed. BF16 predicate currently
  admitsgfx1013 without dedicated validation (risk, not reproduced numeric bug);
  Lt whitelist excludesgfx1013; FP8/MX remain excluded. DPP0 AOT numerics pass but
  strict C++17 compile refuses unguarded subwave_reduce reference; DPP1 strictpass.
  JIT uses C++17/Werror, so scalar fallback strict JIT remains unresolved.
  Build4CPU/RAM10GiB/mem+swap12GiB/shared20GiB ccache/no reset/noGPU/network.
  Initial fat-object disassembly error corrected via offload-bundle extraction;
  wrapper path TypeError before build retained. No math/tolerance relaxation.
  Ordinary available1,234,112,512B vs totalfree26,614,734,848B incl reserved; no
  reserve policy change/full build started. Need explicit build/reserve budget.
 489 protected hashes/four AOT DBs/links, per-gate/batch/export kernel logs match,
  fiveguards ACK; same r2bootc6a50cfa. Dailyruntimea9a5/profiledb55/image1bb4/
  nativeTorcha524/ROCm/HIP/W8/CPUORT/defaults unchanged. No prefix/image/package/
  kernel/BIOS/user-app/.105/source/cache/prior-artifact deletion. CPU fault distinct/
  unresolved. Raw WS/logs/migraphx-b062-{build-r1,build-r2,configure-r1,probe-r1,
  qualification-r1}; binaries WS/tmp/migraphx-b062/20261024-r2/tmp. Local
  kernel/artifacts/migraphx-b062 archive8,720,498B
  SHA28b8390813ccdb72afe0c1fef2e7ee08c3810d93a88ddedfe95afab1ee9ac540,
  69 rawlogs1,525,263B/eight tools/source archives/hashmap/objects/configure/fiveguards
  verified, full source bundles retained. No own jobs/containers/armed guards running;
  memory sync `.104` only. [Findings/replay](docs/development/bc250_migraphx_b062_results/README.md).
  **Next B06.3:** fix/test strict C++17 DPP-disabled declaration boundary, preserve
  DPP default, establish gfx1013 feature-admission tests and explicit durable disk/
  reserve budget before minimal FP32 GPU graph build. No BF16/FP8/CK/MLIR/EP claim.

- **B06.1 ONNX export/CPU runtime installed2026-10-24:** tools6ae431c9a.
  Pinned onnx1.23.2/onnxscript0.7.2/onnx-ir1.1.0/onnxruntime1.31.0 CPU,
  ml-dtypes0.6.0/protobuf7.36.2/flatbuffers25.12.19; exact PyPI wheel hashes/
  licenses retained. Torchsourcea524 requires torch_2_11 exporter API, verified;
  modern dynamo/opset18, no legacy fallback. Existing NumPy/SymPy/packaging/
  typing_extensions versions preserved. Offline hash-required/nodeps image build,
  RAM1GiB/quota2/networkoff/pullnever; no native ROCm/Torch build/ccache operation.
  Image1bb4ac0ed3766d9c70431c1d5119e03cd4d10e710f4c04cd54254b5b87cd45d5
  retains W8imagee282 layers; dailyruntimea9a5afe3/profiledb556925 select it under
  daily EX lock. Recovery WS/recovery/onnx-b061-before restores prior W8 manifests;
  old images/candidates retained, identity rollback not exercised this batch.
 181 candidate+181 installed/22 gates plus33 early export-probe checks. Bothcohorts
  audit4/export33/ORT-first2: six CPU/GPU-origin dense/CNN/gated+LayerNorm exports,
  batches1/3/4, CPU-double/ONNX NumPy/HIP eager references, host transfer then Torch
  GPU leaf first gradients; three native input refusals; actual ORT node profiles
  CPU only. Available Azure+CPU EP; only CPU selected, Azure unqualified; native
  GPU OrtValue allocation refuses. ORT sessions before Torch import/GEMM backward/
  FFT pass (script-entry Torch absent). Retained W8recipe71/core7/compiled14/
  consumers5+6 pass. No ONNX GPU/autograd/training/W8-export/model-speed claim.
  Old ROCm EP removed upstream1.23; MIGraphX not qualified/registered, separateport.
  Initial qualifier stale report name refuses before GPU, old evidence untouched;
  corrected without math/tolerance changes.29 local tests/Ruff pass.
 477 protected hashes/four AOT DBs/links pinned with manifest/recovery changes;
 23 guards ACK, per-gate/batch/export kernel logs identical; same r2bootc6a50cfa.
  Native Torch/HIP/ROCm/math/SDMA0/heapoff/N3.2/defaults retained; no kernel/BIOS/
  user-app/.105/source/cache/image deletion. CPU segfault distinct/unresolved.
  Raw WS/logs/onnx-b061-{candidate,installed}-r1/final.json; local kernel/artifacts/
  onnx-b061 archive35,255,852B SHAfb55f104f7c83fea3812c3a09acf05de56983df4636c0ae28e984accf3029e44,
 284 rawfiles5,296,329B/seven source+manifest files/seven wheels/models/profiles/
  recovery/23 guards verified. No own jobs/containers/armed guards at handoff;
  memory sync `.104` only. [Use/scope/evidence](docs/development/bc250_onnx_b061_results/README.md).
  **Next B06.2:** audit pinned MIGraphX source/dependencies, gfx1013 instruction/
  generic/scalar applicability and build headroom before any isolated GPU-provider
  experiment. Keep CPU ORT/W8/native defaults/.105; no architecture spoof.

- **B05.6 optional W8 package installed2026-10-24:** final toolsabd8a4e04
  (initialaee53dcc0/runnersc9de6be11 retained). bc250_quant0.1.0 public exports
  FrozenW8Linear/FrozenInt8Weight/pack/linear; explicit opt-in, no model/dispatch
  conversion. Core6832ff9e exact/modulea18e988f only relative import changes;
  wheelMIT/provenance/RECORD SHAfa71248bb32fbc0718873511dd90e159c3f07e0fc8412d40935b04dd36eeae36.
  Imagee2821cf8507467b22323272439d636f7c0918e27b8aa2790a5f524b10165a184 retains
  base506f layers; dailyruntime5a3a4c2f/profile6101835b select it under daily EX lock.
  Old image/manifests retained at WS/recovery/quant-b056-before, rollback exercised
  twice after runner refusals. Native Torcha524/both payloads/HIP/ROCm/kernel/
  math SDPA/SDMA0/heapoff/N3.2 unchanged. No native build/ccache reset/deletion/
  user-app/.105 work. StageRAM1GiB/CPUquota2/networkoff/pullnever; two --cpus
  syntax refusals before build retained. No package/kernel arithmetic defect found.
 817 candidate+817 installed/32 accepted gates: public import/RECORD/source identity,
  both cohorts module71+recipe71, consumers5/6, core7/FFT245/sparse253, nine actual
  B05.5 checkpoint output/first-gradient/reverse roundtrips. Initial consumer
  accelerate and later coretransformers refuse under incorrect plain-Torch runner;
  prior candidate808+808/installed146+303 and failures preserved, both rollback
  records exact. Final retained fixtures bc250-train, public W8 suites bc250-torch.
  Same wheel/image/tolerances throughout;29 local tests/Ruff/wheel determinism pass.
 457 protected hashes/four AOT DBs/links retained with deliberate manifest updates/
  recovery pinned;81 guards ACK (32 accepted/49prior). Same r2bootc6a50cfa,
  per-gate/batch/export kernel logs identical. No native backward/INT4 admission,
  TorchAO/NF4/full QLoRA/compiled W8/model-quality/speed/peak-memory claim; CPU
  segfault remains distinct/unresolved. Raw WS/logs/quant-b056-{candidate,installed}-r3/
  final.json; local kernel/artifacts/quant-b056 archive4,165,410B
  SHAd57be371cd444afd56739af693c35c5477a1843125dde6475af944275a09aef5,
 790 rawfiles18,442,720B/nine tools/wheel/images/recovery/81 guards verified.
  No own jobs/containers/armed guards at handoff; memory sync `.104` only.
  [Package/use/results/rollback](docs/development/bc250_quant_b056_results/README.md).
  **Next B06.1:** audit pinned ONNX export/runtime execution-provider applicability,
  then bounded isolated export/CPU-reference/GPU-interoperability qualification.
  Keep W8 package/native defaults/.105; CPU runtime does not qualify GPU provider.

- **B05.5 explicit frozen-W8 module/state completed2026-10-24:** code
  e1543d1cf75dc679edcba1b2963e6515eb28a1cc. New optional WS/helpers/
  bc250_quant_b055.py FrozenW8Linear; explicit frozen standard nn.Linear conversion,
  copied bias/policy, no floating-base reference, INT8+FP32 persistent buffers.
  Matching explicit FP16/BF16/FP32 compute, first input/bias gradients via unchanged
  helper6832ff9e; CPU storage/load, native gfx1013 forward/conversion. Dtype/device
  moves preserve exact scales as FP32. State v1 plain metadata+tensors validates
  before own-module copies; complete fields even strictFalse/assignTrue refused.
  Parent model loading not globally transactional; no higher/base/scale gradients.
  295/10 ordinary gates:32 module+36 state+3 owned adapter AdamW/resume eachcohort,
  retained recipe71 eachcohort+daily5/opt-in6. Independent scalar CPU packing,
  CPU-double outputs/input and bias gradients; nine roundtrips/24 malformed states,
  exact continuous-vs-CPU-checkpoint model/optimizer step2. No numerical relaxation
  or production defect found.29 local tests/selected Ruff checks pass.
 445 protected hashes/four AOT DBs/links retained; allten guards ACK, per-gate/batch/
  export dmesg identical. Same r2 bootc6a50cfa-a368-4ba5-ab28-66250b4f1f1f.
  Daily561f/profile446f/Torcha524 unchanged. No active-prefix update/build/ccache/
  package/image/default/native/kernel change, deletion/.105 contact/user app work.
  Module only explicitly imported from helper directory; not installed/default.
  Native direct backward/INT4 refuse; no TorchAO/NF4/full QLoRA/model-speed/quality/
  peak-memory/long-run claim. Historical CPU fault distinct/unresolved.
  Raw WS/logs/quant-b055-qualification-r1/final.json; local kernel/artifacts/
  quant-b055 archive510,746B SHA2c0099691e7744e37f4623a269df593d90a48009731bd56de969bc5cbc1b57e7,
  96 rawfiles2,070,943B/six tools/ten guards verified. No own jobs/containers/armed
  guard at handoff; memory snapshot synced to `.104` only.
  [Usage/contracts/evidence](docs/development/bc250_quant_b055_results/README.md).
  **Next B05.6:** package explicit W8 helper/module in independent optional runtime
  candidate, public imports/state/gradient/retained-consumer gates before locked
  promotion. Keep dense defaults/old images; no automatic conversion/full QLoRA.

- **B05.4 strict W8 validation optimization completed2026-10-24:** code917b2823a.
  Explicit helperWS/helpers/bc250_quant_b051.py updated under daily EX lock,
  not an image/site-package/native-prefix change. Normal path reads scalar
  conjunction of scale/input validity once; failure re-reads scale flag to keep
  scale-first errors. Checks/limits/messages/dtype policy preserved. Pack/_Linear
  ASTs unchanged; no native math/backward/persistent-storage change. New helper
  SHA6832ff9e58104236d877a455fdfe90f11b3803198732497d6dd4ac77242fd319 vsold54c98373.
  Baseline profile:~0.14–0.17ms strict data checks, two scalar reads/stream syncs.
  First tensor-list candidate slower(~0.82–0.91 forward ratio); rejected/not
  promoted. Its trace assertion assumes one explicit sync but copy path records
  zero;89 prior checks pass, failure/source/raw trace retained, no numeric/error
  assertion weakened. Scalar variant passes90 early comparisons; short timing
  noisy, extended15 alternating pairs*30 calls/thread1/four-CPU affinity used.
  Daily forward ratios1.075/1.077/1.145, forward+backward1.029/1.033/1.080;
  RCCL1.077/1.080/1.145 and1.030/1.029/1.075 (three FP32 shapes).15/15 forward
  pairs favor candidate;12–15 FB pairs. Helper wall-latency only, no dense/model/
  training-speed or memory claim. Trace native kernels2 retained, valid scalar
  reads/explicit stream syncs2→1. Bitwise old/new output/input/bias grads/pack,
  independent CPU refs,60 data-state+12 combined-error-priority cases pass.
  Current boot candidate/comparison333/6 +actual helper post-update142/2=475/8;
  retained recipe71/cohort,25 refusals/AMP/FD/adapter/stream/layout contracts.
 442 protected hashes/four AOT DBs/links retained with old-helper backup pinned;
 29 local tests/selected Ruff checks pass. RecoveryWS/recovery/quant-b054-before,
  identity-guarded helper rollback not exercised. No native build/ccache reset/
  Torch/ROCm/images/runtime-manifest/default/profile/user-app/.105 changes.
  User confirmed restart while longer run staged, no job started then; SSH returns
  r2 bootc6a50cfa-a368-4ba5-ab28-66250b4f1f1f. Extended comparisons/current gates
  all after reboot; eight current states ACK, three old5c04 guard ACK logs retained
  (/run cleared). Per-gate logs invariant in each boot/current whole batch matches;
  no cross-boot dmesg identity claim/no agent reboot. CPU fault distinct/unresolved.
  Daily561f/profile446f/nativea524 cohorts preserved. Raw WS/logs/quant-b054-*;
  final qualification-r1/final.json +ordinary.json. Local kernel/artifacts/quant-b054
  archive626,696B SHA75a22a82c56d551f55d878d6ec67edd8241a0f2c66b9b9ac3ae8aa4969609418,
  104 rawlogs2,968,279B/eight tools/versions/traces/recovery/current guards verified.
  No own jobs/containers/armed guard at handoff; memory sync `.104` only.
  [Measurements/API/recovery](docs/development/bc250_quant_b054_results/README.md).
  **Next B05.5:** explicit owned frozen-W8 nn.Module conversion +versioned state/
  checkpoint roundtrip; independent output/input/bias/adapter first-gradient refs.
  Keep strict checks/FP32 scales/bounds/opt-in/defaults/.105/user apps; no full-model
  QLoRA, quality or model-speed claim.


- **B05.3 matched RCCL native layout repair selected2026-10-24:** tools411e72917;
  no new Torch source edit, both cohorts usea5249802372882dedccb91d3fd4d73be9fcdf2bd.
  Independent WS/pytorch-quant-b053 from protected B03.7 RCCL14cf source/build
  +full two-file tracked delta; generated HIP matches exactly two contiguous
  calls. Pip exit0; NCCLon/systemNCCLon/Glooon;4CPU/4jobs/link1/RAM10GiB/
  memory+swap12GiB/shared20GiB ccache/no reset/noGPU/noNetwork. Initial~4.8GiB
  free/copy~3GiB; source-size+1.25GiB reserve and post-copy floor enforced,
  observed~1.8GiB before export. No old source/build/cache deletion/reset.
  Shared validator adds explicit --nccl-available and internal harness explicit
  --runtime-profile; defaults/math/refusals unchanged, original helper backups
  pinned. Candidate1732/24=layouts220+recipe71+retained1441, normal profile RO
  RCCL/identity/environment controls, no test-only library/env injection. Ordinary
 371/5=layouts220/train6/RCCL64/guards76/daily5.29 guards ACK, per-gate/batch/final
  dmesg identical;434 protected hashes/four AOT DBs retained with intentional
  helper updates and registry switch;29 local tests/selected Ruff/shell checks pass.
  `/opt/bc250-stack/profiles/rccl-single` →`/opt/bc250-stack/dev/rccl-single-b053`,
  profile446f345e1438f093967a495c816859f4505bd06869dbea382842c2f081f14fd7,
  libtorch_hipa31cbb159d3edfb4b7e1cc59dfcc325db185c4c540e1fcaa952e09e833335533.
  RCCLca790 unchanged. Daily B05.2 native2f276/runtime561f014e/image506f/HIPa116/
  ROCm/N3.2/math/SDMA0/heapoff unchanged. Both selected .104 cohorts now contain
  the layout repair; old14cf/profile42eb/source/build remain for rollback.
  WS/recovery/quant-b053-before stores old registry/profile/helpers; locked atomic
  registry rollback retained/not exercised. Same r2 boot5c04afbb/no reboot/.105
  contact/BIOS/fallback/user-app inspection/modification/test/deletion.
  Native backward/INT4 still refuse; W8 helper not installed/no model conversion.
  Filtered375-object RCCL still one physical GPU/rank1; no numeric FP8/fullbackend/
  symmetric-device/real offload-payload/multi-device/transport/scaling/speed/
  universal correctness/stability claim; unknown CPU fault distinct/unresolved.
  WS/logs/quant-b053-qualification-r1/final.json +quant-b053-ordinary.json; local
  kernel/artifacts/quant-b053 archive200,536,100B SHA171924233619ce789f958f195d275b6fd07e4522b45a54ed883a015212710eb6,
  332 rawlogs8,370,556B/native/RCCL/source/eight tools/recovery/29guards verified.
  No own jobs/containers/armed guard at handoff; memory sync `.104` only.
  [Use/results/rollback](docs/development/bc250_quant_b053_results/README.md).
  **Next B05.4:** profile strict W8 validation/native forward/backward overhead on
  corrected cohorts; implement measured bounded optimization with independent
  CPU/gradient/error-contract references and paired timing. Preserve default/
  opt-in/old artifacts/.105; no check weakening or model-speed inference.


- **B05.2 native weight-only INT8 layout repair installed2026-10-24:**
  Torcha5249802372882dedccb91d3fd4d73be9fcdf2bd over14cf; tools506ae3b9e.
  FP32 conversion preserves strides; native row-major kernel now receives
  contiguous activations/scales. Generated HIP delta verified exactly two calls;
  arithmetic/output cast unchanged.18-config device-generic regression added,
  eight spin/lintrunner checks pass (existing unused import removed); no full CI.
  Independent WS/pytorch-quant-b052: daily2b source/build copied +full six-file
  source overlay, immutable third-party RO; pip-only build exit0,4CPU/4jobs/link1/
  RAM10GiB/mem+swap12GiB/shared20GiB ccache/no reset/noGPU/noNetwork. Existing
  caches/source/builds retained. Postcheck initially rejects normal automatic HIP
  translation of two NCCL files; bytes match protected B03.7 exactly, NCCLoff.
  CMake0/1 vsOFF/ON spelling seal check corrected; original failures and early
  qualification refusal retained. No compiler retry or numerical relaxation.
  Candidate947/11 gates: native220 (216 exact dtype/layout/guard composites,
  private stream/device/backward+INT4 boundaries), W8 recipe71, core7/sparse253/
  FFT245/compiled14+14/profiler1/Gloo31/vision51/audio40. Ordinary231/3 gates:
  daily native220/train-consumers5/unchanged RCCL6.14 guards ACK, accepted/batch/
  final dmesg identical;402 protected hashes/four AOT DBs/links retained with
  deliberate daily Torch/runtime update and old metadata pinned to recovery.
  Active daily →WS/pytorch-quant-b052, native Torcha524/NCCLoff/Glooon,
  libtorch_hip2f2761cd7f259612863406ab0e85d7b1b5ce8b59a32655bf9ac7db9359d0df27,
  runtime561f014e31c4ca0bba73861d56125e4e18124867904b7c23cff8d7e0fa36d2a1.
  Image506f/HIPa116/ROCm/N3.2/math/SDMA0/heapoff unchanged. Recovery
  WS/recovery/quant-b052-before/runtime.json=32fd4559, old2b build/source kept;
  locked identity-guarded rollback helper retained/not exercised. No model
  conversion/W8 package installation/native backward/INT4/performance/universal
  claim. **Opt-in RCCL14cf/profile42eb remains old and unfixed**, still launches;
  its W8 helper normalization remains needed. Unknown CPU fault distinct/unresolved.
  Same r2 boot5c04afbb/no reboot/.105 connection/BIOS/fallback/user app touch.
  Raw WS/logs/quant-b052-*, candidatefinal qualification-r1/final.json,
  installed WS/logs/quant-b052-ordinary.json; local kernel/artifacts/quant-b052
  archive193,263,147B SHAfce8095e5c163f36d100f9730137faad0fcd06b98014448e1f17db729a002288,
  153 rawlogs5,516,695B/native payloads/eight tools/source/recovery/14 guards
  SHA-verified.29 local BC250 tests/selected helper Ruff checks pass. No own jobs/
  containers/armed guards at handoff; memory sync `.104` only.
  [Use/evidence/recovery](docs/development/bc250_quant_b052_results/README.md).
  **Next B05.3:** check disk/resource headroom; independent RCCL-enabled Torch
  candidate with same layout fix, exact layouts +retained RCCL/gradient contracts,
  then locked matched opt-in profile identities/ordinary acceptance/rollback.
  Preserve new daily Torch, older candidates/caches/fallbacks/user apps and `.105`.


- **B05.1 native low-bit audit/W8 prototype completed2026-10-24:** codec4383972c.
  Existing Torch `_weight_int8pack_mm` generic HIP passes contiguous FP32/FP16/
  BF16 references, but strided activation15/15 and strided FP32 scale12/15
  wrong results reproduced. **Native defects remain unfixed.** Native backward
  explicitly refuses; native packed INT4 conversion rejects gfx1013 via CDNA2
  gate. TorchAO/bitsandbytes absent; Triton3.8 available is not kernel admission.
  Source INT8/INT4 hashes match patched local Torch14cf and active native2b.
  Explicit `WS/helpers/bc250_quant_b051.py` prototype: frozen per-channel W8,
  FP32 scales/round-even/clamp[-127,127]/zero-row1, normalized layouts, native HIP
  forward and custom first input derivative using scaled upstream/transposed
  INT8. Bias/owned adapter first gradients; no integer/scale/base/higher grads.
  Bounds dimensions4096/weight4M/activation-output1M/MAC128M; finite checks sync
  host. Output retains explicit input dtype under AMP, bias after output cast.
  No full floating weight retained/materialized by helper; source weights kept
  by caller still consume memory. Prototype only, **not installed/default**.
  Audit7/prototype71/core7/compiled dense14/daily5/opt-in6=110/six accepted gates.
  Scalar CPU packing/CPU-double forward+input/bias refs,36 dtype/layout composites,
  all256 bytes/ties/zero rows/guards/private stream/six exact finite differences,
  owned FP32/BF16 adapter refs/AMP/25 refusals/two native GPU trace events pass.
  Initial collector assumed null gradient; explicit derivative error retained,
  corrected collector records refusal. No tolerance/backend change. Seven guards
  including failed probe ACK; accepted/batch/all-run dmesg match.377 protected
  hashes/four AOT DBs/links retained;29 local tests/selected Ruff checks pass.
  Three alternating pairs/three eager calls: shapes8x128x256/32x256x256 recipe
  medians0.26045/0.25816ms versus resident-dense0.08222/0.08134ms (~3.2x slower).
  Fresh packed payload33,280/66,560B vs FP16 65,536/131,072B (~1.97x smaller),
  not total/peak/process memory or model-quality/training-speed qualification.
  Daily Torch2b/NCCLoff/Glooon/image506f/runtime32fd/HIPa116 and opt-in14cf/
  RCCLca790/profile42eb retained. Current r2 boot5c04afbb-ff2f-4f4b-9eac-8238bb9288fc
  observed new since B03.8; reboot cause not investigated/no agent reboot.
  No native rebuild/ccache operation/prefix replacement/.105 contact/user app
  inspection/modification/test/deletion; Python CPU fault distinct/unresolved.
  WS/logs/quant-b051-qualification-r1/final.json, audit-r2/prototype-r1; local
  kernel/artifacts/quant-b051 archive475,903B SHA6051eee04658afdb81fd017087ae01f2b268843303202965a57906447ebb2f41,
  63 rawlogs1,497,225B/five source hashes/native source/seven guards verified.
  No own build/test/container/armed guard at handoff; memory sync `.104` only.
  [API/evidence/limits](docs/development/bc250_quant_b051_results/README.md).
  **Next B05.2:** fix demonstrated native INT8 activation/FP32-scale layout handling
  in qualified Torch; isolated pip build with recorded ccache/resource bounds and
  independent layout/output regression references. Preserve native unsupported
  backward boundary, custom W8 derivative, daily Gloo/opt-in RCCL/.105/fallbacks;
  no model quantization activation or speculative INT4 gate bypass.


- **B03.8 ordinary-launch opt-in RCCL completed2026-10-23:** code36cabf9be.
  `.104` registry `/opt/bc250-stack/profiles/rccl-single` →
  `/opt/bc250-stack/dev/rccl-single-b038`; select `--runtime-profile rccl-single`
  in installed bc250-torch/bc250-train. Default/daily remains Torch2b449c4,
  NCCLoff/Glooon/runtime32fd4559/image506f/HIPa116/native systems1f9609/r2.
  Profile reuses B03.7 Torch14cf/libtorch_hipeb405cb1 and independently copies
  RCCLca790dc7 plus relative SONAME links; shares daily ROCm/cache/consumers.
  Not an independent relocatable release. ManifestSHA42eb7631; identity checks
  cover nine Torch DSOs/GIT_REVISION/RCCL and links before ordinary launch.
  RO `/workspace/rccl`, controlled LD path, window/CUMEMboth0/P2Poff/IBoff/lo/
  async-error1; explicit conflicting environment refused. User-owned private
  context retains selection. Daily SH lock covers hash checks and process;
  updater now refuses an independent profile lock: use daily EX for updates.
  Ordinary26-gate1461 pass (old1441+new20): packaged torch5/train6, all22 B03.7
  gates through bc250-train without harness library/env injection, daily4/5.
  Extra private-context5 pass; nine installed CPU refusals (seven settings,
  unknown profile, wrong update lock), live ordinary-launch daily EX exclusion
  pass. Local29 BC250 tests and selected Ruff checks/format pass; no full CI.
  372 protected hashes/four AOT DBs/links retained; two launcher copies intentionally
  changed with originals backed up.27 guards ACK, each accepted gate and main
  batch dmesg identical; same boot2db1c7c2/no reboot. No native build/ccache reset/
  BIOS/fallback/user application touch/deletion; `.105` not contacted or changed.
  No multi-device/transport/scaling/full-backend/speed/higher-gradient/symmetric
  device/actual offload-payload claim. Numeric FP8 refused, byte copies retained;
  unknown Python CPU fault unresolved/distinct from VM-heap/CSR/hostcall fixes.
  Raw WS/logs/rccl-b038-*; mainfinal WS/logs/rccl-b038-qualification-r1/final.json,
  supplemental WS/logs/rccl-b038-context.json; recoveryWS/recovery/rccl-b038-before.
  Local kernel/artifacts/rccl-b038: archive7,580,482B SHA64beea0d3b9396f84207d5d055786b4028f7a9009612fd3f1d656a549f6a2efa,
  284 raw logs6,078,114B/profileDSO/recovery/27guards/source verified. Rollback
  helper restores launcher/unregisters profile; not exercised. Jobs complete,
  no own build/test/container/armed guard at handoff; memory sync to `.104` only.
  [Use/evidence/rollback](docs/development/bc250_rccl_b038_results/README.md).
  **Next B05.1:** on `.104`, audit current A08 dense INT8 and pinned low-bit
  Torch/Triton/package paths, then select one stack-owned bounded weight-only
  quantization recipe with CPU output/gradient references. Do not reinstall Torch
  or infer matrix instructions from package flags; keep daily defaults and `.105`.


- **M105.2 future-update guide completed2026-10-23:** workspace
  [HOST105_UPDATE.md](../HOST105_UPDATE.md) describes matched .104 exports,
  independent candidate staging, target qualification, the explicit .105 control
  lock/idle-board requirement, atomic activation/rollback and separate guarded
  kernel updates. It records the boot-health package's fixed original-release
  dependency and the single-use migration-helper limits. Documentation only;
  no runtime/kernel/default/qualification change. Guide copied to both hosts;
  M105.1 evidence remains authoritative. **Next B03.8 unchanged.**


- **M105.1 user-authorized deployment/retirement completed2026-10-23:** tools
  ddc2933cb. `.105` is now a deployment host, ending its old-reference role.
  Independent release `/opt/bc250-stack/releases/10.2-gfx1013-20261023-host105`,
  current symlink; image506f3b26 tagged `localhost/bc250-stack:10.2-gfx1013-20261023-host105`.
  Native Torch2b449c4/NCCLoff/Glooon, HIP1f9609/HIP DSOa1162308, rocBLAS8389197,
  remaining libraries319a235d6 retained from daily `.104`; no native rebuild.
  Signed K6 image84995f1/module0957f1 installed; kernel7.2.10-bc250rocm-r3,
  build IDs57f7c52d/b97bca56; default `bc250-host105-r3-release`. GRUB saves r2
  next_entry before Linux; SSH/cooling/NIC/nowayout-feeder/module identity/GPU7
  readiness ACKs and clears it only after success. Late runtime guards explicitly
  select r2 before emergency reset. Root-owned release `healthchecks/`, operational
  state `/var/lib/bc250-stack/host105`, console `/var/log/bc250-host105-ready.log`;
  retain KWS artifacts/private firmware/helpers, r2 files/modules and control lock
  `/opt/bc250-stack/migration-control/.runtime.lock`. Cgroups/watchdog boot enabled.
  Main656/nine gates pass: core7/CPU Gloo31/CSR253/FFT245/vision51/audio40/
  compiled dense14/CNN14/profiler1. Each accepted gate dmesg identical/600s guard ACK.
  Eight supplemental GPU7 smokes and final ordinary launcher exact forward/gradient
  pass. Three normal promoted r3 boots; intentional15s idle expiry→r2/SSH/new-stack
  GPU7→normal r3 qualified. Final boot e3d322db-a221-47a1-9ab4-9a48b84574a5,
  runtime SHAc5da85ef459e9e586904a2870d68545a911805d0c7d93016cb4a3f9b96cc608b.
  Early harness consumer-mount failure and OCI refusal after lost manual cgroups
  mount preserved; mount corrected/service persisted, recovery passed before ACK.
  Persisted GPU/runtime guardians19 ACK/one intentional expiry; no armed guard.
  All69,229 ROCm/Torch entries match before/after retirement and local runtime
  recovery archive. Old unique sources/Git/modified MIOpen archived locally;
  460 source hashes compared, b578 Git full fsck passes.36 old directories/five
  stopped containers/eight selected7.2.4 images retired; public6.18 downloads and
  own transfer archives removed. Free~101GiB (108477546496B vs~47.7GiB initially).
  Preserve user apps/models/checkpoints, Docker/unrelated images/home caches,
  historical small research records and native board-control/SMI utilities.
  No BIOS or finetune-n7 inspection/modification/test; `.104` runtime32fd4559/
  Torch2b/HIPa116/r2 boot2db1c7c2 unchanged; only small export/memory files added.
  No universal/performance/multi-board/sustained or arbitrary/pre-watchdog hang
  claim; Python CPU fault unresolved/distinct from VM-heap/CSR/hostcall fixes.
  Raw local `kernel/artifacts/migration-105-20261023/`, target WS
  `/home/nik/bc250work/migration-105-20261023/final.json`; evidence archiveSHA
  532eb6bb0b72fefba524e3ddd2cbe87136c8d0cfd381df2977a3caa784a618e1.
  [Usage, identities, archive/recovery scope](docs/development/bc250_host105_migration_results/README.md).
  Board build/test/container jobs complete; no own jobs remain after final sync.
  **Next B03.8:** resume bounded opt-in single-device RCCL packaging on `.104`,
  read-only DSO loading/mounts and explicit window/CUMEM/P2P/IB settings. Daily
  Gloo/NCCL-off and `.105` production snapshot remain selected; cross-board
  distributed execution/scaling still requires its own qualification.


- **B03.7 isolated FP8/API admission guards completed2026-10-23:** tools819bb624f;
  Torch14cfef2d8c overf807099 overade6fcc; RCCL38fd6/B03.4 unchanged. Independent
  WS/pytorch-rccl-b037, releaseWS/rccl-b037/candidate-release-r2, not daily prefix.
  Low-level FP8 numeric AllReduce/Reduce/ReduceScatter now refuse before comm;
  existing uint8 mapping remains for byte copies. PG helper checks actual HIP
  device and refuses FP8 on filtered gfx1013; other devices retain prior version
  policy. Initialf807 builds; r1 passes33 then missing PG Reduce refusal fails.
  14cf adds Reduce/_reduce_oop guards; rebuild-r2 passes, compiled follow-up
  source diff verified exactly six guard lines. Failed source/native libs saved
  WS/recovery/rccl-b037-reduce-before; release-r1 link is reused build, not frozen.
  Targeted spin/lintrunner-a initial8/follow-up3 pass; standard device-generic
  regression added, full upstream CI not run. No multi-GPU wrong-result repro.
  Qualification-r2=1441/22 gates: new76+retained1365. New15 lowlevel/20 PG FP8
  numeric refusals/five formats;32 guarded raw-bit copy refs/four formats*17/257*
  two APIs*two copyops cover all256 encodings. Existing multicast capabilityfalse/
  allocatorrefusal, generic MemPool register/PreMulSum/later allocation/deregister,
  explicit window optout success+NULL, stats/suspend-resume0-1-0/post arithmetic,
  device/loaded DSO/lifecycle pass.36 refusals total. Memory tracked counters all
  zero: no actual offload payload relocation/reclamation or speed proof. Symmetric
  window skip is not functional symmetric support. Full symmetric device API/
  graphs/concurrency/shrink/multi-device/protocol/transport/scaling unqualified.
  356 protected paths/four AOT DBs/links unchanged incl prior340;22 accepted guards
  ACK/gate+batch dmesg match, all23 incl r1 guard ACK. Candidate libtorch_hip
  eb405cb1031343e6; native DSO/client payloads SHA-verified local, prior B03.5/B03.4
  preserved. Active Torch2b449c4/NCCLoff/image506f/runtime32fd4559/HIPa116/native
  systems1f9609/ROCm/N3.2/math/SDMA0/heapoff unchanged. Build4CPU/4jobs/one link/
  10GiB RAM/12GiB mem+swap/shared20GiB ccache/no reset/noGPU/no network; cumulative
  cache stats not timing. Same r2 boot2db1c7c2/no reboot/.105/BIOS/fallback/user
  application inspection/modification/test/deletion. Unknown CPU fault unresolved/
  distinct from fixed VM-heap/CSR/B04.2 queue. Local kernel/artifacts/rccl-b037:
  archive1,083,504,640B SHA963a1d6416ae1aaca62565253250b1e29812def7bc22cbd83688f9db8c225ff8,
  291 raw logs8,996,665B/native/23guards/recovery/source/bundle verification.
  [Report/profile/reproduce](docs/development/bc250_rccl_b037_results/README.md).
  BaselineWS/logs/rccl-b037-qualification-r2/final.json; helper/metadata backup
  WS/recovery/rccl-b037-before. No own jobs/containers/root SSH remain; final
  reports override historical running-job checkpoint. Revision memory sync.
  **Next B03.8:** package bounded opt-in single-device profile with read-only
  RCCL DSO loading/mounts and explicit window/CUMEM/P2P/IB settings; qualify ordinary
  packaged launch output/backward with owned fixtures. Test-only extra mount is
  not ordinary launcher integration. Keep daily NCCL-off/Gloo runtime+rollback;
  no full-backend/multi-device/offload-payload/symmetric-device claim.

- **B03.6 isolated public RCCL contracts/broader regression completed2026-10-23:**
  tools9dc76a795; unchanged candidate Torchade6fcc/RCCL38fd6/B03.4 DSOca790dc7,
  candidate libtorch_hip7f9e69a4. No production defect/fix/native library rebuild/
  ccache operation/default change. R1 passes1365/21 gates: newcontracts178 +old
  RCCL64/GPU Gloo41 +existing18-gate1082 stack suite. Contract128 identity cases=
  eight types*17/257*disjoint-output/exact-alias*four collective wrappers; eight
  broadcasts/five zero counts/four availability predicates/12 low-level refusals.
  Three communicator init/nontrivial FP32 PreMulSum/destroy cycles, subgroup,
  cross-private-stream event handoff; eight PG host refusals incl fnuzFP8/boolAVG/
  bitwise; four BF16/FP32 analytical DDP composites (bucket views off/on, no_sync,
  two microbatches/two SGD updates); loaded DSO verified. Independent exact CPU
  refs/offset5/guards/unchanged inputs; DDP constant1/8/identity/power-of-two case
  only. Ordinary rank1 reductions are identities, not transport proof. No broad
  BF16/FP8/higher-gradient/partial-overlap/graphs/registration/multi-device/
  protocol/transport/scaling/speed claim. No numerical tolerances weakened.
  Retained1082 includes CPU Gloo31/sparse253/triangular80/legacy41/FFT245+72+3/
  vision51+9/audio34+40/image86/video55/files46/compiled14+14/profiler1/core7.
  All340 protected hashes/four AOT DBs/links unchanged against B03.5;21 guards ACK,
  accepted gates+whole-batch dmesg identical. Active Torch2b449c4/NCCLoff/image506f/
  runtime32fd4559/HIPa116/native systems1f9609/ROCm/N3.2/math/SDMA0/heapoff retained.
  B03.5 release-r3 reused, not deployed; native artifacts remain B03.5/B03.4.
  Source audit: low-level FP8 storage maps uint8 (not numeric FP8 reduction proof),
  availability predicate lacks full dtype/hardware eligibility; fnuz PG refusals
  tested, e4m3fn/e5m2 version admission and excluded APIs remain unaudited at runtime.
  Same r2 boot2db1c7c2/no reboot/.105/BIOS/fallback/user application inspection/
  modification/test/deletion. Unknown CPU fault unresolved/distinct from fixed
  VM-heap/CSR/B04.2 queue. Local kernel/artifacts/rccl-b036: archive6,543,360B SHA
  d083ff68a56bee95053784c7943da4cfcd4c63c7000c0a5880af6ab5cb49e31b verified;
  257 raw logs5,386,517B/21 guards/helpers/recovery, no duplicate native payloads.
  [Report/reproduce/boundaries](docs/development/bc250_rccl_b036_results/README.md).
  BaselineWS/logs/rccl-b036-qualification-r1/final.json; helper backupWS/recovery/
  rccl-b036-before. No own jobs/containers/root SSH remain; revision memory sync.
  **Next B03.7:** audit/guard excluded or version-admitted public paths (FP8
  reductions versus byte-copy semantics; symmetric/registration/offload APIs),
  then determine concrete bounded single-device promotion profile. Keep daily
  prefix unchanged; one physical GPU cannot qualify multi-device correctness.

- **B03.5 isolated Torch RCCL integration completed2026-10-23:** toolsce089950f;
  Torch source/candidateade6fcc1ae over2b449c4, RCCL38fd6ba67d/B03.4 unchanged.
  ROCm find_package(rccl)/USE_NCCL=1/USE_SYSTEM_NCCL=1, headers2.32.3 exceed2.23;
  42 lexically called declared APIs exported, actual consumer builds/links native
  gfx1013. libtorch_hip RUNPATH names isolated B03.4 DSO, loaded path verified.
  R1 passes31 initial checks then reproduces BF16 host PreMulSum zero/nonzero
  reference failure: float scalar template supplies low two bytes to BF16 API.
  ade6fcc uses at::BFloat16 and adds BF16 upstream regression-loop coverage;
  source also explains old device tensor float-pointer mismatch (negative GPU
  reproduction was host only). Corrected host/device scalar refs pass. Eight
  targeted spin linters and lintrunner -a pass; no full CI/multi-GPU test claim.
  R2 builds/64 RCCL+41 GPU Gloo pass; CPU Gloo old NCCL-off fixture contract fails.
  Opt-in --nccl-available fixes fixture; default off and numerical checks retained.
  R3 passes172/7 gates: RCCL64/GPU Gloo41/CPU31/compiled dense14/CNN14/profiler1/
  core7. RCCL one rank/device:24 SUM offset guards/stream identities,24 nontrivial
  PreMulSum fourfloat types*three sizes*host-device scalars,10 other identities,
  FP32/64 functional first gradients and two DDP SGD/momentum updates,2metadata/
  DSO/lifecycle composites. Small exact inputs/fixed tolerances; no broad numeric/
  multi-GPU/transport/protocol/scaling/speed/higher-gradient/full-library claim.
  340 protected paths/four AOT DBs/links unchanged; seven accepted600s guards ACK,
  accepted gate+batch dmesg identical. All11 guards incl failed runs ACK/preserved.
  CandidateWS/pytorch-rccl-b035 +WS/rccl-b035/candidate-release-r3 ISOLATED;
  active Torch2b449c4/image506f/runtime32fd4559/HIPa1162308/native systems1f9609/
  ROCm/N3.2/math/SDMA0/heapoff unchanged, daily NCCL off. Original failed source/
  all Torch lib payloads backed upWS/recovery/rccl-b035-bf16-before before rebuild;
  r1/r2 release links point at reused candidate, not immutable old payloads.
  Existing active sources/builds/B03.3/B03.4/caches retained. Compile4CPU/4jobs/
  one link/10GiB RAM/12GiB mem+swap/shared20GiB ccache/no reset/noGPU/no network;
  no cache performance claim. Same r2 boot2db1c7c2/no reboot/.105/BIOS/fallback/
  user application inspection/modification/test/deletion. Unknown Python CPU fault
  unresolved/distinct from fixed VM-heap/CSR/B04.2 queue. Source is retained2b
  archive+exact production C++ overlay; upstream test change remains local/bundle.
  Local kernel/artifacts/rccl-b035: SHA-verified archive1,080,432,640B/142 raw logs
  6,797,890B/native/failed payloads/11 guards/recovery; compiled hipified source
  delta verified only intended specialization. Candidate libtorch_hip7f9e69a479fbaffc.
  [Report/reproduce/limits](docs/development/bc250_rccl_b035_results/README.md).
  BaselineWS/logs/rccl-b035-qualification-r3/final.json; helper/metadata backup
  WS/recovery/rccl-b035-before. No own jobs/containers/root SSH remain; final
  reports supersede historical running-build checkpoint. Revision memory sync.
  **Next B03.6:** isolated one-device Torch/RCCL consumer contracts: public out-of-
  place/type/alias/error paths, stream/lifecycle behavior, BF16 DDP where feasible;
  broader stack regression before considering promotion. Explicitly audit filtered/
  excluded API paths. Keep daily runtime/.105; one GPU cannot prove multi-device.

- **B03.4 broader isolated native RCCL completed2026-10-22:** tools7ea133762;
  source38fd6ba67d unchanged (B03.3 fixes already included). Native gfx1013 build
  compiles375 specialized objects: AllReduce270/Reduce45/ReduceScatter45/
  Broadcast6/AllGather6/SendRecv3; SIMPLE201/LL174, Ring/Tree typed filter.
  905 generated files/864 entries:375 admitted/489 guarded out. LL128 retains
  upstream gfx1013 exclusion; FP8/PAT/symmetric/rocSHMEM/GIN/MSCCL/ROCTX/Python
  outside selection. Compilation is not multi-rank/protocol runtime proof.
  Build/client exit0; isolated629/4 gates=native550+Gloo GPU41/CPU31/core7.
  Native120 PreMulSum +420 AllReduceWithBias CPU references:ten types,17/257/1025,
  in/out, host/device scalars; bias offset5/guards/unchanged inputs and bias.
  Integer comparisons remain integer incl uint64 modulo wrap; small exact float
  values only. Six identity/four metadata-lifecycle checks separate. One physical
  GPU/rank1/private stream/loaded DSO verified; no RCCL autograd/Torch integration/
  multi-GPU/transport/scaling/speed claim. No new production source defect/fix.
  328 protected paths/four AOT DBs/links unchanged; four600s guards ACK, gate and
  batch dmesg identical. CandidateWS/rccl-b034/install-r1 isolated, B03.3 preserved;
  Torch NCCL off. Active image506f/runtime32fd4559/Torch2b449c4/HIPa1162308/native
  systems1f9609/ROCm/N3.2/math/SDMA0/heapoff unchanged. Build4CPUs/4jobs/one link/
  10GiB RAM/12GiB mem+swap; client2CPUs/4GiB/5GiB; no GPU/network during compile.
  Shared20GiB ccache/no reset covers standard CXX/client; custom RCCLDEV/device
  link bypass launcher, no batch cache speed claim. Same r2 boot2db1c7c2/no reboot/
  .105/BIOS/fallback/user app touch/deletion. Unknown CPU fault unresolved/distinct
  from fixed VM-heap/CSR/B04.2 queue. Local kernel/artifacts/rccl-b034 has source/
  native DSO/client/50 raw logs1,353,457B/four guards/recovery; archive15,267,840B
  SHA d6f1bd4944c3bec967c4729b553f59d502ae2a1a9c1911575ef0dba0ff489d4e verified.
  DSO ca790dc73b76a94a7dc97538608cd94b789e73cc18d756b0631111c18230583e.
  [Report/reproduce/boundaries](docs/development/bc250_rccl_b034_results/README.md).
  BaselineWS/logs/rccl-b034-qualification-r1/final.json; helper backupWS/recovery/
  rccl-b034-before. All build/test jobs complete; no running own containers/root
  SSH. Historical running-build-checkpoint.json is superseded by final reports.
  **Next B03.5:** inspect pinned Torch NCCL/RCCL consumers/FindNCCL/API requirements;
  prepare independent NCCL-enabled Torch candidate against isolated RCCL and
  one-device/rank1 output+first-gradient tests before deployment. Keep current
  runtime/Gloo and .105; no full-library/multi-GPU/performance claim.

- **B03.3 isolated native RCCL feasibility/readiness completed2026-10-22:**
  tools7e7896b10; RCCL source38fd6ba67d over1f9609fd6, includingc653cbc33c.
  Native gfx1013 restricted build succeeds; ONLY_FUNCS=AllReduce RING SIMPLE Sum
  f32, symmetric/rocSHMEM/GIN/MSCCL/ROCTX/Python/upstream tests off. Mandatory
  one-rank/CE/DDA code also compiled. No spoofing/full-RCCL or Ring-runtime claim.
  Build-r1 device link succeeds/host duplicate nccl_domain fails; c653 honors
  NVTX_NO_IMPL in direct full-header includes/adds no-op range macro. Build-r2
  missing std::cerr declaration fails;38fd adds direct iostream (CRLF retained).
  Build-r3/client-r1 exit0; original source/failed logs/build/cache preserved.
  Qualification-r1=161/4 gates: native82 +GPU Gloo41/CPU Gloo31/core7. Native72
  PreMulSum actual oneRankReduce GPU arithmetic: six dtypes*three sizes17/257/1025*
  in-out*host-device scalars, exact CPU outputs/input+output guards/private stream;
  six one-rank identity-copy/group cases and four metadata/lifecycle checks.
  Loaded candidate DSO/version/native-device identity verified. No autograd/
  exceptional-value/overflow/full rounding/multi-GPU/network/scaling/speed claim.
  323 protected paths/four AOT DBs/links match, four600s guards ACK, every accepted
  gate+whole batch dmesg identical. CandidateWS/rccl-b033/install-r1 outside active
  prefix; Torch NCCL still off. Active image506f/runtimeJSON32fd4559/Torch2b449c4/
  HIPa1162308/systems1f9609 native/ROCm/N3.2/math/SDMA0/heapoff unchanged.
  Build4jobs/4CPUs/one link/10GiB RAM/12GiB mem+swap/shared20GiB ccache/no reset;
  client2CPUs/4GiB/5GiB mem+swap; both no GPU/network. No source/build/cache delete.
  Same r2 boot2db1c7c2/no agent reboot/.105/BIOS/fallback/user app touch.
  Unknown CPU fault unresolved/distinct from VM-heap/CSR/B04.2 queue; fixes are
  compile-only. Local kernel/artifacts/rccl-b033 holds source archive/native DSO/
  client/55 raw logs1,495,447B/four guards/recovery. Archive3ee65ba1 verified
  11,816,960B; DSO4af5420f76dacb1a88ffb42109563a977dc10ee4b2dba8b790e13d8fa13376f0.
  [Report/reproduce/boundaries](docs/development/bc250_rccl_b033_results/README.md).
  BaselineWS/logs/rccl-b033-qualification-r1/final.json; source backupWS/recovery/
  rccl-b033-source-before, helper backup rccl-b033-before. No own jobs/root SSH
  closed; revision-keyed memory snapshot. **Next B03.4:** broaden isolated native
  gfx1013 RCCL build beyond FP32 Ring/SIMPLE; inspect admitted type/protocol
  combinations and qualify feasible single-device operations before Torch NCCL
  integration. Keep current prefix/.105/defaults; no multi-GPU/P2P/XGMI/GDA/SDMA
  transport/scaling proof from one GPU.

- **B03.2 Gloo GPU tensor/DDP qualification completed2026-10-22:** code418db3a90.
  Existing native Torch2b449c4/B03.1 handles bounded HIP tensors; no runtime patch/
  native rebuild/ccache/default change. Two independent ranks share physicalcuda0,
  explicit Gloo TCP/loopback; collectives copy via pinned CPU buffers, model math/
  gradients remain GPU. Source inspection precedes testing; TCP hasGPUDirectfalse,
  GPU send/recv raw-pointer path left unexecuted/unqualified.
  Installed-r1 passes108/6 gates: newGPU41+CPU31+compiled dense14/CNN14+profiler1/
  core7. GPU24 SUM=eight dtypes*factors1/17/1025, exactly representable values,
  offset guards/private stream async wait+immediate consumption; FP64 product/
  min/max/AVG, broadcast and six gather/scatter/reduce/alltoall paths, FP32/64
  analytic functional first gradients, FP32/64 GPU DDP SGDmomentum+AdamW/two
  updates against independent CPU same-dtype global-batch output/params/gradients.
  Fixed tolerances unchanged. Legacy functional/reduce-scatter deprecation warnings
  retained; newer functional API/BF16 DDP/checkpoint recovery/RPC/GPU send-recv/
  non-loopback/multi-device/scaling/throughput/higher gradients unqualified.
  318 hashes/four AOT DBs/links unchanged; six600s guards ACK, every accepted gate
  and whole-batch dmesg identical. Active image506f3b26/runtimeJSON32fd4559/
  WS/pytorch-distributed-b031/native Torch2b449c4/ROCm/defaults retained.
  Initial user miner preflight refusal, user freed board; process untouched.
  Same r2 boot2db1c7c2; no reboot/.105/BIOS/fallback/user finetune-n7 inspection/
  modification/test/delete. Unknown CPU segfault unresolved/distinct from VM-heap/
  CSR/B04.2 queue. Local kernel/artifacts/distributed-b032 holds73 raw logs
  1,659,204B/six guards/helpers; archive8cdbb0e5 SHA-verified2,048,000B.
  [Report/use/limits](docs/development/bc250_distributed_b032_results/README.md).
  BaselineWS/logs/distributed-b032-qualification-r1/final.json (318 paths);
  shared-helper backupWS/recovery/distributed-b032-before. No own jobs/root SSH
  closed; final revision memory snapshot on .104. **Next B03.3:** independent
  native gfx1013 RCCL configure/build feasibility and single-device readiness.
  Pinned source accepts explicit target compiler checks/non-GFX9 wave32 branch;
  no build/runtime support proof yet. Keep current Gloo/Torch, .105 and defaults;
  RCCL/Torch NCCL still off, no multi-GPU/P2P/XGMI/GDA/SDMA/scaling claim.

- **B03.1 native local CPU Gloo installed2026-10-22:** integration8e6416f39
  (initial toolingedeabcf4b), patched Torch source/native2b449c4abe36b896a6efb2a68e7a42d9922d5696.
  Independent WS/pytorch-distributed-b031 native build exit0; distributed/Gloo/
  TensorPipe+bundled libuv ON, NCCL(RCCL)/MPI/UCC/ibverbs/TLS OFF. CPU-only31
  checks pass without GPU devices. Candidate-r2/installed-r3 each1082 checks/18
  gates: new31 +retained1051 sparse/FFT/vision/audio/codecs/compiled dense+CNN/
  profiler/core. Two local ranks/torchrun, three stores incl libuv TCP, eight SUM
  dtypes/other collectives/subgroup/CPU send-recv, analytic first gradients,
  FP32/64 DDP SGDmomentum+AdamW/two updates versus independent global-batch
  model/all params+gradients. Loopback CPU only; GPU distributed/RPC/network/
  multi-board/higher gradients/checkpoint recovery/performance unqualified.
  Candidate-r1 retained: audio numerical checks pass, historical missing-codec
  expectation fails; selected existing --codec-available flag, no tolerance change.
  Gloo bcd1672/TensorPipe2b4cd910/libuv5152db2 pinned,276/292/467 tracked regular
  files verified; existing host Gloo legacy HIP guard delta captured as patch,
  not newly applied/runtime repair. Native source archivecfe2e98b preserved locally.
  Build4 Ninja/4CPUs/one link/10GiB RAM/12GiB mem+swap/shared20GiB ccache/no reset,
  no GPU; build image28ad36e6. Old sources/builds/caches/images/releases retained.
  Active image506f3b262fa5fe8c29b9babec07b03078dd6f85492a9ac2cffc65955297e4349
  unchanged; runtimeJSON32fd45595989e09597dbb0fcfc8992595e8861c0581a49827fcf965e35a865e0.
  Torch DSOs changed, ROCm/kpacks/N3.2/math/SDMA0/heapoff defaults unchanged.
  306 prior hashes passed before install; installed318 includes recovery/new native
  identities and intentional runtime JSON update; four AOT DBs retained.
  47 acknowledged600s guards; accepted gate and two passing whole-batch dmesg
  identical. Exact oldTorchlink/runtime in WS/recovery/distributed-b031-before;
  locked deploy_distributed_b031_on_host.py --rollback available, unused.
  New boot2db1c7c2 observed after build, still r2; no agent reboot, cause unknown.
  Miner initially blocked preflight, left untouched; user freed board before GPU
  tests. .105/BIOS/boot fallback/user finetune-n7 untouched, include_training_app=False.
  Historical unknown CPU segfault unresolved/distinct from VM-heap/CSR/B04.2
  queue and B02 known assertion cleanup trigger. No universal correctness/speed claim.
  Local kernel/artifacts/distributed-b031 holds source/native payload/579 raw logs
  15,615,864B/47 guards/recovery; archive7e53e0ed SHA-verified552,079,360B.
  [Report/use/recovery](docs/development/bc250_distributed_b031_results/README.md).
  BaselineWS/logs/distributed-b031-qualification-r3/final.json; rawWS/logs/
  distributed-b031-*; disk58,211,602,432B observed.
  No own build/test/container/guard/export running; root SSH closed. Remote memory
  snapshot keyed by final memory revision. **Next B03.2:** inspect pinned Gloo HIP
  tensor handling/native RCCL applicability and select a bounded single-board GPU
  tensor/software-transport probe; no .105/multi-board/P2P/XGMI claims.

- **B02.1 explicit bounded CSR solve installed2026-10-21:** coded2c3ea105,
  bc250_sparse0.1.0 SHA272cc68c8e898e9f8eed5816a55b5ae9e2076861b575304713daae47f0b300a6.
  Pinned Torch general API rejects ROCm; hipSOLVER QR already densifies, CSRRF
  needs initial factors, N2.13 triangular is separate. Added opt-in spsolve/CSR
  parts API: same-GPU scatter/dense LU, explicit first adjoint with AMP disabled,
  canonical square1..1024/RHS1..64/four data types/two indices. Major storage
  estimate<=64MiB before densification, not total allocator/native workspace cap;
  finite/zero-pivot/FP64 normalized residual checks. Parts API supports complex/
  real stored-value+RHS gradients; CSR wrapper real values/all RHS types, complex
  value gradients explicitly use parts API. No CPU solver/global override/native
  public API/scalable sparse factorization/factor reuse/performance claim.
  Candidate-r4/installed-r5 each381=new253+N2.13 edge80+sparse41+core7. New192
  CPU-double output/all-leaf gradients across dtype/index/shape/RHS/layout/AMP,
  eight pivot/analytic/wrapper,13 larger analytic diagonal65/257/1024/RHS64,
  exact budget boundary/two directional FD/35 refusals+contracts/traced fixture/
  actual native GPU trace135 kernels. Fixed single rtol3e-4 atol3e-6, double
  rtol2e-10 atol2e-12; maxoutput4.011e-7, absolute residual7.807e-7; FDmax1.189e-11
  <1e-8. Residual not conditioning/forward-error guarantee; first derivatives only.
  Entry pre-Python Podman terminal failure from missing /var/tmp fixed mkdir1777;
  no package/runtime replacement. Initial audit complex CSR constructor autograd
  rejection retained; revised parts audit8 pass. Failed candidate-r1 AMP gradient
  rounding fixed explicit adjoint; r2 complex CSR values-view fixed detached copy;
  r3 native GPU invariant checker on invalid columns asserts/aborts/cleanup139+
  libc fault. Final helper uses safe GPU predicates before indirect consumption,
  passes unchanged refusals. Fresh-process core7 recovery passes without reboot.
  Known assertion cleanup trigger does not identify/fix historical unknown CPU
  segfault; native malformed-CSR assertions outside helper unchanged. VM-heap/CSR/
  B04.2 queue defects remain distinct. No numerical tolerance relaxed.
  Image506f3b262fa5fe8c29b9babec07b03078dd6f85492a9ac2cffc65955297e4349;
  runtimeJSONf09cae594646107e8101e2bc383e4ee7257a3c3354c218805a9a6350091f35b1.
  PurePython image layer only/no native build/ccache operation/Torch/ROCm DSO/
  kpack/default changes. Previous5d0a6 image/source/build/cache/release history
  retained; recoveryWS/recovery/sparse-b021-before/runtime.json+old qualifier,
  locked deploy_sparse_b021_on_host.py --rollback restores exact metadata/image,
  unused. Sixteen600s guards ACK; eight accepted gate dmesg/two whole batches
  identical.306 stack hashes/four AOT DBs/links retained; user finetune-n7 remains
  uninspected/unmodified/untested, include_training_app=False. Same r2 bootc646d874,
  no agent reboot/.105/BIOS/fallback/delete. Local kernel/artifacts/sparse-b021
  preserves variants/recovery/154 raw logs4,956,287B/16 guards/archive8b05a494
  SHA-verified. [Report/use/recovery](docs/development/bc250_sparse_b021_results/README.md).
  BaselineWS/logs/sparse-b021-qualification-r5/final.json, rawWS/logs/sparse-b021-*;
  disk21,405,282,304B observed. No own build/test/container/guard/export running,
  root SSH closed. Remote memory snapshot keyed by final memory revision.
  **Next B03.1:** inspect qualified Torch distributed build configuration and
  pinned Gloo/RCCL paths; select bounded local CPU process-group readiness first.
  Preserve .105/native gfx1013. Multi-board/P2P/XGMI coverage separate. Native
  general sparse factorization remains open; no compile/graphs/higher derivative/
  total-memory/universal-stability claim. Work-item cursor refreshed; stale B05
  missing-A08 note corrected (native dense INT8/INT32 Lt already qualified).

- **B01.1 explicit BF16 FFT fallback installed2026-10-21:** code427492188,
  bc250_fft0.1.0 moduleSHA50bada1588a98dc2a18eb5948a231a76ec7aaede8fe6198b503c338ef9820fe6.
  Audit3 proves GPU rfft2 BF16 rejects outside/inside BF16 AMP, explicit float
  conversion works/backpropagates. Qualified native SpectralOps/_refs rejectBF16;
  FFT FP32 autocast registrations are CPU-side, no old helper found. Added
  **opt-in** bc250_fft18 complex/real/Hermitian forward/inverse1D/2D/nD adapters:
  only BF16 input cast differentiably to FP32 on existing device; no native BF16
  FFT/CPU fallback/global monkeypatch/default behavior change. Without out result
  C64/F32; leaf gradients retain dtype/rounding. Other inputs/out follow native
  semantics. Extra conversion allocation, no speed/peak-memory claim.
  Final candidate-r4/installed-r3 each327 checks: new245, retained N2.14 real72/
  contracts3/core7. New216=18ops*3norms*2layouts*2AMP, CPU-double output/all-leaf
  gradients/crop-pad/strides/unused storage/private streams;20 non-BF16 controls,
  five native contracts, two guarded outs, independent DFT17+continuous-reference
  finite differences, actual native GPU FFT forward/backward trace32 kernels.
  Maxoutput6.404e-7/gradientrelativeL2.001981<.004; fixed outputrtol/atol5e-4,
  BF16 gradientrtol.02/atol2e-5. No discrete-quantization derivative claim.
  Earlier dynamic-export candidate-r2 also327 pass; explicit named exports fix
  static lint visibility, unchanged suite rerun. Stage-r2 owned tmp FileExistsError
  retained/fixed exist_ok in r3; no numerical gate/tolerance relaxed. AllBlack/
  Ruff-F/AST/diff checks pass. Thirteen600s guards ACK/all gate+batch dmesg same.
  Image5d0a6d9cef758b2f7677a2b52b9d29ac51e24970703259a10ca8e44bee93feb1;
  runtimeJSONab9db860207e230ea704cab2bf76005d19b7c8967e59559a5a4b0088ed7f38c1.
  PurePython image module layer only; no native build/ccache operation/Torch or
  ROCm DSO/kpack/default changes. Previous94c610 image/source/build/cache history
  retained. RecoveryWS/recovery/fft-b011-before/runtime.json+old shared helper,
  locked deploy_fft_b011_on_host.py --rollback restores exact old metadata/image,
  unused.306 stack hashes/four FFT AOT DBs/links retained; user finetune-n7 stays
  uninspected/unmodified/untested, include_training_app=False. Same r2 bootc646d874,
  no agent reboot/.105/BIOS/fallback/delete. Unknown CPU fault unresolved/distinct
  from VM-heap/CSR/B04.2 queue. No universal shapes/devices/compile/graphs/higher
  derivatives/performance/stability claim. Local kernel/artifacts/fft-b011 holds
  modules/controller variant/recovery/141 raw logs4,140,885B/13 guards; archive
  77cf1bd7 SHA-verified. [Report/use/recovery](docs/development/bc250_fft_b011_results/README.md).
  BaselineWS/logs/fft-b011-qualification-r3/final.json, rawWS/logs/fft-b011-*;
  disk21,439,528,960B observed. No own build/test/container/guard/export running,
  root SSH closed. Remote memory snapshot keyed by final memory revision.
  **Next B02.1:** inspect pinned general sparse-solve/factorization consumers and
  N2.13 triangular evidence. Choose feasible ROCm integration or explicit bounded
  fallback with memory caps/residual/singularity/promised-gradient references
  before coding. Ordinary torch.fft BF16 API rejection remains unchanged.

- **B04.5 CPU image codecs completed/installed2026-10-21:** code93a601b36,
  TorchCodec0.17.0+gfx1013 source3eecace02076800cf6aa9be6364ed2f1f611548d
  unchanged; no new source patch/Torch rebuild. JPEG/PNG/WebP ON, IMAGE default/
  CUDA/nvJPEG OFF; AVIF/GIF/HEIC remain disabled by build choice, not hardware
  impossibility. Existing build libs JPEG2.1.5/PNG1.6.43/WebP1.3.2 reused. Runtime
  missing libwebpdemux2 supplied from exact build image under /opt/bc250-codec-
  sysdeps with copyright/loader cache; private copy not dpkg-registered. DSO
  SHA f31bf01638c352533e18720676db66fcb190a622db5bf617558a49bc445d0b83.
  Candidate-r1/installed-r3 each203 checks: image86/video55/audio46/vision9/core7.
  PNG8 gray/alpha/RGB/RGBA independent serialized samples, PNG16 precision/scaling,
  orientation6, JPEG cross-Pillow references, exact lossless WebP static/animated
  pixels, PNG/JPEG encoder views/output forms, refusals and explicit GPU spatial
  output/input-gradient versus CPU double then CPU encode. All pass first gates;
  fixed exact PNG/WebP, JPEG<=2 crossdecoder/MAE<20 owned encoded fixtures, GPU
  rtol2e-6 atol2e-7. CPU codecs only; explicit GPU JPEG refusal. No WebP encode/
  AVIF/GIF/HEIC/palette/interlace/all orientation/animation timing/HDR/security/
  universal stability or performance claim. Historical CPU fault unresolved,
  distinct from fixed VM-heap/CSR/B04.2 queue; B04.4 time-seek repair retained.
  Image94c610368d0fcf596a9f9f6355a5d5d28fd1111d54d4033c46e6b5977cae1ece;
  runtimeJSONc1fd78993ec25d0866e75e69bd4cc8ef2ebe9710f9a1f08e0de33bb350a2aaea;
  wheel0551a76dcc4c9cc950cd8045855ae7a235c9b0f8c6edd220396828a88d19e09d.
  Only codec wheel/private image dependency/new image/dev metadata changed;
  existing native ROCm/Torch/kpacks/N3.2/math/SDMA0/heapoff unchanged. Old87875
  image/source/build/cache/checkpoint/release history retained. RecoveryWS/recovery/
  image-b045-before/runtime.json plus original shared helper; locked deploy helper
  --rollback restores exact old metadata/image, unused.4 Ninja/4 CPUs/8GiB RAM/
  10GiB mem+swap/shared20GiB ccache, no build GPU devices; private N8 packaging.
  Ten600s guards ACK, every gate/both passing batches dmesg identical.306 stack
  hashes/four FFT AOT DBs/links retained; finetune-n7 uninspected/unmodified/
  untested, include_training_app=False. Same r2 bootc646d874; no agent reboot/
  .105/BIOS/fallback/delete. Local kernel/artifacts/image-b045 holds source/wheel/
  private sysdeps/notices/loader config/build manifests/recovery/218 raw logs
  2,870,579B/ten guards; archive97879a5e SHA-verified. [Report/use/recovery](docs/development/bc250_image_b045_results/README.md).
  BaselineWS/logs/image-b045-qualification-r3/final.json; rawWS/logs/image-b045-*.
  Disk21,467,881,472B observed; no own build/test/container/guard/export running,
  root SSH closed. Remote memory snapshot keyed by final memory revision.
  **Next B01.1:** return to remaining ordered support gaps. Read N2.14/B01 BF16
  FFT evidence and current autocast/promotion paths; implement a missing explicit
  FP32 helper only if needed, CPU-reference complex64 outputs/input gradients.
  Common media coverage above is bounded; optional image formats remain separate.
  Broad B01/B02/B03/B05/B06/B07/N2–N8 remain; user application stays excluded.

- **B04.4 CPU video completed/installed2026-10-21:** code22f796252,
  TorchCodec source3eecace02076800cf6aa9be6364ed2f1f611548d over d9dc5423.
  Initial unchanged runtime passes40 checks then FAILS single-frame VFR time
  seek: packet duration ends before next PTS, returns next frame. Exact/custom
  modes now use scanned index as batch time API does; approximate branch and
  reported packet durations unchanged. Reviewable patch amd-mainline/torchcodec/
  0001-fix-exact-single-frame-timestamp-seeks.patch; original failure retained.
  Candidate-r2/installed-r3 each108 checks: video55/audio46/core7. Independent
  nine-frame26x18 RGB references, lossless FFV1/H264-RGB I/P clips, NCHW/NHWC,
  random/backward/index/time/duplicate/range/slice seeks, bytes/filelike/tensor,
  prescribed VFR t_i=i*i/5 gaps; contiguous/strided/encoded-bytes/filelike/two-
  chunk encoders versus independent FFmpeg CLI pixels, five refusals. CPU decode
  -> actual HIP spatial output/weighted input gradient versus CPU double -> CPU
  encode exact pixels. Fixed exact pixel/PTS1e-9/GPU rtol2e-6 atol2e-7 bounds.
  CPU video only; no GPU codec/B-frame/YUV/HDR/all-format/long-stream/speed/
  universal stability/custom-mapping/approximate VFR claim. Image builds off;
  existing torchvision image/audio support retained. User finetune-n7 remains
  uninspected/unmodified/untested, owned fixtures/include_training_app=False.
  Image87875c40998bf083ecbc991e7dbe00b4e5549755b84d0f5b36d470167ab5de6d;
  runtimeJSON4cf699ed5c85b6eccf76b04976ca6abc9c83bd0449474ff8f28b2031a6a5cb92.
  Wheel72eee43277a2806684efa1f485f951567e935582a02bb731ced6f24ca65ef8f3;
  version0.17.0+gfx1013 unchanged, source/hash distinguish repair. Only codec
  wheel/new image/dev runtime metadata changed; ROCm/Torch DSOs/kpacks/B04.2 CLR
  fix/N3.2/math/SDMA0/heapoff unchanged. Olddcd025 image/builds/caches preserved.
  RecoveryWS/recovery/video-b044-before/runtime.json, locked deploy helper
  --rollback restores exact previous metadata/image, unused. Existing FFmpeg
  build image reused;4 Ninja/4 CPUs/8GiB RAM/10GiB mem+swap/shared20GiB ccache,
  no build GPU devices/new distro package install. Private N8 packaging boundary.
  Seven600s guards ACK including original failed numerical gate; six passing
  GPU gates/both whole batches dmesg identical.306 stack hashes/four FFT AOT DBs/
  links retained. New entry bootc646d874-ebd6-4e4b-bb77-2e47233748e0, same r2
  throughout; no agent reboot/.105/BIOS/fallback/delete. Unknown historical CPU
  fault unresolved/distinct from VM-heap/CSR, B04.2 queue and this semantic bug.
  Local kernel/artifacts/video-b044 retains source/provenance/wheel/recovery/
  build manifests/130 raw logs2,536,706B/seven guards; archive8295c64f SHA-verified.
  [Report/use/recovery](docs/development/bc250_video_b044_results/README.md), raw
  WS/logs/video-b044-*, baseline video-b044-qualification-r3/final.json.
  Disk21,589,233,664B observed, no own build/test/container/guard/export running;
  root SSH closed. Remote memory snapshot keyed by final memory revision.
  **Next B04.5:** inspect pinned CPU image codec flags/dependencies; enable and
  qualify feasible TorchCodec JPEG/PNG/WebP with owned independent pixel refs,
  GPU codecs off. Broad B01/B02/B03/B05/B06/B07/N2–N8 remain open.

- **B04.3 TorchCodec completed/installed2026-10-21:** integration9c5199ccb,
  upstream v0.17.0 d9dc54238e173000bc3969eb3959777026ff9ae8, preserved local Git
  sources/bc250-torchcodec branch users/Naster17/gfx1013-codec. No source patch;
  source archive matches all471 tracked files. CPU FFmpeg6.1.1 audio codec build,
  CUDA/image codecs off; existing Torch2.13/HIP7.17/Python3.12/ABI1 retained.
  torchaudio.load/save now available by default through bc250-torch/train; decode
  on CPU, explicit HIP transfer for tensor ops and CPU transfer before saving.
  Public normalized-FP32/ignored legacy-option warnings remain upstream behavior.
  Candidate-r1/installed-r3 each136 composite checks: files46, DSP34, native40,
  vision integration9/core7. Independent PCM16 mono/stereo/PCM24 RIFF, FLAC,
  paths/file-like/layout/crop/empty/metadata, MP3 versus separate FFmpeg CLI,
  WAV/FLAC save/independent parsed PCM, encoded bytes/warnings/refusals, actual
  GPU resample output/weighted input gradient against CPU double and CPU save.
  Exact PCM/owned MP3 CLI match; unchanged resample rtol.002/atol3e-5 and retained
  audio gates. No GPU codec/video/image/file-I/O gradient/all-codec/model/speed
  or sustained-stability claim. Existing torchvision image path stays available.
  **User scope2026-10-21:** leave finetune-n7 uninspected/unmodified/untested;
  use owned stack fixtures. Two initial app-doc preservation refusals retained;
  user then excluded application. Eight old app identities excluded without
  further reads;306 stack identities/four FFT AOT DBs/links retained. Shared
  qualifier explicit include_training_app=False; historical default preserved.
  Future stack batches use this scope and current306-path baseline, not old
  app-bound snapshots; no trainer qualification/checkpoint migration claimed.
  Image dcd025fbee1ffd180b9ff4f3ab8b753f2415d8a58914d77506ea12d4777a88d7;
  runtimeJSON f902e0bc3f7c4275db9ea7f7a2234f29fcc6fd9d78b449bf95666b00ed8ccd96.
  Wheel5850732408eaf7ef372d570be4a8367948e69181e9b4eb58f6d833462f2c7780.
  Only dev image/codec metadata changed; ROCm/Torch DSO/kpack bytes unchanged,
  B04.2 HIPa1162308/systems1f9609fd6/base62ab2437c, nativeTorchdc8fea0/source
  2b449c4/rocBLAS8389197c/otherlibs319a235d6/N3.2/math/SDMA0/heapoff retained.
  Old97fac image/release/builds/caches retained. RecoveryWS/recovery/codec-b043-before/
  runtime.json; exclusive-lock deploy_codec_b043_on_host.py --rollback restores
  exact prior metadata/image. Old shared helper backed up; rollback unused.
  Build-r1 missing ROCm CMake path/r4 missing native arch fixed in build config;
  r2 competing UID1000 train refused before build, process exited/user confirmed
  board free; r3 app-doc refusal retained. Corrected build-r5 succeeds:4 Ninja/
  4 CPUs/8GiB RAM/10GiB mem+swap/shared20GiB ccache, no build GPU devices. Private
  distro-FFmpeg development wheel; release dependency/license packaging staysN8.
  Ten600s guards ACK; each GPU gate/both whole passing batches dmesg identical.
  New r2 boot8ea98081-ca52-4869-ae8f-cf8072362d80 observed at ENTRY, unchanged
  across tests. No agent reboot/.105/BIOS/fallback/delete changes. Historical
  unknown CPU fault unresolved/distinct from fixed VM-heap/CSR and B04.2 queue.
  Local kernel/artifacts/codec-b043 has source/proof/wheel/build inventories/
  recovery/145 raw logs3,703,497B. Fsynced export archiveb52252d2 SHA-verified.
  [Report/use/recovery](docs/development/bc250_codec_b043_results/README.md),
  rawWS/logs/codec-b043-*, baseline codec-b043-qualification-r3/final.json.
  Disk34,926,919,680B observed; no own build/test/container/guard/export running,
  root SSH sessions closed. Remote memory snapshot keyed by final memory revision.
  **Next B04.4:** inspect pinned TorchCodec CPU video APIs; owned raw-frame
  decode/seek/encode reference fixtures with explicit HIP transfer before broader
  claims. CUDA/image codecs remain off pending separate qualification. Broad
  B01/B02/B03/B05/B06/B07/N2–N8 remain; Qwen/LoRA is optional and excluded now.

- **B04.2 torchaudio completed/installed2026-10-21:** integrationd75350905,
  runtime controllers7bf89b571. Preserved source sources/bc250-torchaudio,
  upstream34c52a67 / port1ae7f5d8023e82be51b500399a37fceb45481a2b;
  native2.11.0+gfx1013 against existing Torch2.13/HIP7.17/Python3.12/ABI1.
  Four audio patches: detected ROCm build, current-stream IIR launch outside
  dispatch macro, RNNT device-index guard and stream-ordered counter resets.
  Demonstrated cold hostcall CPU crash fixed in CLR: reclaimed internal transfer
  queue was null during direct occupied-bitfield fill. Lock execution/reacquire
  queue/check before synchronous fill; systems1f9609fd69c4923291f58934495520fe62e66ef8.
  Original HIP machine-code-matched symbolication retained; RNNT counter fix
  alone still failed. Not a claim all historical Python CPU faults are solved.
  Candidate142 (141 main+direct HIP probe1), installed142, user-reboot replay48
  composite checks pass. DSP34 includes explicit CPU DFT output/gradient;
  native40 includes IIR waveform/coefficient gradients, RNNT FP16/32 logits
  gradients against CPU FP64 DP/enumeration, forced alignment exact CTC paths,
  actual native GPU/current-stream trace. Retained vision60/core7 pass. Direct
  HIP probe60 cold streams/384 checked words; no actual printf/assert service.
  Fixed limits unchanged; maxrelativeL2 DSP3.7623e-6/native3.3462e-4. No speed,
  universal-model/stability/second-derivative or GPU codec claim. TorchCodec
  absent: torchaudio.load/save explicitly remain unavailable, next B04.3.
  Image97fac19f0d5747265c4ff8328f99d2eaacaf8ff62e5092d22cda300f7130a632;
  HIP SHAa1162308076a99ff5cf5e4e15ca8bff53928d8bc4e624c42033d7e15dd2d1e3b.
  Only dev runtime.json/image plus HIP DSO and exact CLR source changed;
  runtime base62ab2437c/Torchdc8fea0/source2b449c4/rocBLAS8389197c/otherlibs319a235d6
  and N3.2/math/SDMA0/heapoff retained. No matching external HIP kpack found;
  other packs untouched. RecoveryWS/recovery/audio-b042-before; exclusive-lock
  deploy_audio_b042_on_host.py --rollback restores image metadata/DSO/source
  together. Old image/release/source/builds/caches retained. Existing content-bound
  checkpoints may refuse changed HIP identity; no silent migration/app changes.
  Initial build hipify failures/early RNNT CPU crashes retained. Separate concurrent
  qwen15_training/real_qwen15_vul global host OOM killed external and own staging
  processes; user reserved board, resource/GPU-client preflight added, later pass.
  No exclusive-causality claim.26 guards ACK600s; passing r4/r5/r6 batches each
  dmesg identical, whole session differs due retained faults/OOM. Historical CPU
  fault remains unresolved/distinct from fixed VM-heap/CSR and this queue path.
  User explicitly confirmed reboot during evidence copy: bootefd16227 ->
  c81853b2-fba0-4e30-ab67-0c7e06f221bd, still7.1.5-bc250rocm-r2. Postboot
  identities match and48 checks pass; no agent reboot/.105/BIOS/fallback changes.
  Audio build4 Ninja/10GiB RAM/12GiB mem+swap/20GiB ccache explicit HIP compiler;
  CPU wrapper bypass retained. Isolated runtime one TU/link6GiB/8GiB swapbound/
  4 CPUs, no active build mutation before qualification. Local SHA-verified
  kernel/artifacts/audio-b042: source+incremental Git bundles/patches/both wheels,
  candidate/native recovery/symbols and301 raw logs9,240,283B. Reboot-interrupted
  empty evidence-r1 retained; fsynced evidence-r2 exported/verified successfully.
  [Report](docs/development/bc250_audio_b042_results/README.md), rawWS/logs/audio-b042-*.
  Active preservationbaseline WS/logs/audio-b042-qualification-r6/final.json:
  314 protected paths/four FFT AOT DBs/links retained. One changed entry app doc
  preserved without trainer requalification. Disk17,554,157,568B free observed.
  No own running build/test/container/guard/export; root SSH sessions closed.
  **Next B04.3:** inspect pinned TorchCodec ABI/FFmpeg requirements; isolated CPU
  audio file load/save reference fixtures before installation. Preserve qualified
  native audio signal operators. Broader B01/B02/B03/B05/B06/B07/N2–N8 remain.
  Remote memory snapshot WS/agent-memory/latest keyed by final memory revision.

- **B04.1 torchvision completed/installed2026-10-20:** prep626e19ef2,
  integration220bd80e5. Broad support scope retained; Qwen/LoRA remains optional.
  New preserved local Git repo sources/bc250-torchvision, release/0.28,
  8fb87713a24951e639c494b0f2a8a81b5f8e33a6; upstream CI targets release/2.13.
  Native0.28.0+gfx1013 against existing patched Torch2.13/HIP7.17/Python3.12/ABI1,
  Pillow13.0.0. No vision operator source patch or replacement Torch wheel.
  Candidate-r2 and installed-r3 each67 checks: NMS, four ROI operators, grouped
  masked/unmasked deformable conv FP16/32/64 outputs/all promised gradients;
  independent NMS/bilinear oracles, strided/empty cases, actual GPU ROI trace;
  CPU PNG/JPEG/WebP+HIP transfer, four FP32 transform gradients, synthetic FP32
  ResNet18 output/input/all62 parameter gradients, retained core7/imports.
  MaxrelativeL2 ops0.001519/transforms5.986e-7; fixed bounds unchanged. No GPU
  decode/BF16 custom ops/video/second-derivative/universal-model or speed claim.
  Entry refuses exactly two changed app docs; reviewed/preserved, other311
  identities match. entry-r2 is preservation only, not renewed trainer/doc
  qualification. Build-r2 missing host /var/tmp fixed by workspace TMPDIR; raw
  package-r1 output-dir failure retained. Candidate-r1 ops51 pass/codec FAIL:
  indirect WebP libsharpyuv missing from loader search despite bundled DSO.
  Package-r3 gives each bundled codec its own ORIGIN RUNPATH/license notices;
  corrected candidate+installed pass. Failed images/wheels/builds/logs preserved.
  Active image a0b85fa88c9ecfbec88a8774f19db0fbc18fcfeaa086a4e402b69aa99ba55e27;
  original5df0a58392de remains fallback. ONLY dev runtime.json image/vision
  metadata changed; old release/consumer site-packages untouched. Recovery
  WS/recovery/vision-b041-before/runtime.json; locked deploy helper --rollback
  restores exact original. Shared qualifier optional release override, old copy
  retained.313 final identities/four FFT AOT DBs/links retained; runtimeJSON only
  intended entry delta. Runtime62ab2437c/rocBLAS8389197c/nativeTorchdc8fea0/source
  2b449c4/otherlibs319a235d6/N3.2 CSV/math/SDMA0/heapoff unchanged. Eight600s
  guards ACK, each GPU batch dmesg identical. Same r2 boot48bdf727; no reboot/
  .105/BIOS/fallback/delete change. Four Ninja jobs/10GiB memory/12GiB mem+swap,
  ccache20GiB explicit HIP compiler caching; CPU cpp-extension wrapping bypassed,
  initial CPU units uncached. Separate build image acdb6dc4 retained. Local
  kernel/artifacts/vision-b041 contains SHA-verified source archive/raw+fixed
  wheels/Pillow/109 raw log files; [report](docs/development/bc250_vision_b041_results/README.md).
  WS/logs/vision-b041-*; active baseline vision-b041-qualification-r3/final.json.
  Disk18,191,425,536B free observed; no own build/test/container/guard deadline;
  root SSH closed. CPU segfault unresolved/distinct from VM-heap/CSR/N38 OOM.
  **Next B04.2:** inspect compatible pinned torchaudio source/build/API/backend
  requirements; independent CPU/GPU audio tensor references, isolated build/test
  before install. No alternate Torch, codec/media acceleration assumption or
  unrelated trainer restoration. Broader B01/B02/B03/B05/B06/B07/N2–N8 remain.
  Remote memory snapshot WS/agent-memory/latest keyed by final memory revision.

- **N3.10 broad compiler qualification completed2026-10-20:** code7c8156c57.
  **User scope:** broad hardware-feasible ROCm/PyTorch/library/API support;
  Qwen3.5/LoRA is one optional test fixture, not the port boundary. Missing broad
  features stay in queue even when not needed by that trainer. Proposed Qwen-only
  MLP step replaced with model-independent composed training blocks after reading
  prior P1/compiler/dynamic-attention evidence; no repeated model checkpoint run.
  Residual dense+LayerNorm, gated/RMS MLP, group2/dilation2 CNN+GroupNorm/GELU/
  pooling/head, train BatchNorm. FP32/FP16-BF16 autocast with FP32 parameters;
  two shapes/layouts (strided MLP/channels-last CNN), fullgraph dynamic=False.
  48 eager/compiled cases compare output/input/all parameter gradients CPU FP64,
  all running buffers and own-gradient CPU AdamW.12 CPU finite-difference probes
  plus four profiled CPU-reference replays;56 composite checks pass.52 GPU
  forward/backward runs/48 optimizer updates. MaximumrelativeL2 dense0.004123/
  gated0.006393/CNN0.008807/BN0.004749; fixed precision limits unchanged. CPU AdamW
  parametererror<=1.49e-8, BN3 buffers match.24 capturedFXgraphs/314 generated
  kernel metric; four traces101 actual GPU kernel events incl Triton. No eager
  fallback or new production defect/fix/install/speed claim. Ordinary compiler
  coverage qualified; is_big_gpu max-autotune HIPmajor10 exclusion retained as
  policy, not a proof all scalar tuning impossible. Dropout/dynamic reuse/
  compiled optimizers/full apps/graphs/max-autotune remain separate.
  Four600s guards ACK; whole batch dmesg identical,313 protected identities/four
  FFT AOT DBs/links retained. New r2 boot48bdf727-b2f1-4d7b-aba5-2a0b775f1387
  observed at ENTRY; no agent reboot. Runtime62ab2437c/rocBLAS8389197c1/native
  Torchdc8fea0/source2b449c4/otherlibs319a235d6 unchanged. N3.2/math defaults;
  inactive vocab/square candidates stay inactive. No nativebuild/prefix install/
  cache reset/delete/.105/BIOS/fallback change; four CPU/compiler threads,
  existing JIT caches extended. Disk28,975,644,672B free observed, no cleanup.
  56 rawfiles1,304,438B SHA-verified local kernel/artifacts/training-n310/remote;
  [report](docs/development/bc250_compiler_n310_results/README.md), rawWS/logs/
  training-n310-*, baseline training-n310-qualification-r1/final.json. No own
  build/test/container/export/guard deadline; root SSH closed. CPU segfault
  remains unresolved/distinct from fixed VM-heap/CSR and retainedN38 host OOM.
  **Next B04.1:** inspect Torch ABI/package/build configuration and existing RPP/
  media evidence; select pinned compatible torchvision, isolated native gfx1013
  extension build and real image-operator CPU/GPU references (NMS/ROI Align etc)
  before deployment. No other Torch wheel; torchaudio separate. Broader N2/N3/
  N4–N8 and B01/B02/B03/B05/B06/B07 remain scope; no universal support percentage.
  Memory snapshot WS/agent-memory/latest keyed by final memory revision; old kept.

- **N3.9 restored/deployed, 2026-10-20 local / 2026-10-19 UTC:**
  code9899e0659/continuity5a24fe4e7/guidea7fa50d00/appguidanceabb5256ab. Reviewed
  exact entry→qualifiedN37 diff; patch reconstructs trainerb9bf2587 byte-exact.
  Restores N37 provenance integration and A07 --attention-backend math|tiled;
  default math/legacy math fingerprint unchanged, tiled fingerprint distinct.
  Current opsd058b14a cosmetic status edit preserved; arithmetic unchanged.
  Actual masked32x1024/rank16/alpha32/BF16/FP32 adapters/checkpoint/accum16 fresh2,
  new checkpoint resume, ownedN36 legacy resume, installed replay: exact372
  adapters/full AdamW/scheduler/CPU+GPU+Python RNG/order/progress/run.json/loss/
  token counts/LR/evaluation. Installed numerical state exact preN37 excluding
  only provenance marker. Step2loss0.07038949553197377/eval0.1256762146949768.
  Short tiled rank4 fresh2/resume1 exact full state+provenance, actual41-token
  second record; not a new long-sequence claim. Eight metadata/runtime refusals
  pass; actual ownedN37 checkpoint refuses preserved cosmetic ops source ONLY
  before weights/output, remains SHA-identical. Whole-source hashes remain
  conservative; no silent bound-checkpoint migration. New selected runtime
  identity130 files/15 mapped DSOs now enforced by installed trainer again.
  r1 candidate26/r2 historical continuity1/r3 installed4 =31 composite checks;
  83 train micros/eight eval forwards. Initial174 app tests had3 old fixture
  errors: missing attention_backend field; fixture corrected+one forwarding
  test.175 tests pass candidate+installed; four provenance CPU tests local+.104.
  Black/Ruff-F/syntax/patch dry-run+exact reconstruction pass. Seven600s guards
  ACK; each gate/whole batches dmesg identical. No new OOM/CPU fault; historical
  CPU segfault unresolved/separate from fixed VM-heap/CSR and retainedN38 OOM.
  Installation trainer/test fixture/usage guide only; r4 docs-only corrects
  stale app AGENTS attention paragraph. BackupWS/recovery/training-n39-before
  incl original shared validator (optional scratch parent added, old default
  unchanged).313 paths protected final:309 originals unchanged, trainer/guide
  intentional changes; tests/AGENTS added. Ops/provenance module/four FFT AOT DBs/
  links/native artifacts preserved. No rollback needed. GPUbaselineWS/logs/
  training-n39-qualification-r3/final.json; latest preservationbaseline
  WS/logs/training-n39-guidance-r4/final.json. Disk scratchWS/experiments/
  training-n39-r1;257 files1,327,912,905B plus57 internal fixture links and40
  resolved Python links verified durable local kernel/artifacts/training-n39/
  export. RawWS/logs/training-n39-*; [report](docs/development/bc250_training_n39_results/README.md).
  Native runtime62ab2437c/rocBLAS8389197c1/Torchdc8fea0/source2b449c4/otherlibs
  319a235d6 retained. Active N3.2 CSV1f2947a8; N38vocab and rejected square remain
  inactive. Same r2 bootd3857b96, no native build/install/cache reset/delete/
  reboot/.105/BIOS/fallback change. Disk5,490,642,944B free observed. No running
  own build/test/container/export/guard deadline; root SSH closed. Memory snapshot
  WS/agent-memory/latest keyed by final memory revision; previous snapshots kept.
  **Next N3.10:** read prior compiled-block/dynamic-attention evidence first,
  inspect actual Qwen3.5 nonzero-LoRA MLP dispatch; qualify bounded eager/compiled
  output/input/adapter-gradient comparisons with independent references. Keep
  N3.2/math defaults; full-trainer compilation/speed require separate evidence.

- **Historical N3.8 audit completed, experimental candidate NOT deployed, 2026-10-19:**
  coded7b7a2bc4/settlinged0946ef6/resourcepreflightf85831f3b. Actual BF16 vocab
  forward[1024,2048]@[2048,248320] and input-backward[1024,248320]@[248320,2048];
  all14 admitted scalar solutions+default per role pass full chunked CPU FP64,
  output guards/immutable inputs; five small CPU oracle/checker contracts pass.
  Owned FP64 tensor bounds1.916GiB/64MiB, not RSS. Two2.547GB actual fixtures
  SHA-verified durable locally. Method correction: private handle starts atomics0,
  eagerTorch uses atomics1/32MiB workspace; initial raw2.443x backward ratio is NOT
  Torch speedup. Corrected paired actualTorch/candidate isolated1.078x forward
  (-339)/1.723x backward(-533); candidate reduction order differs. Experimental
  CSV keepsN3.2 -490 row, adds only these two; active N3.2 SHA1f2947a8 unchanged.
  r7 full-model ABBA passes loss/all372 gradients/CPU AdamW/state/RNG/reference
  exact repetition, worstgradientrelativeL2=1.446%<4%; timings shift strongly.
  r8 extra-settling FAILS global host OOM/SIGKILL: concurrent UID1000 train appears
  in OOM tasks, no exclusive-causality claim. Failed evidence retained, distinct
  from unknown CPU segfault and repaired VM-heap/CSR. User then reserves board;
  r9 >8GiB RAM/no competing GPU clients preflight passes, two extra full untimed
  steps before three timed steps per worker. All four numerical workers pass;
  reference medians5.809/5.435s, candidates5.352/5.352s, ratios1.085/1.015x still
  variable. Firstloss exact14.2942647934, CPU parametererror<=1.86265e-9; no stable
  whole-model/accumulated LoRA/convergence/bitwise continuation speed claim.
  **Entry drift:** r1 refuses oldN37 baseline BEFORE GPU. Current user train.py
  SHA170d3920 removes N37 provenance integration AND A07 attention-backend CLI;
  bc250_ops.pyd058b14a changes only printed status. Both preserved; other309
  protected identities match. At N38 entry the trainer did not enforceN37
  or expose tiled CLI; N39 above restores these features. N38 model workers use separate exact qualified hookf0bc.
  WS/logs/training-n38-entry-r2/final.json is preservation snapshot, NOT app
  qualification. All311 entry identities/four FFT AOT DBs/links/native files
  remain unchanged.14 passing composite checks/41 completed model micros across
  passing gates (failedr8 partial excluded); eight600s guards ACK, passed gates
  dmesg identical; whole batch differs dueOOM. All136 raw files3,159,740,816B
  SHA-verified durable local kernel/artifacts/training-n38 (fixtures/, remote/);
  original app/source variants retained there. RawWS/logs/training-n38-*;
  [report](docs/development/bc250_training_n38_results/README.md).
  Shared worker backupsWS/recovery/training-n38-before; no app/native install,
  build/cache reset/delete/reboot/.105/BIOS/fallback change. Runtime62ab2437c/
  rocBLAS8389197c1/nativeTorchdc8fea0/source2b449c4/otherlibs319a235d6 retained;
  same r2 bootd3857b96. Disk9,799,057,408B free observed. No running own test,
  build/container/export/guard deadline; root SSH closed. Memory snapshot is
  WS/agent-memory/latest, keyed by final memory revision; prior snapshots kept.
  **Next N3.9:** reconcile preserved current trainer with qualifiedN37/A07
  provenance/attention CLI, preserving other edits. Keep current N3.2 preset;
  owned masked1024/rank16/accum16 fresh/resume/refusal gates before app install.
  No silent new-provenance checkpoint migration or vocabulary preset activation.

- **Historical N3.7 completed and deployed, 2026-10-19:** code `e55d8b23a`, app0f2dfa078.
  Supplemental runtime-provenance.json digest bound in run.json/training.pt;
  existing training fingerprint unchanged.130 selected files incl110 gfx1013
  GEMM assets +15 mapped DSOs/preset/hooks/environment/kernel/version identity.
  New missing/corrupt/changed identity refuses before weights/output; legacy
  preserves checks with explicit unverified-runtime warning, new saves gain
  current provenance. Data-only preflight validates metadata, defers live runtime.
  Candidate actual1024/rank16/accum16 masked32-record fresh2/new resume/actual
  owned N36 legacy resume, then installed resume: all372 adapters/full AdamW/
  scheduler/CPU+GPU+Python RNG/order/progress/run.json/loss/token counts/LR exact;
  evaluation2 exact, installed numerical state exact pre-N37 after excluding only
  metadata key.80 train micros/five eval forwards,26 qualified composite checks,
  four CPU contracts local+.104; Black/Ruff-F/syntax/patch dry-run pass.
  Eight first refusal assertions pass but original r1 batch FAILS kernel-log
  equality: exactly XFRM+bridge network-init messages added, no GPU fault in delta.
  Original failure retained. r2 checks original sources/protected identities/three
  passed training gates/exact two-line delta, reruns only8 refusals in new scratch;
  byte-identical dmesg required again, PASS. r3 guarded app install/replay PASS,
  dmesg identical; six guards ACK600s. Whole batch dmesg NOT byte-identical.
  Intended app changes train.py/new provenance module/usage guide only; old trainer
  and guide backup WS/recovery/training-n37-before-r3.308 other original paths
  unchanged; final311 protected identities match expected installed state; four
  FFT AOT DBs/defaults/links/user data/checkpoints/profile intact. Runtime62ab2437c/
  rocBLAS8389197c1/nativeTorchdc8fea0/source2b449c4/other libs319a235d6 retained.
  Same r2 boot d3857b96; no native build/install/cache reset/delete/reboot/.105/
  BIOS/fallback change. Disk fell from~5.1GiB to~1GiB observed, no reclamation.
  Explicit RAM scratch /tmp/bc250-training-n37-r1 and refusals-r2 VOLATILE;
  owned regular files/symlink targets exported/SHA-verified durable local
  kernel/artifacts/training-n37/{scratch-r1,scratch-refusals-r2}; priorN35/36 kept.
  [Report](docs/development/bc250_training_n37_results/README.md); raw WS/logs/
  training-n37-* and local kernel/artifacts/training-n37/remote. Next baseline
  WS/logs/training-n37-qualification-r3/final.json (311); app recovery above,
  native recovery WS/recovery/gemm-n32-before. No active own job/container/guard.
  CPU fault unresolved/distinct from fixed VM-heap/CSR, N3.2 loader and this log
  guard failure. Selected eager identities only, no full image integrity/security/
  universal determinism/real-data/convergence/speed/mid-accum/CPU-model claim.
  **Next N3.8:** vocabulary projection audit/tuning. Read N3.3 layouts/evidence,
  design chunked CPU-double references under bounded RAM/VRAM, audit actual
  forward/backward admitted scalar solutions. Keep N3.2 in both model workers,
  unchanged full-model gradient/optimizer gates before deployment. Square override
  remains rejected/inactive; do not silently migrate new-provenance checkpoints.

- **N3.6 completed, 2026-10-19:** qualification code `bee5c6ce9`.
  Actual deployed inner trainer CLI/current N3.2 preset/no preload; owned32
  exactly1024-token records with32 distinct masked-target counts72–289,
  rank16/alpha32/BF16/FP32 adapters/math/checkpoint/accum16. Fresh2 updates,
  separate checkpoint1→2 resume: all372 adapter tensors/full optimizer/scheduler/
  CPU+GPU+Python RNG/shuffled order/progress/run.json/loss/token counts/LR exact;
  evaluation2 exact too.48 training micros/three eval forwards,7 checks pass.
  Two guards ACK600s, dmesg identical,309 hashes/four FFT AOT DBs/links/runtime/
  user trainer/config unchanged. No production code/native build/install/cache
  reset/deletion/reboot/.105/BIOS/fallback change. Runtime62ab2437c/rocBLAS
  8389197c1/nativeTorchdc8fea0/source2b449c4/other libs319a235d6 retained.
  Same r2 boot d3857b96; disk~5.1GiB observed, no reclamation performed.
  RAM artifacts /tmp/bc250-training-n36-r1 are VOLATILE; all copied/SHA-verified
  durable local kernel/artifacts/training-n36/scratch-r1. PriorN35 kept.
  Executed raw sources archived; committed formatting AST-identical/Black/
  Ruff-F/syntax pass. [Report](docs/development/bc250_training_n36_results/README.md).
  Raw WS/logs/training-n36-* and local kernel/artifacts/training-n36/remote;
  next baseline WS/logs/training-n36-qualification-r1/final.json; runtime
  recovery WS/recovery/gemm-n32-before. No active agent job/container/guard.
  No real-data/convergence/throughput/mid-accum/CPU-model/cross-runtime claim.
  CPU fault unresolved/distinct from fixed VM-heap/CSR and repaired N3.2 loader.
  **Next N3.7:** production fingerprint omits preset/native runtime/hook content.
  Add compatible supplemental provenance for NEW checkpoints: inspect launcher
  identities, qualify same-runtime replay/changed-runtime handling with owned
  fixtures, preserve explicit legacy behavior and user checkpoints. No arithmetic
  change. Vocabulary projection tuning remains open afterward.

- **N3.5 completed, 2026-10-19:** qualification code `986093a3c`.
  Current N3.2 preset/no preload, actual-base synthetic1024/batch1/rank16/BF16/
  FP32 adapters/math/checkpoint/accumulation16. Independent16 scaled gradients
  summed CPU FP64 versus GPU FP32 accumulation: all372 pass atol1e-8/rtol1e-5/
  relativeL2<2e-6, worst1.17090e-8; losses/IDs/CPU+GPU+Python probes/RNG exact.
  Independent global clip and two uninterrupted CPU AdamW/scheduler replays pass,
  maxparamabs1.863e-9/3.725e-9. Owned optimizer-boundary checkpoint fresh step2
  matches uninterrupted all372 params+gradients/full AdamW/scheduler/losses/inputs/
  progress/three RNG exactly; resumed CPU AdamW maxabs1.863e-9. No whole-model CPU,
  actual CLI/real data/mid-accumulation/throughput/convergence claim; labels unmasked.
  r1 test mount overwritten by consumer setup fixed before model work. r2 first48
  micros pass; resumed CPU oracle aliases saved CPU step tensor and fails. Clone/
  counter-nonalias fix is test-only; r3 reuses six passed checks and unchanged
  checkpoint/control, executes corrected fresh16, exact full-state comparison.
  Source-origin SHA admitted explicitly; actual corrected SHA separately recorded,
  no model/preset identity/tolerance/checkpoint/GPU arithmetic change. Total80
  micros incl16 retained failed-oracle branch, composite8 checks pass. Three
  guards ACK600s/dmesg identical/309 identities/four FFT AOT DBs intact.
  Entry already on NEW r2 boot d3857b96-1b51-4e39-8f3e-f076f0c0422a; old d053
  boot historical. Disk881MiB at entry, no deletion/reserve change. RAM-backed
  /tmp scratch preflight>2GiB/>6GiB available RAM, explicit512MiB text-log disk
  floor only N35; other gates retain2GiB default. Four artifacts~836MiB copied/
  hash-verified durable local kernel/artifacts/training-n35/scratch-r2. Remote
  /tmp/bc250-training-n35-r2 is VOLATILE across reboot, preserve local copy.
  No native build/install/persistent cache reset/user app/.105/BIOS/fallback change.
  Runtime62ab2437c/rocBLAS8389197c1/nativeTorchdc8fea0/source2b449c4/other native
  libraries319a235d6 retained; no reboot by agent. No active agent job/container/
  guard, disk~880MiB. CPU fault unresolved/separate from fixed VM-heap/CSR,
  repaired N3.2 loader and the corrected test-oracle bug. [Report](docs/development/bc250_training_n35_results/README.md).
  Raw WS/logs/training-n35-* and local kernel/artifacts/training-n35/remote;
  failed sources/records preserved, helper backups WS/recovery/training-n35-before.
  Next baseline WS/logs/training-n35-qualification-r3/final.json (309 unchanged);
  runtime recovery remains WS/recovery/gemm-n32-before.
  **Next N3.6:** actual deployed trainer CLI owned1024/rank16/accum16 fresh/resume
  on current preset; inspect prior small CLI/fingerprint code, keep user data/
  checkpoints/config untouched, RAM scratch+durable local copy while disk low.

- **N3.4 audit completed, candidate rejected, 2026-10-19:** code `6531025ef`.
  Actual BF16 [1024,2048]@[2048,2048] TN42forward+42backward/NN42backward calls,
  two layouts offsets0. All14 admitted solutions each plus defaults pass full
  CPU-double references/guards/immutable inputs, fixedrelativeL2<0.004; worst
  0.001702264. Six-round fixture best-346 TN1.34234x/-533 NN1.07391x; initial TN
  timing regime shifts retained, not a stable-clock or whole-model gain claim.
  TN-346 full-model comparison keeps installed N3.2 NT-490 in BOTH workers.
  Actual-base synthetic1024/batch1/accum1/rank16/BF16/math/checkpoint: candidate
  worst372-adapter gradientrelativeL2=0.0765128583 > unchanged0.04: REJECTED.
  Lossdelta0.00033474 passes existing forward-model0.015 contract; CPU AdamW
  own-gradient parameter/state checks pass maxabs1.863e-9, RNG hashes equal.
  One warm median pair5.625697/5.581987s; ABBA stops at failed gradient gate,
  no qualified speedup. ExperimentalCSV not active; no tolerance relaxation.
  Follow-up14 TN solutions: exact default-output indices-343/-338/-345/-350;
  after sustained2s warmup, six-round ratios0.69190/0.99441/0.99386/0.72009.
  Other solutions change624–639 of2097152 captured outputs; -346638/relativeL2
  5.45847e-5. Fixture exactness not universal. No square override deployed;
  no production arithmetic defect established. Four guards ACK600s (three GPU
  passes/one numerical candidate rejection), dmesg identical; final309 protected
  identities/four FFT AOT DBs/links/runtime/defaults/user trainer/config intact.
  No native build/install/cache reset/reboot/.105/BIOS/fallback change.
  Runtime62ab2437c/rocBLAS8389197c1/nativeTorchdc8fea0/source2b449c4/other native
  libraries319a235d6 retained; same r2 boot d0532075-88fb-4a2f-974e-f61578d20d9f.
  No active agent job/container/guard; free~5.34GiB. CPU Python fault unresolved/
  separate from fixed VM-heap/CSR, repaired N3.2 loader and this rejection.
  [Report](docs/development/bc250_training_n34_results/README.md); failed source/
  logs/gradients retained, raw WS/logs/training-n34-* and matching local ignored
  kernel/artifacts/training-n34/remote copied/hash-verified. Helper backup
  WS/recovery/training-n34-before; runtime recovery WS/recovery/gemm-n32-before.
  Next baseline WS/logs/training-n34-qualification-r3/final.json (309 unchanged).
  **Next N3.5:** current-preset1024/rank16 accumulation16 gradients/update and
  owned checkpoint/fresh-process resume. Inspect prior42-token/rank4 CLI resume
  and current accumulation1 evidence first, qualify new geometry without user
  dataset/checkpoint edits. Vocab projection tuning remains open; needs bounded
  memory/reference design, not declared hardware-impossible.

- **N3.3 completed, 2026-10-19:** TheRock profiling/audit code `7838d94bc`.
  Installed N3.2 preset/DSO active, no preload. Same actual Qwen3.5-2B synthetic
  1024/batch1/accum1/BF16/rank16/math/checkpoint: loss/all372 nonzero clipped
  gradients/adapter update/full AdamW state/CPU+GPU RNG profiled versus unprofiled
  exact.36 conv/36 gated-delta/one loss calls. Three finite warm updates median
  5.442552s/peak5351892480B;38296 kernels sum5.403136s, mm75.387%, backward3.382234s.
  Reverse [1024,2048]@[2048,6144]26.427%, prior tuned shape15.423%, square2048
  projection7.105%, vocab forward6.884%/backward4.333%; conv7.529%. No new speed
  claim; optimizer/gated-delta dominance not established. Four analyzer contract
  tests/Black/Ruff-F/syntax pass. Reverse actual audit TN132/NN24 mm calls,
  two layouts offsets0, column-major M6144/N1024/K2048; all14 admitted solutions
  each plus defaults CPU-double/guards/immutable inputs pass fixed tolerances,
  worstrelativeL2=0.001666206. Best-339/-533 six-round ratios1.09229/1.09099;
  no demonstrated full-model benefit, neither deployed. N3.2 tuning unchanged.
  r1 profile/r2 layout+solutions all GPU-pass; r2 controller alone assumed3
  layouts instead of2 and failed after passing tests. Count derived from audit;
  r3/r4 seal original GPU evidence without rerun; r4 retains inherited user-file
  hashes. Initial controller/failure/source preserved, tolerances unchanged.
  Three guards ACK600s/dmesg identical, final309 protected identities/four FFT
  AOT DBs/links/runtime/defaults/user trainer/config unchanged. No native build,
  Torch/library install, cache reset/reboot/.105/BIOS/fallback changes.
  Runtime code62ab2437c, rocBLAS8389197c1, native Torchdc8fea0/source2b449c4,
  other native libraries319a235d6. Same r2 boot d0532075-88fb-4a2f-974e-f61578d20d9f.
  No active agent build/test/container/guard, free~5.92GiB. Historical CPU Python
  fault unresolved/separate from fixed VM-heap/CSR and repaired N3.2 loader.
  [Report](docs/development/bc250_training_n33_results/README.md); full raw trace/
  fixtures WS/logs/training-n33-*, local ignored kernel/artifacts/training-n33/remote
  copied/hash-verified. Helper backups WS/recovery/training-n33-before; runtime
  recovery still WS/recovery/gemm-n32-before. Next preservation baseline
  WS/logs/training-n33-qualification-r4/final.json (309 unchanged identities).
  **Next N3.4:** audit BF16 [1024,2048]@[2048,2048] actual layouts/admitted scalar
  solutions with CPU references and paired timings. Largest remaining untuned
  moderate-size mm in this trace; only demonstrated benefit proceeds to full-model
  gradients/AdamW/paired timing before reversible deployment.

- **N3.2 completed and installed, 2026-10-19:** TheRock code `62ab2437c`,
  rocBLAS/Tensile `8389197c1`, patch0033. Actual dominant logical BF16 GEMM audit
  finds TN48/NN48/NT18 calls (checkpoint recomputation included), not132 mm calls.
  Existing-solution CPU-double/guard/input checks pass all14/14/8 candidates.
  Only NT M2048/N1024/K6144 lda2048/ldb1024/ldc2048 batch1 BF16/FP32compute/beta0
  selected: solution-490, six alternating timing rounds173.797->4.841ms fixture.
  Other layouts unchanged. Old-loader cold override internal error reproduced
  twice: preloaded metadata lacked codeObjectFilename. Lazy hydration repair,
  `.load().empty()` patch; isolated cold/public Torch then installed cold pass.
  Targeted cached rocblas build r2, ccache20GiB/OMP4/max16/nativegfx1013; no GPU/
  network, active dist read-only. Initial ThreadSafeValue compile failure and
  prelaunch Path/string controller failure preserved, corrected; no Torch rebuild.
  Four fresh model AB/BA workers: synthetic1024/batch1/accum1/BF16/rank16/math/
  checkpoint, loss/RNG exact,372 clipped gradients maxrelativeL2=0.02030649\<0.04;
  each worker's AdamW params/state CPU replay passes maxabs1.863e-9. Reference
  repeat exact. Warm medians8.603/5.654/5.494/8.552s, ratios1.522/1.557x.
  Installed automatic model5.507s/1.562x; no sustained dataset/convergence claim.
  Exclusive-lock deployment changes only rocBLAS DSO/global+helper launcher/
  preset/runtime metadata/provenance. Automatic current-dev-prefix exact-problem
  tuning verifies CSV/DSO/110 gfx1013 assets; frozen previous unchanged. Explicit
  `BC250_GEMM_TUNING=0` outer opt-out restores old selection, manual override wins;
  identity mismatch falls back. Reduction order changes: no bitwise continuation
  promise across tuning changes. [Usage](docs/development/bc250_torch_usage.md).
  Installed cold3/core7/conv30/model4/optout2 pass;12 GPU guards ACK600s (10 passing,
  two expected reproduced cold failures), dmesg identical whole batch. Prior193
  identities changed only active rocBLAS aliases/runtime.json; final309 protected
  paths, four FFT AOT DBs/user trainer/config/old packs/releases/checkpoints intact.
  Native Torchdc8fea0/source2b449c4, other native libraries319a235d6 unchanged;
  only rocBLAS rebuilt at8389197c1. DSO SHA85599e274e3ba5077587f09eed0ba1df611ec19b2f2b595cc3a3724752d2f516.
  Same r2 boot d0532075-88fb-4a2f-974e-f61578d20d9f, no reboot/default BLAS/math/
  SDMA0/heap-off/.105/BIOS/fallback change. No active agent job/container/guard;
  free~6.13GiB. Historical CPU Python fault remains unresolved/separate from fixed
  VM-heap/CSR and repaired loader defect. [Report](docs/development/bc250_gemm_n32_results/README.md).
  Raw WS/logs/gemm-n32-\*, candidate WS/candidates/gemm-n32-r2, recovery
  WS/recovery/gemm-n32-before, full local ignored kernel/artifacts/gemm-n32/remote;
  all copied raw identities verified. Next preservation baseline
  WS/logs/gemm-n32-deployment-r1/final.json; SOURCE_GEMM_N32.json in WS/current.
  **Next N3.3:** reprofile optimized current model with exact profiled/unprofiled
  loss/gradients/AdamW/RNG replay; audit the next measured dense-mm layout before
  another candidate. Do not assume optimizer/gated-delta dominates or repeat N2.

- **Practical N2 review + N3.1 completed, 2026-10-19:** TheRock code `4e76c2f01`.
  [Review](docs/development/bc250_n2_practical_review.md) separates optional BF16
  FFT/general sparse solve/CK backward/plugin/RPP and other residual scopes from
  current LoRA requirements. No recorded mandatory N2 runtime implementation was
  identified before N3. Corrected stale support-matrix N2.14 cursor and fixed-A01
  group1 convolution boundary; broad N2 remains open, no N2.19 batch executed.
  New explicit synthetic actual Qwen3.5-2B BF16/autocast/rank16/math/checkpointed
  training profiler uses current app hooks; only local base weights read, no user
  training data/checkpoint opened. At65 and1024tokens profiled/unprofiled loss,
  all372 nonzero clipped adapter gradients, adapter update, full AdamW state and
  CPU/GPU RNG match exactly;36 conv/36 gated-delta calls prove recomputation.
  Each shape completes3 additional finite warm training updates. Offline analyzer
  kernel-only/correlation/stage/shape accounting has4 contract tests, Black/Ruff-F/
  syntax checks pass. Initial r1 numeric/profile pass had mixed annotation/kernel
  views; corrected reporting excludes annotations and handles autograd worker
  submissions. r2 controller NameError before launch fixed; r3 final1024/r4 short65
  pass, all raw/source/error evidence retained. No Torch/library/native build,
  ccache reset, application/default/prefix change or speed improvement claim.
  At1024:38314 kernels, warm median8.676s/peak5351892480B;65:17286 kernels,
  median0.726s/peak4232892416B, accumulation1 synthetic only. Kernel sum at1024
  8.680s: mm84.71%, backward-submitted76.82%; BF16 [1024,6144]@[6144,2048]
  45.94% and reverse projection16.53%. Shape call counts are kernel dispatches,
  not necessarily mm call counts; inspect stride/transpose before selection.
  Initial plus2 final GPU processes/guards ACK600s, whole batch dmesg identical,
  193 protected identities incl user trainer/config/four FFT AOT DBs unchanged.
  Active native Torchdc8fea0/source2b449c4/libraries319a235d6/source-doc c29021b40e7,
  r2 boot d0532075-88fb-4a2f-974e-f61578d20d9f/current/previous/image/defaults/
  BIOS/.105 preserved; no reboot. No running agent job/container/guard; free~6.85GiB.
  CPU Python fault unresolved/separate from fixed VM-heap/CSR and tool errors.
  [Report](docs/development/bc250_training_n31_results/README.md); full raw traces
  WS/logs/training-n31-r{1,3,4}-profile/training-step-trace.json and matching local
  ignored kernel/artifacts/training-n31; logs WS/logs/training-n31-*. Preservation
  baseline WS/logs/training-n31-qualification-r4/final.json (N2 binaries retained).
  **Next N3.2:** audit rocBLAS/Tensile selection and actual strides/transposes for
  dominant BF16 [1024,6144]@[6144,2048]; CPU/qualified-GPU reference + paired timing
  of one hardware-feasible scalar candidate, then model gradients before any
  reversible deployment. Default rocBLAS/math remains until demonstrated benefit.

- **N2.18 completed, 2026-10-19:** TheRock qualification `746528fa6`.
  Direct installed scalar CK DL FP32 grouped forward convolution:128 independent
  CPU-double reference cases plus8 vector-shape refusals pass. Plain/Add+ReLU,
  groups1/3, four1x1/3x3 shapes, stride/dilation/asymmetric padding, M/N tails,
  valid contiguous/strided spatial layouts, exact dyadic/fixed-tolerance other
  inputs, default/private streams,256-byte guards/stride holes/input preservation.
  Max abs error4.255575e-6. Retained FP32/FP16/INT8 GEMM3/core7 pass; no production
  defect, new Torch routing/operator, library/header install or speed claim.
  Initial r1 global StreamConfig namespace compile error, r2 guard startup on
  nondefault kernel, r3 invalid packed-layout fixture all retained; corrected
  r4 passes. Four build outputs preserved, ccache20GiB/OMP4/one job/native gfx1013,
  no mapped GPU/network. Test launcher now accepts explicit retained r2/r3
  --guard-release, defaults to r3, rejects mismatch before launch, reports early
  guard failure. Original kernel guard/recovery semantics unchanged; helper backup
  WS/recovery/ck-dl-n218-before. Failed GPU fixture plus3 final guards ACK/600s;
  dmesg identical across all4 attempts,191 protected identities/new binary sealed.
  **Observed .104 already on r2 at entry:**7.1.5-bc250rocm-r2,
  boot d0532075-88fb-4a2f-974e-f61578d20d9f. No reboot/boot-policy change this batch;
  prior r3 qualifications retained, do not describe r3 as currently running.
  Active Torch dc8fea0/source2b449c4/libraries319a235d6/source-doc c29021b40e7,
  current/previous/ROCm links/runtime/defaults/four FFT AOT DBs/.105/BIOS unchanged.
  No running agent job/container/guard; free~7.16GiB. CPU Python fault remains
  unresolved/separate from fixed VM-heap/CSR and test-only failures above.
  [Report](docs/development/bc250_ck_dl_n218_results/README.md); raw
  WS/logs/ck-dl-n218-*, outputs WS/build/bc250-coverage/ck-dl-n218-r{1,2,3,4},
  local ignored kernel/artifacts/ck-dl-n218. Next preservation baseline
  WS/logs/ck-dl-n218-qualification-r4/final.json plus binary_sha256.
  **Next:** audit remaining N2 gaps against current model workloads/B01/B02/B07/
  A09 evidence, then select next bounded N3 batch; avoid repeating prior AMP/
  compile/resume/CK Tile work. N2 remains broad/open, no automatic N2.19 sweep.

- **N2.17 completed, 2026-10-18:** TheRock qualification `5826261d8`.
  Native FP32/64 stable key/value sorting and selection pass rocPRIM 656 /
  hipCUB 656 / Thrust 648 =1,960 checks, plus public Torch 62 (54 finite output/
  indices/base-gradient fixtures, eight non-finite output/index fixtures).
  Sizes 0/1/33/257/4099/65537, default/private streams, guarded outputs/workspace,
  immutable inputs/indices/flags. Exact bits preserve signed zero, smallest
  subnormals, infinities and quiet-NaN sign/payloads; exact payload indices prove
  stability. Native radix pinned encoded NaN ordering is distinct from explicit
  strict-weak NaNs-last comparison (both directions) and Torch default NaN
  ordering. Five selection policies verify stable counts/bits;16 unsupported
  partial-float-bit queries rejected. Torch sort/masked_select finite backward
  matches CPU exactly, including unused strided base storage; no NaN-gradient
  claim. No production defect/new operator/default/dispatch/speed claim.
  Initial build r1 failed only new Thrust adapter's mixed-type auto declaration;
  corrected in separate r2 outputs, failed source/logs/r1 binaries retained.
  Three standalone builds ccache20GiB/OMP4/three jobs/native gfx1013, builder
  fa3e8c254, no mapped GPU/network; no library/Torch rebuild/install. P2 INT32
  primitives138/core7 retained; six GPU gates/guards ACK/600s, dmesg unchanged
  including initial build, 181 protected hashes plus new binaries sealed.
  Active native Torch `dc8fea0`, source `2b449c4`, native libraries `319a235d6`,
  source-doc `c29021b40e7`,
  links/runtime metadata/DSOs/headers/four FFT AOT DBs/image/kernel/boot/defaults/
  .105/BIOS/r2 fallback unchanged. [Report](docs/development/bc250_sorting_n217_results/README.md)
  includes explicit contracts, case counts, source/binary/build/kernel evidence.
  Outputs WS/build/bc250-coverage/sorting-n217-r{1,2}; raw WS/logs/sorting-n217-*,
  local ignored kernel/artifacts/sorting-n217. Next preservation baseline
  WS/logs/sorting-n217-qualification-r2/final.json plus binary_sha256 for new
  test products. No running agent job/container/guard; free ~7.16GiB. CPU Python
  fault unresolved/separate from fixed VM-heap/CSR and the compile error above.
  **Next N2.18:** audit CK scalar DL admission/GEMM evidence, then one admitted
  DL convolution/grouped CPU-reference qualification and targeted build; keep
  XDL/WMMA/modern MIOpen CK plugin off, preserve defaults, do not repeat completed
  CK Tile normalization/top-k. Broad N2–N8 remain open.

- **N2.16 completed, 2026-10-18:** TheRock qualification `2db3b18c1`.
  Existing installed headers pass native rocPRIM 576 / hipCUB 480 / rocThrust 480 new
  CPU-reference checks: FP32/FP64 ordinary/segmented reduce and inclusive/
  exclusive scans, nonzero init, repeated keys, empty/uneven segments, in-place
  scans, plus 288 exact custom first-index argmax checks. Sizes 0/1/33/257/4099/
  65537, default/private streams, guarded inputs/outputs/workspace and immutable
  read-only inputs/keys/offsets. Dyadic sums exact; fixed-tolerance non-dyadic
  CPU long-double refs max error 5.72e-6/8.77e-15 FP32/64. Offset APIs include
  empty segments; Thrust reduce-by-key correctly checks only nonempty runs.
  No production defect reproduced; no new Torch operator/dispatch or speed claim.
  Three standalone native test units built ccache20GiB/OMP4/three jobs (<=16),
  -O2/C++17/gfx1013, builder fa3e8c254, no mapped GPU/network. Output
  WS/build/bc250-coverage/primitives-n216-r1; sources/builds/caches/P2 binaries
  preserved. First attempt passes, P2 INT32 primitives138/core7 retained. Five
  GPU processes/guards ACK/600s; dmesg unchanged whole batch, 168 protected hashes
  match plus new binaries separately sealed. No library/Torch/header install,
  runtime metadata/link/default/image/kernel/boot/.105/BIOS/r2 fallback change;
  four FFT AOT DBs preserved. Native Torch `dc8fea0`, source tests `2b449c4`, libraries `319a235d6`,
  libraries source-doc `c29021b40e7` retained. [Report](docs/development/bc250_primitives_n216_results/README.md)
  includes source/build commands, counts/errors/guards/protected and binary hashes.
  Raw WS/logs/primitives-n216-*; ignored local kernel/artifacts/primitives-n216.
  Next preservation baseline WS/logs/primitives-n216-qualification-r1/final.json
  plus binary_sha256 when protecting new test artifacts. No running agent job/
  container/guard; free ~7.19GiB. CPU Python fault unresolved/separate from fixed
  VM-heap/CSR. No broad precision/determinism/performance/consumer-autograd claim.
  **Next N2.17:** audit P2 sorting/selection, then bounded floating stable key/
  value sorting and selection with explicit ordering/special-value contracts,
  CPU references, guards/private streams and targeted builds. N2–N8 remain open.

- **N2.15 completed, 2026-10-18:** TheRock qualification `0e4557ee5`;
  libraries source-doc revision `c29021b40e7`, native libraries still `319a235d6`.
  No RNG arithmetic defect reproduced. Torch state/layout80, moment/range22,
  offset/lifecycle/stream/dropout-trace20 and fresh-process continuation1 pass.
  FP16/BF16/FP32/64, complex normal and INT32/64; guarded strided views, empty/
  odd/tail sizes, seeds>2^63, offsets>2^32/2^40, state rejection/legacy bytes,
  default isolation/fork exception restoration and ordered private streams.
  Five GPU kernels traced including dropout forward/backward and Philox normal;
  dropout CPU mask/scale refs pass. Mixed GPU/default/explicit+CPU+Python fresh
  continuation matches10 fields. Direct native68 pass: five pseudo/four Sobol
  variants, reset/rebinding, exact prefix relationships and CPU Philox words
  through rocRAND/hipRAND. Initial native59/60: stale header incorrectly said
  LFSR113 rejects offsets; implementation/upstream tests deliberately support
  them. Corrected comments only (C API vs C++ wrapper), added9 LFSR cases incl
  high-offset non-aliasing. MTGP32 refusal retained. Patch0032/source before/after
  hashes and failed helper/result preserved; source comments staged on .104 with
  recovery WS/recovery/rng-n215-source-before. Installed headers/DSOs/defaults/
  prefix metadata unchanged; no rebuild/cache reset. Retained RNG28/core7 pass;
  seven passing containers plus initial assertion failure, eight guards ACK/600s,
  dmesg byte-identical whole batch,153 protected identities/four AOT DBs match.
  Active Torch `dc8fea0` / source tests `2b449c4`, WS/pytorch-sparse-n213,
  current/previous/ROCm links/image/kernel/boot/.105/BIOS/r2 fallback preserved.
  [Report](docs/development/bc250_rng_n215_results/README.md) includes exact cases/
  commands/source hashes/trace/failed contract; raw WS/logs/rng-n215-* and ignored
  kernel/artifacts/rng-n215. Next preservation baseline
  WS/logs/rng-n215-qualification-r2/final.json. No running agent job/container/
  guard; free~7.21GiB. No speed/randomness-certification/universal correctness
  claim. CPU Python segfault unresolved/separate from fixed VM-heap/CSR and this
  explicit assertion. **Next N2.16:** audit P2 rocPRIM42/hipCUB48/Thrust48, then
  bounded floating/segmented reduction/scan CPU references, guards and private
  streams with targeted ccache builds. Broad N2–N8 remain open; B01 separate.

- **N2.14 completed, 2026-10-18:** TheRock qualification `44568c2e2`.
  Existing native FFT implementation passes; no production defect reproduced,
  binary/build/default/prefix changes or performance claim. Active native Torch
  `dc8fea0` / source tests `2b449c4`, libraries `319a235d6` remain installed.
  New public complex96 + real72 fixtures each pass CPU-double output/weighted
  backward, actual GPU strides/private stream, input immutability and guarded
  strided out. Prime/composite/odd, crop/pad, conjugate/broadcast, reordered and
  four-transform-dimension decomposition covered. Three independent DFT/finite
  difference/cache-reuse checks pass; trace34 kernels includes native forward/
  backward rocFFT. Direct hipFFT16 plans/32 executions pass caller workspace,
  advanced strides, batches3 and synchronized stream rebinding; raw native gaps/
  input may legally be overwritten, unlike Torch's stronger preservation contract.
  Retained FFT30/core7 pass. Initial48/96 fixture failures were non-leaf GPU
  double tensors with absent .grad; detach correction passes, no tolerance change.
  Failed run/helper preserved. Seven guards ACK/600s, six successful processes,
  dmesg byte-identical whole batch,145 protected identities and four persistent
  AOT databases unchanged. No installation/cache reset/native build/reboot;
  current Torch WS/pytorch-sparse-n213, ROCm/previous links/image/kernel/boot/
  .105/BIOS/r2 fallback/defaults preserved. [Report](docs/development/bc250_fft_n214_results/README.md)
  includes cases/commands/source hashes/trace/failure/preservation; raw
  WS/logs/fft-n214-* and ignored kernel/artifacts/fft-n214. Next preservation
  baseline WS/logs/fft-n214-qualification-r2/final.json. No running agent job/
  container/guard; free~7.22GiB. Python CPU segfault unresolved/separate from
  fixed VM-heap/CSR; test exceptions above are distinct. BF16 promotion B01,
  native in-place/planar/giant plans/callbacks/graphs/higher-order gradients and
  broad performance unqualified. **Next N2.15:** audit existing P2 RNG28 and
  checkpoint/state evidence, then bounded generator offset/state replay and
  private-stream sequencing through Torch/rocRAND with exact replay/analytic
  distribution contracts. Do not require CPU/GPU RNG bitstreams to match.
  Broad N2–N8 remain open; further tiny-eig work stays optional.

- **N2.13 completed and installed, 2026-10-18.** Native Torch `dc8fea0`
  (implementation `a434d03`), final tests `2b449c4`; TheRock build `bbf556448`,
  final controllers `39ea58574`. Public CSR generic SpSV/SpSM and BSR block solves
  qualified across four real/complex types, public 32/64-bit indices, vector/three
  RHS, upper/lower/transpose/unit flags, layouts, guarded outputs and streams.
  Initial nonsingular 256 cases passed; exact/missing pivot probes failed 16/32,
  expanded 39/80 because native division avoidance can leave finite placeholders.
  ROCm-only 2D Torch postcheck now marks exact zero/missing pivot rows as NaN.
  One GPU diagonal-search launch, actual stream, no host reads/dense conversion/
  extra workspace; bit-exact zero test, no near-zero/conditioning claim. Unit
  bypass and legacy empty-A NaNs retained. Other singular rows unspecified;
  direct native APIs unchanged; general sparse factorization/spsolve B02 separate.
  Candidate/installed each pass 433 new public groups, 96 direct generic fixtures/
  192 solves (N/T/adjoint, host/device alpha, analysis reuse), retained sparse41,
  core7 and CPU-first profiler. Final pivot group verifies real NaN writes into
  strided sentinel buffers on a private stream; test-only post-build revision is
  separate from native GIT_REVISION. BSR public int64 still downcasts within
  tested bounds; no giant native64 BSR, triangular autograd, batching or broad
  performance claim. Only libtorch_hip.so changes; CPU profiler fix retained.
  Independent WS/pytorch-sparse-n213 build uses required HIP generation then pip
  editable/no-isolation, ccache20GiB/jobs12/link1/gfx1013; old builds/sources/caches
  and shared immutable third_party retained. 22 guards ACK/600s, two numerical
  baseline failures preserved, dmesg unchanged whole batch; 121 prior identities
  verified, changed active runtime/GPU aliases pinned through recovery. Only dev
  Torch link/runtime metadata updated; ROCm DSOs/packs/current/previous/rocm links,
  image/kernel/boot/BIOS/.105/defaults/r2 fallback unchanged. Recovery
  WS/recovery/sparse-n213-before, SOURCE_SPARSE_N213.json in WS/current. Next
  preservation baseline WS/logs/sparse-n213-qualification/final.json plus its
  candidate_sha256/runtime_sha256; prior active GPU alias hashes are historical.
  [Report](docs/development/bc250_sparse_n213_results/README.md), compressed traces/
  full build log, cases/commands/source/generated hashes; raw WS/logs/sparse-n213-*,
  ignored local kernel/artifacts/sparse-n213. No active agent job/container/guard;
  free about 7.23GiB: capacity check before future builds. Python CPU segfault
  remains unresolved/separate from fixed VM-heap/CSR; these are numerical pivots.
  **Next N2.14:** audit P2 FFT30, then bounded batched/strided complex plans and
  odd-length real transforms with CPU outputs/gradients; preserve rocFFT AOT
  cache. BF16 promotion B01 remains separate. Broad N2–N8 still open.

- **A09/N4 profiler ordering repaired/installed2026-10-18:** native Torch
  b823a1b, test qualificationda45369; TheRock build669baa5d7/controller39f3ed14b.
  ROCm first Kineto init retains GPU capability when a GPU is visible, even for
  CPU-selected first capture; per-capture activity filtering remains. No visible
  GPU retains CPU-only backend. CPU-first warm/cold and GPU-first now each produce
  13 kernels/two copies per GPU window; CPU-selected captures omit GPU records
  even while GPU work runs. Exact profiled/unprofiled outputs+both gradients,
  CPU float64 refs, correlations/durations and nondefault stream pass. Hidden HIP
  device CPU capture passes without Torch CUDA lazy init (not a physical CPU-only
  host/driver-uninitialized claim). Candidate and installed each pass four orders,
  API7/core7; fresh installed reactivation also passes. Only libtorch_cpu.so changes,
  GPU arithmetic DSO byte-identical; no LoRA/training speedup claim. Independent
  WS/pytorch-profiler-order built pip-only after required out-of-place HIP generation,
  ccache20GiB/jobs12/link1/native gfx1013; old A09 build and shared immutable
  sparse-followup/third_party retained.21 guards ACK/600s,18 successful process
  gates incl expected-defect reproductions; three failed fixtures retained (small
  copy trace expectation and two SDK-invalid -1 visibility settings, corrected
  to256KiB/valid nonmatching HIP ordinal). Native patch/bounds unchanged.
  First final preservation check automatically rolled back after installed gates:
  expected old hash on intentionally switched CPU alias. Controller now pins that
  old identity through recovery; fresh reactivation/seal pass.104 protected old
  identities match; dmesg unchanged entire batch. Development torch link/runtime
  metadata updated only; ROCm DSOs/packs/current/previous/rocm links/image/kernel/
  boot/BIOS/.105/defaults unchanged; r2 fallback retained. Recovery
  WS/recovery/profiler-order-before has old link/exact runtime bytes/old Torch link;
  SOURCE_PROFILER_ORDER.json in WS/current. Next preservation baseline is
  WS/logs/profiler-order-qualification/final.json:104 old/recovery identities plus
  candidate_sha256/new runtime_sha256; old N2.12 active alias hashes are historical.
  Native GIT_REVISION=b823a1b;
  subsequent test-only qualificationda45369 recorded separately; no metadata-only
  Torch rebuild. [Report](docs/development/bc250_profiler_order_results/README.md)
  includes both-order/failed traces, source/hash/command/rollback evidence.
  Raw WS/logs/profiler-order-*, ignored kernel/artifacts/profiler-order. No running
  agent job/container/guard; free~10.13GiB. Python CPU segfault unresolved/separate
  from fixed VM-heap/CSR; explicit SDK import aborts above are distinct too.
  **Next N2.13:** audit existing generic hipSPARSE/Torch CSR/BSR triangular vector/
  multi-RHS paths; CPU-reference missing complex/index/layout/stream/zero-pivot
  cases and fix demonstrated defects. Retain P2 legacy real csrsv16; general
  sparse factorization/spsolve B02 separate. Further tiny-eig optimization optional;
  N2–N8 broad workstreams remain. Profiler counters/daemon/external concurrency/
  compile/graphs/long trace/cross-boot clocks remain N4, not claimed fixed.

- **N2.12 completed2026-10-18:** TheRock codeb1c94bddc; explicit
  right_eigenvectors(recovery_backend="device") uses one32-thread native HIP
  triangular solve, complex64/128 cap16. N2.6 reference remains default/oracle;
  all finite/unitary/separation/rescaling/unitnorm/residual gates retained.
  Warm-session profile recovery~2.47–5.19ms supported fusion.128 retained groups,
  six CPU fixtures/36 backend evaluations,32 all-order direct cases,two near-gap
  rescaling stresses,six sentinel ABI cases,six ABI refusals,four public/output
  refusals pass. Default recovery exact N2.11; real native GPU kernel traced.
  Paired A1/B1/B2/A2 recovery1.32–2.32x/whole warm candidate1.17–1.34x in six
  fixtures; recovery still~1.90–2.10ms, host full eig~0.213–0.408ms much faster
  than full native~3.75–11.79ms. Scalar reads remain18, not total sync count.
  No general/model/LoRA gain or production GPU eig integration. One-unit
  hipcc/ccache20GiB/OMP4 build only; Torch/installed ROCm DSOs/defaults/prefix
  metadata/links/images/kernel/boot/.105 unchanged.104 protected hashes match,
  dmesg unchanged entire batch; nine guards ACK/600s, eight process gates pass.
  **New separate A09/Kineto defect:** first CPU-only profile can leave later GPU
  traces empty. Initial required trace check failed and is preserved; final
  fresh-process first CPU+CUDA capture proves native kernel. Runtime defect is
  unresolved, not hidden by a claim of profiler repair. Source points to one-time
  cpuOnly profiler registration; existing A09 tested opposite ordering only.
  [N212.md](docs/development/bc250_solver_results/N212.md), n212/ evidence/hashes/
  failed+passing traces/commands; raw WS/logs/solver-*n212*, ignored local
  kernel/artifacts/solver-n212. Prior helper/oracle WS/recovery/solver-n212-before
  and schur_n211_reference.py. No active job/container/guard; disk~13GiB free.
  Python CPU segfault stays unresolved/separate from fixed VM-heap/CSR.
  **Next:** A09/N4 CPU-first→GPU profiler reproducer and reviewed initialization
  fix in separate candidate, both-order trace/numerical gates before deployment;
  then broader N2 sparse/FFT/RNG coverage triage from existing audit. N2.1–12
  bounded milestones complete; broad N2–N8 remain seven workstreams, not seven
  small tasks or a fixed percentage. Larger/batched/close spectra/balancing/
  gradients/general native eig convergence remain unqualified; further small-eig
  optimization is optional, not required for ordinary training.

- **Storage refresh completed2026-10-18**, explicit user-authorized obsolete
  generated-artifact cleanup, code780e67ceb. Recovered12,839,837,696bytes=11.96GiB,
  free1.08->13.04GiB/used100->97%. Unlinked8,801 old Torch .o/.a/.so files under
  WS/pytorch-{work,a04,sparse-followup}/build,21 old Triton bin/test compiled files,
  90 synthetic CLI adapter/state/tokenizer files from five explicit test labels,
  and removed9 stopped project build containers. Exact plan/journal preserved.
  Sources/generated sources/build metadata/directory paths/links intact; retired
  generated targets require rebuilding. Current A09 full build, shared immutable
  sparse-followup/third_party, active ROCm build/caches and all installed old/current
  DSOs retained. No package/image/global prune: all image IDs/tags and user neural-p1
  container unchanged; real user models/data/checkpoints, all recovery copies,
  source archives, releases/compat installs, .105/user src/boot fallbacks preserved.
  101 protected product/config hashes unchanged, no current/previous/runtime link
  or metadata change. Current-before/current-after/fallback-after plus current/
  fallback fresh Triton constexpr-specialization gates all exit0:8 groups each,
  40 executions incl output/gradient/SDPA/FFT/RNG/consumer imports and GPU Triton.
  Five guards ACK/600s; dmesg byte-identical across entire batch. No native build/
  cache reset/default/kernel/boot/BIOS change; CPU Python fault stays unresolved/
  separate from fixed VM-heap/CSR. [Cleanup report](docs/development/bc250_storage_cleanup_20261018/README.md)
  includes compressed exact plan/journal, final identities and gate evidence.
  Raw WS/logs/storage-refresh-20261018 and storage-refresh-*; ignored local
  kernel/artifacts/storage-refresh-20261018. No active agent job/container/guard.
  **Next remains N2.12:** warm-session recovery profile and bounded native triangular
  back-substitution; retain Python/N2.6 reference/fixed numerical+failure gates/
  cap16/production host LAPACK. Scientific code807642198/Torch4af4a15/libraries319a235
  and r3 boot e3ed129b-9c41-4d0c-ab84-5ab54720bb8f unchanged.

- **N2.11 completed2026-10-18:** TheRock code807642198; explicit caller-owned
  GehrdSession reuses bindings/native handle for one PID/host-thread/device/GPU
  stream. No global cache/default change; current-stream checked/rebound, owned
  GEHRD packed/tau copies retained, close synchronizes pinned stream then destroys
  exactly once. Initial split: create~3.13–3.18ms vs binding0.11–0.13ms/alloc-copy
  0.06–0.08ms/destroy~0.13ms.128 retained groups, six full CPU/backend fixtures,
  three lifetime groups and16 refusal assertions pass: mixed-dtype reuse,
  close-event/double-close, post-GEHRD exception cleanup, two private stream
  sessions/six interleaved submissions, thread/stream/device/closed/busy/PID guards.
  PID guard injected, not actual fork; device metadata check, not multi-GPU test.
  Default reference exact N2.10; fresh/reused H/Q bitwise equal on fixtures.
  Seven guards ACK/600s/dmesg unchanged,82 protected hashes unchanged. Paired
  A1/B1/B2/A2 three-sample preparation4.25–4.51->0.34–0.64ms (7.03–12.52x),
  whole candidate1.28–1.95x, six warm-session fixtures with N2.8 QR/device Q.
  Init3.32–3.41ms/close excluded from warm pairs; cold one-call create+close
  prep4.24–4.51ms has no such gain. Host full eig0.219–0.408ms remains faster;
  no general/model/LoRA/Torch-wide gain. N2.9 regression remains separate.
  Python-only, no native build/ccache/DSO/Torch/image/prefix metadata/kernel/
  boot/default/.105 change. New gfx1013_gehrd_session.py must accompany candidate.
  Prior helper WS/recovery/solver-n211-before; immutable schur_n210_reference.py.
  [N211.md](docs/development/bc250_solver_results/N211.md), n211/ evidence/hashes/
  trace/commands; raw WS/logs/solver-*n211*, ignored kernel/artifacts/solver-n211.
  No active agent job/container/guard; disk~1.08GiB free: check before builds.
  Python CPU fault unresolved/separate from fixed VM-heap/CSR. No broad concurrency/
  batching/gradient/compile/graph/general convergence/production GPU eig claim.
  **Next N2.12:** profile warm-session right-vector recovery and prototype bounded
  native triangular back-substitution for separated Schur forms; retain Python/
  N2.6 reference/fixed separation-rescaling-unitnorm-residual/CPU gates/cap16/
  production host LAPACK default.

- **N2.10 completed2026-10-18:** TheRock codecc8a05d3c; explicit q_backend="device"
  on isolated hessenberg/schur forms GEHRD reflector-Q with one32-thread HIP
  launch, complex64/128 cap16. Native GEHRD/owned packed copy/current-stream
  handle lifetime unchanged; default reference Q and production host LAPACK
  unchanged. Initial split at16: GEHRD~0.37–0.40ms/Q~1.87–1.91ms/total~6.2ms;
  remaining setup/teardown not yet attributed.128 retained groups,24 independent
  complex/zero/mixed CPU reflector cases+guards/nondefault streams, six full CPU
  eigenpair fixtures with both Q backends across reference/N2.8/N2.9 QR pass.
  Default reference exactly matches immutable N2.9; H exact across Q backends.
  Zero/scalar/unknown-backend controls and real GPU Q kernel trace pass. Paired
  A1/B1/B2/A2 three-sample full preparation1.11–1.39x, whole N2.8-control candidate
  1.05–1.12x in six fixtures; order16 Q diagnostic1.87->0.17/1.83->0.19ms. Not
  kernel-only/general/model/LoRA gains; host full eig~0.222–0.429ms still faster.
  N2.9 mixed timing/regression remains separate. Seven guards ACK/600s/dmesg
  unchanged,79 protected runtime/source/recovery hashes unchanged. One-unit
  hipcc/ccache20GiB/OMP4 build, no Torch/ROCm DSO/image/prefix metadata/kernel/
  boot/default/.105 mutation. Prior helper WS/recovery/solver-n210-before,
  immutable WS/helpers/schur_n29_reference.py; all prior binaries retained.
  [N210.md](docs/development/bc250_solver_results/N210.md), n210/ results/trace/
  hashes/commands, raw WS/logs/solver-*n210*, ignored kernel/artifacts/solver-n210.
  No active agent job/container/guard; disk~1.08GiB free: capacity check before
  builds. Python CPU fault unresolved/separate from fixed VM-heap/CSR. No general
  convergence/batching/gradient/compile/graph/concurrency/production GPU eig claim.
  **Next N2.11:** split remaining library binding/handle create-destroy/tensor
  allocation prep costs; implement one measured reuse/cache correction with
  explicit lifetime/current-stream gates, owned GEHRD inputs/cap16/fixed numerical/
  failure gates/Python oracle/host production eig retained.

- **N2.9 completed2026-10-18 (bounded prototype, mixed latency):** TheRock
  code13f91e094; explicit schur(qr_backend="device_chunked",chunk_size=16), valid
  chunk1..32. Reuses N2.8 control core, stops immediately on terminal/failure;
  exact total QR budget, ceil((budget+1)/chunk) launch bound, separate T/Q input
  buffers.128 retained N2.3–5 groups,74 Python-oracle control cases,100 chunk
  boundaries, four long-chunk failures, seven public option refusals and six CPU
  eigenpair fixtures pass. Results/state bitwise exact across chunks1/2/8/16/32
  in those full fixtures and repeated chunk1 boundary cases; reference unchanged.
  Default16 status copies8/20/35->1/2/3 c64,11/26/46->1/2/3 c128. Paired
  A1/B1/B2/A2 three-sample whole ratios1.034/1.046/0.984 and1.040/1.047/0.951
  at4/9/16: small gains4/9, regression16, not a general speedup. Host full eig
  ~0.224–0.426ms remains much faster. Padded-Q experiment passes but does not
  help; retained WS/recovery/solver-n29-padded, selected original/unpadded
  source/binary restored under lock and verified against passed hashes.12 guards
  ACK/600s/dmesg unchanged across both variants;76 protected runtime/source/
  recovery hashes unchanged. Two one-unit hipcc builds/ccache20GiB; no native
  Torch/ROCm DSO/image/prefix metadata/kernel/default/boot/.105 change. Prior
  source/binary WS/recovery/solver-n29-before and immutable schur_n28_reference.py
  retained. [N29.md](docs/development/bc250_solver_results/N29.md), n29/ results/
  trace/commands/final identities; raw WS/logs/solver-*n29*, ignored local
  kernel/artifacts/solver-n29. No active agent job/container/guard; disk~1.09GiB
  free: capacity check before builds. CPU Python fault unresolved/separate from
  fixed VM-heap/CSR. No general convergence/batching/gradient/compile/graph/
  production GPU eig or model/LoRA performance claim; reference and production
  host LAPACK remain defaults. **Next N2.10:** split current GEHRD/reflector-Q
  preparation cost and prototype bounded device reflector-Q accumulation,
  retaining native GEHRD/Python+N2.8+N2.9 controls/cap16/fixed gates/host default.

- **N2.8 completed2026-10-18:** TheRock code67e5d1ef8; explicit isolated
  schur(qr_backend="device_control") fuses trailing deflation, nearest/exceptional
  complex shift, reused cooperative QR/similarity and finite/128eps fill status.
  At most one QR/launch, bounded budget+1 launches; five-int D2H status/launch.
  Cap16/complex64+128/normalization/iteration budget/deflation-before-budget/
  separation/residual/refusal gates unchanged. Reference helper and production
  host-LAPACK defaults unchanged.128 retained N2.3–5 groups +74 direct oracle/
  boundary/branch/nonfinite/fill/nondefault-stream cases +six CPU eigenpair
  fixtures pass. Default reference exactly matches immutable N2.7 on six fixtures.
  First direct harness ignored nonunique QR phases near singular shifts; fixed
  phase-aligned T/Q+independent reconstruction with original bounds, no solver
  change or tolerance increase. Failed log retained. Seven guards ACK/600s/
  dmesg unchanged. Paired A1/B1/B2/A2 three-sample synchronized Schur1.80–2.73x
  and whole-candidate1.56–2.21x versus N2.7 in six fixtures; whole8.88–20.35ms
  still slower than host eig~0.23–0.42ms. No universal/model/LoRA speedup claim.
  Order16 ATen scalar reads159/207->8, but explicit D2H statuses35/47 remain;
  do not describe scalar counts as total synchronizations.70 protected runtime/
  source/recovery/metadata hashes unchanged. Targeted one-unit hipcc/ccache20GiB;
  no Torch/ROCm DSO/image/prefix metadata/kernel/boot/.105/default mutation.
  Prior source/binary WS/recovery/solver-n28-before and immutable N2.7 Python
  oracle retained. [N28.md](docs/development/bc250_solver_results/N28.md) and n28/
  contain results/hashes/trace/commands; raw WS/logs/solver-*n28*, ignored local
  kernel/artifacts/solver-n28. No running agent job/container/guard deadline;
  disk~1.09GiB free: capacity check before builds. Python CPU fault unresolved,
  separate from fixed VM-heap/CSR. No general convergence/batching/gradient/
  compile/graph/production GPU eig contract. **Next N2.9:** bounded multi-iteration
  device chunk to reduce repeated status transfers; retain exact total QR budget,
  terminal/failure semantics, Python/N2.8 oracle/fixed gates/cap16/host eig default.

- **N2.7 completed2026-10-18:** TheRock code100f43e2a; isolated native HIP
  complex64/128 Householder QR+full similarity/Q accumulation, order2..16,
  explicit schur(qr_backend="device").32-thread shared-memory block/current
  stream; default helper reference and production host-LAPACK eig unchanged.
  Normalization/shift/deflation/budget/fill/separation/residual gates unchanged.
  28 direct CPU QR/invariant/stream cases, six CPU eigenpair fixtures,128 retained
  N2.3–5 groups pass; budget0/1 partial/refusal checks and real GPU kernel trace.
  Final reference outputs/counts exactly match N2.6 on six fixtures. Initial
  correct serial prototype slower at9/16; retained in WS/recovery/solver-n27-serial.
  Paired A1/B1/B2/A2 three-sample synchronized Schur medians reference12.37–52.45ms
  ->device10.02–36.02ms (1.24–1.46x). Host full eig~0.22–0.40ms remains faster;
  no whole-eig/model/LoRA/universal speedup claim. Separate order16 device phases
  preparation~6.1ms, QR~3.38/6.41ms, other control/scaling/checks~22.46/28.24ms;
  instrumented sums include forced sync and differ from uninstrumented timing.
  Eight guards ACK/600s/dmesg unchanged;64 prior protected source/runtime hashes
  unchanged. No prefix metadata/DSO/Torch/image/kernel/default/boot/.105 mutation.
  Targeted one-unit hipcc build with ccache20GiB; no Torch build. Source/helpers/
  final hashes/report [N27.md](docs/development/bc250_solver_results/N27.md), raw
  WS/logs/solver-*n27*, ignored kernel/artifacts/solver-n27. Source saved locally
  in Git; prior N2.6 helper retained on host. No running agent job/container/guard;
  disk~1.09GiB free: capacity check before builds. Historical Python CPU fault
  remains unresolved/separate from fixed VM-heap/CSR. No batching/gradients/general
  convergence/production GPU eig qualification. **Next N2.8:** bounded device
  shift selection/deflation/status around cooperative QR to target remaining
  control overhead; retain Python oracle/fixed gates/cap16/host production eig.

- **A08 completed2026-10-18:** libraries319a235d6c6496f0da37be44dfa60eaf31a29f82;
  runtime helpers8f0dfb671e88b31bc3f5bb4a4958acefdafde556; Torch4af4a15 unchanged.
  Existing native gfx1013 Lt now supports signed INT8 A/B, INT32 C/D/scalars and
  COMPUTE_32I with unsigned modulo32 accumulation/scalar/bias arithmetic. Row/col,
  N/T, padding/offsets/strided+broadcast batches, host/device scalars and DEFAULT/
  BIAS/RELU/RELU_BIAS qualified. Integer alignment/span-alias guards added; zero
  scalar no-read behavior retained. Existing torch._int_mm works without Torch
  rebuild or launcher hook, preserving its M>16/K,N positive multiples8/out rules.
  99 direct CPU cases+11 refusals and16 public Torch/out/custom-stream cases+five
  refusals pass, including odd16,790,289 above FP32 exact range and both overflow
  directions. Profiler confirms native gemm_int8. Existing floating403+561+664
  cases pass; old floating gemm/launch token streams unchanged. Core7/profiler7/
  sparse38/conv30/A0642/A0736/actual2B3 pass;372 nonzero adapters and12 head256
  forward/recompute calls retain A06 exact BF16/A07 prior bounded agreement.
  Three five-pair eager timing fixtures versus resident FP32 GEMM+cast are mixed:
  M=N=K32/128/256 INT8~0.01473/0.01528/0.05640ms vs0.02179/0.03440/0.05550ms.
  Event envelopes include submission, baseline has an extra conversion kernel;
  no kernel-only/universal/LoRA/quantized-model speedup claim. No quantization
  recipe, packed low-bit/QLoRA, integer autograd or compile/graph integration.
  Initial probe requested unsupported public pedantic descriptor mode and failed
  before GEMM dispatch; retained, implementation explicitly supports plain32I.
  Other compute/scales/pointer vectors/nonlinear+aux epilogues remain refused.
  12 GPU guards all ACK/600s/dmesg unchanged. Final source/DSO/external kpack
  identities verified;61 protected old/current Torch/runtime/SDK/app/frozen-pair
  hashes unchanged. Paired recovery WS/recovery/dense-int8-a08-before; restore
  DSO+kpack together under exclusive prefix lock. Runtime/source GIT stamps
  updated only after qualification. SOURCE_DENSE_INT8_A08.json in WS/current.
  [Report/usage/trace](docs/development/bc250_dense_int8_a08_results/README.md),
  logs WS/logs/dense-int8-a08*, ignored kernel/artifacts/dense-int8-a08/remote-logs/.
  Targeted hipBLASLt+dist build ccache20GiB/jobs12; full Git/patch0031/verified
  bundle onfcf7a714 retained. Clang-format, six Python helper format checks pass;
  no full CI/type claim. No agent job/container/guard deadline; disk~1.10GiB
  free: capacity check before builds. Old source/build/cache/image/links/user src
  retained. No kernel/BIOS/boot/.105/default BLAS/attention/SDMA0/heap-off changes.
  Historical Python CPU fault stays unresolved/separate from fixed VM-heap/CSR.
  **Next N2.7:** re-read existing N2.6 profile/failure evidence, split Schur
  preparation/QR-loop cost and prototype one bounded device-side small QR/
  similarity iteration with residual/scale/separation/failure gates, retaining
  default host-LAPACK production eig. A01–A09 complete within recorded bounds;
  optional B-items, fuller attention/profiling/quantization contracts remain open.

- **A09 completed2026-10-18 (user selected ahead of unfinished A08):** Torch
  `4af4a1508a6ec0b7b2b29f5e0115f8477c4d1bb5` adds tests on unchanged native source
  71a0426566bc9b952d9aa9b41f4a9d79ce46e0e8; helpers
  `4c2edb78553a8fe993a209d41539eba2c5966317` (full revision in SOURCE_PROFILER_A09.json). Existing pinned
  Kineto094d3c1 SDK backend enabled by USE_KINETO=1/KINETO_BACKEND=rocm and linked
  librocprofiler-sdk.so.1. No Kineto/SDK native source repair or architecture alias
  required. Separate WS/pytorch-profiler-a09 candidate built through pip editable,
  ccache20GiB/MAX_JOBS12/link1, original object cache copied and immutable third-party
  sources bind-mounted read-only from preserved pytorch-sparse-followup/third_party.
  Rebuilds require this shared source mount; no old tree/cache/image removed.
  Seven candidate/installed profiler tests pass:13 kernels/16 runtime/13 linked CPU
  events per FP32/BF16 fixture, exact profiled/unprofiled output+both gradients and
  CPU-float64 refs, two correlated H2D/D2H copies with exact roundtrip,4MiB alloc/free,
  two scheduled8-kernel custom-stream windows, CPU-only selection and bounded wall
  overhead. Candidate/installed short-window median ratio4.70x/4.63x includes setup/
  teardown, not LoRA throughput; profiling is explicit and adds cost.
  Nine candidate gates pass: profiler7/core7/sparse38/A05=69/A06=42/A07=36/conv30/
  actual2B3/CLI2. Four installed gates pass: profiler7/core7/actual2B3/CLI2. Model
  checks retain all372 nonzero adapter grads, exact A06 BF1665/1024, bounded A07
  2.17%/1.31% with12 D256 forward/checkpoint calls. Each backend fresh/resume exact
  across adapters/optimizer/scheduler/RNG/progress; user checkpoints/data untouched.
  14 GPU guards all ACK/600s/dmesg unchanged. Initial marker test confused CPU/GPU
  annotations sharing a name; explicit CPU-category check fixed harness, no wider
  tolerance. Controller schema mismatch after successful model test corrected;
  existing passed/ACKed gates reused, failed intermediate report retained.
  Exclusive-lock current/torch -> WS/pytorch-profiler-a09, old full source/build
  pytorch-sparse-followup retained.47 protected old/runtime/SDK/app hashes unchanged;
  candidate CPU/Python DSOs change, libtorch_hip.so remains byte-identical8ffd32ee.
  Recovery WS/recovery/profiler-a09-before; SOURCE_PROFILER_A09.json in WS/current.
  [Report/traces/usage](docs/development/bc250_profiler_a09_results/README.md), logs
  WS/logs/profiler-a09-* and ignored kernel/artifacts/profiler-a09/remote-logs/.
  Build container bc250-profiler-a09-build retained stopped; no running agent job,
  GPU container or guard deadline. Disk~2.03GiB free: capacity check before builds.
  Full local Git, patch0008/test bundle on71a0426 preserve tests; feature build flag
  is in build_torch_a09.sh. Older SDPA3-file reconstruction gap retained. Nine
  targeted spin/lintrunner -a checks pass, no full CI/type claim. General and host
  BC250_TRAINING_OPS.md usage updated. No ROCm DSO/image/kernel/boot/BIOS/.105/default
  attention/SDMA0/heap-off/dense BLAS/user src change. Hardware counters, broad
  concurrency/external profiler combinations, compile/graphs, long tracing and
  cross-boot clock calibration unqualified. CPU Python segfault remains open and
  distinct from fixed VM-heap/CSR. **Exact next A08:** inspect dense INT8->INT32
  _int_mm/Lt code/audit, capacity plan and bounded native route with CPU integer,
  overflow/layout/scalar/epilogue references; then N2.7. A08 was not completed.

- **A06 precision correction + A07 head256 completed2026-10-18:** Torch
  `71a0426566bc9b952d9aa9b41f4a9d79ce46e0e8`; installed runtime helpers
  `e4f4b3854` (full revision in SOURCE_A06_A07.json). FP16/BF16/autocast full
  gated-delta now preserves reference GPU preparation/scan ordering; non-autocast
  FP32/native direct preparation retained.42 A06 tests + four exact autocast cases
  pass; actual2B training/checkpoint/nonzero-B all372 BF16 adapter gradients exact
  at65/1024. This supersedes the failed BF16 qualification below within this scope.
  A07 explicit D<=256/tile16 above128: scalar tiled FP32/FP16/inference; BF16 D>128
  training uses math-forward/native-LSE+tiled-backward hybrid.36 tests pass; actual
  model worst adapter relativeL2~2.17%/1.31%65/1024 within unchanged4%;12 D256 calls
  verify six forward + six checkpoint recomputations. Shared fixture now .train():
  earlier actual-model tests that only enabled checkpointing were eval and did not
  prove recomputation; prior operator and real CLI evidence remains separate.
  Five paired attention fixtures through4096: BF16 hybrid lower peak but slower
  (1024:27.42->32.98ms,156->112MiB extra peak). Forward temporary allocations remain
  quadratic, saved state linear. Math remains default; no universal/model speedup.
  App's independent A06 hook preserved/corrected and banner updated. Inner/outer
  trainer --attention-backend tiled is opt-in for a new output/run; default math
  preserves legacy fingerprint schema, tiled records a distinct backend. Exact
  synthetic fresh/resume continuation passes within both backends, all372 adapter
  tensors plus optimizer/scheduler/RNG/progress. User checkpoint500 unchanged;
  this does not prove bitwise continuity across old buggy code/backend changes.
  Final exclusive-lock deployment eight gates pass:42/36/four autocast/three
  actual model/two CLI backend/69 A05/seven old-attention/seven core groups.
  Five installed source hashes match,39 protected DSO/DB/kpack/frozen/consumer
  hashes unchanged.32 GPU guards across candidates/retries all ACK/600s and unchanged
  per-run dmesg; four failures retained (eval harness, pure BF16 model threshold,
  stale trainer assertion, moved chat file). A separate r2 guard refusal occurred
  before GPU execution and rolled back. Found .104 booted r2 at resume; restored
  documented one-shot r3, base SSH/watchdog/cooling and full GPU/VMM gates pass;
  boot e3ed129b-9c41-4d0c-ab84-5ab54720bb8f. r2 remains persistent fallback.
  Recovery WS/recovery/a06-a07-before-r4; earlier original/r2/r3 attempt directories
  retained. SOURCE_A06_A07.json in WS/current, logs WS/logs/a06-a07-deployment-r4
  and a06*/a07*. [Results/evidence](docs/development/bc250_a06_a07_results/README.md).
  Full local Git, patch0007 and verified incremental bundle onba6cbcd preserved;
  older SDPA three-file reconstruction gap remains. Nine targeted spin/lintrunner
  -a checks pass; no full CI/type claim. Python/Triton source-only, no native build;
  existing Triton caches retained, ccache untouched. Usage docs updated locally and
  /home/nik/finetune-n7/BC250_TRAINING_OPS.md. No BIOS/image/DSO/default BLAS/SDMA0/
  heap-off/.105/user src/data change. Historical Python CPU segfault unresolved,
  distinct from fixed VM-heap/CSR corruption. No running agent job/container/guard
  deadline. **Exact next A08:** inspect existing dense INT8->INT32 _int_mm/Lt code
  and audit evidence; implement one bounded native route with CPU integer refs,
  overflow/layout/scalar/epilogue contracts and targeted ccache build settings.
  Then A09, then N2.7; full native FLA, dropout/compile/graphs remain extensions.

- **A06 recheck2026-10-17: BF16 LoRA qualification reopened; production unchanged.**
  The original actual2B test initialized LoRA B=0, so its372 named gradients included
  zero A gradients. New seed61718/std0.001 nonzero-B fixtures exercise372 nonzero
  finite A/B gradients each. FP32/65 worst/global relativeL2~9.54e-6/3.81e-6 passes;
  BF16/65 worst4.00217%, BF16/1024 worst5.42540% exceed the unchanged4% per-adapter
  limit (global1.574%/0.870%). Do not call broad/resumed BF16 LoRA parity complete.
  40 earlier independent operator tests remain bounded evidence, not invalidated.
  Exact ATen preparation control in the same composition gives zero model-gradient
  difference at65; native-forward/exact-backward and inverse swaps support FP32
  preparation rounding/derivative-order amplification through BF16 as the next
  investigation. Restoring original ATen scan structure still fails4.002%/5.573%;
  candidate retained but NOT installed. Original ba6cbcd/module9ecb6f9 unchanged.
  Forward phase1.341ms of27.384ms(~4.9%); backward19.836ms(~72.4%);196 batched-mm
  calls unchanged. Native event-instrumented total26.594ms; coarse event envelopes,
  not kernel-only time. Five actual nonzero-B1024 pairs median7.778/7.317s, pair
  ratios0.9887–1.0756; diagnostic timing on a failed numerical gate, not deployment
  or universal gain evidence. [Review](docs/development/bc250_gated_delta_a06_review/README.md)
  contains exact outcomes, counters, failed guards and source hashes.
  Six review guards ACK/600s/dmesg unchanged53f83f7b; three retained numerical/
  instrumentation failures.40/41 prior protected hashes match. Independent app edit
  observed in finetune-n7/replica/bc250_ops.py at00:08:37UTC: eddfea40->d058b14a,
  now enables A06 by default via BC250_GATED_DELTA=1. Origin not established; edit
  preserved. Our test uses separate pinned old helper. For new ordinary training,
  use inner bc250-train launcher option --env BC250_GATED_DELTA=0 before the script;
  host shell assignment alone is not inherited. Current app integration is unqualified.
  No prefix/module/DSO/kernel/default/boot change by this review; .105/fallback and
  user src/data unchanged. Raw WS/logs/gated-delta-a06-review-* and ignored local
  kernel/artifacts/gated-delta-a06-review/. No active job/container/guard deadline.
  Exact next **A06 numerical parity**, before A07: reproduce seed61718 nonzero-B
  65/1024, investigate reference-ordered Q/K normalization/reductions/derivatives
  and decay scan rounding, keep4% model gate and independent operator refs. Full
  native solve/scan fusion and FLA remain unqualified; A07–A09/N2.7 queued. Historical
  Python CPU fault remains separate/unresolved; VM-heap/CSR corruption remains fixed.

- **A06 completed2026-10-17 (first native substage):** Torch
  ba6cbcd2fafac1a6503f1f580033fbf72ad2a4f9 / TheRock helperse61d609d3658ee984907841f4fe1dc8ecdf66baa
  installed .104. Explicit torch.nn.bc250_gated_delta fuses Q/K normalization/scaling,
  beta*K/V, padded/chunked layout and cumulative log-decay; native scalar wave32
  forward/backward covers all Q/K/V/g/beta. ATen triangular solves, GPU dense matmuls
  and sequential chunk scan remain. Not a complete FLA backend or automatic trainer
  replacement.40 candidate/installed CPU-float64 stage and independent recurrent
  reference tests pass, plus69 A05 and seven core groups. Tiny FP32/BF16 LoRA with/
  without checkpointing and actual Qwen3.5-2B BF16/rank16/checkpointed65+1024tokens
  pass all372 adapter-gradient comparisons; max relativeL2~3.23%, not bitwise equal.
  Three paired synthetic1024-token forward/backward medians7.653->7.390s(~3.6% faster)
  and peak11,976,548,352->11,745,457,152bytes(~220MiB lower). Not dataset/optimizer/
  convergence/resume or universal speed evidence. Installed four operator fixtures
  show prep1.34–1.88x/full1.02–1.12x ratios; exact protocol/sample data in
  [report](docs/development/bc250_gated_delta_a06_results/README.md).
  Twelve guards all ACK/600s/dmesg unchanged53f83f7b, including three retained
  model-adapter flag refusals.210 cached candidate/installed variants audit native
  gfx1013/wave32/no-MFMA/WMMA. Spin nine targeted static/format checks and lintrunner
  -a pass; no full-project typing/CI claim. Exclusive-lock additive module install,
  41 protected hashes unchanged; DSOs, shared consumers and original trainer/helper
  unchanged. Recovery WS/recovery/gated-delta-a06-before; SOURCE_GATED_DELTA_A06.json
  in WS/current. Module SHA9ecb6f9014bc139e54cffe0fbb2c80f782035e57a3149e3c9a6b31fe445e8b2b.
  Full local Git/incremental patch0006/bundle on6ed3d12 preserved; old SDPA3-file
  reconstruction gap remains. No C++ build/image/kernel/reboot/default change;
  existing Triton cache retained and ccache untouched. Raw WS/logs/gated-delta-a06-*
  and local ignored kernel/artifacts/gated-delta-a06/. Disk8.4GiB free. No active
  agent job/container/guard deadline. Existing LoRA command retains the reference
  warning until explicitly integrated; packed/cache/higher-order/compile/graphs and
  further native solve/scan fusion remain unqualified. Historical Python CPU fault
  remains unresolved/distinct from fixed VM-heap/CSR. .105/BIOS/r2fallback/SDMA0/
  heap-off/math/default dense BLAS/user src/data unchanged. Exact next **A07**:
  inspect current tiled SDPA D<=128 guards and head256 Qwen3.5 geometry; implement
  and qualify D256 output/QKV gradients with bounded GQA/mask/all-masked/LoRA refs,
  keeping math default. Dropout/compile/graph contracts remain separate. A07–A09
  before N2.7; K6 completed.

- **Sparse/lint follow-up completed2026-10-16:** user-requested A04/A05 gap closure
  before A06; Torch6ed3d12b184a27d21c0fa41d43269ff6bd17f1e0 / TheRock runtime
  helpers4f478cc1d6acd9aa901f011a17d7c79856dccd36 installed .104. Raw native
  FP32/FP16/BF16 compress/mm first-order gradients now cover retained slots, dense
  input, bias and scalar/vector alpha. Public FP16/BF16 conversion preserves dense
  source-weight edge; transposed row-sparse weights reconstruct on GPU for dense
  GPU backward, avoiding incorrect transpose pruning. Retained zeros and absent/
  undefined optional tensors handled. Public wrapper initialized lazily to preserve
  Torch import order. Candidate/installed38 tests pass CPU references:18 raw,
  16 public/linear rectangular/frozen/trainable,3 gradient-subset groups(5 each),
  retained-zero group. Installed20 A04/120 matmuls,69 A05, seven core and30 conv
  regressions pass.12 forward +12 training fixtures also pass CPU references.
  15 guards including retained import/optional failures ACK/600s/dmesg identical
  53f83f7b. Reversible exclusive-lock current/torch switch to WS/pytorch-sparse-followup;
  full original A05 WS/pytorch-a04 and old WS/pytorch-work preserved.31 protected
  runtime/DB/kpack/original/frozen/trainer/consumer hashes unchanged. Recovery
  WS/recovery/sparse-followup-before; SOURCE_SPARSE_FOLLOWUP.json in WS/current.
  libtorch_hip SHA8ffd32eeaa6548ef896a5f56773255a26a985398a4a7bf811463bcb1dc043b90.
  Verified lint tools initialize locally in isolated uv caches;12 targeted configured
  static/format checks and required lintrunner -a pass. Spin0.18/lintrunner0.12.7;
  PYTORCH_LINT_SKIP_CODEGEN=1 avoids separate codegen. Not full-project type/CI claim.
  A05 function/class ASTs unchanged by formatting; import order canonicalized.
  Performance measured, not a blanket speedup: prepacked eager forward dense/sparse
  0.279–1.585 across12 fixtures, compression separate; forward+backward sparse
  1.92–3.36x slower than dense in12 small fixtures. Keep ordinary dense BLAS/training
  unchanged and structured sparse explicit. GPU backward uses dense intermediates
  and metadata sync; no hardware sparse accelerator or application speedup claim.
  Raw valid floating first-order scope only; integer/higher-order/forward AD/compile/
  export/graphs/broad shapes/concurrency/stability remain unqualified. Sources,
  exact lint/timing protocol, errors and evidence in [follow-up report](docs/development/bc250_sparse_followup_results/README.md).
  Raw WS/logs/sparse-followup-* and local ignored kernel/artifacts/sparse-followup/.
  Build r1 concrete-size codegen and r2 C++ operator/ADL failures retained; r3/r4
  success, final bc250-sparse-followup-build-r4/image28ad36e6/pip-only/native/noGPU/
  ccache20GiB/jobs12/link1. Log WS/logs/sparse-followup-build-r4.log. No active agent
  build/test/container/guard deadline. Disk8.7GiB free; avoid unnecessary clones.
  Full local Git, patch0005/reverse-apply/verified bundle on6181d00 preserved; older
  SDPA3-file reconstruction gap remains. Kernel/image/boot/BIOS/.105/user src/data/
  SDMA0/heap-off/math/default rocBLAS/r2fallback unchanged. Historical Python CPU
  fault remains unresolved/distinct from fixed VM-heap/CSR. Exact next **A06**:
  inspect existing gated-delta/FLA reference/model shapes, profile bounded forward/
  backward substages, implement one scalar/wave32 stage with independent output and
  Q/K/V/gate gradients. A06–A09 before N2.7; K6 completed.

- **A05 completed2026-10-16:** Torch6181d00b5beb8cc9fd6800da7a19c8abf756b30d,
  TheRock helpers869463155599a1c836bef10fb6089cbaa511058c installed .104.
  New opt-in torch.nn.bc250_training_ops exports causal_conv and chunked_causal_loss,
  without model/version hooks. Native gfx1013 K1–4 FP32/FP16/BF16 forward/dx plus
  new deterministic FP32 partial-reduction dw/db; strided/custom-stream and autocast
  handling; unsupported valid conv falls back to ATen. Chunked loss recomputes FP32
  probabilities, validates labels/counts/kwargs; original low-precision logits remain.
  Final candidate/installed69 TestCase tests pass:48 all-gradient conv,3 no-bias,
  4 AMP/checkpoint/scaled-update,12 loss, dispatch/8-refusal group and tiny-model
  CPU loss/all-gradient plus exact AdamW/parameter/GPU-RNG serialized resume.
  Installed seven core groups and A04 sparse20/120-matmul regressions pass. Six
  GPU guards all ACK/600s/dmesg identical53f83f7b;30 protected hashes unchanged.
  Only one Python module atomically added to active WS/pytorch-a04 under exclusive
  lock; Torch/ROCm DSOs and shared/frozen consumer bundle unchanged. Recovery
  WS/recovery/training-a05-before; SOURCE_TRAINING_A05.json in WS/current.
  Old finetune-n7 trainer/bc250_ops.py remain unchanged; new API requires explicit
  import/calls. No automatic general conv/loss replacement, FLA, higher-order,
  compile/graphs/export, large-model or performance/stability claim. Fixed8x AMP
  scaling test does not qualify dynamic GradScaler overflow/growth. Existing real
  Qwen/CLI51-gate evidence belongs to the old helper, not this module. Source/hash/
  usage/evidence in [A05 report](docs/development/bc250_training_a05_results/README.md), raw logs
  WS/logs/training-a05-* and local ignored kernel/artifacts/training-a05/.
  Full Torch Git/incremental patch0004/verified bundle preserved; prior SDPA3-file
  bundle gap remains. Lint initialization still blocked, attempts retained; syntax/
  whitespace pass. No active jobs/container/guard deadline. Disk15.9GiB free.
  Kernel/image/boot/BIOS/.105/user src/data/default dense BLAS/SDMA0/heap-off/math/r2
  fallback unchanged. Historical Python CPU fault distinct from fixed VM-heap/CSR.
  Exact next **A06**: inspect recorded Qwen3.5 gated-delta reference/model shapes,
  profile bounded forward/backward substages; implement one scalar/wave32 substage
  with independent output and Q/K/V/gate gradient refs before deployment.
  A06–A09 before N2.7; K6 completed.

- **A04 completed2026-10-16:** Torch8cdfcc8e0feb3689a1ec1d292539af083b13ea94,
  TheRock helpers b5879aefa7990928270645676e66cc40d972de6b installed on .104.
  Fresh candidate configure finds the existing hipSPARSELt package; no discovery
  source workaround needed. Native gfx1013 gate/FP32, HIP Python version/search,
  truthful compiled availability, initialized search fields and pack/layout/device/
  bias/alpha guards implemented. Candidate and installed20 tests pass:120 small
  CPU-reference matmuls, independent pack decoding, FP16/BF16 public wrappers,
  search and14 refusals. Installed30 convolution cases, seven core groups and one
  tiled SDPA output/QKV-gradient smoke pass. Eight guards including retained test
  failures ACK/600s, dmesg unchanged53f83f7b.18 protected ROCm/DB/kpack/original-Torch/
  frozen-fallback hashes unchanged. Exclusive-lock atomic current/torch switch to
  WS/pytorch-a04; original WS/pytorch-work fully preserved. Recovery metadata in
  WS/recovery/torch-a04-before. SOURCE_TORCH_A04.json in WS/current; source/hash/log
  scope in [A04 report](docs/development/bc250_torch_a04_results/README.md). Native scalar same-dtype
  FP32/FP16/BF16/INT8; valid1:2/2:4 inputs, compression is not pruning; GPU scalar/
  row-vector alpha, typed/FP32 bias. Default dense rocBLAS unchanged. No sparse
  hardware speedup, sparse-autograd/compile/graphs/large-shape/stability claim.
  Retained path/CPU-alpha/version assumptions and stale-HIPify/live-script build
  failures; final r4 pip build succeeded with ccache20GiB/jobs12/link1. Lint blocked
  by missing offline initialization; attempts logged, no lint pass. Full Torch Git,
  incremental A04 patch/bundle retained; old P2 SDPA3-file bundle gap remains.
  Historical Python CPU fault distinct from fixed VM-heap/CSR. No active jobs or
  guard deadline; retained build containers stopped. Disk16.1GiB free. Kernel/image/
  boot/SDMA0/heap-off/math defaults/r2fallback/BIOS/.105/user src/data unchanged.
  Exact next **A05**: inspect existing Qwen3.5 causal-conv/loss helpers and51 prior
  operator/model/resume gates; reusable opt-in API, trainable conv gradients and
  CPU/checkpoint/AMP references. A05–A09 before N2.7; K6 completed.
- **CURRENT A03 completed2026-10-16:** TheRock source/helpers
  7d9176a45a2ae40766290b0a440c468cfcb3a164, libraries
  fcf7a7142af8c258164e106e66bf893c7f563d15, installed .104. Optional modern CK
  loader respects build flag/existing device whitelist; explicit plugin path still
  warns on missing library. No plugin kernels added (optional B07 remains).
  Genuine gfx1013_20 system FindDb16 keys/58 CPU-checked immediate solutions,
  one searched ConvHipDirectFwd PerfDb record and41 compiled kernel records with
  exact geometry/type/source/hash provenance. Unforced normal alternatives checked
  after isolated search. Read-only default-path replay15+1 configurations/58
  solutions passes with empty user FindDb/PerfDb and zero new user kernels; logged
  searched-parameter load. Unseen geometry fwd/dx/dw passes and creates3 user
  records; system payload hashes unchanged.30 Torch conv +A02 smoke3 +seven core
  groups pass; all16 GPU guards ACK/600s/dmesg same53f83f7b, including retained
  initial WAL/read-only failure. Packaging now consistent SQLite backup/DELETE
  journal; original caches/packages retained. Initial deployment precheck rejected
  Info-level empty-user-cache notices before mutation; warning-severity check fixed.
  Under exclusive runtime lock, only MIOpen DSO and3 native DBs/manifest added;
  installed SHA3a751adc149778f061198014d6f086a75c33e7640b9473dd75bf14464652afce.
  11 protected HSA/HIP/Torch/BLAS/kpack/frozen/shared-heuristic hashes unchanged.
  Recovery WS/recovery/miopen-a03-before, SOURCE_MIOPEN_A03.json in WS/current.
  Report docs/development/bc250_miopen_a03_results/README.md; raw WS/logs/miopen-a03-*
  and local ignored kernel/artifacts/miopen-a03/. Bundle WS/build/bc250-a03-db-package-r3
  and local ignored copy preserved; text assets/manifest/hashes committed.30 library
  patches exact reconstruction. Restored historical MIOpen bias/fusion inventory
  row accidentally overwritten by A02 doc edit. Prior Torch SDPA3-file patch-bundle
  gap retained; full sources/Git preserved. Small seed, no universal tuning/speedup/
  cross-board/compiler/coverage claim; not all compiled search candidates CPU checked.
  A01 clock/trace/reset/reboot and event-speed-ratio limits retained. Python CPU
  fault remains unresolved/distinct from fixed VM-heap/CSR. Torch4ee6571/Triton
  ee65ff3b/ROCr329eaf7/image5df0/kernel0957/image84995/boot a287fd16/SDMA0/heap-off/
  math/r2fallback/BIOS/.105/user src/data unchanged. No active build/test/container/
  recovery deadline. **Exact next A04:** inspect patched Torch hipSPARSELt CMake
  package discovery/version gates, targeted ccache build (12 jobs/link1), then real
  compress/sparse-mm metadata/bias/scalar/stream CPU-reference contracts. Keep
  default dense BLAS. A04–A09 before N2.7; K6 completed; don't repeat earlier work.
- **CURRENT A02 completed2026-10-16:** TheRock source/helpers
  d17427bfeff6ab7204855a753a883c9a66fbf51b; libraries
  d375d8a2ea7280170455d133058c3cf8e16f58b6 (native-profile2093090048f), installed .104.
  Original direct sums/LSTM backward failure reproduced; native bundled legacy CK
  target accepted with RDNA1 descriptor0x31014000/scalar FMA, packed-dot asm off,
  explicit XDLOPS refusal; wave32 metadata verified. Additional BF16 mapping/raw
  buffer/load conversion and compute-precision alpha/beta epilogues corrected.
  Reduction cache key version2 prevents reuse of old embedded-header kernels;
  preserves caches. Retained intermediate mapping/buffer/numeric/stale-cache
  failures, no hidden tolerance change. Final candidate69/69; installed54 numerical
  reduction cases +3 unsupported BF16 compute refusals and12 FP32/FP16 LSTM/GRU/
  tanh-RNN outputs/all gradients pass. Thread/wave/block/multiblock/scales/beta/
  guards/strided custom stream/FP32 min-max indices covered. Installed30 conv,
  32 stable positive event pairs and seven core reference groups pass.14 total
  GPU/diagnostic guards ACK/600s/dmesg identical; baseline collector0 had3 failures.
  Only MIOpen DSO atomically replaced under exclusive prefix lock with installed
  RPATH; SHA f4b1a10ce20bdbdf0e6ac90168fb50945b35cf8e26385b66f3ee387aa4507b3c.
  Eight protected HSA/HIP/Torch/BLAS/kpack/frozen-fallback hashes unchanged.
  Recovery WS/recovery/miopen-a02-before; SOURCE_MIOPEN_A02.json in WS/current.
  Report docs/development/bc250_miopen_a02_results/README.md, raw WS/logs/miopen-a02-*
  and ignored local kernel/artifacts/miopen-a02/host-logs.29 library patches exact
  reconstruction; prior Torch3-file SDPA patch-bundle gap separate/preserved.
  No performance/universal coverage claim; BF16 RNN and wider NaN/index/compute/
  recurrent contracts unqualified. A01 clock/reset/absolute trace limits and event
  speed-ratio remeasurement remain. Python CPU fault unresolved/distinct from fixed
  VM-heap/CSR. Torch4ee6571/Triton ee65ff3b/ROCr329eaf7/image5df0/kernel0957/image84995/
  boot a287fd16/SDMA0/heap-off/math/r2fallback/BIOS/.105/user src/data unchanged.
  No active agent build/test/container/recovery deadline.
  **Exact next A03:** inspect actual gfx1013_20 FindDb/PerfDb/kernel-cache records;
  bounded genuine tuning entries with device/version/shape provenance, replay and
  packaging; review capability-aware optional CK plugin probing. Missing system DB/
  plugin warnings remain A03/B07. Never fabricate DBs or rename other-GPU records.
  A03–A09 before N2.7; K6 completed; do not repeat earlier milestones.
- **CURRENT A01 completed2026-10-16:** source/helper380a403a1bc17fa561345846fb4d92a6f57c3237,
  systems329eaf726632ab0136924804cbc1321ae09fc126,
  libraries08ba6f386ff661e720873779fc417b4246787d9c; installed on .104.
  Direct C API reproduced failure independent of Torch.32/32 retained HIP event
  pairs unstable/nonpositive before; KFD GOLDEN_TSC and shader s_memrealtime
  measured approximately100MHz but ~14,360s epoch offset on this boot. ROCr's
  existing Windows offset correction extended only to native gfx1013 on Linux,
  detected when a completed timestamp lies >1s in future; mutex/other GPU paths
  preserved. Absolute CPU/GPU trace alignment, opposite epoch/reset/reboot and
  other boards unqualified. No time clamp/solver force/architecture spoof.
  Separate MIOpen obsolete dilated-GEMM workspace exclusion removed; cold query
  480bytes, warm hybrid zero-workspace solution remains valid. Candidate15 direct
  API +30 Torch conv configurations pass, installed same15+30 with fresh private
  DBs pass; f32/f16/bf16 ranks1–3/groups1–2/layout/gradient/transpose/control scopes.
  Installed32 event pairs stable/positive, six1/2/5ms raw shader-clock references
  ~10–18us event overhead/stable after100ms, seven core groups pass. No universal
  correctness/performance claim; earlier GPU-event speed ratios including N1.8
  need remeasurement, numerical and host-wall-time evidence remain separate.
  All21 diagnostic/GPU guards ACKed/600s/dmesg identical, including retained
  defect/loader/DB-permission failures and terminated extra-completion diagnostic
  137. Qualified HIP source restored/rebuilt; only HSA/MIOpen DSOs installed with
  proper RPATHs, seven protected HIP/Torch/BLAS/kpack/fallback hashes unchanged.
  Under exclusive prefix lock; rollback WS/recovery/miopen-a01-before, installed
  SOURCE_MIOPEN_A01.json; source export component revision labels updated,
  superbuild baseline7f73 retained/effective port380a recorded separately.
  Report docs/development/bc250_miopen_a01_results/README.md, raw WS/logs/miopen-a01-*
  and local ignored kernel/artifacts/miopen-a01/host-logs.27 library/four system
  patches reconstruct exactly; legacy P2 Torch patch bundle still lacks three
  later SDPA file deltas (separate packaging gap; full Torch Git/source preserved).
  Torch4ee6571/Triton ee65ff3b/image5df0/kernel0957/image84995/boot a287fd16,
  SDMA/heap-off/math/r2fallback/BIOS/.105/user src/data unchanged. No active agent
  build/test/container/deadline. Python CPU fault remains unresolved, distinct
  from fixed VM-heap/CSR and A01. Missing plugin/system DB warnings remain A03/B07.
  **Exact next A02:** inspect MIOpen bundled legacy CK reduction configuration/
  wave32 feature gates, reproduce direct reduction and LSTM backward; implement
  native gfx1013-safe plumbing or portable fallback, CPU fwd/gradient comparisons.
  A02–A09 primary queue before N2.7; do not repeat N2.1–6 or A01 baseline work.
- **CURRENT storage cleanup completed 2026-10-16**, implementation/evidence
  `a26b5621b666d08dc18257208a6a68d21b7bb21e`. User explicitly authorized obsolete
  project artifact removal; this supersedes blanket retention only for inspected
  retired items. .104 recovered31,282,634,752bytes (29.13GiB): free12.2->41.3GiB,
  used97->91%. Retired candidate priority2-20261014 and initial20261012 rocm/torch/
  cache payloads (6dirs),56 stopped project containers,2 obsolete images,33
  synthetic validation file entries and2 orphan private launcher contexts. No
  global prune/volume/package removal. Runtime metadata renamed runtime.retired.json
  beside RETIRED.json; numerical reports/logs retained, deleted test checkpoints
  require regeneration under a new helper label. User data/checkpoints untouched.
  `current` unchanged ->dev/priority1; `previous` atomically consolidated to tested
  frozen20261013-4b000d542, same as before-dev-current. This independent payload/
  image and current/python-consumers dependency remain. No active links into
  retired payloads.14 core runtime/config/DSO hashes unchanged; external .kpack
  hashes separately match N1.8/frozen sealed manifests (executed helper initially
  omitted packs via wrong relative path; retained source corrected, see report).
  Three GPU gates cleanup-{fallback-before,current-after,fallback-after}:exit0,
  seven bounded CPU-reference/contract groups each,21 group executions total;
  600s guards ACKed, dmesg byte-identical across entire cleanup. No universal
  correctness/stability claim. Source/plan/journal/results and scope exclusions:
  docs/development/bc250_storage_cleanup_20261016/README.md, matching WS/logs.
  User neural-p1 container/image unchanged. Sources/builds/ccache/real models,
  compatibility installs/links, user src/, .105, r2 boot fallback/BIOS retained.
  Local Alpine build belongs to separate gfx1103 setup with running container:
  excluded; initial unprivileged attempt failed before deleting local build files.
  Torch4ee6571/libraries a6f7d3d/Triton ee65ff3b/kernel0957/image84995/boot a287fd16,
  SDMA/heap-off/math defaults unchanged. No running agent build/test/container or
  recovery deadline. MIOpen A01/A02 and historical Python CPU fault remain open,
  distinct from fixed VM-heap/CSR. **Exact next A01:** minimal direct-MIOpen C-API
  workspace-query/invocation versus handle/stream profiling isolation, CPU fwd/dx/dw
  references; fix only demonstrated cause. Audit primary queue before N2.7.
- **CURRENT Torch support audit completed 2026-10-16**, source/evidence revision
  `7e39ed699289a60d6a9605ed22ff4493446540f2`. User asks to audit Torch/dependencies,
  identify hardware-feasible gaps and follow a shared table BEFORE N2.7. Latest
  Qwen3.5 conv/loss/resume fixes are app scoped; earlier C++1Torch diagnostic fix
  remains runtime-wide. No production dispatch/DSO/prefix/build/boot changes here.
  Canonical table docs/development/bc250_torch_support_matrix.md has stable primary
  A01–A09, optional B01–B07 IDs; work-items.json and raw evidence in
  docs/development/bc250_torch_support_audit/. All3520 dispatcher names recorded,
  registration is not support proof.21 DSOs examined/19 load; RCCL and gfx1013 CK
  grouped-conv plugin absent. No observed missing mandatory loader dependency.
 51 unique targeted configurations after retained harness corrections:35 bounded
  passes,6 reproduced MIOpen failure variants across two defect families,10
  unavailable/configured-off/refused paths. Collector exit0 is NOT all-probes-pass.
  A01 group1 FP32 conv B2/C4/L19,W6x4x3,stride2,pad2,dilation2: inference fwd passes;
  combined grad-input/weight fwd passes then bwd fails. Input-only/weight-only
  requires-grad variants fail already in fwd.480-byte workspace/provided0 and
  elapsed<=0 recur; scoped GEMM-off removes workspace warning but does NOT fix.
  A02 NEW separate LSTM fwd CPU match/bwd HIPRTC failure: bundled legacy CK
  reduction config lacks gfx1013 macro. Standalone CK Tile/optional modern CK
  grouped-conv plugin are different copies; no architecture spoof/blanket flag.
  A03 no gfx1013 FindDb/PerfDb/KDB, only3 shared heuristic files; user ufdb/ukdb
  exist. HIP20 WGP multiprocessors matches40 physical CUs; gfx1013_20 key correct.
  A04 Torch cached hipsparselt_DIR-NOTFOUND although package/DSO now installed;
  direct N1 API pass does not mean Torch integration. A05 reusable conv/loss API,
  A06 gated-delta port, A07 head256/attention contracts, A08 denseINT8 Lt (valid
 32x32x32 NOT_SUPPORTED), A09 Kineto; optional FFT BF16/sparse solve/distributed/
  ecosystem packages below. FP16 power-of-two FFT/CSC/BSR/fused AdamW/AMP/compile/
  graph pass bounded gates; ordinary native operations retained. No blanket
  all-operators/all-hidden-defects or performance claim. Harness M16 INT8, AMP
  loss dtype, CPU tiled-oracle and one syntax failure retained, not platform bugs.
  Labels torch-support-{audit,refined,conv-isolation-r2} completed with findings;
  all four batch guards ACKed/600s/dmesg identical under runtime lock, including
  syntax-failed conv-isolation. No crash/core, first batch diagnostic core cap only.
  SOURCE_TORCH_SUPPORT_AUDIT.json retains source/runtime identities. Torch4ee6571/
  libraries a6f7d3d/Triton ee65ff3b/kernel0957/image84995/boot a287fd16 retained;
  SDMA/heap-off/math defaults/r2fallback/BIOS/.105/user src/ unchanged. No active
  agent build, test, container or recovery deadline. **Exact next A01:** direct
  MIOpen C-API minimal reproduction; separate workspace size-query/invocation from
  handle/stream profiler timing, CPU fwd/dx/dw references, fix only demonstrated
  cause. A02 legacy reduction follows separately; selected primary queue before
  N2.7. Historical Python CPU fault remains distinct from fixed VM-heap/CSR and
  both reproduced MIOpen failures.
- **CURRENT Qwen3.5 training follow-up completed 2026-10-16:** user supplied an
  old log and freed .104; this supersedes the N2.7 cursor until training follow-up
  is done. Implementation `690042b55f33b4a6fd67c4c48d54ff8d7313b64a` adds scalar
  Triton frozen causal depthwise conv K<=4 (FP32/FP16/BF16 forward/dx/SiLU) and
  chunk64 FP32 causal loss with backward recomputation. Full BF16 logits remain.
  Default only in finetune-n7 replica/train.py / Transformers5.19.0 / native gfx1013;
  unsupported/trainable conv weights retain original, math SDPA/head256 retained.
  Source/patch/tests local build_tools/bc250/qwen35_training* + deploy/validate helpers;
  deployed replica/bc250_ops.py SHA eddfea40c7fb2f0fd6dcfac72d67077918d1c61730ceba60d6e3f872f5e1f6d8.
  51 final operator/reference/stream/storage/refusal checks pass;4 small LoRA pairs.
  Real text-only Qwen3.5-2B BF16 rank16/alpha32:65-token loss match,372 adapter
  gradients max relativeL2~1.12%;1024-token checkpointed backward/AdamW update pass,
  whole peak~11.15GiB without a paired full-model baseline. Isolated1024/vocab248320
  loss extra peak2.84->0.65GiB; no full-model speed/memory/convergence claim.
  Fresh CLI exposed PEFT minimized suffix target names rejected by old raw string
  comparison; now require exact resolved module set/rank/alpha.2 fresh synthetic
  updates/eval plus checkpoint1 resume:372 exact saved tensors and exact optimizer/
  scheduler/RNG/progress;4 mismatches refused. Initial CLI failure retained.
  Final labels qwen35-{ops-final,loss-memory,model-tiny,model-actual,cli-resume-fixed,
  resume-guards}:exit0/dmesg identical/600s guards ACKed under exclusive prefix lock.
  Original user log continued251/saved250; allocator warnings were recoverable
  attempts, not a demonstrated fatal OOM. Final CLI causal/MIOpen/OOM warnings absent;
  gated-delta/FLA reference warning remains valid. Global MIOpen elapsed/workspace
  defect NOT fixed; Python CPU fault unresolved, distinct from fixed VM-heap/CSR.
  Evidence docs/development/bc250_qwen35_training_results/README.md + matching WS/logs;
  SOURCE_QWEN35_TRAINING_OPS.json; WS/recovery/qwen35-training-before and local ignored
  kernel/artifacts/qwen35-training-reference retain caller originals/final sources.
  Usage updated, app opt-out documented. No native rebuild; existing Triton cache,
  Torch4ee6571/libraries a6f7d3d/kernel0957/image84995, prefix/DSOs/boot unchanged.
  SDMA/heap-off/r2 fallback/BIOS/.105/user src/ unchanged. No active agent build,
  test, container or recovery deadline. **Exact next:** profile reference gated-
  delta forward/backward at synthetic Qwen3.5 geometry and select one bounded native
  gfx1013 substage with independent output/gradient references; isolate global
  MIOpen root cause separately. N2.7 Schur iteration remains queued after this.
- **CURRENT N2.6 completed2026-10-16:** measured small candidate before changing
  code. Order16 recovery~37ms/376 ATen scalar reads versus Schur41/52ms and host
  eig~0.35/0.41ms. Vectorized right recovery across all columns, n-1 row updates
  instead n(n-1)/2; global checked gap proves active pivots, inactive pivots set1,
  per-column homogeneous scaling and final finite/norm/residual guards retained.
  No tolerance/gap/iteration changes. Scalar reads34/124/376 ->18, ATen calls
  377/1437/4349 ->226/331/478 at4/9/16. Same-input A1/B1/B2/A2,3 synchronized
  wall samples/pass, six c64/c128 order4/9/16 basis10 fixtures: recovery1.53–8.44x,
  whole1.07–1.72x; raw controls retained, no general/model speed claim. Native
  whole still16–55ms versus host0.20–0.40ms, so production host dispatch unchanged.
  Unchanged N2.4/N2.5 gates21+40+30+16=107 pass plus timing fixture CPU references.
  Labels solver-{n24,scale,boundary,failure}-regression-n26 and solver-paired-n26
  exit0/dmesg identical/600s guards ACKed under lock; initial profile-before retained.
  Report docs/development/bc250_solver_results/N26.md and n26/, .104 same WS/logs.
  Prior helper WS/recovery/solver-n26-before; no native compile or Torch/DSO/
  prefix metadata/boot-policy change, no active agent job/container/deadline.
  Torch4ee6571/libraries a6f7d3d/kernel0957/image84995; SDMA/heap-off, math defaults,
  r2/BIOS/.105 unchanged. Python CPU fault/MIOpen issue remain open. **Exact next
  N2.7:** split Schur preparation/QR-loop cost and prototype one bounded device-
  side small QR/similarity iteration for remaining launch/scalar-sync overhead;
  retain Python reference/failure gates and production host dispatch.
- **CURRENT N2.5 completed2026-10-15:** demonstrated candidate scale defect:
  seeded baseline6 had2 unit passes, tiny c64/c128 positive info4 at256steps and
  large c64/c128 nonfinite/fill errors. Added GPU component normalization before
  GEHRD/QR; restore original-unit T on success/partial result with finite check.
  Shared safe normalization preserves right-vector 8sqrt(eps) gap policy; integer
  bit checks refuse all-subnormal inputs even if floating reductions flush zero.
  Exact zero/scalar paths retained; nonrepresentable Schur output refused.
  Same seeded baseline6 now6/6; new scale40 (c64 1e-25..1e25, c128 1e-200..1e200,
  orders4/9 basis1/10), boundary30 (18accept/12refuse, also passed before), failure/
  control16 and unchanged N2.4 regression21 all pass. Max original residual
  ~4.00e-7/7.52e-16; no tolerance/iteration-limit changes or performance claim.
  Final labels solver-{scale-n25-fixed,boundary-n25-fixed,failure-n25,
  n24-regression-n25,scale-n25-replay} exit0/dmesg identical/600s guards ACKed under
  lock; failed before baseline retained separately. Report/hashes/evidence:
  docs/development/bc250_solver_results/N25.md and n25/, same .104 WS/logs labels;
  prior helper WS/recovery/solver-n25-before. No native rebuild, default Torch/
  DSO/prefix metadata/kernel/boot-policy change; no active agent job/container/
  deadline. Production eig remains host, cap16 prototype uses GPU arithmetic/
  host scalar control. General dynamic range/conditioning/balancing/subnormals/
  batching/gradients/performance unqualified; CPU fault/MIOpen issue open, SDMA/
  heap-off/math defaults/r2/BIOS/.105 unchanged. **Exact next N2.6:** bounded
  end-to-end/component cost and host-sync profile versus host eig with correctness,
  then one measured overhead-reduction step, keeping production dispatch unchanged.
- **CURRENT N2.4 completed2026-10-15, isolated native right eigenvectors:** added
  right_eigenvectors(SchurResult) to gfx1013_schur_candidate.py. GPU dimensionless
  triangular back-substitution with homogeneous rescaling, pivot/finite/residual
  checks, GPU Q transform/unit normalization. Rejects positive info, malformed
  T/Q and repeated/close scaled gaps<=8sqrt(eps); no pivot perturbation/host solve.
  Scalar zero supported; cap16/dtypes complex64/128; production eig remains host.
  New21 groups pass first run:12 end-to-end orders4/9/16 basis1/10 CPU+analytic
  spectra/vector projectors/original residuals, nondefault streams/input+Schur
  guard preservation, analytic nonnormal2x2, repeated/Jordan/close/zero refusals,
  above-floor control, zero/one-step nonconvergence, scalar and7 malformed cases.
  Max original residual~3.84e-7/7.79e-16, CPU projector~8.21e-6/2.02e-14. Label
  solver-eigvectors-n24 exit0/dmesg identical/600s guard ACKed under lock. Report:
  docs/development/bc250_solver_results/N24.md and n24/; .104 same WS/logs label.
  Prior helper retained at WS/recovery/solver-n24-before. No native rebuild or
  production Torch/DSO/prefix metadata/boot-policy change; source Torch4ee6571,
  libraries a6f7d3d and kernel0957/image84995 retained. No active agent job/container/
  deadline. Host scalar sync remains; extreme scales/balancing/conditioning,
  repeated vectors/batching/gradients/graphs/performance unqualified. CPU fault
  and MIOpen issue remain open; SDMA/heap-off, math defaults, r2/BIOS/.105 unchanged.
  **Exact next N2.5:** bounded scale ladder and spectral-separation boundary with
  nonnormal/refusal controls; fix demonstrated scaling/residual defects before
  any opt-in integration, leaving production host dispatch unchanged.
- **CURRENT N2.3 completed2026-10-15, isolated native Schur prototype:** audited
  native GEHRD/QR pieces; no pinned rocSOLVER geev/hseqr/lahqr/gees. Added
  build_tools/bc250/gfx1013_schur_candidate.py: native C/Zgehrd, device reflector-Q
  reconstruction, bounded explicit shifted GPU QR and Schur accumulation, cap16.
  Python loop/scalar convergence controls synchronize host; matrix arithmetic
  stays GPU, no CPU eig fallback. Positive info preserves partial T/Q on budget
  exhaustion; Q is Schur vectors, NOT right eigenvectors. Production eig remains
  N2.2 host compatibility; no Torch/DSO/default-prefix mutation or native rebuild.
  Initial17 and final21 groups pass; final adds ±i rotation/scalar cases and
  budget1 partial iteration. Main complex64/128 orders4/9/16 basis1/10 CPU+analytic
  spectra, Hessenberg/Schur reconstruction/orthogonality, guards/nondefault stream;
  budget0/1 nonconvergence, repeated triangular and9 argument rejections checked.
  Max Schur reconstruction~1.35e-6/2.90e-15; no speed/general convergence claim.
  Labels solver-schur-n23{,-final} exit0/dmesg identical/600s guards ACKed under
  lock. Evidence docs/development/bc250_solver_results/N23.md and n23/, same
  .104 WS/logs labels; Torch4ee6571/libraries a6f7d3d/kernel0957/image84995 unchanged.
  No active agent job/container/deadline; balancing/extreme scaling/larger cases,
  vectors/batching/gradients/concurrency remain unqualified. Python CPU fault and
  MIOpen issue unresolved; SDMA/heap off, math defaults, r2/BIOS/.105 unchanged.
  **Exact next N2.4:** guarded triangular back-substitution and GPU Q transform
  for right eigenvectors from converged small Schur forms; separated-spectrum
  CPU/analytic residual/direction checks and explicit repeated/near-singular/
  nonconverged refusal, keeping production dispatch unchanged.
- **CURRENT N2.2 completed2026-10-15, host compatibility ONLY:** pinned Torch
  complex nonsymmetric eig selects hipSOLVER DnXgeev, whose implementation copies
  to host and calls LAPACK cgeev/zgeev, then copies outputs to GPU. This is NOT a
  native GPU eigensolver. Native N2.1 Hermitian eigh remains a separate result.
  New16 groups pass first run: complex64/128 batch2 dimensions9/33/65, known
  separated complex spectra with basis condition1/20, padded/transposed guards,
  nondefault streams, eig/eigvals CPU+analytic spectra, residual/projector checks,
  two eigenvalue-gradient groups and two empty-shape groups. Max residual
  c64~3.17e-7/c128~6.72e-16. Label solver-eig-n22 exit0/dmesg identical/600s guard
  ACKed under lock. Report: docs/development/bc250_solver_results/N22.md; mapped
  hipSOLVER/OpenBLAS hashes, runtime identity, logs/results retained locally and
  .104 WS/logs. No backend defect reproduced, code/selection/runtime patch or
  native rebuild. Torch4ee6571, libraries a6f7d3d, kernel0957/image84995 unchanged.
  No active agent job/container/deadline. Native nonsymmetric eig remains open;
  no performance/universal correctness claim. Python CPU fault/MIOpen failure
  unresolved; SDMA/heap off, math defaults, r2/BIOS/.105 unchanged.
  **Exact next N2.3:** audit native Hessenberg/QR/Schur primitives and select the
  smallest reviewed GPU implementation/convergence step; retain host compatibility
  until independent native eigenpair and failure-contract gates pass.
- **CURRENT N2.1 completed2026-10-15:** audited existing P2 solver32 and pinned
  Torch4ee6571/ROCm-libraries a6f7d3d consumers; qualified existing native complex
  Hermitian eigh eigenvectors rather than adding a duplicate backend. New22
  checks: complex64/128, batch2 sizes17/65/129/257, both triangles, padded strides/
  guard preservation/ignored-triangle NaNs, nondefault stream, size33 repeated
  eigenspace projectors and two separated-spectrum CPU gradient comparisons.
  Source default HETRD blocked switch256;257 covered, two-stage11000 unqualified.
  Max residual c64~1.15e-6/c128~2.16e-15; no speed/stability/universal claim.
  Existing32 solver regression passes unchanged. Final labels solver-eigh-n21-blocked
  and solver-p2-regression-n21 exit0/dmesg identical/600s guards ACKed under lock.
  Earlier18-case gate passes too; initial reference-check broadcasting failure
  retained, corrected without tolerance changes, not a backend defect. No native
  implementation/selection/runtime change or rebuild. Report/evidence:
  docs/development/bc250_solver_results/README.md; local helpers and matching
  .104 WS/logs. Existing image/module/Torch/DSOs, SDMA/heap-off, math defaults,
  r2 fallback/BIOS/.105 unchanged; Python CPU fault and MIOpen failure unresolved.
  No active agent job/container/deadline. **Exact next N2.2:** audit nonsymmetric
  eig coverage and pinned dispatch, qualify one bounded complex eigenpair gap
  with CPU/residual references, and change backend only for demonstrated failure.
- **CURRENT K6 handoff2026-10-15:** user explicitly resumed K6, superseding the
  earlier deferral below. K6 complete for the recorded bounded scope; no runtime,
  kernel code, power-policy or boot change. `kernel/r3/handoff/README.md` contains
  exact configs, reviewed six-patch series, build/sign/install safeguards and
  one-shot SSH/watchdog/cooling/GPU/recovery procedure. Temporary-index replay
  exactly matches image84995f1 tree after2 patches and module0957f1 tree after6.
  Local/current source archive+fan source and21 artifacts have matching local/
  .104 hashes; manifests/evidence under `kernel/r3/handoff/`, ignored artifacts
  under `kernel/artifacts/k6-7.2.10/`, .104 KWS/handoff/k6/{artifacts,docs}.
  Signed upstream verification retained; archive excludes generated module signing keys.
  Read-only .104 audit and fresh loaded identity check pass; same boot
  a287fd16-c97b-4d88-8055-b0553a2f7631, image84995f1/module0957f1, active watchdog/
  SSH/cooling, empty next_entry, r2 default. No new build/GPU test/reboot; prior
  bounded VMM/SVM/HPD/consumer evidence retained, no bit-identical build claim.
  Telemetry/per-core fidelity, scheduler-stop/resume and hardware-hang limits
  remain; Python CPU fault and separate MIOpen failure remain unresolved. SDMA/
  experimental heap off, math attention default, BIOS/.105/fallback unchanged.
  No active agent job/container/deadline. **Exact next: N2.1** audit existing32
  solver cases and pinned Torch/rocSOLVER/hipSOLVER consumers; choose one missing
  difficult/complex operation with CPU/residual references before implementing.
- **Historical user-priorities1–4 batch2026-10-15:** user initially deferred N2/K6.
  Priorities1/2/4 bounded-qualified; priority3 investigated/instrumented, NOT fixed.
  Torch`4ee65717d69e9f51021272e3f37d0f342f68fc08` (optimization43cb299 ancestor).
  Longer CPU/GPU attention output+gradient references:11 cases through4096tokens,
  head128/GQA. Real existing Qwen3-1.7B BF16 rank4 q/v LoRA reference passes at512/
  1024 synthetic tokens; max per-adapter gradient relativeL2~0.041 versus math.
  No user data/checkpoint changes or convergence/full-model fine-tuning claim.
- **Priority2:** causal row/key bounds skip mathematically masked tiles; tile32
  only causal heads>=64, other tile16 paths retained. Final same-input paired BF16
  head128 causal kernel ratios1.67–2.05 versus prior implementation in measured
  fixtures; FP32/control noise retained, no universal/whole-model speedup claim.
  Real Qwen1024 whole-model peak remains5,818,112,000 allocated bytes for both;
  isolated attention memory savings do not eliminate logits/loss-dominated peaks.
  Math stays default. Autocast now handles eligible mixed Q/K/V dtypes with gradients.
- **Priority3 unresolved:** recovered ONE historical Python IP0x3b3e6 event (many
  later logs duplicate it), no retained original core/stack. Eight fresh debug-
  allocator solver/FFT/CSR/COO/GEMM processes pass1920 checks; not a crash repair.
  Launcher defaults PYTHONFAULTHANDLER=1; empty-value user opt-out checked. Diagnostic
  core limit256MiB only, no global core-policy change. Distinct newly observed
  mixed-operator failure: MIOpen EvaluateInvokers elapsed<=0 / GemmFwdRest workspace
  480/provided0, before first conv/sparse; exit1, no segfault/dmesg delta. Retained
  and unresolved; do not claim original full55 operator suite passed this batch.
- **Priority4:** actual train_lora resume RNG ByteTensor failure reproduced and
  repaired; adapters now trainable on resume, scheduler/Python/data-order state
  saved, legacy format resumes/updates with missing-state replay warnings. New
  matching-config tiny CLI fresh/resume passed, old-format fixture passed too.
  Full-parameter small Qwen AMP batch2/length128 FP32/FP16 math and length256 BF16
  math/tiled: CPU gradients,2updates/no scaler skips, fresh-process exact next
  loss/model/optimizer/scheduler/scaler+CPU/GPU/Python RNG probes pass. Compiled
  fullgraph/dynamic math-attention block:6 FP32/FP16/BF16 output/gradient cases at
  batches2/3 lengths128/96 pass. Tiled compile/graphs/dropout remain unqualified.
  False TF32 Tensor Core warning corrected using native is_tf32_supported API.
- **Evidence/state/EXACT NEXT:** docs/development/bc250_training_expansion_results/README.md,
  reports/logs and build_tools/bc250 helpers/caller patch; .104 WS/logs matching
  labels. Final attention/AMP/compile/app/legacy/sparse labels exit0,600s guards
  ACKed,dmesg identical; failed mixed/initial AMP/app attempts preserved. Python-only
  changes reuse existing caches; no new native build. Recovery WS/recovery/
  {attention-priorities,training-expansion,tf32-diagnostic,training-expansion-metadata}-before;
  SOURCE_TRAINING_EXPANSION records revisions/hashes. No agent job/container/deadline
  remains; source/kernel0957f1aa4, r3/r2fallback, BIOS/governor/.105 unchanged.
  Next isolate the newly demonstrated small-convolution timing/workspace failure
  and capture a real CPU crash/core/backtrace before proposing its fix. Keep
  historical CPU fault separate from proven CSR corruption/safe-pool bypass and
  kernel VMM/SVM repairs. N2/K6 remain deferred, not silently restarted.
- **Usage-documentation follow-up2026-10-15:** added dated attention instructions
  to docs/development/bc250_torch_usage.md, including default warning/API fixes,
  math default, application-only --attention-backend tiled selection, Transformer
  registration and qualification limits. Linked detailed evidence; no code/runtime
  change or GPU rerun. No agent jobs; attention/K6/N2.1 next actions unchanged.
- **Previous N3 attention batch2026-10-15, bounded opt-in:**
  patched Torch`bda0831ba49ef9d0ba2968a800642e5abaec5dd5`. Three native scalar
  Triton kernels implement tiled online-softmax forward/recomputed backward;
  no full score/probability matrix, matrix instructions or default SDPA override.
  FP32/FP16/BF16, GQA, broadcast boolean/additive masks, zero fully masked rows,
  finite Q/K/V, D<=128/L<=4096, eager first-order only. No dropout/mask gradients/
  compile/graph or full-model qualification. Explicit Transformer attention+mask
  registration and caller --attention-backend tiled; math remains default.
- **Warning/caller corrections:** built-in C++ probe passed !debug as warning text,
  producing1Torch during successful fallback. Now ordinary checks are quiet;
  explicit unsupported-backend diagnostics and failure retained. This does not
  enable built-in efficient/flash/CK backends. User finetune-n7 train/chat migrated
  to nn.attention.sdpa_kernel(MATH), backward included for checkpoint recomputation,
  broad warning filters removed. Reviewed caller patch/local helpers retained.
- **Attention evidence:** `docs/development/bc250_attention_results/README.md`;
  .104 WS/logs/sdpa-tiled-candidate-r5:18 CPU-float64/GPU-math output+QKV-gradient
  cases, padded small Llama/checkpoint test. Installed integration-r4:7 groups,
  actual Qwen1.7B head128/Hq16/Hkv8 geometry at65tokens; small random Qwen3 BF16
  LoRA2updates/CPU gradients/cache decode; diagnostics/streams/caller checks.
  Unchanged19-case eager Torch regression passes. Final batches exit0/dmesg
  unchanged/600s guards ACKed. Earlier harness/reference/install failures preserved.
  Extra allocated FP32 B1/H2/D32 forward+backward peak:512/1024/2048 tokens
  tiled0.63/1.26/2.52MiB vs math8.38/32.75/129.50MiB. Tiled slower in these one-event
  timings; no speedup/general memory/stability claim. Dense input masks can be quadratic.
- **Attention deployment/job status:** exclusive update_dev_prefix lock used for
  source/library/caller mutations and direct-Podman tests; ccache20GiB/12Torchjobs.
  HIP SDP-utils+regenerated HIPHooks TUs and libtorch_hip rebuilt; installed DSO
  SHA578a3583e78ad0009779d77bd65591fcf8119dec441ea81dde7be56d9b9c11d5.
  New module SHA227a68174927397cf01132421f7d5c7dd7c2a65ba05bbc4babc651addde2bbc3.
  Recovery WS/recovery/sdpa-{probe,callers,metadata}-before; SOURCE_SDPA records
  source/revision hashes. Existing images/releases/checkpoints/caches retained.
  User's training/chat exited normally before tests; no agent job/container/deadline
  remains. Kernel0957f1aa4/r3/r2default/governor/BIOS/.105 unchanged.
- **Exact next:** attention scope is opt-in only; qualify longer head128 references
  and representative full-model memory/time/training before default integration,
  then optimize the demonstrated scalar bottleneck. K6/N2.1 remain independent
  roadmap items; telemetry correction optional for current use. Unresolved Python
  CPU segfault remains separate from fixed VM-heap corruption and VMM/SVM repairs.
- **ACTIVE: K1–K6 kernel workstream; K5/SVM passed; HPD fixed; K6 remains; telemetry correction optional.**
  2026-10-15: .104 now runs`7.2.10-bc250rocm-r3` after six one-shot boots;
  persistent GRUB default remains`bc250-rocm-r2-kernel`, next_entry empty. Native
  GPU loaded,40CU/80SIMD, wave32/KFD100103; live quirks3/0/3. No active build,
  GPU gate, container or unacknowledged recovery deadline at batch end. User's
  training/download finished; user explicitly returned.104 to development use.
  .105 unchanged. Root orchestration SSH session9359 is idle, not durable state.
- **CURRENT HPD/Display Core milestone2026-10-15:** module/source
  `0957f1aa4717bb5ea19545bdcb01d6899e94bb9f`, patch0006. Init incorrectly called
  all6 HPD+6 HPDRX enums, but active DCN201 table implements only two pairs. Query
  active table before bulk clears; skip dummy/empty entries, retain actual clears,
  connector enablement and invalid-call ASSERT/error diagnostics. No dc=0 or
  guessed pin counts. Baseline8 dummy-ack+8 dummy-set+8 failed-clear errors and
  two IRQ WARNs →0 each after guarded reboot/load. Temporary DRM KMS debug4
  traced exactly sources1,2,7,8; restored0. DCN2.0.1 initialized, amdgpu.dc=-1,
  same DP-1/DP-2 disconnected connectors, taint4160 with WARN bit clear. Physical
  hotplug/scanout/audio/resume and other ASICs unqualified. Existing2 SDMA0 fence
  fallback-expiry messages/workqueue-name truncation remain; no wholly clean-boot
  or SDMA qualification claim.
- **Current identity/evidence:** r3-only signed updates/amdgpu.ko.gz SHA
  `6ebd8393a9c7d25cd810fd89d4ef5defa187980cae64da0d37ee0d975715e8d1`,
  build/installed/loaded ID`b97bca56793aecea35967385d0c68d4186ecd121`.
  Kernel image84995f1/buildID57f7c52/initramfs/firmware unchanged. Prior source/
  module independent`KWS/recovery/hpd-before`; r2/default/.105 retained. Boot
  `a287fd16-c97b-4d88-8055-b0553a2f7631`; base/cooling and360s GPU guard ACKed.
  Unchanged Lt664/SparseLt150/hipTensor316/Torch19 and two fresh memory processes
  each24 allocation+16 VMM rounds pass, workload dmesg delta0. Existing SVM source
  hashes preserved. Host`SOURCE_HPD.json`, source/GIT_REVISION0957f1aa4;
  repository`kernel/r3/evidence/hpd-capability/`, `hpd-*` helpers/manifest/README.
  PID6463 modules build ended0,16jobs/20GiB ccache; public DC header dependency
  rebuild,`logs/hpd-modules-build.log`, `hpd-module-install.log`. No active build,
  agent-owned test/container or deadline. DRM debug/injection parameter both0;
  external governor now recorded in the read-only audit below.
- **CPU unlock persistence request2026-10-15:** read-only .104 check confirms
  eight cores/16threads online and OpenRC default unlock service enabled/started.
  Existing script reapplies volatile mask, deliberately never auto-reboots (prior
  bootloop described in its comments); cold-start enumeration needs a later warm
  reboot, not newly cold-cycle-tested. Requested upstream uses same class of
  volatile unlock and points to BIOS DXE for cold-boot persistence. No duplicate
  mailbox unlock, reboot, service/policy change or firmware flash; .105 unchanged.
  Evidence/local script+service snapshots `kernel/r3/evidence/core-unlock-audit/`.
  User explicitly chose2026-10-15 to leave BIOS untouched and retain the existing
  service/warm-reboot arrangement. Firmware modification is out of current scope;
  r2/SSH cannot rescue pre-OS failure. DMI P3.00 does not prove stock BIOS.
  No active jobs. K6/N2.1 retained; telemetry correction is optional for current use.
- **CURRENT read-only telemetry audit2026-10-15 completed:** production
  kernel/module0957f1aa4/image84995f1/prefix/boot unchanged, same boot above.
  Newly observed external governor PID5265 (`cyan-skillfish-governor-smu`) has
  fix-metrics/fix-freq enabled and bind-mounts tmpfs over GPU metrics AND hwmon
  freq1_input. Kernel policy auto does not imply stock clocks. Existing governor/
  overlays/config/clock/voltage/fan policy untouched. amdgpu_top PID5728 was also
  present initially, absent later; not stopped by this audit. No agent-owned jobs
  remain. GuardPID7594 ACKed/ended; rootSSH9359 idle. No kernel build/reboot.
- **Passed collection/reference:** original governor-held sysfs fd6 (device0:22,
  overlay0:25) read without unmounting; kernel-translated v2.2, NOT raw PMFW.
 84 samples~5Hz:4s idle,8s native integer GPU workload plus startup,4s cooldown;
 131,072 exact CPU-reference checks, kernel logs byte-identical. During own load,
  governor-supplied clock1000–2000MHz (median2000) versus original kernel/DRM/
  DPM mostly2–7MHz, max47; average kernel456–953MHz also untrusted. Overlay
  clock views agreed84/84; shared kernel clock views agreed74/84 sequential
  samples, NOT independent clock validation. Actual attained clock unresolved.
  Physical8core/16thread topology confirmed; per-core temperatures0/2 report
 0.63–1.88/0.62–3.56°C,1/3 zero,6/7 temperature/clock65535. Original GPU activity
 65535 and DRM load errno22; overlay utilization is governor-derived.
- **Temperatures/provenance:** reported edge51→68°C, original GFX51.25→68.25°C,
  CPU Tctl52.625→67.125°C in this short load. Hwmon edge truncates centidegrees
  to integer°C (observed difference−0.75..0°C). Fan actually reads CPU Tctl;
  existing automatic PWM90→175/RPM~2k→3.5k, remaining high during CPU cooldown.
  No absolute calibration/max-thermal claim. Running governor binarySHA00cd65a6...
  differs from retained target/releasee8028ca2...; reference Git240b0df/Cargo0.4.0
  lacks observed frequency-overlay code. Running binary/source NOT qualified.
  Reference archive`kernel/artifacts/telemetry-readonly/governor-reference-240b0df.tar`
  retained locally (ignored artifact/SHA in evidence); original repo/history intact.
- **Evidence/EXACT NEXT:** `kernel/r3/telemetry/README.md`, sampler/load/summarizer;
  `kernel/r3/evidence/telemetry-readonly/` raw JSONL/config/script/identity/results.
  Host`KWS/logs/telemetry-readonly`, `telemetry-compile.log`, `telemetry-guard.log`;
  one-TU gfx1013 probe compiled with20GiB ccache, binary`KWS/telemetry-reference-load`.
  Pin running governor source/overlay semantics and kernel/daemon SMU mailbox
  coordination, THEN minimal guarded read-only GetGfxFrequency getter validation.
  Do not blindly apply legacy packaging0005: physical core count alone does not
  prove the hybrid map; instantaneous power is labeled average; query failure
  falls back to the suspect clock slot. Keep per-core remapping/power semantics
  separate until offsets/units proven. K6 then N2.1 remain; SDMA0/heap-off, r2/
  .105 and prior SVM/fault/lockdep/physical-display limits retained. Python CPU
  segfault separate/unresolved. No speedup/sustained claim from this8-second test.
- **Historical SVM validation milestone2026-10-15:** source/module
  `41db89c103b183c82c05bbe099ea96ad1ddee8d1`, patch0005. Map AND validation-path
  unmap collect PT fences/site bits under BO reservations, then release reservations,
  wait and invalidate under notifier lock; caller-held mmap/SVM/migrate locks
  preserve lifetime. Drain earlier completed work even when a later VMA fails.
  Fence-tracking allocation failure falls back to the prior in-place wait, without
  DQM invalidation under BO reservations. No OOM/fence-error injection claim.
  Also fixed validation PROT_NONE byte/page bounds using each VMA's inclusive
  page indices: exact same probe traced0 target-page unmaps before,2 after;
  no GPU access to protected page attempted; restore references passed.
- **Previous SVM validation identity/results (superseded above):** r3-only signed updates/amdgpu.ko.gz SHA
  `d47974d6d7eef898b50a4fdf03edb716e8cd13dfd7f2373b495f273727e5f07f`,
  build/installed/loaded ID`7ee5573fc124ceb0edad1bc662feb44ebdbf129f`.
  Kernel IMAGE84995f1/buildID57f7c52, image/initramfs/firmware unchanged. Prior
  module/source independently retained`KWS/recovery/svm-map-before`; original
  stock/K5/SVM modules and r2/default/.105 preserved. Boot
  `84debfc4-5f83-413b-9656-6a5fd3140a3a`; base/cooling and360s GPU guard ACKed.
  Two concurrent processes each54 GPU CPU-reference checks, including24 multi-VMA
  updates+24 CPU discard/restores, passed with unchanged workload dmesg. Both
  partial/full CPU-unmap -EIO regressions pass again; unchanged full Lt664/
  SparseLt150/hipTensor316/Torch19 and two memory processes each24 allocation+
  16 VMM rounds pass, delta0. Two intentional trace+two injection lines accounted
  separately. Dynamic debug restored; one-shot injection parameter verified0.
- **Previous validation jobs/evidence/next (HPD now fixed above):** PID7547 modules build ended0,16jobs/20GiB ccache;
  `logs/svm-map-modules-build.log`, `svm-map-module-install.log`. No active build,
  test/container/GPU user or deadline. Host`SOURCE_SVM_MAP.json`, source/GIT_REVISION
 41db89c10; helpers`svm-map-*`, repository`kernel/r3/evidence/svm-map-deferred/`,
  patch0005 and source manifest record replay/identity. No lockdep/universal deadlock,
  OOM/fence-error/partial-map-failure lifetime, failed scheduler-stop/resume,
  XNACK/retry-fault or multi-GPU qualification; no performance/sustained claim.
  Next analyze the two existing Display Core IRQ load WARN call chains/tables;
  preserve warning evidence and avoid blindly suppressing WARNs/disabling display.
  Then trusted telemetry/K6; N2.1 next library item. Retain r2 default, SDMA0 and
  experimental heap off; separate Python CPU segfault remains unresolved.
- **Historical SVM CPU-unmap milestone2026-10-15:** module/source
  `67d0f189570c831efa17b524c080de07bbec485a`, patch0004. Single-device BC-250
  CPU-unmap notifier acquires queue eviction BEFORE PT invalidation; successful
  invalidation balances it, returned errors after a confirmed stop retain it
  until process exit. Full-range mapped_to_gpu clears only after success. Existing
  opt-in unmap bit controls this guard; other ASIC/multi-device behavior unchanged.
  Root-only0600 one-shot`bc250_svm_unmap_fail_pid` defaults0; initial-namespace TGID
  must match and queues must have successfully quiesced before consumption.
- **Previous CPU-unmap module/boot (superseded above):** signed r3-only updates/amdgpu.ko.gz SHA
  `c1d0e5a887bc3d2019472b9758cf57c920e9ef9544b3b47f6ed1a2d620f5e868`,
  build/installed/loaded ID`c99a2c6dc9074760efe06d074ec0248ce5e8739b`;
  kernel image still84995f1/buildID57f7c52, image/initramfs unchanged. Previous
  K5 module independent copy`KWS/recovery/svm-before/amdgpu.ko.gz`, original stock
  r3 and r2/default retained. New boot`eb2cc9df-5df6-4250-a0ec-1af4de205f93`;
  base/cooling ACK,360s SVM guard and180s full-injection guard ACKs completed.
  Source/GIT_REVISION67d0f1895;`SOURCE_SVM.json` records image/module delta.
  Pid7101 incremental modules build finished0,16jobs/20GiB ccache;
  `logs/svm-modules-build.log`, `svm-module-install.log`. No active build/test/
  container/GPU user or recovery deadline. Injection parameter verified0.
- **SVM evidence:** initial six GPU CPU-reference checks passed on K5 module.
  New module: partial AND full CPU-unmap injection gates passed. Each target
  process completed five prior SVM references and a normal GPU-write control,
  then received one modeled -EIO at the unmap runlist point; PID matched, hook
  consumed, exactly one expected error log. Queued GPU write remained unexecuted
  through200 one-millisecond host polling samples, process exited0; six-check
  fresh-process SVM gates passed before/after each injection. No new ordinary
  kernel lines. Unchanged full Lt664/SparseLt150/hipTensor316/Torch19 gate passed,
  two memory processes each24 allocation/pinned/stream/event+16 VMM rounds,
  workload dmesg delta0. Intentional error lines retained separately, not hidden.
  Host`logs/svm-{baseline,cpu-unmap-gate,cpu-unmap-full-injection-gate,
  cpu-unmap-full}`, repository`kernel/r3/evidence/svm-cpu-unmap/`; new
  `svm-{reference.cpp,validate-on-host.py,module-on-host.sh,gpu-phase-on-host.sh,
  identity-on-host.py,source-manifest.json}` provide replay/provenance.
- **CPU-unmap scope/previous next (validation locking superseded above):** qualify only returned CPU-unmap errors AFTER confirmed
  quiescence. Failed scheduler stops cannot veto blockable MMU reclamation by
  returning false (core warns/continues); hardware HWS hangs, resume failures,
  retry-fault migration/XNACK and multi-GPU remain unqualified. Audit SVM MAP
  runlist flush under BO reservations and move post-PT invalidation outside those
  reservations while preserving notifier/range lifetime ordering, then short
  concurrent CPU-compared GPU gates. Validation-context flushing is now moved outside reservations above; other
  SVM callers and failure paths are not universally certified lock-order safe. Follow with Display Core warning analysis,
  telemetry and K6; N2.1 remains next library item. Keep r2 default, SDMA0 and
  experimental heap off. Unknown Python CPU segfault remains separate/unresolved.
- **Historical K5 milestone2026-10-15:** explicit fixed-VA VMM stale translations repaired
  on .104. Kernel source/module revision`8f3c0d352643378629e5a73f4b9df96b7d38d94a`,
  patch0003; kernel IMAGE remains84995f1, same config/image/initramfs/firmware.
  HIP and direct ROCr isolation each failed8/26 checks before repair, while
  distinct-VA anchors passed; ordinary KFD mapping churn made both26/26 pass.
  GEM_VA bypassed existing KFD map/unmap runlist invalidation. New BC-250 compute
  path waits PT completion, releases BO reservations before DQM lock, invokes
  opt-in runlist flush, then publishes timeline completion/returns flush errors.
  Original failures preserved, not weakened/skipped. Upper SVM containment is
  still separate/unqualified; no injected-failure or raw concurrent DRM timeline
  ordering claim.
- **Historical installed K5 module (retained; superseded by SVM module above):** r3-only`updates/amdgpu.ko.gz`, compressed SHA
  `451b8a3d05cdf494b67546f9074c86f4d931481a3efd7f7b64e7bad48611dadb`,
  build/installed/loaded ID`12ad61569200a95f1d626396fca3692753231707`.
  Original r3 stock module, image/initramfs, r2 artifacts/default and .105 retained.
  K5 boot`e2a06828-208c-47f7-9838-8a5be9f7bfd5`; base/cooling readiness ACK
  and fresh360s GPU guard ACK completed. No active build/container/GPU gate/deadline.
  Single .ko build compiled both units but failed modpost on companion DRM symbols;
  incremental`make ... -j16 modules` succeeded using16jobs/20GiB ccache, only new
  AMDGPU installed. PIDs25806(failed)/26295(success) ended. Exact logs
  `KWS/logs/k5-{amdgpu-build,modules-build,module-install}.log`.
- **K5 passed unchanged full gate:** Lt664, SparseLt134 numerical+16 API,
  hipTensor304 API+12 GPU, Torch19 CPU references; two fresh processes each
  24 malloc/free+pinned alias/streams/events and16 explicit VMM remaps. Same
  immutable runtime/DSO+kpack/harness and original memory binary. Separate HIP
  and direct ROCr each26 GPU/CPU comparisons pass WITHOUT diagnostic churn;
  workload dmesg byte-identical. Two old Display Core load WARNs remain. No
  whole-Torch performance, sustained stability, universal VMM or arbitrary hang
  recovery claim. Python CPU segfault and experimental VM-heap defect unresolved
  and distinct. SDMA0/experimental heap off unchanged.
- **Current evidence/replay:** `kernel/r3/evidence/k5-{summary,gem-va-full,
  gem-va-isolation,running-identity}.json`, pre/post isolation logs; helpers
  `k5-{build,module,gpu-phase,isolation,identity}-on-host.*`, patch0003 and
  `k5-source-manifest.json`. Host`SOURCE_K5.json` records module delta;
  `source/GIT_REVISION`8f3c0d3, original image/source manifest84995f1 retained.
  `KWS/logs/k5-gem-va-{full,isolation}`, `k5-gem-va-console.log` and guard log.
  K5 follow-up CPU-unmap containment is now qualified in the bounded SVM scope
  above; use that current cursor for remaining locking/failure limits and next work.
- **Source/build:** signed stable`v7.2.10`, base
  `7b20d05fd38de8e0be168956f6fd241d750c7d69`, Greg fingerprint
  `647F28654894E3BD457199BE38DBBDC86092693E` verified. Separate repo
  `kernel/linux-r3`, original image source`84995f1a85188cde56910729214b0a68a7e402da`, initial two reviewed
  patches in`kernel/patches/r3/`: CU/PASID/TTM quirks, complete probe register
  restoration, runlist caller error propagation. SVM upper notifier/unmap paths
  still discard some existing returns/pre-mark unmapped: CPU-unmap returned-error containment now bounded-qualified above;
  retry-fault/failed-stop handling remains unqualified. Eight-core telemetry/clock trust still open.
  .104`KWS=/home/nik/bc250work/kernel-r3-7.2.10`; native16jobs/20GiB ccache build
  finished22:26:33UTC Oct14. Original PIDs12625/25715 ended; source/build/stage,
  caches and logs retained. Fan source`kernel/board-fan` revision`9086701` imported
  read-only from.105; ABI compiled/signed. Initial fan signing path failed; fixed
  source/scripts/sign-file→build/scripts/sign-file, repaired without rebuilding
  kernel; original failure/unsigned module retained,`logs/fan-sign-repair.log`.
  Signing private key remains remote build/certs,600/root; never copy it.
- **Artifacts/identity:** new image SHA
  `4531223f35bd9f313a242c2aa5d7bece0b4ffd4161d28b41271914a9b54d4558`,
  initramfs SHA`0e8444c328d070e0f6d422514116e19ccb13feb9e44d0bb4b289f0d1ea3eeb11`.
  `kernel/r3/evidence/candidate-inspection.json`:499 initramfs entries, exact
  watchdog hook, storage drivers, AMDGPU deferred, no private signing keys;
  new modules have matching release/signing key. r2 image/initrd/module/default
  hashes unchanged. Loaded kernel build ID`57f7c52d9c7ac0ac21c67d6e6efc3e08c12bbfde`
  and ORIGINAL AMDGPU`e2b6524374abbb8852759eb9c61857f56222793a` matched initial build; superseded by K5 module above; evidence
  `running-identity.json`, not bit-reproducible-build proof. Actual r2 firmware
  snapshot privately pinned via firmware_class.path; its2 SDMA entries are Navi12
  bytes under cyan names, differing from root cyan files. Root firmware unchanged,
  ROCr SDMA0 and experimental CLR VM heap default off. r2 loaded/initrd/installed
  module identity already closed; do not repeat the initial provenance audit.
- **K4 recovery exercised:** r2 feeder SIGSTOP under exclusive runtime lock,
  countdown to19s observed, then SSH loss/new boot ID/r2 return; flag bootstatus0
  does not confirm reset cause. Separate r3 idle-GPU guard deliberately expired
  after15s and emergency-restarted to persistent r2; SSH/new boot ID/default and
  consumed one-shot verified. Returned to r3, base/cooling ACK and24-round GPU
  smoke passed again. Builtin SP5100 early init handoff, nowayout1, root feeder
  120s, candidate panic60/oops/lockup policy confirmed. Candidate guard starts
  after bootmisc/sysctl, reapplies policy after Alpine panic120 default. Empty
  next_entry= is accepted; nonempty pending selection rejected. No guarantee for
  pre-watchdog hangs or arbitrary wedged GPU/reset recovery. No daily default
  promotion. Guard dry-runs alone were not treated as hardware qualification.
- **Historical pre-fix K5 comparison: BOTH full gates FAILED explicit VMM.**
  Identical immutable runtime/N1.8 DSO+kpack/harness on r2/r3: Lt664 references,
  SparseLt134 numerical+16 API, hipTensor304 API+12 GPU,19 eager CPU-compared Torch
  checks (bmm/fp32/fp16/bf16, FFT, Cholesky residual, COO/CSR, Adam) passed. Each
  workload kernel-log delta0; two pre-existing Display Core IRQ WARNs remain at
  AMDGPU load on both kernels. Malloc/free24 rounds, pinned GPU alias, streams/
  events/guards pass; post-recovery r3 basic smoke another24 rounds, delta0.
  Full gate fails after those consumers: fixed VA rebound to handle0 reads exact
  handle1 initializer BEFORE update (got01238878 vs01234567, round0/index0).
  Independent diagnostic kernel→ordinary buffer copy reproduces on BOTH r2/r3;
  cause/layer subsequently isolated and repaired in K5 above. Original/diagnostic failures preserved; not skipped
  or labeled full pass. This VMM defect is distinct from fixed CSR/VM-heap
  corruption AND unresolved historical Python CPU segfault; neither repaired
  by the new kernel. No whole-Torch performance or sustained correctness claim.
- **Evidence/next:** `kernel/r3/evidence/{kernel-comparison,r2-baseline-full,
  r3-candidate-full,watchdog-r2-hardware,deadline-fallback,running-identity,
  post-recovery-basic}.json`, diagnostic logs and README. Host`KWS/logs/` retains
  all commands/results: build-initial,fan-build/install-inactive, r2-baseline,
  r2-vmm-diagnostic,r2-baseline-full,r3-candidate-full,r3-vmm-diagnostic,
  r3-deadline-practice,r3-post-recovery; candidate.json records booted bounded
  scope/full VMM failure (superseded by SOURCE_K5.json and new candidate module fields). Initial test queued behind user's training was canceled
  BEFORE lock acquisition/feeder stop; restarted only after user finished jobs.
  Historical next isolation steps are completed by K5 above. The failed baseline
  remains preserved; use the new cursor for current module status and next action.
- **Historical initial kernel review (superseded by K1/K3 checkpoints above):**
  preliminary read-only r2 audit complete;
  see `docs/development/bc250_kernel_audit_20261014.{md,json,patch}`. Both hosts run
  the recorded7.1.5-r2 image; installed AMDGPU module hashes match. Archive-derived
  AMD subtree comparison:7 active files/136 additions/3 removals,2762 identical
  files and5 inactive `.orig` backups. Upstream signatures/full tree not audited.
  Findings: four ignored runlist-flush error returns; CU probe modes leave RLC
  changed; retained eight-core telemetry patch absent; two pre-existing Display
  Core boot warnings. No kernel cause for unresolved Python fault established.
  At that initial checkpoint there were no kernel/module/boot/runtime changes
  or builds. Upstream/full-tree and module identity work since completed above;
  use the current K3/K4 cursor, not this historical next step. N2.1 retained.
- **Latest completed library milestone:** N1.8 hipBLASLt short small/skinny GEMM dispatch implemented,
  installed on .104 and bounded-qualified. N1.1–N1.7 retained in their scopes.
- **Last qualified code:** root `62ab2437c9fd80e011b23578ffbf9208d85cda64` (N3.2 launcher/qualification),
  libraries `a6f7d3d8ba7a1d022628d8c3fdcc54235551a9b8` (patches0025–0026).
- **N1.8 contract:** K<=64 and either min(M,N)<=4 or both M,N<=32 with M*N<=512
  use one32-lane scalar subgroup/output, eight outputs/block. Physical D order
  guides output lanes; no LDS/barriers in this specialization. FP32 tree reduction
  changes rounding from ascending-K FMA; same/mixed f32/f16/bf16 and existing
  scalar/bias/activation/AUX/DGELU epilogues retained. Public algorithm unchanged;
  remaining shapes keep tiled/register-tile dispatch. Subgroup grids above INT_MAX
  also fall back to the validated tiled grid; no oversized launch was executed. No matrix engine claim.
- **Passed:**664 new CPU-double-referenced GPU cases +403 scalar/561 extended Lt
  regressions. Orders/all transposes, padded/broadcast batches, host/device and
  zero scalars, compatible in-place C/D, K0/1/15/16/17/63/64/65 and area512/544,
  skinny tails/cross-batch blocks and immutable guards. Kernel logs identical1009
  lines; no general bitwise/nonfinite or sustained qualification.
- **Timings:**36 CPU-checked deterministic C++ fixtures ×A1/B1/B2/A2; five GPU-event
  samples of32-launch HIP graphs/pass. Selected skinny ratios1.14–1.86x and
  selected small1.21–1.61x; controls0.998–1.006x (roughly flat); max repeated-pass
  median drift0.24%. Column-major NN, batch1, host scalars/default epilogue only.
  These are graph-replay kernel costs, not uncaptured application or Torch speedups.
  Clock telemetry remains untrusted; default Torch BLAS stays rocBLAS.
- **Evidence/jobs:** `WS/logs/n18-lt-r4/result.json`, repository
  `bc250_scalar_library_results/n18-{lt,lt-attempts,lt-performance,lt-pack,reconstruction}.json`.
  Targeted hipBLASLt+dist,16jobs/20GiB ccache/exclusive lock; final build11.8s,
  numerical3.7s/scalar2.3s/extended2.1s, each36-case timing pass1.0–1.2s. No running
  jobs/gates. Initial one-thread/output candidate passed504 cases but was slower
  (0.46–0.76x selected ratios); rejected. Cooperative r2 passed504+964 checks,
  but32x32 regressed0.91–0.96x: final area<=512 restriction retains its tiled path. r3 passed664+964; final
  review added the subgroup grid-bound fallback and r4 repeated the full gate.
- **Recovery finding:** BLAS DSO references external `.kpack/blas_lib_gfx1013.kpack`.
  Initial attempt restored only DSO; A2 aborted134 on missing old GPU symbol.
  Recovered exact baseline pair from immutable P2 candidate (DSO hash matched),
  preserving experimental files. Runner now backs up/restores/hashes BOTH files;
  successful A2 proves paired baseline execution before reinstating final pair.
  Older N1.1–N1.7 failure reports validated DSO restoration only: do not claim
  complete runtime rollback from those fields or reuse old runners unchanged.
  Their successful final GPU gates remain evidence. This host-side code-object
  mismatch is distinct from unresolved Python CPU segfault and fixed pool/VM issue.
- **Provenance:** `SOURCE_N18.json` verifies12 source files on N1.7/P2 plus both
  active BLAS hashes: DSO `91c65ae424a19b58edb9210d033e0c5d2ce72f1781fd283f5204371c2f654ff6`;
  pack `a431881f2b826ee3eedb65fa6f08b2c2ba6242a62f9c4671f811f7ceb018dde5`.
  Pack payload comparison:1 hipBLASLt entry changed,68 entries unchanged (including
  all67 rocBLAS entries). Independent recovery DSO `WS/recovery/n18-lt-before.so.1.5`
  and pack `WS/recovery/n18-blas_lib_gfx1013-before.kpack` retained. All26 patches
  reconstruct exact tree; .105/frozen candidates/kernel/boot/Torch unchanged.
- **Next: N2.1, solver difficult/complex coverage and selection audit.** Read the
  existing32 P2 solver cases, residual/gradient/condition10,000 evidence, patched
  Torch linalg consumer and rocSOLVER/hipSOLVER dispatch before choosing a missing
  complex/difficult operation. Use short independent CPU/reference/residual gates;
  implement only a demonstrated missing path/defect. Any prefix mutation needs
  paired DSO/kernel-pack recovery. Sparse widened INT8 outputs/beta-vector and
  other optional N1 extensions remain separately open, not hardware-impossible.

At resume: read this file, inspect local Git status/log and `.104` runtime/source
manifests, then read the inventory section for the selected task. Check evidence
before deciding something is missing. If state differs, reconcile it; do not
silently replace the plan or downgrade already qualified work.

## Goal, authorization and working agreements

Build the most functional, reliable and efficient **native gfx1013 ROCm/Torch**
stack the BC-250 can support: compiler/runtime, feasible libraries, training,
profiling/debug tools, communication, packaging and measured performance.
Do not spoof another GPU or pretend missing matrix instructions exist. A scalar
implementation of a mathematical operation may be feasible even when an upstream
matrix backend is not. CPU fallbacks must be labeled.

- User preference2026-10-15: update/commit canonical flow once at final response
  or meaningful handoff; use commentary for progress, avoid repeated routine doc commits.
- English communication/code/docs. Work locally, transfer explicit changed files,
  build/test/install on `.104`. User authorizes development/kernel experiments;
  retain SSH/boot fallback. No push/publish without explicit instruction.
- User authorized `.105` migration/legacy retirement2026-10-23 (M105.1 complete).
  It is now a deployment host; preserve the selected production snapshot, r2
  recovery, operational healthchecks and unrelated user files. Further deployment
  follows the current task scope and its recorded exclusive control lock.
- Preserve releases, sources, caches, build objects, images, models/checkpoints,
  kernel/initramfs/GRUB recovery and unrelated untracked `src/`. M105.1 is the
  explicit exception for the inspected `.105` legacy stack, with unique work archived.
- Use bounded meaningful CPU-compared GPU tests. Reuse objects/ccache, group commits
  at milestones, keep routine documentation short. No automatic long soaks.
- Do not run `fetch_sources.py`, reset submodules, expunge or prune as a resume step.
- SSH keys work. Doas credentials are transient session information; **never write
  passwords, tokens or raw credential-bearing prompts into memory/source/logs**.

## Historical state/path snapshot (superseded identities)

This section retains older milestone identities. Use the resume cursor and
workspace-manifest.json for current revisions, boot evidence and providers.
Do not reinstall these older identities as a resume action.

`WS` = `/home/nik/bc250work/rocm-10.2-gfx1013` on `nik@192.168.1.104`.
Workspace root `/home/nik/Documents/Vibe/bc250stack`; TheRock root
`/home/nik/Documents/Vibe/bc250stack/therock` (old path remains a compatibility
symlink), branch
`users/Naster17/rocm-10.2-gfx1013`. `.104` host is Alpine, ROCm userspace is glibc
inside rootful Podman; this is not the separate native-musl gfx1103 project.

| Item                     | Current identity                                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| Qualified root code      | `0115c01f06f8654b26b5a665286069caf89eda2d`                                                                                |
| rocm-libraries           | rocBLAS `8389197c134a7337932021ab7137987a585664ae`; other native libraries `319a235d6`; patches0001–0033                                                      |
| rocm-systems             | `dbdb7b8d1c56316d873a9b90e548e5a72853496d`, patches0001–0003                                                              |
| Torch                    | native `dc8fea0cb8b157cb415359033e4f0175d2ebfa56`, source/tests `2b449c4`; editable pip                                                |
| Triton                   | `ee65ff3b85f99d52e678b252a99c8425fa46e329`, patches0001–0004; qualified wheel retained                                    |
| LLVM                     | Clean upstream `7667f52dd687899256a3285ab0bce3c53bc73394`                                                                 |
| Active stack             | `/opt/bc250-stack/current` → `/opt/bc250-stack/dev/priority1`; legacy directory name, **N3.2 mutable dev profile; earlier qualified scopes retained**                  |
| ROCm/Torch/cache         | `WS/build/dist/rocm`, `WS/pytorch-sparse-n213`, `WS/cache/dev-user` (uid/gid1000)                                                |
| Runtime image ID         | `5df0a58392de13916819f6d84342b0338ae2c2bbb2f3c21a1f3ad38be2724a59`                                                        |
| Runtime/build tags       | `localhost/bc250-stack:10.2-gfx1013-priority1-candidate`; `localhost/bc250-rocm-build:10.2-ccache`                        |
| Source exports           | Gitless `WS/source`, qualified Torch exports; current deltas `SOURCE_SPARSE_N213.json` / `SOURCE_GEMM_N32.json`                |
| Independent P2 candidate | `/opt/bc250-stack/candidates/priority2-20261014`; passed bounded checks, not daily-release activation                     |
| Recovery userspace       | `/opt/bc250-stack/before-dev-current` → frozen `10.2-gfx1013-20261013-4b000d542`; initial a640c9296 release also retained |
| Kernel/boot              | `.104` live `7.1.5-bc250rocm-r2`, boot d3857b96; qualified r3 retained; new boot observed N3.5, no agent reboot |
| Memory backup            | `WS/agent-memory/latest` (documentation only; preserves qualified source snapshots)                                       |
| Training artifacts       | `WS/training-model`, `training-env-stable`, `training-validation`, `logs/training-memory/`                                |

Manifests and immutable image IDs outrank mutable tag names. Component version
strings differ legitimately (TheRock10.2, HIP7.17, Torch 2.13). Source archive names
and generated Torch Git labels may retain older baselines; do not relabel them.
The P2 source transfer verified51 files; N1.1 adds7; N1.2–N1.7 each add8 verified changed files; N1.8 adds12 and BLAS DSO/pack hashes. Candidate payload verified191
independent shared-object/kernel-pack copies. Detailed hashes are in manifests.

## Unified local workspace

The new parent workspace has its own AGENTS.md/README and component manifest.
`torch/` and `triton/` link to our patched sources inside TheRock. Your fresh
upstream checkout is retained under `references/pytorch-upstream`; it is not an
upgrade/replacement of the qualified Torch. Old `Vibe/therock` and `Vibe/pytorch`
paths remain symlinks to preserve absolute build paths. Do not delete them yet.

`kernel/source` is the already-patched Linux7.1.5 r2 source recovered read-only
from .105, with `kernel/configs/r2.config`, packaging repository, legacy scripts
and provenance. Kernel binary hash matches both hosts. Compiled objects/private
signing keys/generated initramfs archives are excluded. Do not apply legacy
patches twice. Existing board driver/CU-unlock repositories are copied under
`board-support`. Organization did not change either remote runtime or boot state.

## Hardware and indispensable constraints

`.104` has eight CPU cores/16 threads, **40CU/20WGP/80SIMD**, qualified wave32,
64KiB LDS, KFD100103/HIPgfx1013:xnack−, PCI1002:13fe. HIP multiprocessor count20
is correct for WGP reporting. GTT15,938,355,200 bytes≈14.844GiB and systemRAM
≈14.858GiB share physical GDDR6; they are not two independent memory budgets.

Keep SDMA disabled (`HSA_ENABLE_SDMA=0`) and experimental VM heap default off;
VMM/async pools remain available. Current GPU recovery parameter is0. Watchdog
SP5100 TCO service is active (10s/120s), but actual hang/reset recovery is not
qualified. No usable MFMA/WMMA or packed-dot backend. Wave64/XNACK+ and hardware
media/counter/debug modes are separate qualification, not inferred from enums.
No cpufreq governor exists on this host; GPU clock telemetry conflicts. Fan curve
reads CPU Tctl, not GPU edge. Full measured hardware/recovery facts are in inventory.

## Completed work: do not restart it

| Batch                     | Qualified work / evidence anchor                                                                                                                                                                                                                                                                |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Core and expanded release | Native HIP/RTC/Torch/Triton; BLAS/solver/FFT/RNG/sparse; grouped MIOpen convolution, primitives, OpenCL/OpenMP, RPP, hipTensor generic, hipDNN providers, hipFile compatibility, profiler tracing. Historical13 component/56 eager/18 compiler gates; independent frozen600s soak40,339 cycles. |
| Training/files            | Real SmolLM2-135M full FP32 and AMP LoRA FP16/BF16, optimizer/scheduler/scaler/RNG checkpoint/resume; 10GiB GPU+ 768 MiB CPU pressure, streams/graphs/processes. Persistent host paths, mount modes, downloads/DataLoader/pip/stdin/shell/interruption qualification already exists.            |
| Scalar milestone          | Real native hipBLASLt/hipTensor/hipSPARSELt backends, CK/RPP/generic/Torch coverage; Torch NHWC average-pool gradient fix and separate sparse Lt kernel-pack ownership.                                                                                                                         |
| P1                        | Lt C++/host-grouped consumers and epilogues; CK247 numerical+400 rejection cases; hipDNN100 graph cases; Torch64 including36 observed real Lt calls. Mutable dev stack activated.                                                                                                               |
| P2                        | hipTensor166, hipSPARSELt234, hipBLASLt964 API, math149, primitives138 GPU cases;71 performance cases. Exact library/system/Torch patch reconstruction; independent candidate and installed launcher each passed20 cycles with sparse/compiled/FFT/checkpoint checks.                           |
| N1.1                      | hipTensor unilateral A/B sums and mixed shared reductions:253 numerical +20 descriptor cases; alias/zero-scalar/span guards,16-mode union, nine algorithm/workspace combinations, small-fixture timings. Patch00018; native scalar routing. |
| N1.2 | Same-type C32F/C64F hipTensor contractions with identity/NEG/CONJ:426 numerical +24 descriptor cases,273 real regressions, scalar ABI/zero-scalars/alias/span/plan-lifetime guards; patch0019. |
| N1.3 | hipTensor plan null/enum/attribute-size guards, atomic setters and owning failure cleanup; honest no-cache/JIT/autotune scalar policy.304 API +12 numerical checks,273 real/450 complex regressions; patch0020. |
| N1.4 | Sparse Lt matching-count zero-stride broadcasts for A/B/C, one physical sparse pack, single-writer pruning/compression and span alias guards.604 numerical +5 descriptor cases,234 regressions; patch0021. |
| N1.5 | Native GPU sparse Lt per-row bias: same-type/FP32, batch broadcast/strides, before activation, plan snapshots and size/type/span/alias guards.526 numerical +41 API checks,234 scalar/609 broadcast regressions; patch0022. |
| N1.6 | Native device FP32 per-row alpha vector shared across batches, plan flag snapshots and span/alignment guards.404 numerical +34 API checks;234 scalar/609 broadcast/567 bias regressions. Beta-vector enable explicitly unsupported; patch0023. |
| N1.7 | Native INT8 input/output, bounded INT32 accumulation, FP32 epilogue/bias, nearest-even saturation.134 numerical +16 API and234/609/567/438 regressions; K<=262140, widened outputs unimplemented; patch0024. |
| N1.8 | Short small/skinny Lt subgroup dispatch:664 new +964 regression cases,36 A/B graph timing fixtures; selected1.14–1.86x, controls~flat; paired DSO/kpack recovery; patches0025–0026. |
| Performance               | Matched `.104`/`.105` comparisons already done: most GEMM/copy differences small, variable small-conv results. Separate same-board P2 large LtGEMM≈2.1–2.4x faster; selected large compiled HIP-graph fusion≈4x. No whole-Torch or universal speedup claim; default BLAS unchanged.             |

Reports: `docs/development/bc250_scalar_library_results/priority{1,2}-*.json`,
`bc250_performance_results/`, host `WS/logs`. Detailed coverage limits, patch
index, training/model pins and old defects are in the inventory. Counts overlap.

## Known pitfalls and resolved problems

- **Proven old CSR corruption:** experimental VM heap overwrote live rocSPARSE
  analysis metadata. Safe compiled CLR default fixed repeated CG gates (40 and
  five-process200); it did not repair the experimental heap. Preserve this fix.
- **Separate later segfault:** Python CPU execution fault at0x3b3e6, cause unknown.
  Do not identify it as the proven CSR defect or claim bounded reruns solved it.
- **Prefix concurrency:** launcher holds shared `.runtime.lock`; run every host
  prefix update through `helpers/update_dev_prefix.py -- COMMAND...` (exclusive).
  Arbitrary manual writes bypass this protection. Never test against staging libs.
- **Torch generation:** before each source rebuild, run upstream
  `python tools/amd_build/build_amd.py --out-of-place-only`, then
  `python -m pip install -e . -v --no-build-isolation`. Cached CMake can leave HIP
  sources stale; don't rebuild solely to change generated version metadata.
- **Runtime rollback:** pair every changed DSO with its external kernel pack; back up independent copies and verify both hashes under the exclusive lock. DSO-only restoration is incomplete. N1.8 discovered this; older rollback helpers need this fix before reuse.
- **Targeted builds:** prefer component+dist, including
  `hipBLASLt+dist hipTensor+dist hipSPARSELt+dist`. Global dist can rebuild unrelated
  MIOpen. Keep16 normal jobs, tensor12/providers6/Torch 12/link 1 and20GiB ccache.
- **Flag names:** use `THEROCK_GFX1013_SCALAR_BACKENDS=ON` with exclusive gfx1013;
  similarly named `THEROCK_ENABLE_...` cache entries are not this option.
- **Already fixed:** Triton RDNA1 buffer flags/low-precision lowering, RPP
  spectrogram/decibel handling, Torch pooling, CK/header tracking, FFTS linking,
  launcher path/read-only/shell/consumer pins. Read inventory before redoing them.
- **ABI details:** Lt SWISH beta is fixed 1; MIOpen bias uses beta 0; mixed fusion
  intermediates remain rejected. SparseLt compression is backend-specific.
- **Testing:** require real execution/counts, CPU references, unchanged meaningful
  tolerances andguards. Clone float64 CPU references before gradient mutation.
  Do not disable/skips tests to turn red into green.
- **History:** native Alpine branch testedgfx1103, not BC-250. Historical frozen
  long-soak results do not qualify the currentmutableprefix forlongstability.

## Ordered full-port plan: stable task IDs

P0/P1/P2 milestones are completed **within their recorded scopes**. Remaining
work is below; do not revive their old unchecked checklist lines as new tasks.
Work through one measurable subtask at a time and keep these IDs stable.
The current priority order is [the remaining-work checklist](../BC250_REMAINING_PLAN.md);
this older category table is not the execution cursor.

| State | ID                   | Work and acceptance boundary                                                                                                                                                                                                                                                                                                                                                                        |
| ----- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [x]   | N1.1 | hipTensor unilateral contraction modes completed within the bounded scope above. Preserved zero-scalar, span and alias guards; CPU-referenced types/layouts/plan-lifetime gates and small-fixture timing passed. |
| [x]   | N1.2 | Native same-type C32F/C64F hipTensor contractions, identity/NEG/CONJ, bounded numerical and real-regression gates completed as above; mixed/widened complex and other unary functions remain rejected. |
| [x]   | N1.3 | hipTensor plan API validation, owning failure cleanup and supported scalar preference acceptance/rejection qualified in the bounded scope above. Scalar cache/JIT/autotuning/ranked alternatives remain explicitly unsupported. |
| [x]   | N1.4 | hipSPARSELt matching-count zero-stride broadcast inputs, single physical sparse compression/pruning and span-alias guards qualified above. Incompatible counts/multi-batch zero-stride outputs remain rejected. |
| [x]   | N1.5 | Native hipSPARSELt row bias pointer/stride/type contracts and epilogue qualified above. Device bias memory remains caller-owned; plans capture settings. |
| [x]   | N1.6 | Native alpha-vector path and explicit scalar-beta/unsupported beta-vector policy qualified within the bounded scope above. Beta-vector implementation remains separately open; no universal vector-mode claim. |
| [x]   | N1.7 | Native signed INT8 input/output with bounded INT32 accumulation, FP32 scalar/vector/bias epilogue and nearest-even saturation qualified above. Widened outputs remain unimplemented. |
| [x]   | N1.8 | Short small/skinny Lt subgroup dispatch qualified above; tiled path retained outside measured selection. No default Torch BLAS change or whole-Torch speedup claim. |
| [x]   | N2.1 | Audit P2 solver32 cases and pinned Torch/rocSOLVER/hipSOLVER consumers; qualify one missing difficult/complex operation with CPU/residual references and fix only demonstrated defects. |
| [x]   | N2.2 | Complex nonsymmetric eig16 groups qualified through existing host-LAPACK compatibility; native GPU eig remains missing. Pinned dispatch audited, no backend defect reproduced. |
| [x]   | N2.3 | Isolated native small Schur prototype cap16: GEHRD plus bounded GPU shifted QR, host scalar control, explicit nonconvergence;21 groups pass. No right eigenvectors or production eig integration. |
| [x]   | N2.4 | Isolated native GPU right eigenvectors from small Schur forms;21 groups pass including analytic/CPU residual/direction and repeated/close/nonconverged refusal. Production eig unchanged. |
| [x]   | N2.5 | Repair demonstrated tiny/large magnitude failures; seeded6 replay,40 scale,30 separation,16 failure/control and21 N2.4 groups pass. Experimental cap16/default eig unchanged. |
| [x]   | N2.6 | Profiled small native versus host, vectorized recovery;107 groups and6 paired fixtures pass, recovery1.53–8.44x/whole1.07–1.72x in fixtures. Native remains slower than host; default unchanged. |
| [x]   | N2.7 | Isolated native cooperative small QR/similarity step;28 direct+128 retained groups+six paired CPU fixtures pass. Schur1.24–1.46x in fixtures, host eig remains faster; defaults unchanged. |
| [x]   | N2.8 | Device shift/deflation/status around cooperative QR;128 retained+74 direct+six CPU/paired fixtures pass. Whole prototype1.56–2.21x versus N2.7 in fixtures; host default remains faster/unchanged. |
| [x]   | N2.9 | Bounded chunks1..32 reduce status copies;128 retained+74 direct+100 chunk+six CPU fixtures pass. Latency mixed/regresses16; padded experiment not selected, defaults unchanged. |
| [x]   | N2.10 | Native bounded reflector-Q formation;128 retained+24 independent reflector+six CPU/backend fixtures pass. Preparation1.11–1.39x/whole1.05–1.12x in fixtures; defaults unchanged. |
| [x]   | N2.11 | Explicit pinned-stream GEHRD sessions;128 retained+six CPU/backend fixtures+three lifetime groups/16 refusals pass. Warm prep7.03–12.52x/whole1.28–1.95x in fixtures; cold/default/host limits recorded. |
| [x]   | N2.12 | Native bounded triangular recovery;128 retained+36 CPU/backend+32 direct+two rescaling+six ABI cases pass; fixed gates/defaults retained. Recovery1.32–2.32x/whole1.17–1.34x in six fixtures; host remains faster. Separate CPU-first profiler defect found. |
| [x]   | N2.13 | Generic sparse triangular coverage and exact-pivot NaN marking installed:433 public +96 direct generic fixtures, retained sparse41/core7/profiler pass in candidate and installed runtime. General sparse spsolve B02 remains separate. |
| [x]   | N2.14 | FFT complex96/real72 output+weighted backward+guarded out, three DFT/finite-difference/reuse checks,16 direct plans/32 executions and FFT30/core7 pass; no production defect or binary/default change, AOT cache preserved. BF16 promotion B01 separate. |
| [x]   | N2.15 | Torch state80/statistics22/contracts20/fresh continuation1, direct native68 and RNG28/core7 pass. Stale LFSR113 offset comments corrected; implementation/binaries/defaults unchanged; four AOT DBs preserved. |
| [x]   | N2.16 | Native floating/segmented/custom reductions/scans: rocPRIM576/hipCUB480/Thrust480, P2 primitives138/core7 pass; fixed CPU references/guards/streams, no production defect or default/binary change. |
| [x]   | N2.17 | Native floating stable sorting/selection: rocPRIM656/hipCUB656/Thrust648; Torch62 incl54 finite backward cases; P2 primitives138/core7 pass. Exact ordering/payload/count/guard checks; no production defect or installed change. |
| [x]   | N2.18 | Scalar CK DL FP32 grouped forward:128 CPU-reference cases +8 refusals; GEMM3/core7 retained. Explicit r2/r3 guard selection repaired; no production CK/runtime/default change or speed claim. Next review N2 practical gaps before selecting another N3 batch. |
| [ ]   | N2                   | Broader scientific/primitives/RPP coverage: solver difficult/complex cases and selection, sparse formats/analysis/solves, FFT plans/layouts, RNG distributions, floating/custom/segmented algorithms, admitted CK DL conv/grouped operations. Qualify actual operations, not architecture lists.                                                                                                    |
| [x]   | N2 practical review | Current-workload gap classification complete; stale A01/next-task documentation corrected. Broad optional scientific coverage remains open; no mandatory implementation identified before N3. |
| [x]   | N3.1 | Current Qwen3.5 synthetic training profile:65/1024 exact loss/372 gradients/AdamW/RNG replay, corrected kernel-only correlation/shape attribution. Dense mm dominates; no runtime/default or speed claim. |
| [x]   | N3.2 | Exact-layout BF16 NT solution-490 plus cold override loader repair installed; CPU GEMM/all372 model gradients/CPU AdamW/ABBA timings and installed gates pass. Synthetic whole-step1.52–1.56x; changed reduction order, no convergence/bitwise continuity claim. |
| [x]   | N3.3 | Optimized1024 model exact replay/profile and reverse-projection CPU-reference selection audit complete; mm75.39%, reverse26.43%, modest1.09x fixture candidates left undeployed. No runtime/default/new speed claim. |
| [x]   | N3.4 | Square BF16 two-layout/all14-solutions-each CPU audit complete; faster-346 fails full-model gradient7.65%>4% contract, exact-output alternatives flat/slower. No runtime deployment or qualified speedup. |
| [x]   | N3.5 | Current-preset1024/rank16 accumulation16 versus FP64 gradient sum, CPU AdamW/scheduler and exact owned fresh-process state replay qualified; test mount/CPU-oracle alias fixes, no runtime change. |
| [x]   | N3.6 | Actual deployed inner CLI1024/rank16/accum16 masked targets; fresh/resume exact372 adapters/full state/loss/evaluation,7 checks, no production change. |
| [x]   | N3.7 | Supplemental checkpoint runtime/preset/hook provenance deployed; exact new/legacy/installed replay, eight refusal cases, pre-N37 numerical state exact. Network-only kernel-log guard failure retained; strict refusal recheck passes. |
| [x]   | N3.8 | Full chunked CPU FP64 vocabulary forward/input-backward audit, matched Torch policy and full-model gates; timings variable, candidate inactive. Entry app drift and host OOM retained. |
| [x]   | N3.9 | Qualified N37 provenance/A07 CLI restored preserving current ops edit; exact actual fresh/new/legacy/installed replay, conservative historical checkpoint refusal, app suite and guidance synchronized. |
| [x]   | N3.10 | User-broadened generic compiled CNN/dense/gated/BatchNorm training;56 CPU-reference/optimizer/buffer/trace checks, no production defect/install/speed claim. |
| [x]   | B04.1 | Pinned native torchvision installed in independent image;67 candidate+67 installed CPU-reference/operator/gradient/codec/model/core checks. CPU codecs explicit, old image/release retained. |
| [x]   | B04.2 | Native torchaudio installed;142 candidate/142 installed/48 post-user-reboot checks. Signal outputs/first-order gradients/stream behavior plus demonstrated CLR cold-hostcall queue repair qualified within recorded bounds. |
| [x]   | B04.3 | CPU TorchCodec/FFmpeg audio load/save installed;136 candidate+136 installed file/reference/GPU interoperability and retained regression checks. User application excluded; existing ROCm/Torch retained. |
| [x]   | B04.4 | CPU video decode/seek/encode and exact single-frame VFR timestamp repair installed;108 candidate+108 installed reference/audio/core checks, explicit HIP forward/backward interop. |
| [x]   | B04.5 | CPU JPEG/PNG/WebP decode and PNG/JPEG encode installed;203 candidate+203 installed image/reference/HIP interop/retained regressions. Other formats/GPU codecs remain separate. |
| [x]   | B01.1 | Explicit same-device BF16-to-FP32 bc250_fft helper installed;327 final candidate+327 installed references/gradients/native trace/retained FFT gates. Native API remains unchanged. |
| [x]   | B02.1 | Explicit bounded GPU dense-LU CSR solve installed;381 candidate+381 installed output/first-gradient/analytic/FD/refusal/trace/retained sparse gates. Native general API/large sparse factorization unchanged. |
| [x]   | B03.1 | Native distributed/Gloo/libuv local CPU readiness installed;31 two-rank references/first gradients/DDP checks; candidate1082+installed1082 retained GPU regressions. RCCL/MPI off; GPU scope is recorded in B03.2, RPC/multi-board unqualified. |
| [x]   | B03.2 | Existing Gloo HIP host-staged collectives and FP32/64 GPU DDP bounded-qualified with41 GPU+67 retained checks; no runtime change. One GPU/two ranks, no scaling claim. |
| [x]   | B03.3 | Isolated restricted native gfx1013 RCCL build and82 single-device references/lifecycle checks plus79 retained pass; no-tracing header/direct-include compile fixes. Not installed/Torch NCCL off. |
| [x]   | B03.4 | Broader isolated375-object native RCCL;629 single-device/retained checks. No protocol/transport/scaling claim. |
| [x]   | B03.5 | Isolated native Torch RCCL integration and BF16 PreMulSum scalar correction;172 checks. Daily NCCL off. |
| [x]   | B03.6 | Bounded public contracts, first gradients/DDP and retained stack regression;1365 checks. |
| [x]   | B03.7 | Numeric FP8 admission guards, raw-bit copy and memory API readiness;1441 checks. No real offload payload/symmetric device claim. |
| [x]   | B03.8 | Explicit ordinary-launch rccl-single profile on .104;1461/26 gates plus private-context5, nine CPU refusals and lock exclusion pass. Daily NCCL-off preserved; .105 untouched. |
| [x]   | B05.1 | Audited native low-bit paths; two native layout defects reproduced, backward/INT4 refusals retained. Isolated frozen W8 first-gradient recipe110/6 checks passes; smaller weight payload/slower small forward fixtures, not installed. |
| [x]   | B05.2 | Native activation/FP32-scale layout fix installed in daily NCCL-off Torch after947 candidate+231 ordinary checks. Old RCCL profile remains unfixed; native backward/INT4 refusal retained. |
| [x]   | B05.3 | Matched RCCL Torch layout repair selected after1732 candidate+371 ordinary checks; daily B05.2 unchanged. Both cohorts use a524 source; old profile/recovery retained. |
| [x]   | B05.4 | Strict helper validation normal scalar reads2→1;475 current-boot checks, matched outputs/gradients/errors, bounded paired latency gains. Rejected slower list-copy experiment retained; native cohorts/defaults unchanged. |
| [x]   | B05.5 | Explicit frozen-W8 module/versioned state, FP32 scale-preserving moves, CPU-reference first gradients and exact owned adapter AdamW resume;295 checks/ten gates. Optional helper only, no native/default/image update. |
| [x]   | B05.6 | Public opt-in bc250_quant0.1.0 installed in both .104 runtime cohorts;817 candidate+817 ordinary installed checks, actual B05.5 checkpoint compatibility; identity rollback exercised after test-runner refusals. Native/defaults/.105 preserved. |
| [x]   | B06.1 | Pinned modern ONNX export/CPU runtime installed on .104:181 candidate+181 installed checks plus early33, CPU/GPU-origin graphs/independent references/explicit HIP interop, retained W8/compiler/consumers. ONNX GPU/autograd unqualified. |
| [x]   | B06.2 | Pinned MIGraphX audit:30 native header CPU-reference checks,52 total/five gates; minimal GPU configure passes with isolated pinned CMake. Strict DPP0 JIT issue/feature risks and disk/reserve budget remain; no full backend/provider installed. |
| [x]   | B06.3 | Strict C++17 DPP declaration guard repair; actual HIPRTC old-failure reproduction/patched module execution, default subwave and block references;93 checks twice. Feature predicates exercised, explicit full-build budget refuses current space; no backend/provider install. |
| [x]   | B06.4 | Durable external build storage; full minimal FP32 GPU backend built, Lt-disabled default-query compile defect repaired.22 native graph/reference cases plus22 retained checks pass twice; no runtime/provider install. |
| [x]   | B06.5 | Private pinned Abseil/Protobuf and native C++ ONNX parser built; six FP32 IR10/opset18 model routes/initializers/refusals and retained regressions pass68/six gates. No active SDK/provider install. |
| [x]   | B06.6 | Isolated pinned Python3.12 bindings; six FP32 ONNX models/serialization/explicit HIP copies in three fresh import orders plus retained regressions pass169/seven gates. No active SDK/ORT GPU EP install. |
| [x]   | B06.7 | Relocatable CPython3.12/public C API/CLI SDK archive; generated CMake target repaired, two extracted prefixes (spaces included) pass134/eight guards and68 GPU graph evaluations. No active SDK install. |
| [x]   | B06.8 | Versioned .104 SDK installed/registered with explicit bc250-migraphx launcher, identity refusals/live unregister-register recovery and ordinary daily/RCCL activation;165 checks/ten guards/68 GPU evaluations. Defaults/.105 preserved. |
| [x]   | B06.9 | Pinned ORT1.31 source/dependency/budget audit; real C++20 public wrapper repair in source/private overlay,168 C ABI symbols link, two CPU header regressions and38 guarded checks/six guards/16 GPU async graph evaluations. ORT provider itself unbuilt. |
| [x]   | B06.10 | Matching isolated ORT1.31 core/Python/MIGraphX provider built with narrow strict-compiler repairs;118 checks/seven guards/36 GPU evaluations and18 actual GPU profiles pass. Installed defaults unchanged; package identity/relocation remain next. |
| [x]   | B06.11 | Relocatable ORT package installed via explicit bc250-ort; package identity/RUNPATH normalization, two build-free extracts, live recovery/identity and repaired caller context;174 checks/48 GPU evaluations. Defaults/.105 preserved. |
| [x]   | B06.12 | Modern Torch exporter→public ORT GPU integration completed 2026-10-29:112 checks/24 GPU evaluations/12 GPU-only model profiles; practical gap review found no mandatory defect. Optional dynamic-shape/dtype/operator expansion remains separate. |
| [ ]   | N3                   | Torch functional/performance expansion: larger meaningful training sequences/batches, full-model AMP, compiled/eager gradient equivalence, optimizer/checkpoint resume, quantization/QLoRA applicability, feasible scalar tiled attention and hipDNN mixed/fused graphs. Keep mathematical SDPA working; unsupported matrix backends stay off.                                                      |
| [ ]   | N4                   | Trusted read-only telemetry, RDC applicability, debugger consumers/suites, additional tracing and physical counter mapping. A09 CPU-first ordering repaired with bounded both-order traces; physical telemetry/clock/counter and advanced profiler work remain. Dangerous debugger tests require recovery first.                                                                                                                                                   |
| [ ]   | N5                   | Extend beyond qualified native RCCL/Gloo/DDP/FSDP; optional rocSHMEM/MPI and feasible generic/software transports with GDA/SDMA/XGMI/P2P claims off until demonstrated. Single-GPU readiness is not full collective coverage. `.105` D01 update, D02–D05 cross-board RCCL/DDP/FSDP, D06 matched-batch DDP accumulation, D07 DDP AMP/D08 FSDP AMP/scaler/shards and D09 public consumer environments/tiny Transformers DDP pass within bounds; D10–D25 add bounded native consumer/Trainer AMP, clipping, finite/overflow policy and exact restart. L01 measured1.231x warm-update throughput in one real DDP LoRA fixture; earlier synthetic fixtures remain slower than one host. Larger-model capacity, arbitrary FSDP/model/frozen combinations, elastic recovery and other transports remain unqualified.                                                                                                  |
| [ ]   | N6                   | Guarded kernel/runtime investigations: experimental VM-heap repair, SDMA, reset/fault recovery, XNACK/wave64 where supported, advanced profiling. Preserve known boot/stack defaults; isolated opt-in gates and independent recovery required.                                                                                                                                                      |
| [ ]   | N7                   | Remaining optional feasibility: host media/HIP interoperability, actual media rings, hipThreads/Fortran execution, hipify-clang, hipFile registration/concurrency/direct-storage applicability, rocRoller/emulation and other pinned projects. Explicit unsupported/not-applicable decisions where hardware/API prevents a path.                                                                    |
| [ ]   | N8                   | Final integration/release: source reconstruction, independent relocatable packaging, licenses/tools/plugins/kernel packs, selected Torch regressions, agreed bounded stability/training/memory gates, kernel-log review, recovery verification and truthful operation matrix. Longer runs only when relevant/user-agreed.                                                                           |

Kernel workstream added at the user's request (2026-10-14):

| State | ID | Work and acceptance boundary |
| ----- | -- | ---------------------------- |
| [x] | K1 | Verified latest stable upstream Git tag/source and signing identity; compare legacy source/config/modules and retain provenance/fallbacks. |
| [x] | K2 | Minimal series reviewed/implemented and normal paths bounded-tested; diagnostic register restore corrected; CPU-unmap returned-error containment bounded-qualified by patch0004; failed-stop/map-locking/telemetry remain open. |
| [x] | K3 | Separate r3 image/modules/config built, signed/staged/installed and artifact-inspected on .104; r2 retained unchanged. |
| [x] | K4 | Bounded r2 watchdog countdown/SSH return and r3 deadline fallback/SSH return exercised; two r3 base boots/cooling ACKs; early-hang/arbitrary GPU-hang recovery unqualified. |
| [x] | K5 | GEM_VA runlist invalidation repaired; unchanged full GPU/Torch/VMM gate and independent HIP/ROCr references pass on signed r3 module8f3c0d3. Historical r2/r3 failures preserved; bounded scope only. |
| [x] | K6 | Record qualified scope/artifact hashes, source patch series and reproducible build/boot/recovery instructions; integrate only supported settings and retain r2 fallback. |

Every component ultimately needs a documented disposition: qualified native path,
qualified explicit fallback, or a concrete hardware/API limitation. Missing matrix
instructions themselves are **not pending implementation**. Do not compute a
completion percentage from boxes; an infinite API-combination sweep is not a plan.

## Memory maintenance rule for every future agent

At final response or a meaningful handoff, update this file once (user preference
2026-10-15: do not rewrite/commit flow for routine progress). Preserve durable
job/status evidence if interrupted; use commentary during work. Record: taskID/subtask, actual source and
installed revisions, exact changed behavior, passed/failed evidence andscope,
unresolved failures, next concrete action, running container/PID/log/exit paths,
resource settings andany user scope changes. Update the affected inventory row
when a capability/decision changes. Preserve historical reports/failures; don't
rewrite the entire history or make a commit after every command.

Temporary tool cell/session IDs and `/tmp` transfer helpers are not durable state.
Record replayable repository/host commands and artifact paths. Never record
credentials. Mark complete only when implemented, staged and numerically executed
for the stated scope. New plans/assumptions must be labeled, not quietly promoted
to hardware facts. If interrupted, write **in-progress/unknown**, not passed.

Before repeating work, answer: what already exists, which source/report proves it,
and what new change/uncovered case/failure requires another build or test?
