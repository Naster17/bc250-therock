# B07.13: installed explicit FP32/FP16 bc250-ck option

Completed 2026-11-12 on .104, tools `be602f7ee`. The already-qualified **B07.12
coherent FP32/FP16 CK + repaired-MIOpen** archive is installed and activated as the
optional `bc250-ck` command, replacing the B07.7 FP32-only registration. Native
libraries `8389197c1`, plugin source `742ee1e2c`, stack source `96e5a7dfc`, the
daily/RCCL runtime images and the production architecture whitelist remain
unchanged. Ordinary `bc250-torch`/`bc250-train` defaults are preserved. The B07.7
FP32 package/tool/version records are **retained** as the live-rollback target.
.105 was not contacted.

[Use, identities and recovery](../bc250_ck_plugin_b0713_usage.md);
[candidate artifact scope](../bc250_ck_plugin_b0712_results/README.md).

## Installation, activation, identity and recovery

- Versioned package: `/opt/bc250-stack/sdks/bc250-miopen-dl-gfx1013-b0712`.
- Registration: `/opt/bc250-stack/sdks/miopen-dl.json`, plus the independently
  retained version record `bc250-miopen-dl-gfx1013-b0712.json`.
- Tools: `/opt/bc250-stack/tools/ck-b0713`; `/usr/local/bin/bc250-ck` points to its
  root-controlled `bc250_ck.py`. The B07.7 tools (`ck-b078`) are not overwritten.
- Recovery record: `/opt/bc250-stack/sdks/miopen-dl-b0713-rollback.json` captures
  the exact pre-activation registry bytes (B07.7) and wrapper symlink target.

Install/activate/unregister/rollback acquire the canonical exclusive update lock;
launches hold the shared lock through identity checking and application exit.
`install` stages the package/tools/record without switching activation; `activate`
moves **both** `miopen-dl.json` and the `bc250-ck` symlink together (the b0713
wrapper refuses the old b077 registration) and captures the rollback state once.
Installation verifies the accepted archive SHA
`a0d47ed4ba4df354cc7ea898a285eefa051f49693667386567ea392f66ddd91e` and manifest SHA
`52db5d4262bd29e6307b13a5862cc0652f0e0b6f9c8509761d03d39965e02846`, the selected
runtime metadata hashes (`86d94f43…` daily, `f183bd3f…` rccl-single) and images, and
the exact 38-file payload. The format2 entry verifies the 27 actual mapped ABI
libraries (including the private repaired `runtime/libMIOpen.so.1`), native
`gfx1013:xnack-` and API5/66 exports inside a fresh worker.

The wrapper reuses the qualified caller-context handling to preserve bare
environment variables across `doas`, supports mount/env/env-file/shm-size, Python
script files, `--check`, `--ck-info` and non-merging `--seed-db NEWDIR`, and
delegates through the installed ordinary `bc250-torch` launcher (byte-identical
`bc250_torch.py`, SHA `9c8b3c78…`). It does not force a solver or add the frozen
training consumer overlay.

Live unregister causes activation to refuse; `rollback` restores the captured B07.7
registry bytes and wrapper symlink exactly; re-`activate` restores b0712. The
retained old wrapper is the D49 cohort variant that reads its own cohort registry,
so the rollback gate asserts the restored live registry bytes/package/symlink and a
non-b0712 `--ck-info` rather than a literal b077 `--ck-info`.

## Qualification evidence

18 CPU checks pass: `--ck-info` identity; five unsupported/protected CLI refusals;
unregister → unregistered refusal → activate; shared-lock exclusion of an exclusive
updater; runtime-identity drift, payload byte-drift and extra-file refusals; and the
live rollback cycle (rollback → B07.7 registry+wrapper restored → cohort `--ck-info`
→ reactivate → b0712 `--ck-info`). Owned drift fixtures are retired afterwards.

Nine main 600 s guarded groups pass **72 checks**: host-cli 8 (check/ck-info/seed/
existing-refusal × daily/rccl-single), installed-daily 15 and installed-rccl 15
(wrapper env/read-only mount/native origin; entry `--check` 27 identities + private
MIOpen; same-process public solver137 FP32 vs CPU double; eight-key mixed seed; six
probe batteries b072/b073/b076/b0710_public/b0710_half/b0711 against the B07.12
reference counts; four conv output/all-first-gradient fixtures; native backward),
plain-daily 6 and plain-rccl 6 (no plugin env, four gradient fixtures, backward),
core 7, onnx audit 4, daily-consumers 5, rccl-consumers 6. A separate ordinary
UID1000 interactive `doas` run passes **2 checks** with its own guard (bare env,
env file, argument with spaces, script import dir, installed activation, native
backward). **92 checks total, 10 acknowledged guards.**

### Fixes made while qualifying against the D61 release

- `deploy activate` captures the rollback state **once**; the qualification's
  unregister→activate→rollback→reactivate cycle otherwise clobbered the original
  B07.7 recovery target with an empty registry.
- The fresh baseline protects `current/torch/torch/__init__.py` by hash and reads
  links only for real symlinks: in D61 `current/torch` is a directory (the patched
  Torch source tree), not the B07.8-era symlink.
- The CLI validator invokes `--ck-info` bare; the wrapper accepts it only as the
  sole argument, not after `--runtime-profile`.
- The installed validator spawns the nested entry `--check` with the **clean
  container base** `LD_LIBRARY_PATH` (stripping the run-worker prefix). Inheriting
  the polluted worker path maps three extra ROCm dependencies
  (`rocprofiler-sdk`/`dw`/`lzma`) and drifts from the pinned 27-library identity;
  the fresh-container `bc250-ck --check` already passed.

## Preservation and durable evidence

133 protected hashes, 5 links and 4 AOT databases match; `current/torch` payload
identity is hash-protected. No native ROCm/Torch build, cache reset or deletion.
Same boot `ed0d130a-f36e-480f-9594-8fd27c03d7f6`, kernel `7.1.5-bc250rocm-r2`.
Kernel logs are unchanged through qualification/export/user-guard; no KFD warning
recurred. No reboot, .105/user-application access or boot changes. No owned
jobs/containers/armed guards remain. Ordinary free 75,098,808,320 → 74,579,607,552 B;
no existing source/build/cache/checkpoint/release/image/link removed. The B07.7
package, tools and version record are retained intact.

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`; exact completed commands:

```sh
python3 helpers/deploy_ck_plugin_b0713_on_host.py install \
  --archive /home/nik/bc250work/ck-plugin-b0713-inputs/bc250-miopen-dl-gfx1013-b0712.tar.gz \
  --receipt logs/ck-plugin-b0713-install-r1/final.json
python3 helpers/deploy_ck_plugin_b0713_on_host.py activate \
  --receipt logs/ck-plugin-b0713-activate-r1/final.json
python3 helpers/qualify_ck_plugin_b0713_on_host.py
python3 helpers/guard_ck_plugin_b0713_user_on_host.py arm
# recorded ordinary-user command (interactive doas, not persisted):
BC250_CK_BARE=caller-value bc250-ck --env BC250_CK_BARE \
  --env-file logs/ck-plugin-b0713-user-r1/user.env \
  helpers/validate_ck_plugin_b0713_user.py --sentinel "argument with spaces" \
  --output logs/ck-plugin-b0713-user-r1/result.json
python3 helpers/guard_ck_plugin_b0713_user_on_host.py finish
python3 helpers/export_ck_plugin_b0713_on_host.py
```

Existing attempt directories/receipts are preserved. Qualification PID 12043 exited
0. Logs: `WS/logs/ck-plugin-b0713-qualification-r1/final.json`, the nine
`ck-plugin-b0713-r1-*` gate directories, `ck-plugin-b0713-user-r1`, install-r1 and
activate-r1. Export at `WS/tmp/ck-plugin-b0713/export-r1`. Local evidence in
`kernel/artifacts/ck-plugin-b0713/evidence.tar.gz`: 739,076 B, SHA256
`bab9d8d20e78bd719590d2ba40c98ad2406e0c995794cb858e10a25065f9563d`; 221 raw files
(30 guard, 8 helper, 177 host-log, 3 installed, 3 installed-tool). Installed/source
helpers, manifest, registration, rollback record and archive pins verified.

## Scope and next

The plugin supplies native FP32/FP16 packed2D forward (exact alpha1/beta0,
restricted descriptors/pointers/workspace); backward uses retained native solvers.
Neither universal correctness nor graph/concurrency coverage follows from these
fixtures; no global solver choice or whole-model speedup is promised. B07.13 was the
one genuinely pending optional package; all ten work packages are now complete.
Remaining optional work (fork CI, upstream tracking, full-prefix distribution,
distributed benchmark, bottleneck optimization) is not started here. Preserve
bc250-ck/defaults/.105/user applications.