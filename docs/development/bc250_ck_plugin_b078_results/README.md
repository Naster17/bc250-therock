# B07.8: installed explicit bc250-ck option

Completed2026-10-29 on .104, tools `36da14fb2`. Only the accepted B07.7 r2 archive
was installed. Native libraries `8389197c1`, source libraries `3616b2f9f09`, Torch
`a524980`, daily/RCCL images and production architecture whitelist remain unchanged.
**New command: `bc250-ck`**, an explicit optional FP32 packed2D forward plugin;
plain Torch/train defaults are preserved. .105 was not contacted.

[Use, identities and recovery](../bc250_ck_plugin_b078_usage.md).

## Installation, identity and recovery

- Versioned package: `/opt/bc250-stack/sdks/bc250-miopen-dl-gfx1013-b077`.
- Registration: `/opt/bc250-stack/sdks/miopen-dl.json`, plus the independently
  retained version record `bc250-miopen-dl-gfx1013-b077.json`.
- Tools: `/opt/bc250-stack/tools/ck-b078`; `/usr/local/bin/bc250-ck` points to its
  root-controlled `bc250_ck.py`. Existing launcher/SDK tools are not overwritten.

Install/register/unregister acquire the canonical exclusive update lock; launches
hold the shared lock through identity checking and application exit. Installation
verifies the accepted archive SHA
`90f0d323d0bb4ffd3a8ae04f1acd4e17805a10d7dcb62d577840cf2120e06984` and manifest SHA
`f1e0f751bdb3896fa6ccc164070d08fee2f1c07de0cf65d30285162975751cb6`.
The rejected first artifact cannot satisfy those pins. Every package path/file is
checked against the root-controlled registration/manifest, along with the selected
runtime metadata hash and image; the unchanged artifact verifies eight actual ABI
libraries, native architecture and API5/66 exports inside the runtime.

The wrapper reuses the already-qualified caller-context handling to preserve bare
environment variables across `doas`. It supports normal mount/env/env-file/shm-size
options, Python script files, `--check`, `--ck-info` and non-merging `--seed-db NEWDIR`.
It delegates through the installed ordinary `bc250-torch` launcher, mounts the
package read-only and runs the unchanged verified same-process artifact entry.
It does not force solver137 or add the frozen training consumer overlay.

Live unregister causes activation to refuse; re-register restores the exact same
record and installed bytes. Data/package/tool files and recovery records remain.
Shared-lock exclusion of an exclusive updater, runtime identity drift, payload
byte drift, extra-file admission and unsupported/protected CLI cases pass. Deliberate
invalid scratch copies are retained; live native/package data was not corrupted.

## Ordinary consumer evidence

Nine main600s guarded groups pass60 checks. A separate ordinary UID1000 interactive
`doas` run passes two more checks with its own guard: bare env, env file, argument
containing spaces, script import directory, selected installed plugin path and exact
native Torch backward. No credentials are written to source, receipts or evidence.
13 host registration/identity/CLI/lock checks bring the batch to **75 checks and
10 acknowledged guards**.29 local unit tests, Black/critical Ruff/Python compilation/
diff pass; full upstream CI not run.

Both daily and rccl-single ordinary `bc250-ck` calls prove read-only installed
mounts and native Torch origin. Public solver137 executes **inside the already
verified Python/HIP process** and matches an independent CPU FP64 convolution;
max absolute error2.11277e-6 under unchanged N2.18 tolerance. This directly tests
activation after the entry's library/architecture checks, not just an environment
variable in a separate executable.

Installed reference probes exercise586 native convolution CPU-reference executions
across the two profiles, including those two same-process calls, plus244 per-ID API
and492 runtime refusals/304 retained API contracts. Genuine example DB records
replay in fresh processes without searching. Ordinary Torch output/input/weight/bias
first gradients match CPU FP64 for groups1/3 and1x1/3x3:16 fixtures across installed
and plain daily/RCCL paths. The plugin is forward-only; existing retained solvers
provide backward. Native arithmetic backward also passes after the probe batches.

Plain `bc250-torch` runs have no plugin environment and retain their existing
outputs/gradients. Core7/CPU ONNX4/daily5/RCCL6 consumer regressions pass. CPU
references/tolerances/256-byte probe guards are unchanged. There is no new speed,
arbitrary type/shape/graph/concurrency or automatic fastest-solver claim. B07.6's
two moderate performance fixtures remain the only public-route timing claim.

## Preservation and durable evidence

3804 protected hashes/36 links/four AOT databases and136 workstation backing paths
match; the extra link is the new owned command. No native ROCm/Torch build, cache
reset or deletion. Same boot `e327c5de-46b8-4e05-94ad-d999a12a06e7`,
kernel7.1.5-bc250rocm-r2. Kernel logs are unchanged through qualification/export/
cleanup; no KFD warning recurred. The B07.7 exec-after-HIP warning's underlying
kernel return remains untraced, separate from the historical Python CPU fault and
fixed VM-heap corruption. No reboot, .105/user-application access or boot changes.
No owned jobs/containers/armed guards remain. Ordinary free12,650,000,384→
12,628,176,896B; no existing source/build/cache/checkpoint/release/image/link removed.

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`; exact completed commands:

```sh
python3 helpers/deploy_ck_plugin_b078_on_host.py install --receipt logs/ck-plugin-b078-install-r1/final.json
python3 helpers/qualify_ck_plugin_b078_on_host.py
python3 helpers/export_ck_plugin_b078_on_host.py
```

Existing attempt directories/receipts must be preserved. User guard uses
`guard_ck_plugin_b078_user_on_host.py arm|finish` around the recorded ordinary-user
command in the usage fixture; authentication is interactive and not persisted.
InstallationPID7929/main qualificationPID8319/exportPID10985 exited0. Logs:
`WS/logs/ck-plugin-b078-qualification-r1/final.json`, corresponding
`WS/logs/ck-plugin-b078-r1-controller.log`, user-r1 and install-r1 directories.
Export at `WS/tmp/ck-plugin-b078/export-r1`. Local evidence in
`kernel/artifacts/ck-plugin-b078/evidence.tar.gz`: 979,892B,
SHA256 `22e958b6f3ed827fd1500ded77288cfc71d983b365466808b2a924997bca7bfb`;
164 raw files/5,386,751B. Installed/source helpers, manifest,
registration/re-registration, archive pins and retained bytes verified.

B07.9 planned: inspect existing native scalar FP16 DL forward factory/accumulator/type admission and N2.18/B07 evidence, then a bounded isolated CPU-reference factory proof if hardware-feasible before changing any registered FP32 artifact or production admission. Preserve bc250-ck/defaults/.105/user applications.
