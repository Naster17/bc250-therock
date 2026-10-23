# B03.8: ordinary-launch opt-in single-device RCCL

`.104` now has an explicit `--runtime-profile rccl-single` selection in
`bc250-torch` and `bc250-train`. The daily NCCL-off/Gloo-on runtime remains
selected by default. No `.105` connection or change belongs to this batch.

Tooling revision:`36cabf9be01522bf071e090655ca88558bf70f20`.

## Packaging

Registry `/opt/bc250-stack/profiles/rccl-single` points to
`/opt/bc250-stack/dev/rccl-single-b038`. This dependent development profile reuses
the qualified B03.7 Torch build, daily ROCm/cache/consumers and the existing image.
It independently copies the13,271,416-byte RCCL DSO and relative SONAME links;
it is not a standalone relocatable distribution. No native rebuild, compiler or
ccache operation was needed. Existing sources/builds/caches/releases remain.

The installed launcher mounts `profile/rccl` read-only at `/workspace/rccl`,
prepends `/workspace/rccl/lib` to its controlled library search path, and hashes
the manifest's RCCL/Torch native identities before launch. It checks both SONAME
links and requires explicit profile selection. A user-owned private context
preserves that selection across the existing doas elevation. Startup hashing
has an I/O cost; no performance improvement is claimed.

The profile fixes `NCCL_WIN_ENABLE=0`, `NCCL_CUMEM_ENABLE=0`,
`NCCL_CUMEM_HOST_ENABLE=0`, `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`,
`NCCL_SOCKET_IFNAME=lo`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`.
Explicit conflicting launcher environment values are rejected. The ordinary
launcher holds the **daily** mutable runtime's shared lock before hashing and
through process exit. All profile/dependency updates must use that same daily
prefix's exclusive updater; the updater rejects an independent profile lock.

## Use and bounds

```sh
bc250-torch --runtime-profile rccl-single /path/to/owned_script.py
bc250-train --runtime-profile rccl-single /path/to/owned_script.py
```

An RCCL script explicitly selects `backend="nccl"`, `cuda:0`, rank0/world_size1.
The launcher option does not rewrite applications or guarantee arbitrary
world sizes. Ordinary invocations without it retain the daily Torch build.
`bc250-train` retains the same frozen Transformers5.2.0/PEFT0.18.1/Accelerate1.12.0
consumers; user applications were not inspected, modified or tested.

This is a bounded one-device profile. No multi-GPU/network transport/scaling,
full375-object protocol coverage, speed, universal BF16/FP8 numerical, higher
gradient, broad graph/concurrency, multicast/symmetric device execution or actual
offload-payload relocation claim follows. Numeric FP8 reductions remain refused;
FP8 raw-bit copies retain their qualified semantics. Historical unknown Python
CPU segfaults remain unresolved and distinct from fixed VM-heap/CSR/hostcall bugs.

## Qualification and recovery

The host qualifier uses installed launchers, without `build_command` test
injection, test-only RCCL mounts or environment patching. B03.5/B03.6/B03.7
fixtures add an explicit `--packaged` mode that directly uses the launcher's
configuration and verifies the loaded `/workspace/rccl` library. Their historical
isolated mode remains available. Independent numerical references and existing
tolerances are retained. See `verification.json` for the completed acceptance
counts, hashes, guard state and retained baseline. All1,461 checks across26
ordinary gates pass, plus five private-context GPU checks. Nine installed CPU
refusals and live shared-lock exclusion pass.372 protected hashes/four AOT DBs/
links are retained; two launcher copies are intentional changes with backups.
All27 guards ACK; accepted gate/main-batch kernel logs match. Local29 BC250 tests
and selected Ruff syntax/name/format checks pass. No full CI claim.

Package installation ran through the existing exclusive daily prefix lock.
The ordinary-launch qualifier intentionally runs without that exclusive lock:
it holds a shared daily lock for the batch, compatible with the public
launchers' shared locks. Each gate uses the existing600s watchdog guard.

On .104, `WS=/home/nik/bc250work/rocm-10.2-gfx1013`:

- `WS/helpers/package_rccl_b038_on_host.py`: single-use package installer;
  refuses an existing profile/registry and first verifies B03.7 preservation.
- `WS/helpers/qualify_rccl_b038_on_host.py`: single-use26-gate ordinary replay.
- `WS/helpers/qualify_rccl_b038_context_on_host.py`: installed refusal and
  user-owned private-context acceptance.
- `WS/logs/rccl-b038-*`: raw commands, outputs, guards and before/after evidence.
- `WS/recovery/rccl-b038-before`: original launcher/helper/validator copies and
  daily manifest. Candidate build/source/library and all logs remain preserved.

Default fallback requires no mutation: omit `--runtime-profile rccl-single`.
To restore the previous launcher and unregister the profile, use the guarded
helper (rollback not exercised in this batch):

```sh
WS=/home/nik/bc250work/rocm-10.2-gfx1013
doas python3 "$WS/helpers/update_dev_prefix.py" -- \
  python3 "$WS/helpers/package_rccl_b038_on_host.py" --rollback
```

Rollback verifies current and backup identities before replacing the two launcher
copies; it leaves the packaged release/native builds/cache/evidence intact.

Local raw evidence is preserved in `kernel/artifacts/rccl-b038`:7,580,482-byte
archiveSHA256`64beea0d3b9396f84207d5d055786b4028f7a9009612fd3f1d656a549f6a2efa`,
284 raw log files/6,078,114bytes, packaged RCCL, recovery and guard/source copies.
The archive hash and all ten host tooling hashes match local source. Daily
manifestSHA`32fd45595989e09597dbb0fcfc8992595e8861c0581a49827fcf965e35a865e0`
and boot`2db1c7c2-1b0e-47ac-a153-bc06aaa9973c` remain unchanged. No own jobs,
containers or armed guards remain at handoff. Next B05.1 audits pinned low-bit
quantization applicability on .104; `.105` remains untouched.
