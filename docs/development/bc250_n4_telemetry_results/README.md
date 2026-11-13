# N4: read-only telemetry provenance and semantics on the r2 release

Completed 2026-11-13 on .104 (board 1), kernel `7.1.5-bc250rocm-r2`, release
`10.2-gfx1013-20261111-138a66f52`, boot `a3142efc-e1e2-4378-8534-f184ed8c99f1`.
This re-establishes the running governor/source/overlay provenance and validates
read-only attained-clock/temperature/power/fan semantics on the **current r2
release**, updating the 2026-10-15 r3-era
[read-only telemetry audit](../../../kernel/r3/telemetry/README.md). It is a
read-only audit: **no clock, voltage, fan-policy, governor, overlay, prefix,
kernel or boot change**, and no raw SMU mailbox commands. .105 not contacted.

Helper `build_tools/bc250/qualify_n4_telemetry_on_host.py` adapts the proven
`kernel/r3/telemetry/sample-on-host.py` methodology (governor's retained original
sysfs descriptor bypasses the tmpfs overlay without unmounting; 128-byte v2.2 ABI;
whitelisted sensors + read-only DRM info ioctls; idle + one bounded 8 s native
integer GPU load + cooldown).

## Governor / source / overlay provenance

The external `cyan-skillfish-governor-smu` (PID 3044) still bind-mounts tmpfs over
`gpu_metrics` and AMDGPU hwmon `freq1_input`; those readings are governor-supplied.
`power_dpm_force_performance_level=auto` does **not** imply stock clocks — the
governor uses its SMU policy (`set-method=smu`, `fix-metrics/fix-freq=true`,
`busy-flag` utilization, safe-point 1000 MHz/700 mV, throttle 85 °C).

**The running governor's source is still not pinned**, confirming and strengthening
the r3-era finding:

- Running binary `00cd65a6…` (4,410,760 B, 2026-08-27) is a **musl cross-build**
  (`/rust/deps`, BuildID `a1de19b4…`, not stripped).
- Retained repo `/home/nik/cyan-skillfish-governor-smu` HEAD `240b0df` (branch
  `smu`, Cargo 0.4.0) native build is `e8028ca2…` (4,926,104 B, 2026-04-25) — a
  different binary; `config.toml` is locally modified (2 lines).
- The `gpu_metrics` overlay code exists in repo `src/gpu_usage_fix.rs`, but the
  **`freq1_input` overlay code is absent from `src/` of every branch** (smu,
  gpu-metrics, fix_11, dbus-performance-mode) even though the running binary
  exhibits it. The running binary's source is therefore not in the retained repo.

Full detail: [`provenance.json`](../../../kernel/artifacts/n4-telemetry-r2/provenance.json).
The retained repo is evidence, not a qualified production import.

## Read-only semantics (84 samples, ~5 Hz: 20 idle / 44 load / 20 post-load)

| Reading | Idle | GPU load active | Post-load |
| --- | --- | --- | --- |
| Governor-supplied clock | 1000 MHz | 1000–2000 (median 2000) | 1400–2000 (median 1800) |
| Kernel-translated `current_gfxclk` | 1000 MHz | 1000–2000 (median 2000) | 1400–2000 (median 1800) |
| Kernel `average_gfxclk` | 1000 MHz | 1000–2000 | 1400–2000 |
| AMDGPU hwmon `freq1_input` | 1000 MHz | 1000–2000 | 1400–2000 |
| DRM gfx clock ioctl | 1000 MHz | 1000–2000 | 2000 MHz |
| GPU edge temperature | 52–53 °C | 53–70 °C | 56–62 °C |
| CPU k10temp Tctl | 53.875–54 °C | 54–69.25 °C | 68–69.25 °C |
| Fan PWM / RPM | 90 / ~2010 | 90→175 / 2003–3458 | 175 / 3428–3488 |

**Key r2 difference from r3:** on the r3-era kernel the original
`current_gfxclk` read 2–47 MHz and `average_gfxclk` 456–953 MHz, disagreeing with
the governor. On r2 the kernel-translated clock reads sensible discrete steps
(1000/1400/1600/1800/2000 MHz) and **all clock views agree** — patched overlay ==
hwmon freq in **84/84** samples, and kernel/DRM views track the governor-set SMU
clock. The governor sets the clock via SMU and the r2 kernel reads it coherently.

## Trust boundary

- **Coherent but governor-set, not independently calibrated:** the clock views
  agree because the governor drives the SMU and every view reads that policy
  output. This is *not* physical attained-clock calibration — no raw PMFW
  `GetGfxFrequency` mailbox read was issued (kernel/daemon mailbox coordination
  remains unestablished, per the r3 audit's caution).
- **Utilization is governor-derived:** original `gpu_activity` stays the 65535
  sentinel (84/84) and the DRM GPU-load sensor returns errno 22; `gpu_busy_percent`
  is EOPNOTSUPP on this board (also recorded by PLAN-10). The governor supplies a
  derived `busy-flag` percentage through its overlay.
- **Temperatures/fan:** edge derives from `Current.GfxTemperature` (integer °C
  truncation); the fan script reads CPU Tctl, not GPU edge. Agreement is not
  absolute sensor calibration. Per-core hybrid fields remain unvalidated.

## Preservation and safety

Kernel log byte-identical before/after; mounts, governor binary/config and
fan-script hashes, and `power_dpm_force_performance_level` all unchanged; GPU
reference load passed `checks=131072 launches=58592 bounded_seconds=8`
(gfx1013:xnack-, wave32); temperature bound <80 °C held; no container/guard left
running. The boot guard acknowledged and exited.

**Safety fix made during this batch:** the first helper draft wrote a plain
`acknowledged` marker, but the boot guard only accepts `ack.json` carrying its
exact `boot_id`/`nonce`/`phase` and triggers an **emergency restart** on an
unacknowledged deadline. The guard was acknowledged correctly before its deadline
(no restart occurred) and the helper now writes the nonce-matched `ack.json`. The
clean re-run passed (84 samples).

## Evidence and reproduction

`kernel/artifacts/n4-telemetry-r2/`: `final.json` (SHA
`6130552ffcec1a309eeb0a15d99b855ccc94daa300ce51cb9bddbab695c90757`), `samples.jsonl`
(SHA `71d2b9705d332f751fadf5a48f28a0d7cd34ce6e28683120182dccb7ab9ddac8`),
`result.json`, `gpu-load.log`, byte-identical `kernel-before/after.log`,
`provenance.json`. Host attempt `KWS/logs/telemetry-n4-r2`
(`KWS=/home/nik/bc250work/kernel-r3-7.2.10`); the r3-era `telemetry-readonly`
attempt is preserved. Run `qualify_n4_telemetry_on_host.py` on .104 as root (it
takes preflight + a fresh boot guard + the exclusive runtime lock); the probe is
`KWS/telemetry-reference-load`.

## Remaining gap / next

To resolve *physical attained clock* and fully pin the governor: (1) obtain the
running governor's actual build source (the `freq1_input` overlay branch is not in
the retained repo) and qualify it; (2) a minimal guarded read-only driver
`GetGfxFrequency` (PMFW queue0 0x37) live-clock getter under the recovery procedure,
establishing kernel/daemon SMU mailbox ownership first — without changing power
policy. Keep hybrid per-core remapping and power-field semantics separate until
offsets/units are proven. Do not apply the legacy 8-core telemetry patch blindly.
Physical counters, debugger/PC-sampling/RDC applicability remain separate.