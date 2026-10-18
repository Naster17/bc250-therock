# BC-250 durable implementation and hardware inventory

Audited 2026-10-14 through **code milestone `ce0025df5`**. This is the detailed
project memory; [flow.md](../../flow.md) is the short resume cursor and ordered
plan. Read the relevant section here before investigating a supposedly missing
feature. Update affected entries after substantive work, preserving evidence.

## Evidence, precedence and audit coverage

Current host manifests, exact source snapshots, committed patches and executed
reports outrank prose from older milestones. The audit covered 143 BC-250 branch
commits after base `7a803357a6e0de10c780a683aa7492da39579938`, all 17 library,
three system, two Torch and four Triton patches, component histories, retained
validation/benchmark reports, `.104` manifests/kernel/sysfs and a read-only
`.105` legacy-source inventory. Twenty project threads were present in the local
Codex state index. Private transcripts are recovery sources, not support proof.
The unrelated untracked `src/` was preserved and excluded from modification.

Distinguish **implemented**, **built**, **executed/numerically qualified**,
**unqualified**, and **hardware unavailable**. Counts describe particular scopes;
counts from different milestones overlap and must not become a coverage percentage.
No claim below establishes every public operation, shape, type or concurrency mode.

The durable entry point is the root AGENTS.md rule plus flow.md; keep the detailed
inventory on demand instead of injecting its entire history into every turn.
OpenAI documents project instructions and workspace files as continuity tools:
[AGENTS.md discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md),
[workspace memory and compaction](https://developers.openai.com/cookbook/examples/agents_sdk/building_reliable_agents_memory_compaction).
Files persist independently of conversational summaries; no hidden permanent
model memory or guaranteed perfect recall is claimed. A versioned documentation
copy under `WS/agent-memory/latest` is separate from qualified source snapshots.

This audit changed documentation only. A later documentation commit does not
invalidate the installed code revision or require a ROCm/Torch rebuild.

## Hardware: measured board versus background descriptions

Community reverse-engineering documentation describes BC-250 as an ASRock mining
APU board with Zen 2 CPU cores, shared GDDR6, DisplayPort, Ethernet and USB. Its
stock core counts and marketing architecture names describe a different
configuration from our unlocked board. No manufacturer datasheet was recovered;
use measurements and native compiler capabilities for implementation decisions.
See the [community hardware documentation](https://github.com/H6-Technologies/BC-250).

The compiler execution model is **gfx10.1.3**, not a substitute gfx1010/gfx1030
identity. LLVM lists gfx1013 as an APU and exposes wave64 as a target option;
our native qualification uses wave32/XNACK−. Target options alone do not prove
wave64 or fault-migration execution. See [LLVM's processor table](https://llvm.org/docs/AMDGPUUsage.html#processors).

| Property                    | Measured `.104` value / durable evidence                                                                                                                        |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU                         | AMD BC-250, family 23/model 71, eight cores/16 threads, SMT active; lscpu and `/sys/devices/system/cpu/online`                                                  |
| CPU instruction/cache facts | AVX/AVX2/FMA; 256 KiB L1d, 256 KiB L1i, 4 MiB L2, 8 MiB L3; one NUMA node. Do not assume AVX512.                                                                |
| PCI identity                | `0000:01:00.0`, vendor/device `1002:13fe`, subsystem `1022:0000`; DRM device uevent/sysfs                                                                       |
| VBIOS string                | `113-AMDRBN-003`; not a complete BIOS image provenance claim                                                                                                    |
| Native GPU                  | KFD target 100103; HIP `gfx1013:xnack-`                                                                                                                         |
| Topology                    | 80 SIMDs, two SIMDs/CU → 40 CU, arranged as 20 WGP; HIP multiprocessor count 20 is not a lost-CU bug                                                            |
| Execution                   | Qualified wave32, 64 KiB LDS; KFD max 20 waves/SIMD; historical HIP L2 report 4 MiB                                                                             |
| Compute queues              | KFD 24 CP queues; DRM compute ring field is a mask, not a queue count                                                                                           |
| SDMA enumeration            | Two engines, eight queues/engine; zero XGMI SDMA engines and zero P2P links. Enumeration is not user-SDMA qualification.                                        |
| Memory                      | MemTotal 15,579,344 KiB ≈14.858 GiB; GTT 15,938,355,200 bytes =15,200 MiB ≈14.844 GiB; VRAM aperture 512 MiB; KFD local_mem_size 0                              |
| Memory implication          | CPU/GPU share physical board memory. GTT and system RAM are overlapping views, not separate ~16 GB budgets. OS/cache/CPU allocations consume the same capacity. |
| PCI link sysfs              | Current/max 16 GT/s, width 16. This is an advertised link property, not measured shared-memory bandwidth.                                                       |
| Firmware fields             | KFD fw_version 144, sdma_fw_version 44, num_xcc 1. Firmware-file presence does not independently prove which image loaded.                                      |

Hardware-relevant sources include `compiler/amd-llvm/llvm/lib/Target/AMDGPU/GCNProcessors.td`,
`AMDGPU.td`, ROCr `core/runtime/isa.cpp`, the native HIP probes and
`WS/logs/baseline-torch.log`. `WS` means
`/home/nik/bc250work/rocm-10.2-gfx1013` throughout this file.

| Capability                          | Current conclusion                                                                                                                                                               |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scalar FP32/FP64, FP16/BF16 storage | Native numerical GPU paths qualified for documented operations; availability is not a throughput guarantee                                                                       |
| Low-precision atomics               | Existing HIP CAS fallback passed contention, returned-value and neighbor-guard tests; no newly implemented/native FP16/BF16 atomic instruction claim                             |
| Packed arithmetic versus dot        | Some scalar/packed-half arithmetic is usable; unsupported packed dot lowering is replaced by widened scalar operations. Do not equate all packed arithmetic with matrix support. |
| MFMA/WMMA                           | No usable native path. Never whitelist another architecture or enable matrix-only engines to bypass this. Scalar implementations of the mathematics remain possible.             |
| Wave64 / XNACK+                     | Compiler/runtime registration exists; our board execution/fault migration is unqualified                                                                                         |
| SDMA                                | Keep `HSA_ENABLE_SDMA=0`; ROCr also intentionally defaults gfx1013 user SDMA off. Explicit override is an experiment, not a supported default.                                   |
| VMM / async pools                   | APIs retained; fixed-VA two-backing VMM remap fails on both r2/r3 (K5). Experimental VM heap default stays disabled.                                                                                                     |
| GPU reset                           | Current boot and live parameter `amdgpu.gpu_recovery=0`; reset behavior unqualified                                                                                              |
| Media                               | DRM VCN decode/encode, JPEG, UVD and VCE queries return −EINVAL with zero exposed rings. Current driver exposure is not proof silicon lacks every media block.                   |
| Profiling                           | API/kernel/copy tracing works. Hardware counter tables, PC sampling, ATT and SPM are unqualified.                                                                                |
| Debugger                            | Native architecture/ELF registration patched; consumers and attach/step/wave suites not qualified                                                                                |

## Kernel, unlock, cooling and recovery

Active experimental .104 kernel: `7.2.10-bc250rocm-r3`, six guarded one-shot
boots2026-10-15. Current module0957f1aa4: bounded GPU/Torch/VMM and prior SVM
milestones pass; HPD init dummy errors and two IRQ ASSERT WARNs repaired below.
Physical display/hotplug and broader failure/recovery remain unqualified.
Persistent default/retained fallback and .105 reference remain
`7.1.5-bc250rocm-r2 #2 SMP PREEMPT_DYNAMIC` (built2026-09-16). Its recovered
original configuration is `bc250stack/kernel/configs/r2.config`; r3 migrated
config/diff and evidence are in `bc250stack/kernel/r3/evidence/`. Do not infer
configuration from unrelated7.0.10 headers.

Retained r2 fallback boot files:
`/home/nik/bc250work/bc250-rocm-lab/{vmlinuz,initramfs}-7.1.5-bc250rocm-r2`.
New r3 artifacts are `/home/nik/bc250work/kernel-r3-7.2.10/artifacts/`; entry
`bc250-rocm-r3-7.2.10-trial` defers AMDGPU until base SSH/cooling ACK.
GRUB persistent default is `bc250-rocm-r2-kernel`, timeout three seconds.
`WS/recovery/` retains kernel/initramfs, grub.cfg, grubenv, grub and cmdline.txt.
Additional lts6.18.52/n17/stable7.0.10 boot files are present, but their presence
alone is not a verified fallback boot. Use one-shot GRUB selection for a new
kernel, preserving the known working persistent default until SSH gates pass.

Relevant current boot arguments (see `/proc/cmdline` for exact root UUID):

```text
amdgpu.sg_display=0 mitigations=off amdgpu.gttsize=15200
ttm.pages_limit=3891200 ttm.page_pool_size=3891200
```

The exact TTM arguments are `ttm.pages_limit=3891200` and
`ttm.page_pool_size=3891200`; additional arguments are
`amdgpu.gpu_recovery=0`, `amdgpu.bc250_cc_write_mode=3`,
`amdgpu.bc250_flush_pasid_kiq=0`, `amdgpu.bc250_flush_by_runlist=3`, and
`bc250.vulkan_v33_r1=1`. Preserve this known profile when investigating changes.

SP5100 TCO watchdog: OpenRC feeder10s/120s, `/dev/watchdog0`, config
`/etc/conf.d/watchdog`. r3 driver builtin/nowayout1, early initramfs arms180s
until root feeder handoff; candidate-only post-sysctl guard restores panic60
and oops/lockup panic. Actual r2 feeder-countdown/reset/SSH return observed
(reset-cause flag0, not a cause confirmation). r3 idle-GPU15s deadline expiry
emergency-restarted to r2; SSH/default/new boot ID verified, then r3 rechecked.
These bounded gates do not qualify pre-watchdog hangs or arbitrary GPU hangs,
AMD reset, SDMA or fault experiments. Persistent r2 default retained.

CPU unlock service `/etc/init.d/bc250-8core-unlock` calls
`/usr/local/bin/bc250-8core-unlock boot`. The script describes SMU mailbox 0x98,
mask register 0x0115A870, stock mask 0x77→0xff and warm-reset persistence;
these descriptions are not newly executed register experiments. Read-only
recheck2026-10-15: service enabled in OpenRC default/started; eight cores/16threads
online. Installed boot path never reboots (comments retain an earlier bootloop).
Cold-start mask reapplication needs a subsequent warm reboot for enumeration;
that sequence was not newly tested. Evidence/script/service snapshots:
`kernel/r3/evidence/core-unlock-audit/`. Requested upstream
`rw-r-r-0644/bc250-core-unlock` is volatile too; its permanent option is a BIOS DXE
mod. No firmware flash or duplicate unlock applied. BIOS-level persistence is
excluded from current scope by the user2026-10-15: leave BIOS untouched and
retain the existing service/warm-reboot arrangement;
r2/SSH fallback cannot recover a pre-OS firmware failure. DMI P3.00 alone is not
proof of unmodified firmware or mod compatibility. The 40-CU
patch is retained at
`/home/nik/bc250work/bc250-gfx1013-fix/patches/kernel/40cu-bc250-unlock.patch`,
with older work under `/home/nik/bc250-40cu-unlock`. Imported performance claims
in that patch are not measurements from this audit.

Important operational facts:

- No cpufreq policy or scaling_governor exists on `.104`; do not report a
  configured CPU performance governor.
- Kernel GPU power policy is auto, but an external SMU governor is active on
  .104 (observedPID5265). Its fix-metrics/fix-freq options bind tmpfs replacements
  over `gpu_metrics` and hwmon freq1_input. Those views are governor-supplied;
  DPM/DRM use the suspect original Current clock slot. True attained clocks
  remain unqualified; auto does not mean stock clocks. Source/provenance below.
- The SMU OC service requests CPU 3500 MHz, scale −19, max_temperature 90;
  these are settings, not verified operating frequency/voltage.
- `/usr/local/bin/bc250-fan-curve` reads CPU k10temp/Tctl despite comments saying
  GPU temperature. NCT6686 pwm2/CPU_FAN1 controls the fan. The existing curve
  uses PWM90 below60°C,175 below70°C,250 below80°C,255 above. Do not assume
  hwmon numbering survives reboot. AMDGPU hwmon has no PWM attributes here.
- Retain cyan_skillfish2 CE/ME/MEC/MEC2/PFP/RLC/SDMA firmware files under
  `/lib/firmware/amdgpu`; file presence is not verified runtime load identity.
- Retain `/home/nik/bc250work/{bc250-rocm-lab,bc250-gfx1013-fix,bc250-vulkan-v33,strata-port}`,
  `/home/nik/{bc250-40cu-unlock,bc250_smu_oc,rocm-systems}`. Older Vulkan7.0.10
  module/initramfs artifacts are distinct from retained r2 and the new r3 candidate.

K6 packaging/handoff complete2026-10-15: `bc250stack/kernel/r3/handoff/README.md`
consolidates current scope, exact configs, source/signature provenance, build/sign/
install and one-shot boot/recovery recipe. Image84995f1 and module0957f1 remain
separate identities. A disposable Git index reconstructs both exact trees from
signed base+six patches;21 source/fan/binary/config/firmware artifacts checksum-
verified locally and on .104. Current source archive and binary snapshots remain
in ignored `kernel/artifacts/k6-7.2.10/`; .104 KWS/handoff/k6 retains copies.
No generated module signing private keys exported; no runtime/build/reboot/GPU requalification.
Fresh read-only artifact and loaded identity checks pass; watchdog/SSH/cooling
active, same boot a287fd16, r2 default and empty next_entry. Historical failures
below are superseded only by their recorded later fixes. Telemetry fidelity,
Python CPU crash and MIOpen timing/workspace defect remain unresolved; bounded
kernel failure/recovery limits remain. Next N2.1 solver coverage audit.

Clean-kernel workstream K1–K6, bounded outcome2026-10-15: fresh signed stable
Linux7.2.10 base`7b20d05fd38de8e0be168956f6fd241d750c7d69`, Greg fingerprint
`647F28654894E3BD457199BE38DBBDC86092693E`; separate source
`bc250stack/kernel/linux-r3` revision`84995f1a85188cde56910729214b0a68a7e402da`.
Native .104 workspace`/home/nik/bc250work/kernel-r3-7.2.10`,16jobs/20GiB ccache;
image/modules/fan driver built/signed/installed,499-entry initramfs inspected.
Initial fan sign-file path failed, then repaired using O=build tool; logs/unsigned
module retained, no kernel rebuild. Image SHA
`4531223f35bd9f313a242c2aa5d7bece0b4ffd4161d28b41271914a9b54d4558`;
initramfs SHA`0e8444c328d070e0f6d422514116e19ccb13feb9e44d0bb4b289f0d1ea3eeb11`.
Loaded kernel/AMDGPU build IDs match build artifacts; signed NCT6687 works with
existing cooling service. r2/default/artifacts/.105/ROCm/Torch preserved.

Minimal series retains CU/PASID/TTM quirks, restores diagnostic probe registers
and propagates runlist errors through existing partial-operation paths. Some
upper SVM notifier/unmap callers still discard returns/pre-mark ranges unmapped;
end-to-end failed-flush containment and failure injection remain unqualified.
Full93,517-file r2 source comparison found7 active AMD changes+5 backups versus
archived7.1+7.1.5 (archive signatures not verified). Retained/installed r2 module
payload and loaded build ID match; not bit-reproducible proof. Actual r2 SDMA
firmware bytes are Navi12 under cyan names, differing from root cyan files;
r3 privately pins actual r2 bytes. Root firmware unchanged, ROCr SDMA off.

Same N1.8 runtime/image/DSO+kpack/harness compared on r2 and r3: Lt664 references,
SparseLt134 numerical+16 API, hipTensor304 API+12 GPU and19 eager Torch CPU
references passed; workload kernel-log deltas0. Two existing Display Core IRQ
WARNs persist at AMDGPU load. Malloc/free24 rounds+pinned GPU alias/streams/events
passed, including another24-round smoke after actual recovery/second r3 boot.
Historical pre-fix full gates failed explicit fixed-VA VMM on both kernels:
handle0 saw handle1 data before update; independent GPU copy confirmed it.
K5 isolation: HIP and direct ROCr each8 mismatches/26 checks, distinct-VA anchors
correct, ordinary KFD map/unmap churn restores26/26. This identifies stale compute
translations on DRM GEM_VA, which bypassed the existing KFD runlist workaround.

Patch0003/module source`8f3c0d352643378629e5a73f4b9df96b7d38d94a` waits for PT
completion, drops BO reservations before DQM locking, flushes through the existing
opt-in BC-250 runlist path and publishes timeline completion after success.
Only AMDGPU rebuilt/installed; kernel IMAGE remains84995f1, same signed image and
initramfs. New signed module is r3-only`updates/amdgpu.ko.gz`, build/installed/
loaded ID`12ad61569200a95f1d626396fca3692753231707`, compressed SHA
`451b8a3d05cdf494b67546f9074c86f4d931481a3efd7f7b64e7bad48611dadb`.
Original r3 module and r2 artifacts/default retained; .105 unchanged.

Original full gate now passes unchanged: library counts above, Torch19, two fresh
memory processes each24 allocation/pinned/streams/events rounds+16 explicit VMM
remaps. HIP/direct ROCr each26 checks pass without diagnostic churn; workload
kernel logs identical. New boot`e2a06828-208c-47f7-9838-8a5be9f7bfd5` base/cooling
and GPU guard acknowledged; no running jobs. Evidence`kernel/r3/evidence/k5-*.json`,
pre/post isolation logs, patch and source manifest; host`SOURCE_K5.json` separates
image/module provenance. Single .ko target failed modpost companion symbols;
incremental modules target succeeded with16jobs/20GiB ccache, preserved logs.

No universal VMM, concurrent raw DRM timeline ordering, injected-failure,
performance or sustained claim. Fixed CSR/VM-heap corruption and historical
unknown Python CPU segfault remain separate; experimental heap/SDMA remain off.
Next: upper SVM notifier/unmap failed-flush lifetime containment, then Display
Core warning analysis, telemetry and K6 integration; N2.1 library cursor retained.

SVM CPU-unmap follow-up/source`67d0f189570c831efa17b524c080de07bbec485a`
(patch0004) now bounded-qualifies one returned-error path: single-device BC-250
CPU unmap holds an independent queue eviction before PT invalidation and retains
it on error after successful quiescence; successful invalidation balances it.
Full-range mapped_to_gpu commits only after all invalidations succeed. Root-only
one-shot injection targets an exact host TGID and only fires while queues are
stopped; default/verified final value0. Initial scheduler-stop failure, retry-fault
migration and map-side BO/DQM ordering are NOT thereby qualified.

Current signed r3-only module SHA
`c1d0e5a887bc3d2019472b9758cf57c920e9ef9544b3b47f6ed1a2d620f5e868`,
loaded/build/installed ID`c99a2c6dc9074760efe06d074ec0248ce5e8739b`, image84995f1
unchanged. K5 module retained independently at`KWS/recovery/svm-before`, stock
r3/r2/default/.105 preserved. New boot`eb2cc9df-5df6-4250-a0ec-1af4de205f93` base/
cooling/GPU guards ACKed, no active jobs. SOURCE_SVM.json separates image/module.

Partial and full CPU-unmap injection each pass: five prior GPU references plus
normal write control; -EIO consumed once, exactly one expected kernel error line;
queued write unexecuted through200 one-millisecond host polling samples; process
exit0 and six-reference fresh-process SVM gates before/after. Baseline six SVM
checks had passed on K5 module. Unchanged full library/Torch/VMM regression gate
passes counts above, no workload kernel-log delta. No performance, hardware-hang,
scheduler-stop, retry-fault or universal SVM safety claim. Evidence
`kernel/r3/evidence/svm-cpu-unmap/`, helpers/source manifest/README. Next audit:
SVM map runlist flush under BO reservations; move with notifier/lifetime ordering
preserved before concurrency qualification. Display warnings/telemetry follow.

SVM validation locking follow-up/source`41db89c103b183c82c05bbe099ea96ad1ddee8d1`
(patch0005): validation map/unmap now capture PT fences under BO reservations,
release reservations, then wait and invalidate under notifier lock. Caller-held
range/mmap/SVM lifetime locks retained; prior updates drain despite later VMA
errors. Tracking-allocation failure falls back to an in-place fence wait; it does
not invoke DQM with caller BO reservations held. Validation PROT_NONE bounds
corrected from bytes/pages mismatch to current-VMA inclusive page indices.
Same probe:0 target-page unmaps before,2 after; no GPU access while PROT_NONE,
restoration references passed. Two concurrent processes each54 GPU references
(24 multi-VMA attribute rounds+24 CPU discard/restores) pass, no workload dmesg
change. Both prior partial/full error-containment gates and unchanged full
Lt664/SparseLt150/hipTensor316/Torch19+two memory processes each24 allocation/
16 VMM rounds pass again. Intentional2 trace+2 error lines retained separately.

Current r3-only module SHA
`d47974d6d7eef898b50a4fdf03edb716e8cd13dfd7f2373b495f273727e5f07f`,
loaded/build/installed ID`7ee5573fc124ceb0edad1bc662feb44ebdbf129f`; image84995f1
unchanged, previous module/source at`KWS/recovery/svm-map-before`. Boot
`84debfc4-5f83-413b-9656-6a5fd3140a3a` base/cooling/GPU guard ACKed; no active
jobs. Source/GIT_REVISION and SOURCE_SVM_MAP.json record41db89c10 module delta.
Evidence`kernel/r3/evidence/svm-map-deferred/`, helpers/README/patch0005.
No lockdep/universal deadlock, OOM/fence-error/partial-map failure-lifetime,
failed-stop/resume, XNACK/retry migration or multi-GPU qualification; no performance/
sustained claim. Next: analyze existing Display Core IRQ WARNs without blanket
suppression, then telemetry/K6. r2/default/.105/runtime/SDMA0/heap-off preserved;
unknown Python CPU segfault remains separate and unresolved.

HPD/Display Core follow-up/source`0957f1aa4717bb5ea19545bdcb01d6899e94bb9f`
(patch0006): bulk init iterated six HPD+six HPDRX enums despite DCN201 only
implementing two pairs. New active-table capability query rejects dummy/empty
entries before bulk clearing, preserving real set/ack operations, connector
enabling and invalid-call diagnostics. Baseline8 dummy-ack+8 dummy-set+8 failed
clear errors and2 IRQ ASSERT WARNs →0 each after guarded boot/load. Temporary
DRM KMS debug4 traced clears1,2,7,8 and restored0. DCN2.0.1 still active with
dc=-1; both DP connectors still enumerate disconnected; taint4160, WARN bit clear.
No physical hotplug/scanout/audio/resume or other-ASIC qualification. Existing
2 SDMA0 fence fallback-expiry and workqueue truncation messages remain; SDMA off.

Current r3-only module SHA
`6ebd8393a9c7d25cd810fd89d4ef5defa187980cae64da0d37ee0d975715e8d1`,
loaded/build/installed ID`b97bca56793aecea35967385d0c68d4186ecd121`; image84995f1
unchanged, previous module/source at`KWS/recovery/hpd-before`. Boot
`a287fd16-c97b-4d88-8055-b0553a2f7631` base/cooling/GPU guard ACKed, no jobs.
SOURCE_HPD.json/sourceGIT_REVISION0957f1aa4;16jobs/20GiB ccache dependency rebuild.
Unchanged Lt664/SparseLt150/hipTensor316/Torch19 plus two processes each24 ordinary
allocation+16 VMM rounds pass; workload dmesg delta0. Evidence
`kernel/r3/evidence/hpd-capability/`, source/patch/helpers/README. r2/default/.105,
userspace, SDMA0/heap-off and separate unknown Python CPU segfault preserved.
Next: read-only eight-core SMU/live-clock/k10temp audit against retained packaging
patch0005 before policy changes, then K6; N2.1 next library item. Prior SVM failure/
lockdep/XNACK/multi-GPU limitations remain.

Read-only telemetry audit2026-10-15 completed on0957f1aa4/same boot/prefix.
Original sysfs metrics read through governor-held fd6 bypassed overlays without
unmounting/stopping it. Device0:22 vs overlay0:25 verified; this is translated
v2.2 data, not raw PMFW.84 sequential samples~5Hz around an8-second gfx1013
integer workload;131,072 exact CPU references passed, kernel log identical,
config/executable/fan-script hashes and mounts unchanged. Existing governor
PID5265 preserved; no policy/source/prefix/boot changes. Guard ACKed, no own jobs.

Governor clock reports1000MHz idle→2000MHz median during own load; original
kernel/DPM/DRM mostly2–7MHz, max47; average kernel456–953MHz remains untrusted.
Overlay views agree84/84, shared kernel views74/84 sequential samples; neither
proves physical clock. Eight CPU core groups/16online threads confirmed, while
kernel copies6 core slots: temperatures0/2 report0.63–1.88/0.62–3.56°C,1/3 zero,
6/7 clock/temperature65535. GPU activity65535/DRM load errno22; overlay utilization
is governor-derived. Legacy hybrid-layout claims fit these symptoms but full
PMFW offsets/core identities were not qualified. No CPU-offline/raw-mailbox test.

Reported edge51–68°C versus original GFX51.25–68.25°C; hwmon truncation gives
−0.75..0°C difference in samples. CPU Tctl52.625–67.125°C; fan uses that CPU
sensor, existing PWM90→175/RPM~2k→3.5k. Absolute calibration/maximum thermal or
stability claim excluded. Installed governorSHA00cd65a6... and candidate built
SHAe8028ca2... differ; reference repo240b0df/Cargo0.4.0 lacks observed freq-overlay
code, so running executable/source identity unresolved. Local read-only archive
`kernel/artifacts/telemetry-readonly/governor-reference-240b0df.tar` (SHA in report)
preserves reference source, original repo/history untouched. Evidence/report:
`kernel/r3/evidence/telemetry-readonly/`, `kernel/r3/telemetry/README.md`.

Next pin active governor source/overlay behavior and kernel/daemon mailbox
coordination before a minimal guarded live-clock getter. Retained packaging0005
cannot be applied blindly: core count does not prove PMFW map; instantaneous
power is labeled average, and live-query failure falls back to the suspect slot.
Keep per-core/power semantics separate. K6/N2.1 retained; SDMA0/heap-off, prior
SVM failure/lockdep/physical-display limits and Python CPU segfault remain open.

Read-only kernel review2026-10-14:
[report](bc250_kernel_audit_20261014.md), [hash/file evidence](bc250_kernel_audit_20261014.json),
[active AMD delta](bc250_kernel_audit_20261014.patch). Reconstructed AMD subtree
from .105's archived7.1+7.1.5 patch:7 active changed files (+136/-3),2762 identical
and5 inactive `.orig` backups. Archive signatures/rest of Linux remain unaudited.
Installed AMDGPU modules match across hosts; loaded/initramfs identity not proved.
That initial identity gap was subsequently closed by the byte/build-ID comparison
above; retain this audit as a historical snapshot, not the current cursor.
Source concerns: ignored runlist-flush errors at all four custom call sites;
probe modes1/4 fail to restore RLC (current boot uses3); retained eight-core
telemetry patch0005 is absent. Existing boot log has two Display Core IRQ WARNs,
so unchanged workload logs are not evidence of a warning-free boot. No kernel
cause for Python segfault established; no live changes/build/reboot this review.
Before a new kernel: verify upstream base/full delta, review minimal patches and
prepare separately versioned candidate with one-shot boot/SSH fallback preserved.

## Installed stack and precise provenance

| Item                   | Current code / location                                                                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Branch                 | `users/Naster17/rocm-10.2-gfx1013`                                                                                                                           |
| Qualified TheRock code | `0115c01f06f8654b26b5a665286069caf89eda2d` (N1.8); newer docs commits need no runtime rebuild                                                                       |
| Libraries              | `08ba6f386ff661e720873779fc417b4246787d9c`                                                                                                                   |
| Systems                | `329eaf726632ab0136924804cbc1321ae09fc126`                                                                                                                   |
| LLVM                   | Upstream `7667f52dd687899256a3285ab0bce3c53bc73394`; no custom LLVM architecture patch                                                                       |
| Torch                  | `4ee65717d69e9f51021272e3f37d0f342f68fc08`; release/2.13 baseline `c779020750d7dc99a39ea8c104ef8df1fc9d0755`                                                 |
| Triton                 | `ee65ff3b85f99d52e678b252a99c8425fa46e329`; baseline `66b3f9ff575eb8db6adec658bbf0badd9a8d2e25`                                                              |
| Distribution/version   | TheRock 10.2.0; HIP component reports 7.17.0; Torch 2.13.0a0+gfx1013, qualified Triton 3.8 wheel. Different component version strings are not a failed port. |
| Active link            | `/opt/bc250-stack/current` → `/opt/bc250-stack/dev/priority1`; directory name retained, runtime profile `gfx1013-priority2-n18-dev`                              |
| ROCm / Torch           | `WS/build/dist/rocm`, editable `WS/pytorch-work`                                                                                                             |
| Runtime image ID       | `5df0a58392de13916819f6d84342b0338ae2c2bbb2f3c21a1f3ad38be2724a59`                                                                                           |
| Runtime tag            | `localhost/bc250-stack:10.2-gfx1013-priority1-candidate`; tag name predates P2, immutable ID above is authoritative                                          |
| Build image            | `localhost/bc250-rocm-build:10.2-ccache`; Ubuntu24.04/glibc inside Alpine host                                                                               |
| Triton wheel SHA256    | `2d65d6fefe359116521640e3cc2257bd03b4b24f6877fc8e28946ee9c57544d8`                                                                                           |
| Candidate              | `priority2-20261014` payload retired20261016; `runtime.retired.json` and validation retained as historical evidence, no longer executable                     |
| Recovery link          | `/opt/bc250-stack/before-dev-current` → retained `10.2-gfx1013-20261013-4b000d542` release                                                                   |
| Software fallback      | `previous` and `before-dev-current` → retained `10.2-gfx1013-20261013-4b000d542`; initial20261012 payload retired20261016                                       |

Storage refresh20261018 (code780e67ceb) recovered11.96GiB: free1.08->13.04GiB.
Explicit user cleanup authority supersedes retention only for inspected obsolete
compiled outputs:8,801 .o/.a/.so files in three inactive Torch build trees,
21 Triton bin/test compiled files,90 synthetic CLI fixture files and9 stopped
project build containers. All source/generated source/config/directory/link paths,
current A09 build/shared third_party source, ROCm build, caches, installed binaries,
images/releases, user data/recovery/compat/boot fallbacks/.105/src remain. Missing
retired generated targets must rebuild; deleted synthetic payloads require a new
fixture label.101 protected hashes unchanged; all images/user neural-p1/current/
previous/runtime metadata unchanged. Five8-group current/fallback/fresh-Triton GPU
gates pass,600s guards ACK, whole-batch dmesg identical. No native build/cache reset/
package/default/boot change or active job/container/guard. [Exact plan/journal and
report](bc250_storage_cleanup_20261018/README.md). Next N2.12 unchanged; CPU Python
fault remains unresolved/separate from fixed VM-heap/CSR.

Storage cleanup20261016 (implementation/evidence `a26b5621b666d08dc18257208a6a68d21b7bb21e`)
recovered29.13GiB on .104; available12.2->41.3GiB. User explicitly authorized
retiring inspected obsolete payloads,56 stopped containers,2 image records and
our synthetic test binaries/2 orphan contexts. Runtime hashes stayed unchanged;
the active runtime and retained frozen fallback passed bounded GPU gates; `previous` now selects that
newer fallback. All sources/incremental builds/caches, genuine user checkpoints,
user neural container, boot fallback, .105 and `src/` remain. Local Alpine/gfx1103
artifacts belong to a separate project and were excluded. Full deletion journal,
protected scope, limitations and test evidence:
[storage cleanup report](bc250_storage_cleanup_20261016/README.md).

`.104` exported source trees are gitless. Inspect `WS/source/GIT_REVISION`,
`WS/source/rocm-libraries/GIT_REVISION`, and both Torch GIT_REVISION files.
`SOURCE_PRIORITY2.json` records 51 transferred file hashes and code provenance;
workspace-root/dev/candidate copies exist. `SOURCE_PRIORITY1.json` and
`SOURCE_CANDIDATE.json` retain their older meanings. Source archives are baselines,
not magically updated by patching extracted trees. Generated torch.version's
Git label may retain the original export revision; use source manifests instead.

Reconstruction:27 library patches and four system patches in
`patches/amd-mainline/{rocm-libraries,rocm-systems}` reconstruct the exact A01
component trees. Two base Torch patches and four Triton patches remain in
`patches/{bc250-pytorch,bc250-triton}`. The legacy P2 helper does not include three
later SDPA file deltas in Torch4ee6571; its current overall mismatch is retained
in the A01 reconstruction report as a separate packaging gap. Use the complete
qualified patched Torch Git history/source, not that incomplete patch subset.
Apply each recorded component series in order. The P2 reconstruction helper verified
exact library/system/Torch trees using private indexes; it did not recreate the
entire compiler or claim a clean-room full rebuild. Nested committed branches
and `.smrev` metadata preserve the port independently from root submodule pins.
Do not reset sources or casually change pinned submodules. N1.1 adds
`SOURCE_N11.json` (7 verified changed files, qualified root/library revisions,
active hipTensor hash) on the P2 snapshot; P2 manifests/candidate stay unchanged.
N1.1 library reconstruction matched all18 ordered patches exactly. N1.2 adds
`SOURCE_N12.json` (8 verified changed files) on N1.1/P2 and exact19-patch
library reconstruction; retained earlier manifests keep their original meaning. N1.3 adds
`SOURCE_N13.json` (8 verified changed files), exact20-patch reconstruction.

## Library patch index

| Patch | Durable change / commit anchor                                                         |
| ----- | -------------------------------------------------------------------------------------- |
| 0001  | Native rocBLAS/Tensile target, capability/library predicates: `55e9abea68b`            |
| 0002  | MIOpen HIP-direct convolution applicability: `b6b37dcdf0d`                             |
| 0003  | Honest hipBLASLt HOST_API_ONLY build: `9ca9c61165f`                                    |
| 0004  | rocRAND native host/device dispatch: `039b79bcb43`                                     |
| 0005  | CK scalar DL/generic CK Tile architecture integration: `3f6e5c01b3c`                   |
| 0006  | Tensile CMake native target validation: `f4302e457a9`                                  |
| 0007  | Omit device-management discovery for host-only Lt: `2efc3717efe`                       |
| 0008  | hipTensor generic-only native profile: `a9e578151e5`                                   |
| 0009  | Execute generic arithmetic cases rather than silently skip: `b35368dd8f5`              |
| 0010  | RPP audio FFTS object dependency: `0a912025cc0`                                        |
| 0011  | Isolate generated headers/read-only sources and generic CK requirements: `e0eb4c93e7a` |
| 0012  | rocPRIM native dispatch in standalone and two hipCCL copies: `a8fc909de24`             |
| 0013  | RPP spectrogram destination row stride: `b9c8ad41e5b`                                  |
| 0014  | Native scalar Lt/tensor/sparse backends and generic CK operations: `a6656565253`       |
| 0015  | Wider types/epilogues/pruning, generic operators, decibel fixes: `7bc253117ee`         |
| 0016  | Lt C++/grouped consumers, CK Tile, hipDNN fusion/provider integration: `f12e43e3f9b`   |
| 0017  | Register-tiled Lt GEMM, tensor unary expansion, sparse activations: `01566e3144d`      |
| 0018  | hipTensor unilateral scalar contraction modes: `681d2f8a25d` |
| 0019  | hipTensor native C32F/C64F scalar contractions and identity/NEG/CONJ: `c6ac34f0ba3` |
| 0020  | hipTensor plan API validation and honest scalar preference contracts: `18983cf7497` |
| 0021  | Sparse Lt zero-stride broadcast/physical compression and span guards: `40e2c04af95` |
| 0022  | Sparse Lt native row bias pointer/stride/type and GPU epilogue: `11e165bd0bf` |
| 0023  | Sparse Lt device FP32 per-row alpha-vector flag/indexing and guards: `a9a1ace297c` |
| 0024  | Sparse Lt INT8 input/output, bounded INT32 accumulation, FP32 epilogue and saturating round-even conversion: `2fd5619d7c8` |
| 0025  | Lt 32-lane subgroup short small/skinny GEMMs and bounded dispatch: `48e1e564d6f` |
| 0026  | Preserve validated launch-grid bound for Lt subgroup dispatch: `a6f7d3d8ba7` |
| 0027 | A01 MIOpen dilated GEMM workspace query: `08ba6f386ff` |
| 0028 | A02 native gfx1013 bundled legacy CK scalar profile: `2093090048f` |
| 0029 | A02 BF16 reduction conversions/scaling and cache-key isolation: `d375d8a2ea7` |
| 0030 | A03 capability-aware optional CK plugin probing: `fcf7a7142af` |
| 0031 | A08 native dense INT8/INT32 Lt, modulo arithmetic and typed contracts: `319a235d6c6` |

System patches: build-time ROCr hotswap default; native ROCdbgapi architecture
registration (`fa95eec5a2`); gfx1013 safe pool default (`dbdb7b8d1c`). The first
patch has a synthetic format-patch ID; its applied tree, not that ID, is evidence.
Torch patches: NHWC average-pool gradient padding (`b36877506d3`), qualified
native Lt backend selection (`4f13bf8647`). Triton patches: widened low-precision
dot (`c815e192`), signed/rounding tests (`63915ab7`), BF16 multiplication fallback
(`270bfb9d`), RDNA1 buffer flags (`ee65ff3b`).

## Component coverage: implemented versus existing native backend

**A09/N4 CPU-first profiler fix installed2026-10-18:** native Torchb823a1b,
final testsda45369, TheRock build669baa5d7/final controllers39f3ed14b. First ROCm
Kineto initialization retains available GPU capability, independent of first
capture selection; no visible GPU keeps CPU-only backend. CPU-first warm/cold,
GPU-first and hidden-device cases pass against candidate/installed. Each regular
GPU window13 kernels/two pinned copies, correlated positive-duration records;
exact profiled/unprofiled output/both gradients, CPU float64 references and
nondefault stream. CPU-selected GPU work omits GPU records; cold CPU capture
leaves Torch lazy init false (HIP availability/backend can initialize). API7/
core7 pass in both mounts, plus fresh installed reactivation. Only libtorch_cpu.so
changes; libtorch_hip.so identical. Independent pip-only candidate
WS/pytorch-profiler-order, ccache20GiB/jobs12/link1/native gfx1013, old full A09
build and shared immutable third_party preserved.104 original identities verified,
old runtime.json and CPU alias pinned via WS/recovery/profiler-order-before;
SOURCE_PROFILER_ORDER.json records active/recovery cohorts.21 guards ACK/600s,
18 successful process gates and three honest failed fixtures retained (small
copy expectation/two invalid -1 visibility values). First controller final hash
check auto-rolled back an old-hash-on-active-alias accounting mistake; corrected
old identity pin, qualified installed hash and fresh reactivation pass. Dmesg
unchanged whole batch. No ROCm DSO/packs/image/kernel/boot/default/.105 changes;
only development Torch link/runtime metadata. Free~10.13GiB/no active jobs.
[Report/results/recovery](bc250_profiler_order_results/README.md); patch0009 mbox
on4af4a15 includes nativeb823 plus two test-only fixture corrections; native
GIT_REVISION staysb823, final testsda453 separately identified. No performance/
LoRA gain or physical CPU-only host/counters/daemon/external concurrency/graphs/
long trace/cross-boot qualification. Python CPU segfault unresolved/separate
from fixed VM-heap/CSR; explicit SDK import aborts are separate. Next N2.13
generic sparse triangular vector/multi-RHS coverage, preserving legacy P2 and
keeping general sparse spsolve/factorization B02 separate.

**A08 completed2026-10-18: native dense INT8->INT32.** Libraries
`319a235d6c6496f0da37be44dfa60eaf31a29f82`, runtime helpers
`8f0dfb671e88b31bc3f5bb4a4958acefdafde556`; Torch4af4a15 unchanged.
The native gfx1013 Lt backend admits I8 A/B, I32 C/D/scalars, COMPUTE_32I,
modulo32 arithmetic, row/column/N/T/padded/offset/batched/broadcast layouts,
host/device scalar alpha/beta and DEFAULT/BIAS/RELU/RELU_BIAS. Existing
`torch._int_mm` works without Torch rebuild/launcher hooks.99 direct exact CPU
cases+11 refusals,16 public Torch configurations/out/custom streams+five
refusals pass; GPU trace confirms gemm_int8.1628 existing floating Lt cases,
core7/profiler7/sparse38/conv30/A0642/A0736/actual2B3 pass.12 guards ACK/600s/
dmesg unchanged,61 protected hashes unchanged. DSO+kpack paired recovery in
WS/recovery/dense-int8-a08-before, SOURCE_DENSE_INT8_A08.json in WS/current.
[Report/usage/timings](bc250_dense_int8_a08_results/README.md). Small eager timing
is mixed versus FP32 GEMM+cast; no universal or LoRA speedup. No quantization
recipe, packed low-bit/QLoRA, integer autograd or compile/graph qualification.
Other scales/compute modes (including public pedantic), pointer vectors and
nonlinear/aux epilogues remain refused. No active job/container/guard deadline;
disk~1.10GiB free, preserve existing sources/builds/caches. No kernel/.105/default
BLAS/SDMA0/heap-off/image change. Historical CPU Python fault remains separate
from fixed VM-heap/CSR. A01–A09 bounded scopes complete; next **N2.7**, existing
small Schur/device-side QR iteration roadmap with default host dispatch retained.

**A09 native Torch GPU profiler — 2026-10-18.** User selected A09 ahead of
unfinished A08. Torch `4af4a1508a6ec0b7b2b29f5e0115f8477c4d1bb5` adds tests on
unchanged native source71a0426; TheRock runtime helpers `4c2edb785` enable
USE_KINETO=1/KINETO_BACKEND=rocm with the pinned rocprofiler-sdk backend.
Seven profiler tests require actual GPU kernels/copies, positive durations,
CPU/GPU correlation, FP32/BF16 output/gradient references, allocation/free events,
scheduled capture, custom streams and CPU-only selection. Profiling is explicit;
ordinary training does not automatically collect traces or become faster.
Candidate short-window setup/teardown-inclusive wall overhead was~4.70x;
not application throughput. Existing convolution/sparse/training/A06/A07/model/
resume gates pass within their recorded scopes. [Evidence and usage](bc250_profiler_a09_results/README.md).
Hardware counters, broad simultaneous/external profiling, compile/graphs,
long traces and cross-boot absolute-clock calibration remain unqualified.
Final exclusive-lock switch to WS/pytorch-profiler-a09 passes four installed
gates;47 protected old/runtime/SDK/app files unchanged. Candidate CPU/Python
DSOs changed, GPU DSO byte-identical.14 guards ACK/600s/dmesg unchanged, no active
agent job/container/deadline. Recovery WS/recovery/profiler-a09-before and
SOURCE_PROFILER_A09.json retain provenance. Disk~2.03GiB free; preserve old build
and shared read-only third-party sources when rebuilding.
A08 remains next, then N2.7. Historical Python CPU fault stays unresolved and
separate from fixed VM-heap/CSR corruption. No .105/kernel/BIOS/image change.

**A06 correction and A07 head256 qualified on .104 — 2026-10-18.** Torch
`71a0426566bc9b952d9aa9b41f4a9d79ce46e0e8`, runtime helpers `e4f4b3854`.
Low-precision/autocast full gated-delta uses reference-ordered GPU preparation and
scan; non-autocast FP32/native standalone preparation remains. All372 nonzero
adapter gradients match exactly in BF16 actual2B65/1024 fixtures. A07 extends
explicit attention through D256: scalar tiled FP32/FP16/inference, BF16 D>128
training math-forward/native-LSE+tiled-backward hybrid.36 operator/dispatch tests,
42 A06 tests, four exact autocast cases, three actual-model train/checkpoint cases
and exact fresh/resume CLI checks within math and tiled pass. Actual-model A07
worst gradient relativeL2 is2.17%/1.31%, within unchanged4%;12 head256 calls prove
forward+checkpoint recomputation. The fixture now explicitly calls `.train()`;
earlier actual-model tests that only enabled checkpointing did not establish its
execution. BF16 hybrid lowers isolated peak memory but is slower; forward temporary
allocations remain quadratic. Math stays default; `--attention-backend tiled` is
an explicit new-run option with a distinct fingerprint. No automatic speedup claim.
Full FLA/dropout/compile/graphs/higher-order remain unqualified. [Report and evidence](bc250_a06_a07_results/README.md).
Final exclusive-lock deployment passes eight installed gates; five source/app
hashes match and39 protected runtime hashes remain unchanged. Recovery
`WS/recovery/a06-a07-before-r4`, provenance `WS/SOURCE_A06_A07.json`. At resume
r2 was active; documented one-shot r3 base/GPU/VMM recovery gates passed, with r2
still persistent fallback.32 GPU guards all acknowledged with unchanged per-run
dmesg; failed harness/numerical/legacy-caller attempts retained separately. No
active agent job/container/deadline. Next primary **A08**, then A09, then N2.7. Historical Python CPU segfault remains
unresolved and distinct from the fixed VM-heap/CSR corruption. Earlier failed A06
review and original timing evidence below remain historical and are superseded
only within this corrected scope.

**A06 recheck2026-10-17: BF16 LoRA qualification reopened; production unchanged.**
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
or universal gain evidence. [Review](bc250_gated_delta_a06_review/README.md)
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

**A06 completed2026-10-17 (first native substage):** Torch
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
[report](bc250_gated_delta_a06_results/README.md).
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

**Sparse/lint follow-up completed2026-10-16:** user-requested A04/A05 gap closure
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
exact lint/timing protocol, errors and evidence in [follow-up report](bc250_sparse_followup_results/README.md).
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


**A05 completed2026-10-16:** Torch6181d00b5beb8cc9fd6800da7a19c8abf756b30d,
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
usage/evidence in [A05 report](bc250_training_a05_results/README.md), raw logs
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

**A04 completed2026-10-16:** Torch8cdfcc8e0feb3689a1ec1d292539af083b13ea94,
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
scope in [A04 report](bc250_torch_a04_results/README.md). Native scalar same-dtype
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


**A03 completed20261016:** libraries fcf7a7142af8c258164e106e66bf893c7f563d15,
source/helpers7d9176a45 (initial452ac72e7); installed .104. Optional CK probe now
respects build flag/existing solver whitelist, explicit path retains diagnostics.
No modern plugin kernels added; B07 remains optional. Genuine native gfx1013_20
system FindDb16 keys/58 CPU-checked immediate solutions, one searched HIP-direct
PerfDb record and41 compiled kernel records installed with provenance. Normal
unforced alternatives checked after the isolated search. Read-only installed
replay uses empty user FindDb/PerfDb and zero new user kernels; searched config
load logged. Unseen geometry fwd/dx/dw falls back and writes three user records.
30 Torch convolutions, A02 smoke3 and seven core groups pass. All16 GPU guards
ACK/600s and dmesg identical. Initial WAL-mode package failed readonly replay;
consistent SQLite backup now uses DELETE journal mode; prior cache/package kept.
Initial deployment precheck mistook expected Info-level empty-user notices for
system warnings and stopped before mutation; severity check corrected. No global
warning suppression. Source/DSO and three native DBs plus manifest installed under
exclusive runtime lock; rollback WS/recovery/miopen-a03-before prepared/not used.
DSO SHA3a751adc149778f061198014d6f086a75c33e7640b9473dd75bf14464652afce,
11 protected runtime/fallback/shared-heuristic hashes unchanged. Exact payloads:
[report/manifest](bc250_miopen_a03_results/README.md); raw WS/logs/miopen-a03-* and
local ignored kernel/artifacts/miopen-a03/. Binary bundle retained locally/.104;
text assets and hashes committed.30 library patches reconstruct exactly. Existing
historical private caches preserved, not promoted. No universal tuning/speedup/
coverage or cross-board/compiler qualification; compiled search candidates are
not all numerically checked. Python CPU fault separate from fixed VM-heap/CSR;
A01 clock/reboot/trace and Torch SDPA patch-bundle limits retained. No active agent
build/test/container/deadline. Exact next A04: inspect patched Torch hipSPARSELt
CMake discovery/version gates and qualify real compress/sparse-mm integration.
A04–A09 before N2.7; K6 completed. Kernel/BIOS/.105/user files unchanged.

**A02 completed20261016:** libraries d375d8a2ea7280170455d133058c3cf8e16f58b6,
TheRock source/helpers d17427bfeff6ab7204855a753a883c9a66fbf51b; installed .104.
Bundled legacy CK accepts the native gfx1013 target, uses RDNA1 descriptor0x31014000
and scalar FMA, disables unsupported packed-dot asm defaults and rejects XDLOPS.
Wave32 assembly checked. BF16 enum/raw-buffer/load conversions and compute-
precision alpha/beta epilogues corrected. Reduction cache keys versioned because
embedded header changes otherwise reuse older compiled kernels; caches preserved.
Installed54 numerical FP32/FP16/BF16 reductions with FP32 compute,3 unsupported
BF16 compute refusals and12 FP32/FP16 LSTM/GRU/tanh-RNN output/all-gradient cases
pass. Thread/wave/block/multiblock, scales/beta, guards, one strided custom stream
and FP32 min/max indices covered.30 convolution,32 stable event pairs and seven
core reference groups pass. All14 diagnostic/GPU guards ACK/600s, dmesg identical,
including retained baseline/intermediate failures. No general speedup/coverage
claim; BF16 RNN, wider NaN/index/compute/recurrent contracts unqualified.
Only versioned MIOpen DSO replaced under exclusive lock, installed SHA f4b1a10ce20b;
eight protected HSA/HIP/Torch/BLAS/kpack/frozen-fallback hashes unchanged. Recovery
WS/recovery/miopen-a02-before; installed WS/SOURCE_MIOPEN_A02.json.29 library
patches reconstruct exactly. [Report and evidence](bc250_miopen_a02_results/README.md).
No active agent build/test/container/deadline. Python CPU fault remains distinct
from fixed VM-heap/CSR; prior Torch SDPA patch packaging gap retained. A01 clock
limits and remeasurement caveat remain. Exact next A03: inspect actual gfx1013_20
DB/cache records, bounded tuning/replay/package provenance and capability-aware
optional plugin probing; missing system DB/plugin warnings remain A03/B07. No
fabricated DBs/architecture renaming. A03–A09 before N2.7, K6 completed.

**A01 completed20261016:** source/helper380a403a1, systems329eaf7266,
libraries08ba6f386ff installed in current development prefix. Direct C API and
retained events isolated an approximately14,360s shader/KFD counter epoch offset,
not a Torch dispatch fault. ROCr's existing epoch correction now applies to native
gfx1013, with Linux detection requiring a completed timestamp >1s in the future.
MIOpen's obsolete dilated-GEMM workspace exclusion removed; cold query480 bytes,
warm hybrid zero-workspace selection remains valid. Installed15 direct CPU fwd/
dx/dw configurations,30 Torch convolutions,32 stable event pairs/six raw-counter
references and seven core groups pass;21 total diagnostic/GPU guards ACKed and
dmesg identical. Failed attempts retained. No kernel/boot change; old DSOs and
metadata in WS/recovery/miopen-a01-before, active SOURCE_MIOPEN_A01.json records
exact hashes/revisions. Frozen fallback/HIP/Torch/BLAS/kpack hashes unchanged.
[Report, source, failures and limits](bc250_miopen_a01_results/README.md).
Absolute CPU/GPU trace correlation, opposite epoch/reset/reboot unqualified;
earlier GPU-event speed ratios (including N1.8) require remeasurement. Numerical
qualification and host-wall-time results remain separate. A02 legacy CK reduction/
LSTM backward is corrected above; A03/B07 plugin/DB gaps and Python CPU fault remain
open, distinct from fixed VM-heap/CSR. Source export component GIT_REVISION labels
updated; the unchanged superbuild baseline label remains7f73 with effective port
source380a403a1 recorded separately. No active agent job/deadline; N2.7 queued.

**Baseline Torch dependency/dispatch audit2026-10-16**, source7e39ed699
(A01–A04 updates above supersede its resolved failure/availability status):
[support matrix and ordered A01–A09/B01–B07 queue](bc250_torch_support_matrix.md),
[raw evidence](bc250_torch_support_audit/README.md).21 DSOs examined,19 load;
RCCL/gfx1013 CK grouped-conv plugin absent. All3520 dispatcher names recorded,
not a coverage percentage.51 unique targeted configurations:35 bounded passes,
6 reproduced MIOpen failure variants across two families,10 unavailable/refused.
Group1 convolution needs grad: missing workspace/elapsed<=0/no suitable solver;
GEMM-off diagnostic still fails. LSTM backward separately fails HIPRTC because
MIOpen's bundled legacy CK reduction config has no gfx1013 macro. This is not the
standalone CK Tile port or optional CK grouped-conv plugin. No gfx1013 system
FindDb/PerfDb/KDB; only three shared heuristic files, user ufdb/ukdb exist.
HIP20 multiprocessors reflects40-CU WGP mode; gfx1013_20 db key is correct.
Torch hipsparselt_DIR-NOTFOUND despite installed CMake package/DSO; direct N1
qualification does not imply Torch integration. Valid dense INT8 reaches Lt but
is unsupported; BF16 FFT/general sparse solve/efficient-flash/head256 attention/
distributed/Kineto gaps retained. FP16 power-of-two FFT, CSC/BSR, fused AdamW,
AMP/compile/graph and ordinary operators pass bounded probes. Existing Qwen
conv/loss/resume changes remain app scoped. No runtime/source-library/kernel
change or native rebuild; guards ACKed/dmesg identical/no active agent job.
Baseline reproductions were followed by A01–A04 corrections above. These baseline
counts remain historical; exact next A05. Selected Torch/library queue precedes N2.7.

| Component                                   | What exists and was executed                                                                                                                                                                                 | Boundary / where to continue                                                                                                                                |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLVM/Clang/device libs/COMGR/HIPRTC         | Existing upstream gfx1013 support; native compile/runtime compile, builtin probes, Triton instruction-selection checks and GPU kernels                                                                       | Not a newly written compiler target; wider builtin/debug/offload suites remain                                                                              |
| ROCr/HIP/CLR                                | Native allocation/copy/stream/churn tests; safe pool default; VMM and async APIs retained                                                                                                                    | SDMA/reset/VM heap/fault migration are separate controlled experiments                                                                                      |
| Low-precision atomic APIs                   | Existing HIP CAS fallback qualified by `hip_atomic_low_precision.cpp`, contention/returned values/neighbor guards                                                                                            | Do not patch working atomic headers or claim native low-precision atomic instructions                                                                       |
| rocBLAS/Tensile + hipBLAS                   | Native identity and scalar capability/predicate patches, 570 fallback kernels; large SGEMM and four Torch precisions; P2 BLAS18                                                                              | No complete tuned database/BLAS suite. hipBLAS uses this backend, not custom new kernels.                                                                   |
| hipBLASLt | Prior964 C/C++ cases +N1.8 new664,403/561 regressions; f32/f16/bf16 same/mixed output, layouts/transposes/broadcast batches, host/device scalars, bias/activation/AUX/DGELU, grouped lifetime/events; tiled/register and short subgroup paths.36 paired graph timing fixtures | Subgroup K<=64 and min(M,N)<=4 or both<=32/area<=512; FP32 tree order differs. Selected graph kernel ratios1.14–1.86x, controls~flat; no whole-Torch/uncaptured latency claim. FP8/INT8/FP64/device groups/tuning unsupported. Default Torch BLAS remains rocBLAS. |
| hipTensor generic                           | Historical154 rank-two and23 higher-rank reduction/permutation/binary/trinary checks in generic-only profile                                                                                                 | Wider rank/layout/operators remain; these counts are separate from contractions                                                                             |
| hipTensor contraction | N1.1 real273 API cases (253 numerical +20 descriptor); N1.2 complex450 (426 numerical +24 descriptor) and273 real regressions. Same-type C32F/C64F, identity/NEG/CONJ, public complex scalars, shared/unilateral sums, padded/permuted layouts, zero/imaginary scalars, span/alias/guard bytes, in-place conjugated C/D,16-mode union with unit axes, plan lifetime and3algorithms×3workspace preferences/type with JIT NONE | N1.3:304 plan API checks +12 numerical executions across four real/complex types; pointer/size/enum guards, atomic failed setters, owning plans, zero required workspace. Scalar cache NONE/JIT NONE/autotune NONE/count0/rank0; alternative requests explicitly unsupported. Broader cache/JIT implementations remain optional. Mixed/widened complex and other complex unary ops unsupported; small-fixture timing is not general performance. |
| hipSPARSELt | f32/f16/bf16 retained; N1.7 INT8→INT8/COMPUTE_32I:134 numerical +16 API;234 scalar/609 broadcast/567 bias/438 alpha-vector regressions. Sparse A/B, layouts/transposes/broadcasts, strip/TILE/partial tiles, FP32 bias/alpha vector and plan/alias/guard checks | INT8 K multiple4 <=262140, FP32 alpha/beta/bias, nearest-even saturation. INT8 sigmoid/tanh enable and widened outputs unsupported. Matching logical batch counts; beta vector/split-K/multi-stream unimplemented. Borrowed vectors/bias live until completion; separate sparse pack, no hardware sparse speedup claim. |
| CK scalar DL                                | CPU-checked FP32/FP16/signedINT8 GEMM examples; native architecture applicability changes                                                                                                                    | Admitted DL convolution/grouped classes are not automatically executed/qualified                                                                            |
| CK Tile                                     | Prior19 generic runs; P1 numerical247: top-k60, LayerNorm18, scalarGEMM160, Pool3D9;400 argument rejections; installed public ScalarGemmKernel                                                               | Explicit generic API, not automatic matrix-kernel replacement; more ops/types/layouts remain                                                                |
| MIOpen | CPU-reference convolutions, A02 native legacy reductions/recurrent gradients; A03 genuine seed FindDb16 keys/58 checked solutions, PerfDb1/KDB41 and normal replay/fallback | Small recorded scope; wider tuning/types/layouts/NaN/index/recurrent contracts unqualified; modern CK plugin B07 absent |
| hipDNN providers                            | Historical pass-by-value5 +HIP9 +MIOpen9, scalar-expansion14; P1 Lt48 +HIP18 +3D18 +fusion16; six host parameter tests                                                                                       | Normalization/training/backward/provider/fusion subsets only. Mixed FP32 virtual/low-precision output rejected; no automatic Torch provider routing claim   |
| RPP/FFTS                                    | Native image layouts16 plus audio26; decibel zero/reference/reduction fixes, spectrogram row-stride and FFTS build/link fixes                                                                                | Broader image/audio/volume operations, types and layouts remain                                                                                             |
| rocPRIM/hipCUB/rocThrust                    | Historical reductions/scans and208 block-reduction tests; P2 int32 cases42/48/48 with guards/streams, signed stable sorts, selection, scans, blocks/wave32 and Thrust policies                               | Floating/custom/segmented algorithms and broader tuning remain. hipCCL is header packaging, not RCCL.                                                       |
| rocFFT/hipFFT                               | Existing native backend/AOT cache; P2 FFT30 plus N2.14 public complex96/real72 output+weighted gradients+guarded out, three independent checks and16 direct advanced plans/32 executions | No production defect found; native in-place/planar/callbacks/giant plans and broader performance remain; BF16 promotion B01 separate |
| rocSOLVER/hipSOLVER                         | Existing native/BLAS backend; P2 solver32 plus N2.1 complex Hermitian eigenvectors22: batch2/17–257, triangles/strides/streams, repeated subspaces and CPU gradients                                                                                                          | N2.1 native eigh22; N2.2 nonsymmetric eig16 host-LAPACK compatibility only. Native nonsymmetric eig, extreme scaling/conditioning, larger/two-stage and performance remain                                                                                         |
| rocSPARSE/hipSPARSE | Existing native backend: P2 Torch25+legacy16; N2.13 public triangular433 and direct generic96 fixtures/192 solves, exact pivot NaN policy in Torch | More formats/analyses/general factorization remain; BSR public int64 is bounded downcast, triangular autograd/performance unqualified; old corruption fix was CLR |
| rocALUTION                                  | CG40 ordinary cases andfive fresh-process200 passes after pool-default fix                                                                                                                                   | Larger systems, other solvers/preconditioners/concurrency remain                                                                                            |
| rocRAND/hipRAND                             | Native dispatch; P2 RNG28; N2.15 Torch state80/statistics22/contracts20/fresh continuation1 and direct native68 incl CPU Philox, offsets/streams and LFSR113 source-comment repair | No arithmetic defect reproduced; broader generators/orderings, large statistical suites, graphs/concurrent mutation/cross-version replay remain |
| OpenCL                                      | Native CL1.2 kernels, LDS/atomics/images/events/profiling checked                                                                                                                                            | OpenCL2 completeness and other feature combinations remain                                                                                                  |
| OpenMP/Flang                                | Mandatory GPU SAXPY/reduction/task dependency checks; bundled Flang compiled                                                                                                                                 | Fortran device execution and hipThreads remain unqualified                                                                                                  |
| HIPIFY                                      | hipify-perl CUDA translation → native compilation → exact GPU check                                                                                                                                          | hipify-clang requires parser/SDK input qualification                                                                                                        |
| hipFile                                     | Five synchronous/asynchronous compatibility I/O roundtrips                                                                                                                                                   | Not a direct-storage/DMA fast-path claim; registration/concurrency/ordering remain                                                                          |
| rocprofiler/ROCTX                           | Native marker/API/kernel/copy tracing with 3628 APIs,263 dispatches,814 copies, CPU workload correlations                                                                                                    | Counter mapping/advanced profiling/debug consumers separate                                                                                                 |
| AMD SMI / RDC                               | AMD SMI built without EPYC ESMI; disposable generated-header source copy                                                                                                                                     | Read-only telemetry remains incomplete/untrusted; RDC not built/qualified                                                                                   |
| RCCL/rocSHMEM/MPI                           | Future applicability work, not built in current profile                                                                                                                                                      | Local readiness/software transports first; no XGMI/GDA/P2P/SDMA claims; two-board work deferred                                                             |
| rocDecode/rocJPEG                           | Current driver does not expose media rings                                                                                                                                                                   | Qualify feasible host decoding/HIP interoperability separately; do not invent an upstream JPEG software fallback                                            |
| rocWMMA / matrix-only engines               | Not applicable as native MFMA/WMMA execution                                                                                                                                                                 | Distinct scalar emulation/math paths could be work; never enable them by architecture spoofing                                                              |
| rocRoller/emulation/other optional projects | No blanket qualification                                                                                                                                                                                     | Audit pinned API/instruction applicability before enablement; CPU fallback must be identified                                                               |

Code roots for actual port backends:

- `rocm-libraries/projects/hipblaslt/library/src/amd_detail/gfx1013_scalar.{cpp,hpp}`
- `rocm-libraries/projects/hiptensor/library/src/contraction/contraction_scalar.{cpp,hpp}`
- `rocm-libraries/projects/hipsparselt/library/src/hcc_detail/gfx1013_scalar.hip`
- `rocm-libraries/projects/composablekernel/include/ck_tile/`
- `rocm-libraries/dnn-providers/miopen-provider/engines/plans/`
- `rocm-libraries/projects/rpp/src/modules/hip/`
- `rocm-systems/projects/clr/rocclr/device/rocm/rocdevice.cpp`
- `sources/bc250-pytorch/aten/src/ATen/{native/cuda/AveragePool2d.cu,cuda/detail/CUDAHooks.cpp}`
- `sources/bc250-triton/third_party/amd/` (verify repository path before edits)

Supported scalar contraction type combinations (A/B share the input type; C/D
share the output type): f16→f16, bf16→bf16, f32→f32, f16→f32 and bf16→f32 use
f32 computation; f32→f64 and f64→f64 use f64 computation. Reduced-precision
accumulation, f16→f64 and narrowed outputs are not supported in this profile.
The contraction source currently accepts 23 real unary operators (19 added in P2).
Low-precision Lt inputs may likewise use f32 outputs; sparse Lt uses its documented
backend-specific FP32 1:2 / FP16-BF16 2:4 compression, not a portable hardware format.

## Torch, training, files and resource utilization

**Current Qwen3.5 training follow-up2026-10-16**, implementation690042b55:
application-scoped native scalar Triton frozen depthwise causal conv K<=4,
FP32/FP16/BF16 forward/dx/SiLU; unsupported/trainable weights keep original.
Chunk64 FP32 causal loss recomputes backward probabilities; full BF16 logits remain.
Enabled by default only in .104 finetune-n7 replica/train.py, Transformers5.19.0,
math SDPA/head256. New51 operator/reference/guard groups,4 small LoRA pairs and
real Qwen3.5-2B BF16 rank16 geometry pass:372 adapter gradients max relativeL2~1.12%
at65tokens,1024-token checkpointed backward/update. Isolated1024/vocab248320 loss
extra peak2.84->0.65GiB, not whole-model savings/speed/convergence. PEFT shortened
target names exposed a separate resume bug; compare exact resolved modules now.
Synthetic deployed CLI fresh/resume gives372 exact adapter tensors plus exact
optimizer/scheduler/RNG/progress;4 mismatches refused. Final logs have no causal
fallback/MIOpen/OOM warnings; FLA reference warning retained. Original user log
continued through251/saved250: allocator retries were not a proven fatal OOM.
All final guards ACKed/exit0/dmesg identical; failed initial resume retained.
Sources/patch/local caller copies, hashes and rollback:
[report](bc250_qwen35_training_results/README.md), SOURCE_QWEN35_TRAINING_OPS.json,
WS/recovery/qwen35-training-before. No native rebuild/DSO/prefix/boot change or
active agent job. Global MIOpen timing/workspace and Python CPU fault remain open,
distinct from fixed VM-heap/CSR corruption. Exact next: profile reference gated-
delta forward/backward and choose one bounded native gfx1013 substage; N2.7 queued.

Current user-priorities1–4 follow-up2026-10-15, Torch4ee65717d69e9f51021272e3f37d0f342f68fc08:
long head128/GQA references through4096tokens and real Qwen3-1.7B BF16 rank4 LoRA
512/1024-token synthetic reference passes completed. Causal tile bounds/tile32
selection improve paired BF16 head128 kernel fixtures1.67–2.05x versus the original;
no broad/whole-model speed claim, math default retained. Qwen1024 whole-model
allocated peak unchanged5,818,112,000 bytes: other allocations dominate. Autocast
mixed-QKV conversion and gradients qualified; tiled compile/graphs/dropout remain open.

Small full-parameter Qwen FP32/FP16/BF16 AMP, larger batch2/length128–256, CPU
float64 gradients,2updates and fresh-process model/optimizer/scheduler/scaler and
all RNG continuation passed. Six dynamic compiled/eager math-attention gradient
cases at batch2/3 passed. Native TF32 capability API fixes a false Tensor Core
warning. Actual user LoRA CLI resume defect reproduced (GPU RNG ByteTensor) and
fixed, with is_trainable=True plus scheduler/Python/shuffle state persistence;
new-format and legacy fixture resumes update adapters. Legacy missing state
cannot provide exact replay; matching schedule arguments still matter.

Python CPU IP0x3b3e6 remains **unresolved**. Recovered single historical event,
no original core/stack;8 fresh allocator-instrumented solver/FFT/sparse/GEMM
processes pass1920 references without reproduction. Launcher defaults Python
faulthandler on (user override allowed); diagnostic-only bounded core capture.
Separate retained failure: full mixed-operator attempt stops at MIOpen conv
EvaluateInvokers elapsed<=0 and workspace480/provided0, exit1, not SIGSEGV.
No false full55-operator pass or crash-repair claim. All final scoped gates
exit0/dmesg unchanged/guards ACKed. Next isolate this convolution failure and
capture crash evidence; N2/K6 deferred by user. Sources/builds/cache/user files/
checkpoints/BIOS/kernel/governor/.105 preserved. [Full report and replay](bc250_training_expansion_results/README.md),
SOURCE_TRAINING_EXPANSION and WS/recovery/*-before retain state/rollback identities.

Historical first opt-in attention batch (superseded within the scopes above):

Opt-in scalar tiled SDPA2026-10-15, Torch`bda0831ba49ef9d0ba2968a800642e5abaec5dd5`:
three native Triton scalar kernels implement tiled online softmax and recomputed
backward with FP32 accumulation and grouped KV gradients, without full score/
probability storage. Built-in flash/efficient/CK flags stay off; ordinary functional
SDPA remains math. Explicit Transformer attention+mask registry helper and user
finetune-n7 --attention-backend tiled provide opt-in access. D<=128, sequence<=4096,
finite FP32/FP16/BF16, broadcast boolean/additive masks, no dropout/mask gradients/
higher-order/compiled/graph qualification. Dense masks still cost their input storage.

18 independent CPU-float64/GPU-math output+gradient cases pass; small padded FP32
Llama/checkpoint gradients pass. Installed7-group gate adds BF16 Qwen1.7B head128/
16-to8GQA at65tokens, two updates of a **small random** Qwen3 BF16 LoRA model with
CPU-float64 references and cached decode, streams and diagnostics. Unchanged eager
Torch19-reference regression passes; all final gates exit0/dmesg unchanged/guards
ACKed. Extra allocated FP32 B1/H2/D32 forward+backward peak at2048tokens2.52MiB vs
math129.50MiB; tiled slower in measured single-event fixtures. No full Qwen1.7B,
sustained stability, universal memory or speedup claim. Next: longer head128 and
representative full-model training/memory/time qualification before default use.

Exact warning bug repaired: !debug was passed as text to TORCH_WARN_ONCE, printing
1Torch during successful fallback. Normal selection now quiet; explicit diagnostic
checks still explain unavailable built-in kernels. User train/chat use new
nn.attention.sdpa_kernel(MATH), backward included for checkpoint recomputation;
broad warning filters removed. Math stays default. Targeted12-job/20GiB-ccache
rebuild linked SDP-utils plus regenerated HIPHooks; loaded libtorch_hip SHA578a3583...
Exclusive lock used for changes/tests. Existing source/build/cache/images/checkpoints,
BIOS/kernel/governor/.105 preserved. Recovery WS/recovery/sdpa-{probe,callers,metadata}-before,
SOURCE_SDPA metadata, and [full report/replay/evidence](bc250_attention_results/README.md).
User training/chat finished before tests; no agent jobs remain. Earlier harness/
reference/staging failures retained; unknown Python CPU segfault still separate.

[Training evidence](bc250_training_memory.md), root `d744f0952`, proves real
SmolLM2-135M full FP32 fine-tuning and FP16/BF16 AMP LoRA for a bounded example.
Model revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, weights SHA256
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`;
134,515,008 base parameters. LoRA rank 8 / alpha 16 / q_proj+v_proj gives 460,800
trainable parameters. Batch 1 / 32 tokens / one text, mathematical SDPA and AdamW:
four updates plus one fresh-container resume update. This is not a convergence
study or proof of larger-model/full-model-AMP/QLoRA support.

Full FP32 loss fell 4.647392→2.845694→2.507597, peak~2.92GB. LoRA loss fell
~4.647392→4.0175→3.803, peak~704MB; frozen base remained unchanged, activation
checkpointing used, FP16 scaler skipped no updates. Resume preserved
model/adapter, optimizer, scheduler, scaler and CPU/GPU RNG; logits and exact RNG
probes were checked. CPU/GPU logit relative L2 was5.56e−6; representative gradient
relative L2 was8.36e−5–4.68e−4. Near-zero elementwise reference failures were
retained and replaced with justified aggregate/max-error checks, not ignored.

Memory/concurrency gates passed 10 GiB GPU pressure + 768 MiB CPU storage, fragmented
reuse, four streams, eight basic graph replays and two GPU processes in 123.99 s.
This respects shared memory; near-OOM/hang recovery was not established.
No CPU/RAM cgroup cap is imposed by the normal launcher; its default 1 GiB shared
memory mount is separately configurable. GPU use still competes with host RAM.

Retain `WS/{training-model,training-env-stable}`, validation configs/reports and
`WS/logs/training-memory/`. Obsolete generated validation checkpoint weights/state
and duplicate tokenizer files were retired20261016; regenerate synthetic fixtures
under new helper labels when needed. Genuine user checkpoints were untouched.
Qualified consumer pins are Transformers 5.2.0,
PEFT 0.18.1, Accelerate 1.12.0. `bc250-train` mounts the frozen consumer bundle
through `current/python-consumers`; it does not replace our custom Torch.
Do not redownload or recreate these because conversation context was compacted.

P1 Torch evidence:64 checks, including 52 eager/compiled dynamic-shape indexing,
normalization, training andmath SDPA cases, plus 12 explicit Lt cases with 36
observed actual successful library calls. LD_PRELOAD tracing is test-only and
must resolve RTLD_LOCAL dependencies; never enable it in the ordinary launcher.
Additional prior eager 56/compiler 18/operator 55 checks remain historical scopes.

Launcher `build_tools/bc250/bc250_torch.py` handles Python/script/-c/-m/stdin/shell,
retains host HOME/cwd/absolute paths, mounts script parents, supports explicit
mount modes and private environment handoff across doas. It runs rootful Podman
as uid/gid 1000 plus device groups. Arbitrary data outside automatic mounts needs
an explicit mount; do not claim an unrestricted host filesystem. Thirteen IO
scenarios include persistent packages, HTTP downloads, DataLoader spawn,
interruption handlers and checkpoint resume. Ctrl-C/termination checkpoints
require the user's application to implement a handler.

Current P2 launcher source added a shared runtime lock; cooperating host updates
must use `helpers/update_dev_prefix.py -- COMMAND...` with the exclusive lock.
This prevents overlap of launcher workloads and prefix replacement, not arbitrary
manual writes. `/usr/local/bin/bc250-torch.before-priority2` and recovery launcher
copies are retained. Metadata `/opt/bc250-stack/launcher.json` predates the P2
lock change; installed-script hash plus committed source and P2 launcher report
are newer evidence. Do not overwrite historical metadata to fake provenance.

N1.1 .104 gate: `WS/logs/n11-hiptensor-r3/result.json`, root report
`bc250_scalar_library_results/n11-hiptensor.json`. Old library rejected the new
case; targeted12-job/20GiB-ccache build17.1s and GPU gate2.7s passed under the
exclusive runtime lock. Kernel logs identical (1009 lines). Thirty-repeat HIP
stream-event timings across seven types: A-only12.6–13.5µs, mixed27.7–29.3µs;
includes submission gaps, no previous implementation/speedup comparison.
Independent prior library: `WS/recovery/n11-hiptensor-before.so.0.1`.
Two initial container failures (`WS/logs/n11-hiptensor{,-r2}`) happened before
compilation: cgroups were unmounted after a host reboot, and runc rejected
NoCgroups. Existing cgroups service started in unified mode and enabled in boot;
no kernel change or reboot, SSH/fallback intact. Do not use disabled cgroups
with the installed runc. N1.1 did not rebuild Torch or alter .105/frozen candidates.

N1.2 .104 gate: `WS/logs/n12-hiptensor-r2/result.json`, root reports
`bc250_scalar_library_results/n12-{hiptensor,reconstruction}.json`. C32F/C64F
scalar path uses HIP complex addition/multiplication, shared geometry/guards;
public host scalar ABI and FP64 capability check retained.450 complex cases
(426 std::complex<double> references,24 descriptor rejections), then273 real
regressions passed. All27 identity/NEG/CONJ A/B/C combinations executed on seven
layouts/two types; mixed/unilateral sums, guards, signed-zero/NaN and imaginary
scalars, null zero-beta C, in-place conjugated C/D,16-mode union with unit axes,
destroyed source descriptors/preferences, nine algorithm/workspace pairs/type.
First17.1s build passed; gate stopped on test-only false alias expectation for
adjacent scalar C/D tensors (`WS/logs/n12-hiptensor`, preserved attempt1 report).
Prior active library restored before releasing lock. Fixed assertion checks
multi-element spans; force source timestamps on retry to regenerate staging
rather than retain the restored old dist hardlink. Cached restage4.9s,
complex2.3s/real2.8s gates passed; kernel logs identical1009 lines.12jobs/20GiB
ccache, targeted hipTensor+dist/exclusive lock. Thirty-repeat tiny stream-event
timings: GEMM8.7–9.9µs, mixed21.5–23.2µs including submission gaps; no speedup
or large-performance inference. Independent prior library:
`WS/recovery/n12-hiptensor-before.so.0.1`. No new host/kernel/Torch changes.

N1.3 plan contracts: common plan APIs validate pointers, enums, exact attribute
sizes and workspace preferences before copying/allocating; failed setters/query
leave state/output unchanged, failed plan creation returns null with owning
cleanup. Native scalar defaults use cache NONE, JIT NONE, autotune NONE, count0,
rank0; PEDANTIC/DEFAULT-JIT/INCREMENTAL/positive count or rank return NOT_SUPPORTED.
Three algorithms remain deterministic aliases, not new performance models.
`WS/logs/n13-hiptensor-r2/result.json`:304 API checks +12 CPU-referenced GPU
executions (R32F/R64F/C32F/C64F ×3algorithms; limits0/1/UINT64_MAX); mutated/destroyed
source preferences/descriptors do not invalidate existing plans.273 real/450
complex regressions passed, kernel logs identical1009 lines. Initial17.6s build;
first probe passed316 but runner's count assertion failed (API checks were not
printed individually), preserved in attempt1 report and `WS/logs/n13-hiptensor`;
prior prefix restored before unlock. Reporting fixed, production code unchanged;
cached restage4.9s, plan1.5s/real2.7s/complex2.3s gates under exclusive lock,
12jobs/20GiB ccache. Prior library `WS/recovery/n13-hiptensor-before.so.0.1`.
Common validation changed, but non-gfx1013 numerical execution and every internal
initialization-failure injection are not qualified. No new performance claim.

N1.4 .104 `WS/logs/n14-sparselt/result.json`, repository
`bc250_scalar_library_results/n14-{sparselt,reconstruction}.json`:604 new numerical
broadcast cases +5 descriptor rejections,234 prior sparse regressions passed.
Native API now admits zero batch stride for A/B/C with matching logical counts;
upstream `rocsparselt_auxiliary.cpp` requires matching counts and compress/prune
sources collapse stride0 to one physical matrix. Native compression size/indexing
and strip/TILE launch counts now follow that physical layout; no duplicate sparse
writers. Multi-batch D stride0/incompatible counts remain rejected. Span checks
reject partial prune/compress and matmul input/D or C/D overlap; exact in-place
pruning remains valid. All7 masks ×sparse A/B×orders×transposes×five type/prune
combinations, selected batch1/stride0 and partial tiles, six activation epilogues;
CPU double references, independently decoded pack, plan/descriptor compression
agreement, pack/D guards and poisoned original sparse buffers. Targeted16-job
hipSPARSELt+dist build6.9s, new gate3.1s/prior1.6s under exclusive lock;20GiB ccache.
Kernel logs identical1009 lines; no performance claim. `SOURCE_N14.json` verifies8
changed files;21 patches reconstruct exact tree. Prior library retained at
`WS/recovery/n14-sparselt-before.so.0.1` (backup name; original runtime file0.2).
No host/kernel/Torch/hipTensor rebuild or .105 changes in this batch.

N1.5 .104 `WS/logs/n15-sparselt-r2/result.json`, repository
`bc250_scalar_library_results/n15-{sparselt,reconstruction}.json`:526 independent
CPU-double-referenced native GPU bias cases +41 API checks;234 scalar and609
broadcast regressions passed. Per-output-row bias is added before activation;
stride0 shares the vector, positive stride>=D rows allows padded per-batch bias.
Default type matches the matrix; f16/bf16 may also use FP32. Public pointer/stride
comments are swapped: upstream memcpy pointer-size/int64 stride and row mapping
are authoritative. Null pointer disables bias. Exact sizes, admissible types,
alignment, stride cap and D/bias span overlap guarded; failed setters/getters
preserve state/output. Plan captures pointer/type/stride while device data stays
caller-owned. Source bias settings mutated after plan creation; typed/FP32 bias,
orders/transposes, sparse A/B, strip/TILE/partial tiles, six activations,
broadcast inputs, poisoned zero-alpha/beta and valid in-place C/D passed with
unchanged tolerances and immutable bias/pack/D guards. First6.6s build and probe
passed567 checks but stdout buffering mixed PASS/stderr records (565 parseable);
runner restored prefix before unlocking. `WS/logs/n15-sparselt`/attempt1 report
retained; line-buffered reporting fixed accounting, production code unchanged.
Cached restage2.8s, bias2.7s/scalar1.6s/broadcast3.0s gates,16jobs/20GiB ccache,
targeted hipSPARSELt+dist/exclusive lock; kernel logs identical1009 lines.
`SOURCE_N15.json` verifies8 changed files,22 patches reconstruct exact tree;
prior library `WS/recovery/n15-sparselt-before.so.0.2`. No performance, new
matrix-type/vector-scaling or long-stability claim; .105/kernel/Torch unchanged.

N1.6 `WS/logs/n16-sparselt-r5/result.json`, root
`bc250_scalar_library_results/n16-{sparselt,reconstruction}.json`:404 native GPU
alpha-vector numerical cases +34 API checks;234 scalar/609 broadcast/567 bias
regressions passed. Public flag int32 0/1, captured plan mode; scalar alpha stays
host FP32, enabled alpha argument becomes borrowed device FP32 vector[M] indexed
by logical output row and shared across batches. No host vector dereference.
Upstream `clients/include/spmm/testing_spmm.hpp` allocates M device elements and
passes the vector as alpha; `tensile_host.cpp` supplies it as scaleAlphaVec with
scalar multiplier1. Beta enum has no pinned upstream handler/client evidence:
scalar beta preserved, getter/disable0 supported, enable1 NOT_SUPPORTED. A native
beta-vector extension is still unimplemented, not a hardware impossibility.
Flags/sizes/null buffers/alignment and D/vector exact/partial spans guarded;
source mode mutation after plan creation, mixed signed-zero/negative/positive
rows, poisoned all-zero alpha vectors and zero-beta C, layouts/transposes/types,
pruning/compression, bias/activations and compatible in-place C/D passed.
Initial/r2/r3 CPU reference mismatches were solely GPU+0/reference-0 from unused
zero-alpha multiplication; r3 repeated after a failed local edit helper. r4
reached zero-beta cases and found GPU-0/reference+0 from adding unused beta*C.
Hex diagnostics/failed reports retained in `n16-sparselt-attempts.json` and host
logs. Reference skips both unused terms to honor zero-scalar semantics; strict
bitwise/padding comparisons and nonlinear tolerances retained. Production GPU
source/library hash unchanged across attempts; every failed gate restored N1.5
under the exclusive lock. Do not repeat the diagnosed reference defect.
Initial6.6s build, successful cached restage2.8s, vector2.2s/scalar1.7s/broadcast3.0s/
bias2.7s gates,16jobs/20GiB ccache. Kernel logs identical1009 lines.
`SOURCE_N16.json` verifies8 changed files,23 patches reconstruct exact tree;
prior library `WS/recovery/n16-sparselt-before.so.0.2`. No general performance,
new-type or long-stability qualification; .105/kernel/Torch unchanged.

N1.7 `WS/logs/n17-sparselt-r4/result.json`, repository
`bc250_scalar_library_results/n17-{sparselt,reconstruction}.json`:134 native GPU
INT8 numerical cases/16 API checks and234/609/567/438 scalar/broadcast/bias/vector
regressions. Pinned upstream type table admits INT8 inputs with compute_i32;
`spmm/rocsparselt_spmm.cpp` still instantiates FP32 scaling arguments. Integer
bias defaults FP32; CPU GEMM/bias/activation clients round with nearbyint then
saturate INT8. Native backend now implements signed INT8 input/output only, INT32
retained-product accumulation, FP32 scaling/bias and nearest-even saturation.
2:4 retains K/2 products, each <=16384 in magnitude: K<=262140 guarantees all
partial sums fit INT32. Descriptor rejects K262144; K262140 GPU execution passed.
K4096/-127 products and tailored FP32 scale/bias expose loss from incremental
FP32 accumulation (>2^24), checking exact integer sum then one FP32 conversion.
Signed half ties, ±128/127 extrema, saturation, orders/transposes, sparse A/B,
padded/broadcast inputs, strip/TILE/partial tiles, row alpha vector, FP32 bias,
RELU/ABS/leakyRELU/GELU, zero scalars, snapshot/alias/in-place/immutable guards and
independent pack/prune references passed. Integer sigmoid/tanh enabling rejected
per upstream; widened INT8→FP16/BF16/INT32 outputs remain unimplemented, not
hardware-limited. Nonfinite epilogues/exhaustive conversion edges unqualified.
Initial TILE fixture converted fractional entries to zero and violated its own
nonzero-cardinality assertion. r2 extra legacy CPU assertion used a truncating
cast; fixed nearest-even saturation. r3 passed all150 checks, then count parsing
failed because numerical PASS lines were absent; reporting fixed. Failures and
restorations retained in `n17-sparselt-attempts.json`; production source/library
hash unchanged across all four attempts. Every failure restored N1.6 under lock.
Initial build6.9s, successful cached restage2.8s, integer1.6s/scalar1.6s/broadcast3.0s/
bias2.7s/vector2.2s gates,16jobs/20GiB ccache/exclusive lock. Kernel logs identical
1009 lines. `SOURCE_N17.json` verifies8 changed files,24 patches reconstruct exact
tree; prior independent library `WS/recovery/n17-sparselt-before.so.0.2` retained.
No performance/automatic Torch routing/new matrix-instruction/long-stability
claim; .105/kernel/Torch unchanged. Next N1.8: inspect existing Lt scalar/register
shape dispatch and timing evidence; select one measured small/skinny improvement.

N1.8 `WS/logs/n18-lt-r4/result.json`, repository
`bc250_scalar_library_results/n18-{lt,lt-attempts,lt-performance,lt-pack,reconstruction}.json`:
664 CPU-double-referenced GPU cases plus403 scalar/561 extended Lt regressions.
Native 32-lane scalar subgroup/output (eight/block) for K<=64 and either min(M,N)<=4
or M,N<=32 with area<=512; no LDS/barriers, physical D-order output mapping. FP32
tree reduction changes rounding from ascending-K FMA; shared existing epilogues
retain scalar/bias/activation/AUX/DGELU. Same/mixed f32/f16/bf16, orders/transposes,
padded/broadcast batches, host/device/zero scalars, compatible in-place C/D, tails,
cross-batch blocks, K0/1/15/16/17/63/64/65, area512/544 and guard checks passed.
Initial direct one-thread/output candidate passed504 cases but selected ratios
0.46–0.76x; rejected. Its DSO-only restore left a mismatched external BLAS pack:
A2 aborted134 on missing GPU symbol. Exact prior DSO SHA matched frozen P2, whose
retained pack restored the baseline pair under lock. r2 cooperative path passed
504+964 cases and paired A2 restoration, but32x32 ratios0.91–0.96x regressed; final
area<=512 bound keeps32x32 tiled. r3 passed664+964; final host grid-bound
review added fallback to the tiled path when subgroup blocks exceed INT_MAX. r4
repeated664+964 and all36 paired timing fixtures; oversized launches not executed.
Every attempt/log preserved; no GPU fault/reset
or new kernel messages. N1.8 runner restores/hashes both DSO and pack on failures.
**Correction to historical rollback wording:** N1.1–N1.7 reports only checked DSO
restoration; they did not verify/restore external packs. Do not treat those fields
as complete runtime-pair recovery. Their successful final GPU gates remain valid
recorded evidence; old runners need paired recovery before reuse. This missing
code-object-symbol abort is separate from unresolved Python CPU segfault0x3b3e6
and the already fixed allocator corruption/default-pool issue.
36 deterministic same-input C++ fixtures, column-major NN/batch1/host scalars/default
epilogue, A1/B1/B2/A2 order; five GPU-event samples of32-launch HIP graph/pass.
Selected skinny ratios1.14–1.86x, selected small1.21–1.61x; controls0.998–1.006x;
max repeated-pass median drift0.24%. Kernel-focused graph replay, not uncaptured
application latency or Torch speedup; clocks untrusted, no clock attribution.
Final targeted16-job/20GiB ccache build11.8s, numerical3.7s/scalar2.3s/extended2.1s;
timing passes1.0–1.2s, all under exclusive lock. Kernel logs identical1009 lines.
`SOURCE_N18.json` verifies12 source files plus BLAS DSO/pack hashes;26 patches
reconstruct exact tree. Pack payload comparison:1 Lt entry changed,68 unchanged
including all67 rocBLAS entries. Recovery DSO `WS/recovery/n18-lt-before.so.1.5`
and pack `WS/recovery/n18-blas_lib_gfx1013-before.kpack` independent copies retained.
No .105/kernel/Torch/frozen candidate changes, automatic consumer routing, universal
bitwise/nonfinite or sustained qualification. N1 bounded roadmap items complete;
optional widened INT8/beta-vector extensions remain open. Next N2.1 solver audit.

## Defects and traps already investigated: do not rediscover them

| Issue                           | Finding / resolution                                                                                                                         | Evidence to preserve                                                                                                                           |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Intermittent CG/CSR zero output | Proven live CSR analysis metadata overwrite associated with experimental VM heap; compiled default now uses the established pool backend     | Root `47925cbd5`, systems patch0003, `rocalution-step-trace.log`, allocation/raw-metadata/first-kernel probes, 40 plus fresh-process200 checks |
| Experimental VM heap            | Default bypass is a mitigation, not a repair. VMM/async pools retained; explicit opt-in remains experimental.                                | `rocdevice.cpp`, root `bd6e4c8a4`, release gates                                                                                               |
| SVM CPU-unmap returned error | Patch0004 retains queue eviction after failed invalidation following confirmed stop; partial/full targeted -EIO tests and fresh GPU references pass. Failed stop/retry migration remain open; validation locking qualified separately below. | `bc250stack/kernel/r3/evidence/svm-cpu-unmap/` |
| SVM validation invalidation | Patch0005 flushes validation map/unmap after BO unlock; PROT_NONE bounds fixed (trace0→2); two concurrent54-reference GPU processes and full regressions pass. No global lockdep/failure-path claim. | `bc250stack/kernel/r3/evidence/svm-map-deferred/` |
| HPD IRQ initialization | Patch0006 queries active IRQ capability before bulk clear; dummy ack/set/clear errors8 each→0 and IRQ ASSERT WARNs2→0. Four real clears/DC/two disconnected DP connectors retained. Physical display/resume unqualified. | `bc250stack/kernel/r3/evidence/hpd-capability/` |
| Explicit fixed-VA VMM remap | K5 GEM_VA runlist fix8f3c0d3: unchanged two-process VMM gate and HIP/ROCr26-check references pass; historical r2/r3 failures preserved. No broad/failure-path claim. Separate from VM-heap corruption. | `bc250stack/kernel/r3/evidence/k5-summary.json`, `k5-gem-va-full.json`, original/diagnostic failures |
| Later Python CPU segfault       | Execution fault at0x3b3e6, no identified stack/operator; cause remains unknown. Stable-prefix sparse gates and independent candidate passed. | Older operator-candidate/kernel logs, P2 candidate/launcher20 cycles each; do not merge with the proven CSR defect                             |
| Triton incorrect buffer reads   | Missing RDNA1 descriptor flag branch; globals passed while buffer path failed. Fixed.                                                        | Triton patch0004/root `9fdae0b5d`; both memory paths validated                                                                                 |
| Triton low-precision lowering   | Unsupported gfx13 packed dot/BF16 multiply replaced by widened scalar arithmetic; signed limits and rounding-tie/NaN tests retained          | Four Triton patches and MLIR/LLVM/assembly/GPU gates                                                                                           |
| NHWC avgpool backward           | Padding correction in Torch source                                                                                                           | Torch patch0001, operator coverage                                                                                                             |
| MIOpen bias/fusion              | Legacy bias beta1 doubles output; beta0 is correct. BF16 and FP16 NHWC pointwise required private scalar HIP fallback; output routing fixed. | Library patch0016, provider CPU references; mixed-intermediate rejection                                                                       |
| Lt SWISH                        | Public beta is fixed1; activation arg0 is not Swish beta                                                                                     | 465 then561 extended consumer cases; hipDNN Lt48                                                                                               |
| RPP audio                       | Spectrogram destination stride and decibel reduction/reference/zero handling fixed; FFTS dependencies/linking fixed                          | Patches0010/0013/0015 and audio26                                                                                                              |
| Source generation               | Torch CMake cache can leave HIPHooks stale even after pip; explicitly hipify before editable rebuild                                         | `build_torch.sh`, `rebuild_torch_pooling.sh`                                                                                                   |
| Installed headers               | CK public headers and provider TOML/.hip require explicit superbuild tracking; noop build alone does not prove staging                       | `math-libs/CMakeLists.txt`, `ml-libs/CMakeLists.txt`                                                                                           |
| Kernel packs                    | hipSPARSELt owns a separate sparselt artifact; extracted pack collisions fixed                                                               | `math-libs/BLAS/CMakeLists.txt`, descriptors; do not merge it back into generic sparse                                                         |
| Reference/test mistakes         | CPU.double() can alias float64 inputs; gradients need detached clones. Some upstream mixed-inference CPU fixtures were limited.              | Corrected P2 math references and meaningful training fixtures, not all-skipped success                                                         |
| Build resources                 | Global dist restarted unrelated MIOpen and was stopped137; targeted component+dist passed0                                                   | P2 stage logs; preserve objects and resume global builds only when needed                                                                      |
| Old/new timing                  | Seed-only fixtures differed; replaced with identical serialized bytes. Clock reports conflict.                                               | Matched reports; excluded raw runs are not comparison proof                                                                                    |

## Performance already measured, with the correct interpretation

Root `24ea38d84`, [performance report](bc250_performance.md), and
`bc250_performance_results/` record nine matched workloads on `.104` 10.2/Torch 2.13
versus `.105` 7.2.4/Torch 2.9.1. GEMM/copy mostly differed ~1–6%; copy~0.2%.
Small CNN results were highly variable on `.105` (59.9% IQR); an apparent 3.4x ratio
is not a stable improvement. Different boards, thermal states and untrusted
clock telemetry prevent attribution of small differences to ROCm alone.

SmolLM2 batch 1/32-token timings were ~131.56ms full FP32 update,~111.72ms FP32 LoRA
without checkpointing,~260ms AMP LoRA with checkpointing and~139–141ms without.
These are training-step processed tokens, not generation tokens/second. The
smallest model may benefit from FP32 because overhead outweighs low-precision compute.

P2 is a separate same-board scalar-kernel comparison: 71 CPU-checked timing cases,
964 Lt API checks, medium/large Lt GEMM~2.1–2.4x faster than prior Lt. Small shapes were
roughly flat/slightly worse. Default Torch BLAS did not change; this is not a 2.4x
whole-Torch or old/new board speedup. Plain compiled reductions often lost to eager.
Explicit HIP graph replay for compiled SiLU/square/sum at 1,048,576 elements measured
~4x over eager; tested large sum/dot~1.35–1.4x. Captured storage/shapes must remain fixed.
For known SPD matrices, Cholesky factorization/reuse was cheaper than general LU solve;
never silently assume all matrices are SPD. The benchmark source encodes exact cases.

## Evidence registry and reproducible workflow

| Scope                             | Durable location                                                                                                                       |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Initial/expanded native gates     | `WS/logs/`, `bc250_port.md`, historical support matrix/roadmap                                                                         |
| Pool corruption/regressions       | `rocalution_diagnostic.cpp`, `sparse_analysis_trace.cpp`, `WS/logs/*rocalution*`, allocator/raw-metadata logs                          |
| Training/memory                   | `bc250_training_memory.md`, `WS/logs/training-memory/`, retained model/adapter artifacts                                               |
| Matched old/new/model performance | `bc250_performance_results/*.json`; `/home/nik/bc250work/rocm-performance-20261013` on both hosts                                      |
| P1/P2 numerical evidence          | `bc250_scalar_library_results/priority1-*.json` and `priority2-*.json`; host logs                                                      |
| Independent frozen release        | 600-second soak40,339 cycles; prior candidate39,657; review46,378 file hashes. These do not qualify current mutable P2 long stability. |
| P2 candidate (retired payload)    | `candidates/priority2-20261014/runtime.retired.json`,191 historical DSO/pack hashes; validation/candidate.json retained after20261016 retirement |
| P2 launcher                       | `/home/nik/bc250work/priority2-launcher.{json,pt}`,20 cycles/40 sparse matmuls, compiled gradients and checkpoint                      |
| Kernel comparison                 | P2 report records82 new bridge/veth messages, no new fault/reset/OOM; logs are not byte-identical                                      |

Always back up and restore changed DSOs together with the external `.kpack` files
they reference, under the exclusive prefix lock. Verify both hashes; a DSO hash
alone cannot establish runtime rollback. N1.8 paired A2 execution checks recovery.

Build without mapped GPU devices; run bounded GPU gates serially after staging.
Use existing objects/20 GiB ccache:16 normal jobs, hipTensor 12, hipDNN providers 6,
rocALUTION 8, Torch 12/link 1, Flang 4/LLVM link 1. rocFFT HIPRTC AOT cache has its own
persistent 16-worker pool, not ccache. Evaluate setup_ccache.py in each builder shell.
Native bootstrap compiler remains under `/usr/lib/llvm-18`; masquerade aliases preserve
old LLVM command lines, not architecture identity.

Exact opt-in: `THEROCK_GFX1013_SCALAR_BACKENDS=ON` with exclusive
`THEROCK_AMDGPU_FAMILIES=gfx1013`. A leftover cache variable named
`THEROCK_ENABLE_GFX1013_SCALAR_BACKENDS=OFF` is not this option. Top-level group
flags can be OFF while dependencies/selected child projects are activated; inspect
actual targets/artifacts, not just group flags. HIPBLASLt provider is enabled;
RCCL/rocSHMEM/MPI/debug/media are off in this profile.

Prefer component `hipBLASLt+dist hipTensor+dist hipSPARSELt+dist` targets to global
dist for local changes. `.104` source exports include root-owned directories: transfer
explicit changed files via tar/checksums, not rsync directory timestamps over them.
Update docs/provenance separately; do not relabel older archive baselines.

Torch rebuild must use explicit HIP generation:

```bash
python tools/amd_build/build_amd.py --out-of-place-only
python -m pip install -e . -v --no-build-isolation
```

Triton lowering changes require `make` before MLIR/LLVM tests. Keep qualified wheel
unless source changed. Exact patch reconstruction helper is
`build_tools/bc250/verify_priority2_reconstruction.py`; it needs local Git objects,
not gitless host exports. Do not rerun all GPU gates just to edit memory/docs.

## Earlier agents, legacy work and archive recovery

BC-250 milestones:

| Anchor                          | Work already done                                                 |
| ------------------------------- | ----------------------------------------------------------------- |
| 7a5c2ca0f                       | Native gfx1013 target/bootstrap                                   |
| a640c9296                       | Initial independent ROCm/Torch/Triton deployment                  |
| 47925cbd5 /4b000d542            | Safe allocator default; expanded candidate/release gates          |
| b2707998a /761ec5803 /f1895f712 | Persistent host paths/files/read-only mounts/shell                |
| d744f0952                       | Real SmolLM2/AMP LoRA/resume/memory concurrency                   |
| 24ea38d84                       | Matched old/new and training performance                          |
| c8dc436d3                       | Frozen snapshot review/pause policy, subsequently revoked by user |
| fdde5561c                       | Scalar backends/operators expansion                               |
| 71f9f0ee4                       | P1 consumers/CK Tile/DNN/Torch/dev activation                     |
| ce0025df5                       | P2 algebra/primitives/math/performance/candidate/runtime-lock     |

Separate branch `users/nik/alpine-musl-port`, tip `36c62955f`, has 22 commits after
the common base. It is **not an ancestor of the active BC-250 branch**. It contains
native-musl toolchain/runtime/COMGR-stack/portable FFT/SMI/OpenCL work, qualified
on Ryzen 8645HS/Radeon 760M **gfx1103**, not BC-250. Examples: `a05baf1ae`,
`7aacbbd89`, `ef108c4ff`. Recover its historical doc with
`git show 36c62955f:docs/development/alpine_port.md`. This work is reusable research,
not current BC-250 native-musl qualification; don't switch branches or integrate it
silently. The current port runs glibc userspace in Podman on Alpine.

Local project thread index: `/home/nik/.codex/state_5.sqlite`, read-only lookup by
cwd `/home/nik/Documents/Vibe/therock`. Raw rollouts live under
`/home/nik/.codex/sessions/YYYY/MM/DD/`. Retrieve only relevant project transcripts if
source/evidence cannot answer a question; do not copy titles/prompts/credentials into
committed memory. Root BC-250 thread ID `01a12a83-7fad-70d1-ab81-57e2e3cf6253`;
rollout filenames may change across compaction/resume, so find by ID rather than hardcode
one current filename. Prior roles include build_guidance, triton_review,
remaining_runtime_audit, remaining_libraries_audit, hipblaslt_scalar,
hiptensor_scalar, hipsparselt_scalar, p1_hipblaslt/p1_ck_tile/p1_torch_compiled,
p2_communication/p2_tensor_sparse/p2_primitives. The P2 communication agent was
redirected to math coverage; its name is not proof RCCL was ported.

`.105` remains the old reference; read-only audit found Alpine 3.24.2/current r2 kernel,
legacy sources under `/home/nik/bc250work/rocm-libraries-gfx1013`, branch
`gfx1013-cyanskillfish`, revision `b578bcd410d0641337ce0f34be055b88560f5c24`.
Other retained legacy work: `/home/nik/bc250-stack`, `pytorch-build-gfx1013`,
`research-handoff`, `bc250-rocm-lab`, `bc250-gfx1013-fix`. Do not copy old patches
blindly or upgrade/prune/reconfigure the host. Its private cgroup2 mount namespace
benchmark workaround was ephemeral, not persistent migration.

## Historical documents and how to avoid stale tasks

`bc250_port.md`, `bc250_release_20261013.md`, `bc250_development_handoff.md`,
`bc250_support_matrix.md`, `bc250_scalar_library_expansion.md`, the old roadmap,
training memory and performance docs are valuable milestone evidence. They may
say the frozen release is active, Lt GEMM unsupported, tensor contractions absent, sparse Lt
excluded, CK Tile untested, Torch unpatched or performance not started. **Those statements
are superseded by the current inventory for its specific scopes.** Do not delete snapshots
or mark every old box done; map completed work to evidence and select an actual remaining gap.

Before repeating any task: identify the component/operation, search this inventory,
patch subjects and validation registry, compare current code/profile, then state why
new work/test is necessary (new change, uncovered case, genuine failure or invalid old
reference). An old unchecked box, context compaction, stale PID, mutable tag name or
component version number mismatch is not a reason to restart the port.

## Unified workspace organization (after the memory audit)

Current local workspace: `/home/nik/Documents/Vibe/bc250stack`. The existing
TheRock checkout moved intact to `therock/`; `/home/nik/Documents/Vibe/therock`
is a compatibility symlink. The separate fresh PyTorch checkout moved intact to
`references/pytorch-upstream`, with its original `Vibe/pytorch` path preserved.
Canonical `torch/` and `triton/` aliases point to the qualified patched sources
inside TheRock, so there is no duplicated or silently upgraded port checkout.

Kernel source was found on .105 at
`/home/nik/bc250work/alpine-rocm-7.1/sources/linux-7.1-bc250-test` and copied
read-only into `kernel/source`. Five source/configuration anchors were verified.
The kernel binary SHA256 is
`792f904835f27574b680d2fd5cfd7e42b348fb3a3d9b39d8e8c87348873c3b0d`, identical
on both hosts. Configuration, legacy scripts and the existing bc250-linux
packaging repository are preserved separately. This is an already-patched source
snapshot, not a clean upstream checkout; generated build products/private signing
keys/initramfs archives are excluded. No remote installation or reboot occurred.

Workspace README/AGENTS and workspace-manifest.json explain component identities.
The workspace repository tracks coordination files; source changes stay in their
component repositories. Board support copies retain driver and CU-unlock history.
New session history may use the new workspace cwd; the earlier Codex thread-index
lookup by the old therock cwd remains useful for archived sessions.

## N2.1 complex Hermitian eigenvectors — 2026-10-15

P2 solver32 audited and rerun unchanged. Native Torch4ee6571 complex eigh dispatch
uses hipSOLVER generic64 DnXsyevd to rocSOLVER complex heevd; no backend
code/selection change.
New22 CPU-complex128/eigenpair/reconstruction/orthogonality/projector/gradient
checks pass for complex64/128, batch2 at17/65/129/257, both triangles, padded
inputs/guards/ignored-triangle NaNs/nondefault stream and repeated subspaces33.
257 crosses source default blocked threshold256; two-stage11000 unqualified.
Final22+32 processes exit0 with identical dmesg and acknowledged600s guards under
runtime lock. Initial harness broadcast failure retained;18-case intermediate
pass retained. Source, mapped-library hashes, dispatch audit and fixed tolerances:
[solver results](bc250_solver_results/README.md). No native rebuild, prefix/boot/
power-policy change or active job. Original CPU crash and separate MIOpen defect
remain unresolved; no universal/performance claim. Next N2.2 nonsymmetric eig
coverage/dispatch audit and one bounded complex eigenpair gap.

## N2.2 nonsymmetric eig compatibility — 2026-10-15

Pinned Torch4ee6571 -> hipSOLVER DnXgeev -> host LAPACK cgeev/zgeev -> device
outputs. This is a CPU compatibility eigensolve, not native GPU; N2.1 Hermitian
eigh remains native separately. New16 groups pass: complex64/128 batch2 at9/33/65,
normal/nonnormal known spectra with basis condition1/20, padded/transposed guards,
nondefault streams, eig/eigvals CPU/analytic eigenvalues and residual/projectors,
two eigenvalue-gradient and two grouped empty-input cases. Max residual3.17e-7/
6.72e-16. First run exit0/dmesg identical/600s guard acknowledged under lock.
No runtime/backend change or native build; no active job. Mapped hipSOLVER and
OpenBLAS hashes, dispatch audit/source/test/results:
[N2.2 report](bc250_solver_results/N22.md). Native non-Hermitian eig, defective/
repeated/extreme-scale/convergence-failure/performance scope remain unqualified;
Python CPU crash and MIOpen defect remain open. Next N2.3 native Hessenberg/QR/
Schur feasibility and smallest reviewed implementation/convergence step.

## N2.3 native small Schur prototype — 2026-10-15

Audited native GEHRD/GEQRF/UNGQR pieces; no pinned native nonsymmetric Schur
iteration found. Added isolated gfx1013_schur_candidate.py, cap16 complex64/128:
owned native GEHRD copy/current-stream private handle, GPU reflector-Q formation,
bounded explicit complex shifted GPU QR, similarity accumulation and checked
roundoff deflation. Python loop/scalar checks synchronize host; no CPU matrix
solve. Positive info on budget exhaustion retains partial T,Q. Schur vectors
are not right eigenvectors. Native production eig remains unimplemented.
Initial17/final21 groups pass with unchanged dmesg and acknowledged600s guards:
orders4/9/16 basis1/10, CPU/analytic spectrum/Hessenberg/Schur invariants, guards/
nondefault stream, zero/one-step exhaustion, repeated triangular, ±i rotation,
scalar and9 argument rejections. Max Schur reconstruction1.35e-6/2.90e-15.
Source/evidence/limits: [N2.3](bc250_solver_results/N23.md). No default dispatch/
Torch/DSO/prefix/boot policy change or native rebuild; no active jobs. Extreme
scaling/balancing/general convergence/vectors/batching/gradients/performance not
qualified. CPU fault/MIOpen defect remain open. Next N2.4 guarded small-Schur
right eigenvector recovery and explicit spectrum/convergence refusal contracts.

## N2.4 native small-Schur right eigenvectors — 2026-10-15

Extended isolated cap16 candidate with GPU triangular back-substitution,
dimensionless scaling/homogeneous rescaling, checked gap/pivots/residuals and GPU
Q transform/unit norm. Positive Schur info and malformed/nonunitary T/Q refused;
scaled pairwise gaps<=8sqrt(eps) reject repeated/close spectra without perturbation
or host solve. All21 groups pass first run: complex64/128 orders4/9/16 basis1/10,
CPU/analytic spectra and vector projectors, original residuals, stream/guard/
Schur immutability, analytic nonnormal2x2, repeated/Jordan/close/zero refusals and
above-floor control, budget0/1 refusal, scalar zero/nonzero and7 malformed cases.
Max original residual3.84e-7/7.79e-16. Exit0/dmesg identical/600s guard ACKed;
no active jobs. Source, hashes, report and evidence: [N2.4](bc250_solver_results/N24.md).
Prior helper retained in WS/recovery/solver-n24-before. Native GPU arithmetic
with Python scalar control, no production Torch eig/DSO/prefix/kernel change or
native rebuild. General scaling/balancing/conditioning/repeated eigenvectors,
batching/gradients/performance unqualified; CPU fault/MIOpen defect remain open.
Next N2.5 bounded scale/separation ladder with nonnormal/refusal controls before
any opt-in integration. Host-LAPACK default retained.

## N2.5 native small candidate scaling — 2026-10-15

Reproduced4/6 seeded baseline magnitude failures (tiny256-step exhaustion, large
nonfinite/fill) with unit controls passing. Fixed isolated Schur driver by GPU
component normalization before native GEHRD/QR, finite checked original-unit T
restoration on success/partial results, and safe right-recovery normalization
with unchanged8sqrt(eps) separation. Bitwise subnormal/zero distinction and
nonrepresentable-output refusal tested; no tolerance/iteration changes.
Seeded6 replay now6/6; scale40 (c64 1e-25..1e25, c128 1e-200..1e200,4/9 basis1/10),
boundary30 (18accept/12refuse; passed before too), failure/control16 and N2.4
regression21 pass. Final logs unchanged/guards acknowledged, no active jobs.
[Source/evidence/limits](bc250_solver_results/N25.md); baseline helper retained
at WS/recovery/solver-n25-before. Production Torch eig/DSOs/prefix/kernel/boot
unchanged; no native rebuild or performance claim. General dynamic ranges,
balancing/subnormals/conditioning/batching/gradients remain unqualified; original
CPU fault and separate MIOpen defect unresolved. Next N2.6 bounded cost/profile
versus host eig and one measured overhead-reduction step before any integration.

## N2.15 RNG offset/state/stream qualification — 2026-10-18

TheRock `0e4557ee5`; libraries documentation `c29021b40e7` on native319a235.
State/layout80, analytic distribution22, offset/isolation/stream/dropout20 and
fresh-process continuation1 pass. Low-precision/complex normal/integer, guarded
strides, empty/odd/tail, seed>2^63/offset>2^40; exact GPU/CPU/Python continuation
ten fields, native dropout output/backward CPU mask refs and five GPU kernels.
Direct native68 includes five pseudo/four Sobol variants, offset reset/stream
rebinding, exact prefix relations and scalar CPU Philox through both wrappers.
Initial native59/60: LFSR113 offset accepted despite stale rocrand.h restriction;
implementation and upstream offset tests confirm intentional support. Comments
corrected, C++ wrapper distinction retained, nine extra LFSR probes pass incl
high-offset non-aliasing. MTGP32 TYPE_ERROR still expected. No arithmetic/ABI/
binary/default/prefix/header-install change; comments staged in source only with
hash-verified recovery WS/recovery/rng-n215-source-before. Patch0032 retained.
RNG28/core7 pass; seven successful GPU containers plus initial assertion failure,
eight guards ACK/600s, dmesg unchanged whole batch;153 protected hashes/four FFT
AOT DBs match. Active Torchdc8fea0/source2b449c4/libraries319a235/kernel/boot/
fallbacks/.105 preserved. No active agent jobs; free~7.21GiB. CPU Python fault
unresolved/separate from fixed VM-heap/CSR and explicit contract exception.
[Report/evidence/limits](bc250_rng_n215_results/README.md); raw WS/logs/rng-n215-*,
ignored kernel/artifacts/rng-n215. Current preservation baseline
WS/logs/rng-n215-qualification-r2/final.json. No broad statistics/tail/ordering/
concurrent/graph/compiled/cross-version replay or speed claim. Next N2.16 audit
P2 primitives138 then bounded floating/segmented reduction/scan references.

## N2.14 FFT plan/layout/odd-real qualification — 2026-10-18

TheRock `44568c2e2`; no native/build/default/prefix change required. Installed
Torch native `dc8fea0` / test source `2b449c4`, libraries `319a235d6` unchanged.
Complex96/real72 each verify CPU-double output, weighted first backward into
the full leaf, private-stream GPU layouts, input immutability and guarded out.
Includes odd17/45/7x13, three norms, crop/pad, conjugate/broadcast and four-dim
decomposition. Three explicit DFT/finite-difference/cache-reuse checks pass;
trace34 kernels includes native rocFFT forward/backward. Direct hipFFT16 plans/
32 executions check single/double real/complex, caller workspace, batches3,
advanced strides and synchronized stream rebinding. Native rocFFT permits
input/gap writes; direct tests guard exterior spans, public tests all gaps.
FFT30/core7 retained. Initial48 double fixture errors were non-leaf GPU .grad,
fixed by detaching test leaf; no production or tolerance change, evidence kept.
Seven guards ACK/600s, six passed processes, dmesg unchanged including failure;
145 hashes/four AOT DBs/runtime/links/kernel/boot/r2 fallback/.105 preserved.
No running agent job/guard/container, free~7.22GiB. No speed/universal-correctness
claim; B01 BF16 promotion/native in-place/planar/giant/callbacks/graphs/higher
gradients remain unqualified. [Report/evidence](bc250_fft_n214_results/README.md).
Current preservation baseline WS/logs/fft-n214-qualification-r2/final.json;
raw WS/logs/fft-n214-*, ignored kernel/artifacts/fft-n214. Historical Python CPU
segfault unresolved/separate from fixed VM-heap/CSR and this fixture exception.
Next N2.15 audit P2 RNG28/checkpoint coverage; bounded offset/state replay and
private-stream sequencing against exact replay/analytic distribution contracts.

## N2.13 sparse triangular coverage and exact pivots — 2026-10-18

Installed native Torch `dc8fea0` (implementation `a434d03`), final tests `2b449c4`;
TheRock build `bbf556448`, controllers `39ea58574`. Existing generic CSR SpSV/SpSM
and BSR paths pass bounded four-dtype/index/layout/stream/flag CPU references.
Pre-fix healthy256 pass; singular/empty failures16/32 then39/80 expose finite
placeholders after division avoidance. New ROCm-only2D Torch GPU postcheck marks
exact zero/missing scalar pivot rows as NaN using compressed diagonal binary
search and bit-exact zero tests; no host reads/dense conversion/workspace/pivot
perturbation. One extra GPU launch on nonunit/nonempty solves; no throughput or
near-zero/conditioning claim. Other singular rows unspecified; legacy empty-A
NaNs including unit mode retained. Unit solves bypass. Direct raw library APIs
unchanged and general sparse spsolve/factorization B02 remains separate.

Each candidate/installed mount passes433 new public groups (256 matrix,80 pivot/
empty,32 larger CSR65/257+BSR66/258 block3,64 implicit unit,one real kernel trace),
96 native generic fixtures/192 solves (N/T/adjoint, host/device alpha, private
stream, two submissions/analysis, sentinels), sparse41/core7/CPU-first profiler.
Final pivot group tests NaN writes into strided sentinel output/private stream;
only that group rerun after test-only strengthening, native binaries unchanged.
BSR int64 public metadata is downcast within admitted bounds; no giant native64
BSR, batching, sparse triangular autograd/compile/graphs/universal performance
qualification. Only libtorch_hip.so changes; CPU profiler fix remains identical.
121 prior identities preserved,22 guards ACK/600s and unchanged dmesg whole batch.
Independent WS/pytorch-sparse-n213, pip-only after required HIP generation,
ccache20GiB/jobs12/link1/native gfx1013; old sources/builds/caches/third_party/
releases/fallbacks retained. Only dev Torch link/runtime metadata changes; ROCm
DSOs/packs/image/kernel/boot/defaults/.105 unchanged. Recovery
WS/recovery/sparse-n213-before, SOURCE_SPARSE_N213.json in WS/current; final
preservation baseline WS/logs/sparse-n213-qualification/final.json with candidate/
current-runtime hashes. [Report/results/recovery](bc250_sparse_n213_results/README.md).
Raw WS/logs/sparse-n213-* and ignored kernel/artifacts/sparse-n213. No active jobs;
free~7.23GiB. Python CPU segfault unresolved/separate from fixed VM-heap/CSR.
Next N2.14 FFT plan/layout/odd-real output+gradient coverage after P2 FFT30 audit;
AOT cache preserved, BF16 promotion B01 separate. Patch0010 incremental mbox on
Torch da45369 retains all three native/test commits; historical reconstruction
gap remains.

## N2.12 native triangular recovery — 2026-10-18

TheRock codeb1c94bddc adds explicit recovery_backend=device to isolated
right_eigenvectors; one32-thread HIP solve, cap16 complex64/128. Per-row
homogeneous column scaling retained; existing finite/unitary/triangular/gap/
unitnorm/residual checks common to both paths. N2.6 reference/default and
production host-LAPACK eig unchanged.128 retained groups, six CPU fixtures/36
backend evaluations,32 all-order direct cases,two near-gap rescaling stresses,
six sentinel/native ABI cases,six ABI refusals,four public/output refusals pass.
Default exact immutable N2.11; real GPU kernel traced. Paired three-sample
A1/B1/B2/A2 recovery1.32–2.32x/whole warm candidate1.17–1.34x in six fixtures;
recovery~1.90–2.10ms, host eig~0.213–0.408ms remains faster than native full
~3.75–11.79ms. No general/model/LoRA gain/integration; scalar reads remain18,
not all synchronizations. Nine guards ACK/600s, eight successful process gates;
initial required trace check honestly failed: CPU-only-first profiler leaves
later GPU captures empty. Source points to one-time cpuOnly registration; final
fresh process starts CPU+CUDA and traces native kernel. **A09 initialization
ordering defect remains unresolved**, independent of solver arithmetic. Both
traces/logs retained.104 protected hashes match; dmesg unchanged entire batch.
One-unit hipcc/ccache20GiB/OMP4 build; no Torch/installed ROCm DSO/prefix/default/
image/kernel/boot/.105 mutation. Prior helper and oracle retained under
WS/recovery/solver-n212-before and helpers/schur_n211_reference.py. [Report](bc250_solver_results/N212.md)
and n212/ contain full results/commands/hashes; raw WS/logs/solver-*n212*.
No active job/container/guard, disk~13GiB free. Python CPU segfault unresolved/
separate from fixed VM-heap/CSR. Next A09/N4 first-CPU→GPU initialization
reproducer/fix in separate candidate with both-order numerical/trace gates,
then broader N2 sparse/FFT/RNG coverage triage. N2.1–12 complete within recorded
bounds; N2–N8 still seven broad workstreams. Further small-eig optimization is
optional; no larger/batched/close-spectrum/balancing/gradient/compile/graph/
concurrency/general convergence contract added.

## N2.11 explicit GEHRD sessions — 2026-10-18

TheRock code807642198 adds explicit caller-owned GehrdSession, one PID/thread/
device/stream; native handle/bindings reused, fresh owned GEHRD copies retained,
close waits pinned stream then destroys once. No global cache/default mutation.
Create3.13–3.18ms dominates binding0.11–0.13/alloc-copy0.06–0.08/destroy~0.13ms.
128 retained groups, six CPU/backend fixtures, three lifetime groups/16 refusals
pass: mixed-dtype reuse, close synchronization/idempotence, exception after GEHRD,
two private sessions/streams interleaved, wrong thread/stream/device/closed/busy/
PID. PID guard injected, no real post-fork GPU test; device mismatch metadata-only.
Default exact N2.10 and fresh/reused H/Q exact on fixtures. Seven guards ACK/600s/
unchanged dmesg;82 protected hashes unchanged. Six paired three-sample warm prep
7.03–12.52x/whole1.28–1.95x with N2.8 QR/device Q; creation3.32–3.41ms/close outside
warm pairs. Cold one-call prep4.24–4.51ms gives no warm gain; host eig0.219–0.408ms
still faster. No general/model/LoRA/Torch-wide gain; N2.9 regression unchanged.
Python-only/no build/ccache/DSO/Torch/image/prefix metadata/kernel/boot/default/
.105 mutation. gfx1013_gehrd_session.py accompanies candidate. Prior helper/oracle
retained WS/recovery/solver-n211-before and schur_n210_reference.py. [Report and
reproduction](bc250_solver_results/N211.md), raw WS/logs/solver-*n211*. No active
job/container/guard; disk~1.08GiB free. No general concurrency/batching/gradient/
compile/graph/convergence/production GPU eig contract. Python CPU fault remains
unresolved/separate from fixed VM-heap/CSR. Next N2.12 profile warm-session recovery,
bounded native triangular back-substitution, Python/N2.6 reference/fixed separation/
rescaling/unitnorm/residual/CPU gates/cap16/host LAPACK default retained.

## N2.10 device GEHRD reflector-Q — 2026-10-18

TheRock codecc8a05d3c adds explicit q_backend=device to isolated hessenberg/schur:
one32-thread HIP complex reflector product, cap16/complex64+128. Native GEHRD,
owned packed input/current-stream handle lifetime and reference/production host
LAPACK defaults unchanged. Initial order16 split GEHRD~0.37–0.40ms/Q1.87–1.91ms/
preparation~6.2ms; remaining setup/teardown not attributed.128 retained groups,
24 independent CPU complex/zero/mixed reflector cases with strided guards/streams,
six complete eigenpair fixtures across both Q/three QR backends, exact old-default
reference/H controls, zero/scalar/refusal and GPU kernel trace pass. Fixed bounds
unchanged. Six paired three-sample preparation1.11–1.39x, whole N2.8-control
candidate1.05–1.12x; diagnostic Q1.87->0.17/1.83->0.19ms at16 includes submission/
sync. Host full eig0.222–0.429ms still far faster; no general/model/LoRA gain or
production integration. N2.9 regression stays distinct. Seven guards ACK/600s/
unchanged dmesg;79 protected hashes unchanged. One-unit hipcc/ccache20GiB/OMP4,
no Torch/ROCm DSO/image/prefix metadata/kernel/default/boot/.105 mutation. Prior
helper/oracle retained WS/recovery/solver-n210-before and schur_n29_reference.py;
all previous binaries retained. [Report/reproduction](bc250_solver_results/N210.md),
raw WS/logs/solver-*n210*. No active job/container/guard; disk~1.08GiB free.
No general convergence/batching/gradient/compile/graph/concurrency contract.
Python CPU fault unresolved/separate from fixed VM-heap/CSR. Next N2.11 profile
binding/handle creation-destruction/allocations and one measured reuse/cache fix,
retaining explicit lifetime/current-stream tests/owned GEHRD inputs/Python oracle/
cap16/fixed numerical/failure gates/host production eig default.

## N2.9 bounded iteration chunks — 2026-10-18

TheRock code13f91e094 adds explicit device_chunked backend, chunk1..32/default16,
reusing N2.8 control and stopping immediately on convergence/budget/failure.
Separate T/Q input buffers; unchanged cap16/scaling/total QR budget/deflation/
shift/fill/separation/residual gates.128 retained groups,74 control oracles,
100 chunk boundaries, four long-chunk failures, seven public option refusals
and six full CPU eigenpair fixtures pass. Complete results/counters bitwise
exact across chunks1/2/8/16/32 on fixtures, boundary chunks exact vs repeated1.
Copies8/20/35->1/2/3 c64,11/26/46->1/2/3 c128; whole ratios1.034/1.046/0.984
and1.040/1.047/0.951 at4/9/16 are mixed, with16 regressing. Host eig0.224–0.426ms
remains much faster. Padded-Q variant passes but does not improve timing; retained
in WS/recovery/solver-n29-padded, original restored/verified against passed hashes.
12 guards ACK/600s/unchanged dmesg;76 protected runtime/source/recovery hashes
unchanged. Two one-unit hipcc/ccache20GiB builds; no Torch/ROCm DSO/image/prefix
metadata/kernel/boot/default/.105 mutation. Prior source/binary/oracle retained
WS/recovery/solver-n29-before and WS/helpers/schur_n28_reference.py. [Report and
reproduction](bc250_solver_results/N29.md), raw WS/logs/solver-*n29*. No active
job/container/guard; disk~1.09GiB free. No general convergence/gradient/batching/
compile/graph/production GPU eig/model speed contract. Python CPU fault remains
unresolved/separate from fixed VM-heap/CSR. Next N2.10 split GEHRD/reflector-Q
preparation and prototype bounded device Q accumulation, preserving native GEHRD,
Python/N2.8/N2.9 controls/cap16/fixed gates/production host LAPACK default.

## N2.8 bounded device Schur control — 2026-10-18

TheRock code67e5d1ef8 adds explicit device_control backend: trailing deflations,
nearest/exceptional complex shift, reused cooperative QR/similarity and finite/
128eps fill status in one HIP launch. At most one QR/launch, budget+1 launches,
one five-int status copy/launch; unchanged cap16/dtypes/scaling/budget/deflation/
separation/recovery residual/refusal rules. Default reference and production host
LAPACK unchanged.128 retained groups,74 direct state/boundary/root/failure/stream
oracle cases and six CPU eigenpair/paired fixtures pass. Reference matches N2.7
exactly on final six fixtures. Initial direct T/Q comparison ignored nonunique QR
phases; phase-aligned similarity and independent reconstruction fix harness with
unchanged bounds, no solver change. Failed log retained. Seven guards ACK/600s/
unchanged dmesg. Schur1.80–2.73x, whole candidate1.56–2.21x versus N2.7 in fixtures;
whole8.88–20.35ms still much slower than host full eig0.23–0.42ms. Order16 scalar
reads159/207->8 excludes explicit35/47 status copies; not a total-sync count.
70 protected runtime/source/recovery/metadata hashes unchanged; one-unit hipcc/
ccache20GiB, no prefix metadata/DSO/Torch/image/kernel/boot/default/.105 mutation.
Prior source/binary WS/recovery/solver-n28-before; immutable N2.7 oracle retained.
[Report/reproduction/evidence](bc250_solver_results/N28.md), raw WS/logs/solver-*n28*.
No active job/container/guard; disk~1.09GiB free. No general convergence/gradient/
batching/compile/graph/production GPU eig or universal/model/LoRA performance
claim. Python CPU fault unresolved/separate from fixed VM-heap/CSR. Next N2.9
bounded multi-iteration chunk to reduce status copies, preserving exact QR budget,
terminal/refusal semantics, Python/N2.8 oracle/fixed gates/cap16/host default.

## N2.7 cooperative device QR/similarity — 2026-10-18

TheRock code100f43e2a adds explicit device-step Schur backend, complex64/128
order2..16:32-thread HIP shared-memory Householder QR and full similarity/Q
update. Reference backend remains default; final six fixtures match prior N2.6
outputs/counts exactly. Normalization, iteration/deflation/fill, separation and
recovery residual/refusal gates unchanged.28 direct CPU QR/invariant/stream cases,
128 unchanged N2.3–5 groups and six complete eigenpair fixtures pass; budget0/1
partial/refusal contracts and profiler kernel evidence retained. Correct serial
prototype was slow; source/binary retained before cooperative rewrite.
Paired three-sample A1/B1/B2/A2 Schur1.24–1.46x in six fixtures; device10–36ms
still far slower than host full eig0.22–0.40ms. Separate instrumented order16
preparation~6.1ms, QR3.38/6.41ms, remaining checks/scaling/control22.46/28.24ms;
forced sync phase sums are diagnostic, not kernel-only/model/general gains.
Eight guards ACK/600s/unchanged dmesg;64 prior protected source/runtime hashes
unchanged. One-unit targeted hipcc build/ccache20GiB; no prefix metadata/ROCm
DSO/Torch/image/kernel/boot/default/.105 change. No active job/container/guard;
disk~1.09GiB free. Prior helper WS/helpers/schur_n26_reference.py; serial recovery
WS/recovery/solver-n27-serial. [Report/reproduction/evidence](bc250_solver_results/N27.md),
raw WS/logs/solver-*n27*. No production native eig, batching/gradient/general
convergence contract. Python CPU fault stays unresolved/separate from fixed
VM-heap/CSR. Next N2.8 bounded device shift/deflation/status to reduce measured
control overhead, preserving Python oracle/fixed gates/cap16/host eig default.

## N2.6 small native eig profile/recovery vectorization — 2026-10-16

Initial order16 recovery~37ms/376 Torch scalar reads; chose all-column rowwise
GPU back-substitution, n-1 instead quadratic row updates. Checked global gap
covers active pivots, masked inactive pivots set1; homogeneous scaling/final
finite/residual checks retained. Scalar reads4/9/16:34/124/376 ->18;107 unchanged
N2.4/N2.5 groups pass. Six synchronized wall-clock A1/B1/B2/A2 fixtures with CPU
references show recovery1.53–8.44x and whole1.07–1.72x improvements, raw3-sample
passes/controls retained. Native whole16–55ms remains far slower than host0.20–
0.40ms; no default integration/model/general speed claim. All logs identical,
guards ACKed, no active jobs. [Report/source/evidence](bc250_solver_results/N26.md).
No native compile, Torch/DSO/prefix/kernel/boot change; prior helper retained at
WS/recovery/solver-n26-before. Existing CPU fault/MIOpen issue remain open. Next
N2.7 split Schur preparation/QR-loop cost and bounded device-side iteration to
reduce remaining measured launch/scalar synchronization overhead.
