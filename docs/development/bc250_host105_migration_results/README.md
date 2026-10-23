# M105.1 — .105 private deployment and legacy retirement

Completed 2026-10-23 with explicit user authorization to replace the old reference
stack. Tools: `ddc2933cb`. `.104` remains the development host; `.105` now runs an
independent pinned daily userspace snapshot and guarded r3 default. N8 distributable
packaging and sustained qualification remain separate work.

| Item | Installed identity |
| --- | --- |
| Host | `nik@192.168.1.105`, hostname `bc500`, Alpine 3.24.2 |
| Release | `/opt/bc250-stack/releases/10.2-gfx1013-20261023-host105` |
| Selection | `/opt/bc250-stack/current` points to that release |
| Image | `506f3b262fa5fe8c29b9babec07b03078dd6f85492a9ac2cffc65955297e4349` |
| Discoverable image tag | `localhost/bc250-stack:10.2-gfx1013-20261023-host105` |
| Torch | `2.13.0a0+gfx1013`, native source `2b449c4abe36b896a6efb2a68e7a42d9922d5696` |
| HIP/CLR | Native `1f9609fd69c4923291f58934495520fe62e66ef8`, existing hostcall repair retained |
| Kernel | `7.2.10-bc250rocm-r3`; image source `84995f1`, signed AMDGPU override `0957f1a` |
| Image/module build IDs | `57f7c52d9c7ac0ac21c67d6e6efc3e08c12bbfde` / `b97bca56793aecea35967385d0c68d4186ecd121` |
| Final runtime JSON SHA256 | `c5da85ef459e9e586904a2870d68545a911805d0c7d93016cb4a3f9b96cc608b` |
| Final boot | `e3d322db-a221-47a1-9ab4-9a48b84574a5` |

This copies the current `.104` ROCm-10.2 workspace snapshot, including compiler,
libraries, assets/databases, native Torch and existing torchvision/torchaudio/
TorchCodec/FFT/sparse helpers. `torch.version.hip` reports component version
`7.17.0`; package identity is pinned by revisions and hashes above. Triton is3.8.0;
frozen training consumers include Transformers5.2.0, PEFT0.18.1 and Accelerate1.12.0.
Daily RCCL remains off/Gloo on; B03.7's isolated RCCL candidate is unchanged.
Math SDPA, SDMA0, experimental-heap-off and N3.2 tuning remain the safe defaults.

## Acceptance

All69,229 ROCm/Torch file/link entries matched `.104` before selection and after
legacy retirement: ROCm24,664, Torch44,565. The origin inventory was independently
repeated under `.104`'s exclusive runtime lock. The local recovery runtime archive
also matches all69,229 entries. Image and installed healthcheck
sources were also verified. No native rebuild, ccache reset or `.104` runtime,
boot, source/build/cache/image change was required; only small export/audit files
were added there.

| Target r3 gate | Checks |
| --- | ---: |
| Core FP32/BF16 GEMM, math SDPA, normalization, FFT gradients and RNG/consumers | 7 |
| Two local CPU Gloo ranks, collectives, first gradients and DDP updates | 31 |
| Bounded GPU CSR helper, dense/analytic/finite-difference references | 253 |
| BF16 FFT promotion/helper contracts and first derivatives | 245 |
| Vision operators and gradients | 51 |
| Native audio operators and gradients/codec interface | 40 |
| Compiled dense training against CPU FP64/optimizer references | 14 |
| Compiled CNN training against CPU references | 14 |
| CPU-first native profiler capture/references | 1 |
| **Main suite** | **656** |

Each accepted gate had an acknowledged600s guardian and identical before/after
kernel logs. Additional seven-check smokes passed on r2, the cold r3 trial,
ordinary `bc250-train`, three promoted normal boots, recovery r2 and after
retirement. Final ordinary `bc250-torch` produced1496 and the exact CPU gradient
for the owned17-element square/sum case. These are bounded output/first-gradient/
optimizer/compile checks, without universal, throughput, convergence, multi-board
or sustained-stability claims. Hardware watchdog countdown on `.105` was not tested.

Two contained setup failures are preserved. The first new harness omitted the
separate frozen consumer mount; native math ran, then Transformers import failed.
The corrected mount passed. The first r3 container could not start because the
previous manual cgroup mount was lost at reboot. Existing unified `cgroups`
service was started and enabled in boot; CPU/memory/pid limits then worked. No
kernel rebuild or disabled-cgroup workaround was used. Failed guardians were
released only after passing recovery smokes and unchanged workload kernel logs.

## Boot and SSH recovery

Persistent GRUB default is `bc250-host105-r3-release`. Before entering Linux, that
entry saves `next_entry=bc250-rocm-r2-kernel`. The boot guardian and builtin
SP5100/nowayout watchdog protect the deadline. With AMDGPU deferred, the release
readiness service checks SSH, cooling/NIC/watchdog ownership, signed module
identity and a bounded GPU reference smoke. It acknowledges the deadline and
clears the saved fallback only after success. Runtime guardians explicitly select
r2 on expiry, including after successful boot readiness has cleared `next_entry`.

Exercised transitions: r2 → r3 SSH trial → normal promoted r3 → deliberate15s
idle-GPU expiry/emergency reset → r2/SSH and GPU smoke → normal r3 → post-retirement
normal r3. Three normal promoted boots passed. r2 files/modules were retained;
image/initramfs hashes remained unchanged. The consumed one-shot and final empty
`next_entry` were checked. Direct SSH and access through `.104` both worked at end;
temporary port2222 listeners were not installed persistently. Arbitrary GPU hangs
and failures before watchdog arming remain outside this evidence.

Operational files to preserve:

- `/etc/init.d/bc250-host105-ready`, `/etc/init.d/bc250-r3-bootguard`, watchdog and cgroups boot services.
- Versioned release `healthchecks/`, `/var/lib/bc250-stack/host105`, `/var/log/bc250-host105-ready.log`.
- `/home/nik/bc250work/kernel-r3-7.2.10/{artifacts,firmware-from-r2,helpers}` and both kernel module trees.
- r2 boot files in `/home/nik/bc250work/bc250-rocm-lab/`, GRUB recovery entries and native board controls.
- `/opt/bc250-stack/migration-control/.runtime.lock`; prefix/boot updates retain the exclusive-lock procedure.

Readiness executes root-owned release files and writes operational data under
`/var/lib`; it does not depend on the migration scratch fixture directory.
For an intentional one-shot r2 boot, stop workloads, select r2 through the recorded
exclusive update helper, then reboot:

```sh
doas python3 /home/nik/bc250work/migration-105-20261023/update_dev_prefix.py \
  --release /opt/bc250-stack/migration-control -- grub-reboot bc250-rocm-r2-kernel
doas reboot
```

## Retirement and use

Five inspected stopped legacy containers and eight selected7.2.4 images were
removed, with an unused parent removed by Podman. Thirty-six explicit legacy
source/build/prefix/packaging directories were retired after archive/recovery
checks; two old public Linux6.18 download archives were also removed. Legacy
retirement freed75,132,190,720bytes; this batch's duplicate transfer archives were
then removed. Available root space rose from about47.7GiB at entry to101.0GiB
(108,477,546,496bytes before the small evidence export).

Old source histories and modified MIOpen files are preserved locally under
`kernel/artifacts/migration-105-20261023/legacy-archive/`. Archives contain540,714
and4,867 entries;460 MIOpen source hashes were compared to the live old tree before
retirement. Archived rocm-libraries Git passed full fsck at`b578bcd410d0641337ce0f34be055b88560f5c24`.
Generated objects/libraries/wheels and signing material were omitted;13 PEM test/
certificate files were removed from working-copy archives. A generated root-only
`usr/initramfs_data.cpio` was unreadable; existing boot images are preserved separately.
[Archive identities](legacy-sources.json) and raw evidence retain the exact scope.

User applications/models, checkpoints, Docker storage, unrelated base images,
home caches, historical small research records and native governor/fan/unlock/SMI
utilities were retained. The small native host SMI library supports board tools;
it is separate from the replaced compute distribution. `.104`'s active runtime
SHA32fd4559/Torch2b449c4/HIPa1162308/r2 boot2db1c7c2 remain unchanged. No BIOS or
user `finetune-n7` inspection/modification/test occurred.

Use the same launchers on `.105`:

```sh
bc250-torch script.py
bc250-train script_using_the_frozen_training_consumers.py
```

Raw target evidence archive: SHA256
`532eb6bb0b72fefba524e3ddd2cbe87136c8d0cfd381df2977a3caa784a618e1`,
local `kernel/artifacts/migration-105-20261023/evidence.tar.gz`, extracted `raw/`.
Machine outcomes: [final](final.json), [recovery](recovery.json),
[payload](payload-verification.json), [post-retirement](post-retirement.json),
[local verification](verification.json), [revisions](revisions.json).
The historical Python CPU segfault remains unresolved and distinct from the fixed
VM-heap/CSR/hostcall defects. Resume **B03.8**, the bounded opt-in RCCL profile;
no new cross-board distributed qualification is claimed by this migration.
