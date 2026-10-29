# B07.7: relocatable explicit MIOpen DL artifact

Completed2026-10-29 on .104; tools `fd8c3fe8c`. Accepted **package-r2**,
qualification-r3. Source libraries `3616b2f9f09`, installed libraries `8389197c1`,
Torch `a524980` unchanged. No plugin/probe/native code rebuild or installation.

## Accepted artifact and activation boundary

`bc250-miopen-dl-gfx1013-b077.tar.gz` contains23 files/6,568,985B: the unchanged
qualified plugin, three accepted reference probes, explicit Python entry, licenses,
focused matching MIOpen/CK source snapshot, configs/build script and genuine
B07.6 two-key daily/RCCL example records. It requires the matching external native
ROCm/SDK and CK instance archive to rebuild; it is not a complete ROCm rebuild kit.
The copied ELF RUNPATH still targets the required qualified native runtime prefix;
there is no original private candidate directory dependency.

**Accepted archive:**5,031,322B, SHA256
`90f0d323d0bb4ffd3a8ae04f1acd4e17805a10d7dcb62d577840cf2120e06984`.
Only this archive is qualified; package-r1 with the same filename is rejected.

The entry verifies every payload, actual mapped MIOpen/HIP/HSA/C++/C/libm/libgcc/
loader binaries (eight SHA pins), native gfx1013:xnack- and API5/66 required exports
before enabling the private library for one process. These checks detect drift;
the separately pinned archive hash anchors the manifest. This is not a signature
or a security boundary against hostile manifest/code edits, and not every transitive
GPU library is pinned by the artifact. The qualified runtime remains a prerequisite.

`check`, `seed-db` and `run` are **artifact subcommands**, not installed host commands.
`seed-db` requires a nonexistent destination; it never merges into user databases.
`run` executes a Python script as `__main__` with its argv/import directory in the
verified process. It uses `runpy` to preserve the initialized HIP/KFD context.
See the packaged [usage/source README](../../../build_tools/bc250/ck_plugin_b077_README.md).
FP32 packed2D forward/exact alpha1/beta0 only; other groups refuse. No new backward,
automatic Torch routing, general shape/performance or training-speed claim.

## Relocation and final acceptance

Two separate archive extractions, one with spaces in its path, each run daily and
rccl-single. The container mounts only that read-only artifact, owned scratch/logs,
validation helpers and the required runtime; original B07.3/B07.6 candidates,
`WS/source` and `WS/build` are absent. No native plugin/probe build occurs. The
runtime's normal MIOpen kernel JIT/cache facilities remain available; build-free
relocation does not mean a compiler-free standalone runtime.

Each extraction/profile verifies identity, native/hash/architecture/path refusal,
genuine seed-copy/existing-destination refusal, explicit public137 and all six
per-ID CPU-double references, unsupported API/runtime contracts, normal PerfDb
replay, absent-path private-solver refusal and an owned-script native backward.
Four artifact groups7+7+7+7 plus core7/CPU ONNX4/daily5/RCCL6 = **50 checks**,
eight accepted600s guards.1168 convolution reference executions,488 per-ID API
refusals/984 runtime refusals,608 retained API checks/128 absent-path refusals,
20 entry/seed refusal cases. Max absolute error2.35537e-6, unchanged N2.18 tolerance,
dyadic exact and256-byte guards/immutable inputs pass.29 local tests, Black,
critical Ruff/Python compilation/diff pass; full upstream CI not run. Existing
B07.6 performance evidence remains narrow; no new timing gain claimed here.

## Rejected first attempt and remaining kernel uncertainty

Package-r1's test process returned0 and numerical checks passed, but its strict
kernel gate failed on a new message:

```text
amdgpu_amdkfd_restore_userptr_worker: Failed to resume KFD
```

The first entry initialized HIP for architecture checking, then used `exec` to
replace that live process address space. That lifecycle was removed: r2 runs the
script in the verified process. The successful r2 daily check and all final r3
checks have no new kernel messages. The first failure and both package archives
remain preserved; no log clearing/reboot was used.

This is bounded evidence for avoiding that lifecycle, **not proof of a kernel
root-cause fix**. Local r2 kernel source allows either missing-mm `-ESRCH` or queue
restore failure at the emitting path; the observed return was not traced. Keep
this as a N6 investigation input if the pattern recurs. It remains separate from
the historical Python CPU fault and already fixed VM-heap corruption.

Qualification-r2 then stopped before its RCCL container because the controller
passed daily metadata to the opt-in profile. R3 selects the proper release/manifest;
no artifact or native numerical change was made for that controller correction.
Initial CPU identity capture also had a background TTY/container-attach error;
a foreground/noninteractive r2 capture exited0 and supplies the pinned metadata.

## Preservation, commands and evidence

3772 protected hashes/35 links/four AOT DBs/136 workstation backing paths unchanged.
Shared runtime lock/resource preflight; no active-prefix update. Ordinary free
12,888,330,240→12,653,342,720B; no deletion. Same boot
`e327c5de-46b8-4e05-94ad-d999a12a06e7`, kernel7.1.5-bc250rocm-r2. Kernel logs
unchanged **since final r3 entry**, not since the rejected first attempt. No owned
jobs/containers/armed guards at cleanup; .105/user applications untouched.
Ten total guards ACK, eight belong to the final accepted batch. Original binary,
build/source/cache/releases/path links remain. No compilation/ccache reset.

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`; staged helper aliases preserve earlier
versions. Exact accepted commands:

```sh
python3 helpers/package_ck_plugin_b077_r2_on_host.py --attempt r2
python3 helpers/qualify_ck_plugin_b077_r3_on_host.py --attempt r3 --package-attempt r2
python3 helpers/export_ck_plugin_b077_r3_on_host.py --attempt r3 --package-attempt r2
```

Existing destinations must be preserved. These helpers correspond to committed
`package_ck_plugin_b077_on_host.py` and qualification/export counterparts; runner
`ck_plugin_b077_entry_r2.py` corresponds to committed `ck_plugin_b077_entry.py`.
New attempts need new helper/input/output versions. Package1PID1940/package2PID3233
exit0; qual1PID2297/qual2PID3406 exit1; qual3PID3971/exportPID6306 exit0.
`WS/tmp/ck-plugin-b077/{package-r1,package-r2,r2-extract-a,r2-extract with spaces,export-r3}`;
`WS/logs/ck-plugin-b077-qualification-r3/final.json` and matching
`WS/logs/ck-plugin-b077-r3-controller.log`. Initial/retry logs, bad manifests and
original artifacts are retained; repeated bad-copy payloads are omitted from the
local evidence archive because the original archives retain those identical bytes.

Local `kernel/artifacts/ck-plugin-b077/` contains accepted/rejected package archives,
manifest, verification, cleanup and evidence: 6,819,295B,
SHA256 `eebcf955438a9a5655bc69ca84334d14da83b61c54b629a5a0a55f7576fd73b4`,
345 raw files/16,958,503B. Payload, original accepted binary,
source/helper alias, example-record and dependency pins verified.

B07.8: install only the accepted SHA-pinned r2 artifact as a versioned explicit .104 option, with identity/registration/refusal/rollback and ordinary daily/RCCL consumer activation checks under the exclusive update lock. Keep production whitelist/default routing/native payloads/fallbacks/.105/user applications unchanged; update usage documentation.
