# B05.3: matched RCCL Torch layout repair

The `.104` opt-in `rccl-single` profile now includes the native weight-only INT8
activation/FP32-scale layout correction already installed in daily Torch by
B05.2. Both cohorts use source`a5249802372882dedccb91d3fd4d73be9fcdf2bd`;
NCCL-off daily and RCCL-enabled native payloads remain distinct. Daily runtime,
ROCm, kernel and `.105` are unchanged. [Verified results/identities](verification.json).

Tooling:`411e72917cb6b10d8cb36b7c99127720f03ed163`.

## Build and isolated qualification

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`. Independent
`WS/pytorch-quant-b053` copies the protected B03.7 RCCL source/build, excluding
third-party dependencies, which remain read-only. The full two-file tracked
source delta from14cfef2 toa524980 is applied once. Generated INT8 HIP source
matches the original with exactly two contiguous calls added. Native arithmetic,
output casts, RCCL FP8 admission/byte-copy/BF16 scalar corrections are retained.
No new Torch source edit, RCCL DSO rebuild, architecture override or matrix-only
engine enablement is introduced.

Pip-only build exits0 using the pinned28ad36e6 build image, gfx1013,
NCCL/RCCL+Gloo on, four CPUs/four jobs/one linker,10GiB RAM/12GiB memory+swap,
shared20GiB ccache/no reset, no GPU/network. CMake flags and source hashes are
checked. Initial free disk~4.8GiB, copied tree~3GiB; pre-copy estimate plus1.25GiB
reserve and post-copy1.25GiB floor are enforced. Observed headroom~1.8GiB remains
before the evidence archive. No source/build/cache cleanup occurs; cumulative
ccache statistics are not a performance or new-error claim.

The shared native validator accepts explicit `--nccl-available`, retaining its
default NCCL-off contract and exact numerical assertions. The existing guarded
harness accepts explicit `--runtime-profile rccl-single` for an isolated release;
its default behavior is unchanged. Original helper versions are preserved in
`WS/recovery/quant-b053-before`, and intentional helper changes/backup hashes
are included in the preservation baseline.

New root-owned profile `/opt/bc250-stack/dev/rccl-single-b053` has an independent
RCCL copy and relative SONAME links; Torch targets the independent build. ROCm,
cache, consumers and tuning refer to the daily developer prefix. It is a dependent
development profile, not a standalone relocatable distribution.

## Results and activation

**1,732 candidate checks/24 gates**:native220, W8 recipe71 and the retained
B03.7 suite1441. Native checks include216 exact CPU-double dtype/layout/guard
composites, private stream and preserved backward/INT4 boundaries. Retained gates
include FP8 refusals/raw-bit copies, public contracts, nontrivial PreMulSum,
first gradients/owned DDP, GPU/CPU Gloo, sparse/FFT/vision/audio/image/video/files,
compiled models, profiler and core. No numerical tolerances are weakened.

Candidate testing uses the explicit profile's normal identity checks/RO RCCL
mount and fixed environment. The old test-only RCCL mount/environment injection
is not used; retained RCCL fixtures use their packaged mode and verify the loaded
`/workspace/rccl` DSO. After qualification, only the registry symlink is atomically
switched under the daily exclusive lock.

**371 ordinary checks/five gates** after selection:public opt-in native220,
frozen training consumers6, RCCL first-gradient/DDP64, guard/readiness76, default
daily launch5.29 guards ACK; accepted gate/batch/final kernel logs are identical.
434 protected hashes/four AOT DBs are retained. The registry switch and two helper
updates are intentional; original profile, source/build/native files and helper
backups remain preserved. Same r2 boot5c04afbb-ff2f-4f4b-9eac-8238bb9288fc,
no reboot.29 local BC250 tests, selected Ruff checks/format and shell syntax pass;
no new Torch source lint/full upstream CI/other-GPU claim.

Selected registry `/opt/bc250-stack/profiles/rccl-single` now points to
`/opt/bc250-stack/dev/rccl-single-b053`:

- Profile manifestSHA`446f345e1438f093967a495c816859f4505bd06869dbea382842c2f081f14fd7`.
- Native libtorch_hip SHA`a31cbb159d3edfb4b7e1cc59dfcc325db185c4c540e1fcaa952e09e833335533`.
- Same RCCL DSO SHA`ca790dc73b76a94a7dc97538608cd94b789e73cc18d756b0631111c18230583e`.
- Daily manifestSHA`561f014e31c4ca0bba73861d56125e4e18124867904b7c23cff8d7e0fa36d2a1`
  and native B05.2 Torch remain unchanged. Image506f3b26, HIPa1162308/ROCm/N3.2/
  math SDPA/SDMA0/experimental-heap-off defaults are retained.

No user command change is needed:

```sh
bc250-torch --runtime-profile rccl-single /path/to/owned_script.py
bc250-train --runtime-profile rccl-single /path/to/owned_script.py
```

Omit the option for the unchanged daily NCCL-off/Gloo-on runtime. Existing
window0/CUMEMboth0/P2Poff/IBoff/loopback/async-error1 settings, startup identities,
RO mounts and daily runtime-lock integration remain. Native backward still
refuses; native packed INT4 still rejects gfx1013. The W8 helper is still an
explicit/not-installed prototype; no automatic model conversion or TorchAO/
bitsandbytes/NF4/complete QLoRA integration is supplied.

RCCL is the same filtered375-object subset, bounded to one physical GPU/rank one.
No numeric FP8, actual offload-payload relocation, symmetric device API,
multi-device/protocol/transport/scaling, speed, higher-gradient or universal
correctness/stability claim follows. Compiled-model regression is not compiled
quantized-op qualification. Historical Python CPU faults remain unresolved and
distinct from fixed VM-heap/CSR/hostcall defects. User applications were not
inspected, modified or tested; `.105`, BIOS, fallbacks and user-owned sources
remain untouched.

## Recovery and next action

Old profile `/opt/bc250-stack/dev/rccl-single-b038` and B03.7 build remain intact.
`WS/recovery/quant-b053-before/profile-runtime.json` and `registry-target.txt`
record the old selection. Identity-guarded rollback restores only that registry
selection and was not exercised:

```sh
doas python3 "$WS/helpers/update_dev_prefix.py" -- \
  python3 "$WS/helpers/deploy_quant_b053_on_host.py" --rollback
```

Build/stage/activation/export use the daily exclusive updater. Internal isolated
candidate gates avoid nested shared locks; public acceptance runs outside EX
with ordinary launcher SH locks. Main reports:
`WS/logs/quant-b053-qualification-r1/final.json` and
`WS/logs/quant-b053-ordinary.json`; raw `WS/logs/quant-b053-*`.

**Next B05.4:** profile the strict W8 recipe's validation/native forward/backward
cost on the corrected stack, then implement a measured bounded optimization with
independent CPU/gradient/error-contract references and paired timings. Preserve
daily/opt-in defaults, numerical/refusal contracts, old artifacts and `.105`;
do not claim model speed or weaken checks based on small fixtures.

Local `kernel/artifacts/quant-b053` preserves the verified200,536,100-byte archive
SHA256`171924233619ce789f958f195d275b6fd07e4522b45a54ed883a015212710eb6`,
332 raw log files/8,370,556bytes, nine native DSOs/extensions, RCCL copy, source,
CMake provenance, eight tools, original helpers/old selection and29 guards.
All tool/native/profile/generated-source hashes match the extracted evidence.
No own build/test/container or armed guard remains at handoff.
