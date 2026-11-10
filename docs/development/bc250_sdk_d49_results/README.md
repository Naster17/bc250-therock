# D49: SDK sentinel repair activated on .104

Completed 2026-11-10. The D48-qualified native SDK is now selected through
ordinary .104 launchers. .105 was not accessed or changed.

## Installed change

- SDK source: `143f1d878296176c738274202092bb247b7b771c`.
- New SDK SHA256: `2a975370ba4eb1c2fe39e406ff36b175f554a49d0aa19febbd326f641fcb590b`.
- Old SDK SHA256: `a6dc1c884c4f446e23a2afc8185b26ef9fd55b57cdd1d15324d46054b902b9c2`.
- Shared public/D09 helper SHA256: `d33f2504de50eac21baf6f43ff4eb30a05c602881cda9b12b3df13360a2c3588`.
- New cohort: `/opt/bc250-stack/candidates/sdk-d49-r1`.
- Preserved fallback: `/opt/bc250-stack/candidates/miopen-plan02-r4`.

The new ROCm tree shares unchanged file inodes with the original tree. Only
`lib/librocprofiler-sdk.so.1.4.2` was atomically replaced with an independent
inode; all 27,905 tree entries were compared. This is a structural/inode
preservation check, not a new full-byte rehash of every original library.
No original library was overwritten. Native Torch, MIOpen, RCCL, images, caches,
source/build trees, kernels, boot selections and application data were preserved.

Both profile `runtime.json` files retain their exact baseline bytes. Adjacent
`native-overrides.json` records the SDK-only delta; read both files together.
This avoids falsely advertising a complete rocm-systems rebuild and preserves
existing CK/MIGraphX/ORT package registration identities. The new cohort has
independent wrapper copies and registrations. D01 now selects its RCCL profile,
retains the exact baseline runtime hash/native Torch identity, and explicitly
records the native SDK override. No two-host requalification was performed here.

The public shared helper admits exact `HIP_VISIBLE_DEVICES=-1` only when the
selected baseline manifest, repaired SDK hash and both SONAME aliases match.
It rechecks under the stable runtime lock before building the container command.
Legacy/missing/corrupt/mismatched cohorts keep the D44 refusal. No global guard
removal, CI disable, architecture override or LD_PRELOAD deployment was used.
Custom preloads/auditors and custom native Torch overlays are outside this
qualification; retain the ordinary loader order.

## Qualification

[Replayed results](verified-results.json), [altered-archive refusals](negative-controls.json).

| Phase | Fresh processes | Numerical checks | Scope |
| --- | ---: | ---: | --- |
| First activation | 12 | 84 | Both profiles: default/0/0,-1 GPU and HIP/CUDA/ROCR -1 CPU-only imports |
| Actual rollback | 2 | 26 | Both original GPU profiles, old SDK mapped; UID1000 legacy HIP -1 refused before worker |
| Final reselection | 4 | 28 | Both profiles: default GPU and HIP -1 CPU-only |
| Total | 18 | 138 | 20 raw GPU traces; eight negative-selector imports |

All GPU output/input-gradient/weight-gradient digests match the actual restored
old cohort and the retained D48 ordinary controls exactly. Fixed FP32 matmul/sin
references use CPU FP64 with atol=2e-6/rtol=2e-4, exact profiled replay and pinned
H2D/D2H on a nondefault stream. Raw traces have kernels, copies, positive finite
durations and runtime correlations. Every worker binds the loaded SDK path/hash.
All before/after kernel logs are byte-identical; all three independent guards ACK.
Six actual CK/MIGraphX/ORT `registered()` checks passed on inactive new profiles;
these are identity checks, not new provider GPU/operator qualification.

Limits: EX board reservation + SH runtime for tests, EX board + EX stable runtime
for each selection/rollback; 4 CPU cores, 4 GiB RAM with zero container swap
allowance, 512 PIDs, 256 MiB shm, 60-second child/420-second batch/600-second
independent recovery limits, 100 ms sampling, 3 GiB host/5 GiB disk reserves.
Minimum sampled host available: 14,229,344,256 bytes; disk: 24,240,054,272 bytes.
Phase wall times: 87.318/15.064/29.172 seconds, not performance measurements.
No new cgroup peak, host paging-I/O, model/AMP/compiler/collective/counter,
universal correctness or stability claim. UUID selection remains unavailable:
HSA reports `GPU-XX`, not a supported board UUID.

134 local tests pass, including three new identity/alias/lock-time regressions.
D44 tests explicitly mock a legacy cohort so their result is independent of the
machine running the tests. Static checks and three altered source/trace/rollback
archives pass/refuse as expected. The archived executed probe/controller retain
three unused imports removed from the committed harness; the installed helper
is byte-identical to the committed source.

## Historical defects reviewed

| Defect | Current disposition |
| --- | --- |
| Exact -1 import SDK CI fatal/secondary libc fault | Native repair qualified D48, activated D49 on .104; legacy fallback still protected |
| GPU traces disappear when SDK is preloaded | D48 reproduces with unchanged old SDK too; ordinary loader resolution selected, no preload deployment |
| Python CPU IP0x3b3e6 | No original core/backtrace; cause unknown. Existing faulthandler diagnostics retained. Passing D49 does not repair it |
| KFD mm/queue fault around exec after HIP | Frontend runpy mitigation already exists; kernel cause untraced, not repaired by SDK activation |
| .105 clocksource remote-CPU timeout | Open; .105 unchanged. `pr_info_once`/quiet logs cannot establish absence of recurrence |
| Prior VM-heap corruption, D05 agent lifetime, D21 optimizer restore | Separate previously fixed defects; not reopened or conflated with the unknown Python fault |

No speculative kernel/runtime edits were made for faults without a demonstrated
cause. A recurrence requires stage/faulthandler/core/backtrace and kernel evidence
before selecting a repair; do not repeat broad stress sweeps merely to collect
more quiet runs.

## Evidence, recovery and next action

Host tools: `~/bc250work/tools/sdk-d49`; controller PID 7740 finished successfully.
Host evidence: `~/bc250work/evidence/sdk-d49-{controller,install,activation-r1,rollback-r1,final-r1,export}`.
All owned tests/containers/guards are finished. Boot ID remains
`2774d201-60ab-4285-a2f6-63ffcaf9873c`, kernel `7.1.5-bc250rocm-r2`.

Metadata archive, workstation `kernel/artifacts/sdk-d49/104-evidence-r1.tar.gz`,
host `~/bc250work/evidence/sdk-d49-evidence-r1.tar.gz`: 1,658,051 bytes,
SHA256 `759648024bd274dee3a0e7710d6d5f6f253dccdec86cc81dd793738313aa71b6`.
The native SDK package itself remains the D48-qualified archive; no rebuild needed.

```sh
python3 therock/build_tools/bc250/verify_sdk_d49_export.py \
  kernel/artifacts/sdk-d49/104-evidence-r1.tar.gz --output /tmp/d49-replay.json
```

The actual rollback was exercised before reselection. If this cohort needs to be
reverted later, with the board free, the existing helper has one unused rollback
transaction label and preserves the stable locks:

```sh
doas python3 /home/nik/bc250work/tools/sdk-d49/activate_sdk_d49_on_host.py \
  --action rollback --label failure-rollback \
  --from-transaction /home/nik/bc250work/evidence/sdk-d49-install/activate-r1/before.json
```

It restores all recorded pointers, both shared helper copies and D01 bytes.
Do not run a launcher while holding its EX runtime lock. Existing transaction
labels are never overwritten; further recovery attempts need a reviewed new label.

Next D50 / PLAN-07: close the portable Torch dependency contents/license inventory
from D47's 37 direct pins and 27 immediate nested gitlinks; record missing objects,
recursive contents and the Gloo legacy-platform overlay/build-version inputs.
Package/replay only admitted inputs; do not claim whole-stack offline build closure
until the remaining toolchain/wheel/license inputs are accounted for. PLAN-07
remains open; PLAN-07 through PLAN-10 are the four important remaining packages.
