# PLAN-02: native MIOpen FP16 repair deployed

Completed 2026-11-08, within the bounds below. Implementation `72c4276ad`,
MIOpen source `e1d8d1ff920` on the retained B07.11 accumulation repair `96e5a7dfc43`.
[Verified rollout metadata](verified-rollout.json). The initial candidate evidence
below remains historical; its “not deployed” statements no longer describe the
selected profiles.

Both `.104` and `.105` select
`/opt/bc250-stack/candidates/miopen-plan02-r4/profiles/daily`; explicit RCCL/D01
selects its `profiles/rccl-single`. Installed MIOpen SHA256:
`381eac05240e33acced9472fb7c3071afe8bc4f434407e67a29ce64423b935ac`.
Torch native a524, RCCL, runtime image and kernel/boot policy remain the recorded
payloads. The runtime descriptor keeps the base native revision and adds the
specific `miopen_precision_repair` source/hash/cache version; this is not a full
ROCm rebuild from e1d8d1ff920.

## What changed and why retained-cache tests matter

The gfx1013 HIP-direct FP16 forward kernel accumulates in FP32 before rounding
its result. The original half accumulator misses the existing reference bound;
this does not establish memory corruption or invalidate every historical run.
A fresh-cache installed-layout repair initially passed all consumers, but a copy
of the old user KDB still loaded the old arithmetic while the repaired DSO was
mapped. Cohort r2 was therefore rolled back and must remain inactive.

MIOpen keys binaries by filename and compile options, not embedded source bytes.
The solver now adds `-DBC250_DIRECT_FWD_FP32_ACCUM_CACHE=1` for gfx1013 FP16
forward only. It separates the repaired user/system-cache key without deleting
old entries. FP32/BF16/backward/other-device compile options retain their original
behavior. The isolated build reuses ccache and rebuilds one solver object plus
the DSO link in a preserved copy; the original build and DSO remain intact.

On each host, daily and RCCL tests independently clone the owned old cache,
retain all 12 old logical records, add six versioned FP16 entries and pass cold
and fresh-process warm precision checks. Warm runs log 12 binary hits, six with
the new marker, no binary saves and unchanged logical records. Original control
cache hashes remain unchanged. `.105` uses transferred owned `.104` controls,
not user application caches. SQLite can change physical bytes during bookkeeping;
logical record equality and no saves establish warm reuse, not byte-identical
SQLite storage. Failed probe r1 used raw-byte equality; r2 omitted quotes in the
log matcher. Both retained GPU phases passed math; the final r3 checks pass.

The original decisive result is 0.8427734375 versus CPU FP64 0.8424274250864983
(bound 0.0003062619962543249). Repaired warm-cache result is 0.84228515625,
inside that unchanged bound. No tolerance was relaxed.

## Qualification and reversible deployment

| Host / phase | Guarded runs | Scope |
| --- | ---: | --- |
| .104 retained-cache migration | 4 | 12 direct CPU-reference cases/run, cold/warm daily and RCCL |
| .104 inactive then public cohort | 10 + 10 | Each matrix has 98 top-level groups: direct FP16/FP32, Torch FP32/FP16/BF16 outputs and ALL first gradients, CK FP32, MIGraphX and retained ORT GPU graphs |
| .104 live rollback / final reactivation | 2 + 4 | Original pointers/D01 bytes/library and known old mismatch; repaired public direct/backward on both profiles |
| .105 retained-cache migration | 4 | Same copied-old-cache reference and fresh-process reuse checks |
| .105 inactive / public / final reactivation | 4 + 4 + 4 | Each matrix has 26 groups: direct references and ordinary Torch output/ALL-first-gradient configurations on daily/RCCL |
| .105 live rollback | 1 | Original pointers/D01 bytes/library and old daily mismatch restored |
| Both hosts public distributed regression | 1 per host | Eight groups/rank: four-dtype GPU sums, small/1MiB sums, three DDP/SGD steps against global CPU FP64 gradients/parameters |

All **49 final r4 guards** acknowledged; kernel logs unchanged. Eight actual
registration/identity refusal groups and two actual activation refusals pass on
.104; cache-unsafe cohort and failed migration evidence refuse before pointer
publication. 62 local tests, Black, critical Ruff+F401 and diff checks pass.
No full upstream CI, training-convergence, broad performance, higher-order
gradient or arbitrary-geometry/special-value claim. Historical CPU Python faults
remain distinct from fixed VM-heap corruption and D05/D21 issues.

Staging verifies SONAME/NEEDED/public C/C++ exports and normalized installed
RUNPATH. Recompiling changes one compiler-generated `__hip_cuid_<hex>` name;
only that anchored content-ID pattern is excluded from the export comparison,
with raw exports retained. Unchanged ROCm files use host-local hardlinks; the
changed MIOpen inode is detached before replacement. **Never modify shared
inodes in place** in a later update: detach every changed file or stage independent
copies. Old prefixes and fallbacks depend on their preserved contents.

.104 CK/SDK/ORT registrations and immutable package metadata are rebased as a
coherent versioned cohort; no identity check is weakened. Public optional
launchers select the cohort's private registries. Old SDK registrations remain
historical and preserved. CK stays explicit and FP32; B07.13 FP16 installation
is separate. `.105` has no optional SDK installation; this batch adds none.
Its old ordinary daily launcher remains, and public RCCL qualification uses the
existing D09 core with the new alias. New target descriptors enable cooperative
locking through hardlinks to its recorded migration-control lock. The frozen
underlying target payloads remain preserved.

Every activation/rollback holds EX reservation and stable runtime/D01 locks;
GPU tests hold SH locks and release them before transactions. Transaction journals
are `COHORT/transactions/{cache-safe-initial,cache-safe-rollback,cache-safe-final}`
on .104 and `{target-initial,target-rollback,target-final}` on .105. To return to
the pre-PLAN-02 stack on an idle board, use this helper with a **new unique label**:

```sh
doas python3 /home/nik/bc250work/tools/plan02/activate_miopen_plan02_cohort_on_host.py \
  --root /opt/bc250-stack/candidates/miopen-plan02-r4 --action rollback \
  --label operator-rollback-r1 \
  --from-transaction /opt/bc250-stack/candidates/miopen-plan02-r4/transactions/cache-safe-initial/before.json
```

Use a new unique lowercase label for each transaction. On .105, use
`transactions/target-initial/before.json`. This is a userspace selection rollback,
not a kernel rollback. Run corresponding reference checks afterward. Kernel r2
fallbacks, native controls, healthcheck package/data, old paths, user applications,
images, builds and caches were preserved. No user app inspection, foreign stop,
cleanup, reboot or download in this rollout.

GPU resources: four CPUs/cpuset0–3,10GiB RAM/12GiB RAM+swap,512PIDs,256MiB shm,
600s recovery/540s child/15s termination. Resource injection is test-only; ordinary
consumer defaults are preserved. .104 boot afcc7be4/r2 and .105 boot8c396bd1/r3
remain unchanged. Exact commands/PIDs/ACKs and retained failures are in the
metadata archives listed in verified-rollout.json; no tensor/checkpoint binaries
or private application data are exported. Final boards idle, no owned jobs,
containers or armed guards.

## Next action

PLAN-02 is complete for this scope. Proceed to **PLAN-03 / D26**: actual Trainer
classic FSDP eager GPT2 causal LM, tied parameters, shifted global token loss,
CPU ALL gradients/clipping/AdamW and exact restart under explicit FP32/BF16/FP16
policies. Checkpoint bitwise continuation across changed MIOpen arithmetic is
not promised; preserved old cohorts remain available for exact old environments.

---

# Initial candidate qualification (historical)

2026-11-08 batch, implementation19de82f88. **PLAN-02 remains in progress; no
ordinary native deployment or .105 change.** [Recorded results](verified-candidate.json).

Read-only audit confirms installed MIOpen3a751adc1497 vs retained repaired native
build797335e2b713. Reused B07.11 source repair/resources, no rebuild, no source
patch reapplied. Current installed-layout RUNPATH uses $ORIGIN/rocm_sysdeps/lib;
retained build candidate still has build-tree RUNPATH. It is not ready to copy
into the active prefix without independent installed-layout packaging.

Fresh direct public solver218 calls reproduce the exact original FP16 mismatch
in BOTH daily/RCCL profiles:0.8427734375 vs FP64 reference0.8424274250864983,
error0.0003460124 exceeds unchanged0.000306262 bound. Twelve cases/profile use
six group/filter/size fixtures and FP16/FP32, actual rounded operands, independent
CPU convolution, guards and immutable inputs/weights. Old runs are intentional
negative controls, not numerical passes at the strict bound.

Isolated candidate runs the same12 direct cases/profile at the original strict
bound, all pass. Separate ordinary Torch runs12 configurations/profile: FP32,
FP16,BF16 ×groups1/3 ×dilation1/2, weighted output and ALL input/weight/bias first
gradients against CPU FP64. FP16 bounds retain B07.10 forward.002/.003,
gradients.003/.006; FP32 3e-6/5e-5, BF16 forward.003/.01, gradients.01/.02 declared
before execution. All pass. Forward repair does not claim changed backward
arithmetic, universal geometry/exceptional values, convergence or speed.

Two setup guards fail before candidate GPU numerical work: r1 ordinary launcher
correctly refuses protected LD_LIBRARY_PATH override; r2 fresh child lacks
librocm_sysdeps_zstd.so.1. r3 uses a fresh child before Torch/HIP import with an
explicit private library+sysdeps path. No global environment relaxation or
exec-replacement after HIP initialization. Both failed logs retained.

Six accepted guards (two old negative controls, two direct candidate, two Torch
candidate), two setup failures, all eight acknowledged, kernel logs unchanged,
no leftover containers. 62 local tests, Black/critical Ruff+F401/py_compile/diff
pass. Resources4CPUs/cpuset0–3/10GiB RAM/12GiB RAM+swap/512PIDs/256MiBshm,
600s recovery/540s child, PLAN-01 reservation/SH runtime lock. Native .104 prefix,
runtime descriptors and SDK/CK/ORT registrations remain unchanged. No user app
inspection, foreign stop, reboot, download or native build. Historical Python CPU
fault remains separate from fixed VM heap/D05 exit/D21 optimizer loading.

Evidence /home/nik/bc250work/evidence/plan02/{final,candidate-final,backward-final,
verified-candidate}.json and controller logs; exact phase commands/result/exit/
kernel logs under WS/logs/plan02-*, ACKs /run/plan02-* copied into archive.
Direct initial controller20838exit1, candidate21414exit1, corrected21705exit0,
backward22359exit0. Final bootafcc7be4/r2. Local kernel/artifacts/plan02/
evidence-r3.tar.gz354657B SHAfe5f48b34620cb42fb47c10deb24601686ff9b300ba4bed5b7e6c41ca2de09c2.
Originals and candidate caches kept; archive contains no model/checkpoint data.

## Exact next action

Stage a NEW installed-layout candidate, preserve existing source/build/artifact,
normalize its RUNPATH to the installed layout using the recorded packaging tool,
verify SONAME/NEEDED/public ABI and rerun these references. Audit and qualify a
coherent native cohort: installed FP32 bc250-ck pins old MIOpen bytes; SDK pins
runtime descriptor hashes; ORT registry AND immutable manifest embed the SDK
registration. Do not merely relax pins, edit one hash or replace the DSO.
Qualify retained consumers/identity refusals and reversible cohort transitions
under the documented exclusive writer lock, then ordinary .104 activation and
live rollback; deploy .105 only after matched target qualification. Keep PLAN-02
unchecked until this deployment scope passes; D26 is not the next action yet.
