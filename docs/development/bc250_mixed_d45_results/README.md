# PLAN-06 / D45: bounded mixed forward/backward and optimizer stability

Completed 2026-11-10 on .104. The predeclared PLAN-06 scope is complete; unknown
historical causes remain open. Next is PLAN-07 / D46 source reconstruction audit,
including the concrete profiler visibility-crash provenance lead below.

Six fresh owned Python processes alternated daily/RCCL profiles, seeds4500–4505.
Every process compared FP32 GPU outputs, loss and ALL input gradients against
CPU-float64 references for matmul32×24×16, convolution2×2×9×11 with3×2×3×3 weights
and bias, mathematical SDPA with non-square7/9 query/key lengths, and real FFT33.
GPU SDPA used the native math backend; CPU used explicit scaled matmul/softmax.
Two dense AdamW updates compared loss, gradients, ALL parameters and both moments
against independently updated CPU-double moments/parameters, and checked native
step counters. Identical FP32-rounded initial values; atol5e-5/rtol5e-4 fixed before
GPU results, unchanged. No AMP/compile/distributed collective/model/app claim.

All162 comparisons pass, including12 optimizer updates. Per-operation begin/pass
JSON is flushed; faulthandler emits a stack after45s if a process stalls. Six
distinct container IDs and child PIDs establish fresh executions. Each phase's
kernel log is byte-identical before/after; no log filtering or clearing.

| Profile / seed | Guarded child s | GPU allocated peak MiB | Sampled cgroup peak MiB | Min sampled host available GiB |
| --- | ---: | ---: | ---: | ---: |
| daily /4500 | 11.936 | 64.019 | 889.418 | 12.940 |
| RCCL /4501 | 8.507 | 64.019 | 706.824 | 13.081 |
| daily /4502 | 8.103 | 64.019 | 700.070 | 13.047 |
| RCCL /4503 | 8.509 | 64.019 | 704.340 | 13.018 |
| daily /4504 | 8.083 | 64.019 | 696.281 | 13.056 |
| RCCL /4505 | 8.531 | 64.019 | 705.112 | 13.101 |

These are test costs, not controlled performance comparisons. Cgroup values are
the largest observed `memory.peak` reads while the container existed, not an
after-exit retained accounting snapshot. Sampled container swap/OOM-kill counters
are zero. No host paging-I/O measurement or absence-of-all-paging claim.

EX board reservation + SH runtime lock covered tests/export. Four CPU0–3,4GiB
hard RAM/4GiB combined (no container swap allowance),512PIDs/256MiB shm;100ms
samples enforced3GiB host/5GiB disk reserves. Child60s, independent recovery600s.
Only owned-label containers may be stopped. A foreign workload invalidates the
test; after owned work has exited, postflight failure is recorded and the guard
ACKed so it cannot reboot an unrelated workload. This foreign-arrival branch was
not exercised here. Final preflight idle except allowed native board service,
watchdog active. Controller19038 exited, recovery ACKed; no running test jobs.

No native source/library/default/profile/registry/prefix/image/kernel/boot change,
rebuild, package installation, user application access, foreign stop or reboot.
.104 boot2774d201/r2 unchanged. .105 unaccessed/unchanged; its prior foreign Docker
ownership must be rechecked before future GPU work. D44 safeguard remains active
on .104 only. Sources, caches, histories, checkpoints and all old evidence retained.

## Historical issues: what the new evidence establishes

- **Python CPU IP0x3b3e6:** still unknown, original core/stack absent. No recurrence
  in these six small processes; this cannot establish a root-cause repair.
- **KFD resume warning:** B07.7 retained a failure after exec-after-HIP; the public
  package switched to same-process runpy and passed its final gates. Kernel queue/
  mm cause remains untraced; D45 does not exercise exec-after-HIP or claim repair.
- **.105 clocksource remote CPU read timeout:** unchanged open issue. Source uses
  `pr_info_once` and retries; quiet logs cannot prove no recurrence. D45 did not
  test .105 or change TSC/kernel settings.
- **D44 visibility import crash:** exact input already refused by .104 launchers.
  Source inspection now identifies the warning-to-fatal mechanism:
  `logging.hpp:76–81` turns `ROCP_CI_LOG(WARNING)` into FATAL under `ROCPROFILER_CI`.
  Retained SDK CMake has `ROCPROFILER_BUILD_CI=ON`; ninja defines
  `-DROCPROFILER_CI=1`. However retained build DSO SHA `bcbb20f9...` differs from
  active SDK SHA `a6dc1c88...` (17,575,368B). Do not attribute that exact build to
  the active binary without resolving packaging/source provenance. No native
  parser repair was implemented or installed. This lead is distinct from the
  old unexplained CPU fault and already fixed VM-heap corruption.

## Evidence and next action

[Verified records](verified-results.json), [mutated-archive refusals](negative-controls.json).
Black, critical Ruff+F401, syntax/diff checks pass. Independent archive replay checks
source digests, ordered coverage, fixed bounds, stage records, freshness, resources,
logs and ACK. It does not recompute the mathematical references offline; the
source-bound worker executes those comparisons. Three private source/ACK/worker
alterations are refused. No full upstream suite or new120-unit-test run is claimed.

Owned tools: `~/bc250work/tools/mixed-d45/`; results:
`~/bc250work/evidence/mixed-d45-r1/`; controller:
`~/bc250work/evidence/mixed-d45-controller.log`.
Final archive local `kernel/artifacts/distributed-d45/104-evidence-r1.tar.gz`,
253,974B, SHA `f37a05accdf5e6e111854f2b9e06b2ccfb76fd0d96a642ef0d5321ab60baea47`.
Same archive retained as host `evidence/mixed-d45-evidence-r1.tar.gz`.

```sh
python3 therock/build_tools/bc250/verify_mixed_d45_export.py \
  kernel/artifacts/distributed-d45/104-evidence-r1.tar.gz \
  --output /tmp/d45-replayed.json
```

**Next PLAN-07 / D46:** CPU-only inventory of source-to-installed-artifact mapping;
audit the historical SDPA three-file reconstruction gap against actual patched
Torch Git, configs and qualified sources. Resolve active-versus-retained SDK
identity/configuration, classify packaging changes versus source drift, then
prepare the smallest native visibility-parser repair/build plan if justified.
Current .104 free space is only~98MiB above the5GiB reserve; no native build should
start without an explicit incremental/cache/link/staging/rollback storage budget.
Preserve the working refusal and do not disable CI checks globally to suppress a
specific parser failure. Four important packages remain (PLAN07–10), optional
scientific/feature expansion separate. Unknown historical faults remain listed
through reconstruction/release rather than being declared fixed by a quiet test.
