# PLAN-06 / D44: prelaunch refusal of the retained HIP visibility crash

Completed 2026-11-10 on .104. **PLAN-06 remains open** for the declared D45 mixed
stability pass. .105 was not accessed or changed in this batch.

The public .104 launchers now reject exactly `HIP_VISIBLE_DEVICES=-1` before
elevation or container creation. This covers the inherited environment, `--env`,
environment files, and privileged context revalidation. Last explicit override
still wins. Other HIP/CUDA/ROCR filters retain their existing semantics; their
acceptance by the frontend does not qualify their native execution.

## Demonstrated failure and boundary

The original D12 CPU diagnostic had retained exit139 with this input. The new
comparison through the unchanged public launcher reproduced exit139 in 0.945s:
rocprofiler SDK `agent.cpp:216` reports that `-1` is not recognized, Python reports
an abort followed by a segmentation fault during `torch` import, and the kernel
records a libc general protection fault. The corresponding source is
`rocm-systems/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/agent.cpp:196–222`.
That parser uses `ROCP_CI_LOG(WARNING)` for an unrecognized token. This identifies
the first fatal diagnostic in this reproduction; it does not prove the secondary
signal-handler/teardown fault's root cause. No native SDK/ROCr repair was made.
The refusal text retains the historical “ROCr import crash” description, while
this report distinguishes the observed SDK diagnostic from ROCr itself.

Do not generalize a rejection rule from ordinary CUDA conventions. In particular,
the retained ROCr filter documentation explicitly defines empty and `-1` filters;
CLR also has ordinal/UUID/termination behavior. D44 therefore does not replace
these parsers or blacklist every negative-looking token.

For ordinary CPU-only Python work, unset the problematic HIP variable and avoid
CUDA initialization. This does **not** remove container GPU device access. For the
board's GPU, use default selection or explicit `HIP_VISIBLE_DEVICES=0`.

The historical Python execution fault at IP0x3b3e6 remains cause unknown: one
retained event, no original core/stack, followed historically by1,920 passing
bounded solver/FFT/sparse/GEMM references. It is distinct from this reproducible
import failure, the fixed VM-heap/CSR corruption, D05 agent exit and D21 optimizer
restore. KFD uncertainty is not repaired by this launcher change. .105's D42
clocksource remote-read timeout remains open; `pr_info_once` and later quiet logs
cannot establish absence of recurrence. Kernels and clocksource were unchanged.

## Tests, deployment and evidence

The candidate refusal exited2 with zero observed containers. CPU import without
visibility flags stayed CUDA-uninitialized. Three fresh GPU checks—daily default,
daily explicit0 and RCCL profile explicit0—passed CPU-float64 loss and ALL input/
weight gradient comparisons for a17×13×7 FP32 matrix fixture, plus CPU-first
initialization checks:12 checks total, fixed atol/rtol1e-5. Native gfx1013:xnack-
was verified. These are functional checks, not distributed communication,
long-run stability, convergence or performance measurements.

Candidate phase kernel logs were byte-identical before/after; the baseline crash's
new libc fault is preserved separately, not filtered. Lowest sampled host memory
available was14,505,086,976B. Each container was capped at4CPU0–3/4GiB RAM with
4GiB combined memory+swap (no container swap allocation),512PIDs and256MiB shm.
Samples every100ms enforced3GiB host and5GiB disk reserves; child deadline60s,
independent recovery guard600s. No cgroup peak/paging-I/O or performance claim.
EX cooperative board reservation + SH runtime lock covered comparison/exports.
Owned containers exited, native-control-only preflight passed and the guard ACKed.
No development reboot or foreign workload stop occurred.

Under EX board reservation + EX runtime lock, both .104 copies were backed up and
replaced atomically:

- `/usr/local/bin/bc250-torch`
- `/opt/bc250-stack/tools/distributed-d09/bc250_torch.py`

Old SHA256 `c5167f171eefb6df728eea2c1063f2af1ca1317a03ea8d69ab67f36a700f6106`;
installed SHA256 `a7b3a5b8db0bf178356271258da3b0dd6fb816461192e817a92a1c40d908a5cb`.
Backups: `~/bc250work/evidence/visibility-d44-install-r1/{0,1}-bc250_torch.py`.
Actual unprivileged public `bc250-torch` and `bc250-train` refusals exit2;
`bc250-distributed` refuses through the shared validation with its existing
ValueError traceback/exit1. No doas prompt or worker was reached. Rollback-on-failed
installation checks is implemented; no rollback was needed or exercised here.
Native Torch/ROCm/MIOpen/RCCL/image/default profiles and distributed registry
were unchanged. .104 remains kernel7.1.5-r2, boot2774d201. No rebuild was necessary.

Local120 unit tests, Black, critical Ruff+F401, syntax and diff checks pass.
The independent archive reader passes; three mutated source/guard/worker archives
are refused. See [verified results](verified-results.json) and
[negative controls](negative-controls.json).

Preserved preliminary r1: the extensionless old launcher was not recognized by
the test adapter's automatic Python loader. It exited1 before importing the
launcher. This was a test-harness failure, not a runtime reproduction. The adapter
now uses an explicit SourceFileLoader. R1 candidate checks passed; authoritative
comparison is r2, which reproduced the native crash. Both attempts and adapter
sources are retained. Do not count r1's top-level “passed” as an old-runtime test;
only r2 is accepted by the independent reader.

Owned tools: `~/bc250work/tools/visibility-d44/`; evidence roots
`evidence/visibility-d44-r1`, `visibility-d44-r2`, `visibility-d44-install-r1`.
Controllers23508/26957 finished, guard ACKed, no running jobs. Final preflight
idle except the allowed native cyan-skillfish service/watchdog. Final archive:

`kernel/artifacts/distributed-d44/104-evidence-r1.tar.gz`,492,032B, SHA256
`99b1b95f3df6e6da1700099f83f46868d3cbf6f83f52a312b6ad5f881ee3e214`.
Same archive retained under .104 `~/bc250work/evidence/`.

```sh
python3 therock/build_tools/bc250/verify_visibility_d44_export.py \
  kernel/artifacts/distributed-d44/104-evidence-r1.tar.gz \
  --output /tmp/d44-replayed.json
```

## Next: PLAN-06 / D45, predeclared bounded mixed-workload pass

Planned, not executed: six fresh owned Python processes on idle .104, alternating
daily and RCCL profiles, each running small FP32 dense matmul, convolution,
mathematical SDPA and real FFT output/first-backward checks against independent
CPU-float64 references, plus two dense AdamW updates/state checks. Fix inputs and
atol5e-5/rtol5e-4 before execution; justify any operator-specific bound before GPU
results, never relax a failed bound afterward. Record per-operation stage/results,
allocator/resource samples, Python faulthandler and exact kernel logs. No user
application, model download, compiler build or deliberate OOM/fault injection.

Use EX board/SH runtime, existing independent600s guard and per-child60s deadlines,
4CPU/4GiB no container swap/512PIDs/256MiB shm and3GiB host/5GiB disk reserves.
Stop only owned work on failure, preserve the failing process/log and do not
automatically extend the run. ACK after owned work exits; if a foreign workload
appears, fail the exclusivity gate and leave it untouched. Scope remains one board,
two profiles and these small operations. A quiet pass cannot close the unknown
historical Python/KFD/clocksource causes. .105 ownership must be rechecked before
any separately authorized tests there; no .105 test is required for this pass.
