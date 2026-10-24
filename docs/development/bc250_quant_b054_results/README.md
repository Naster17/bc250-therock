# B05.4: strict W8 validation overhead

The explicitly imported W8 helper now reads one combined validity scalar on a
valid call, retaining the same scale/input checks and scale-first errors. Packing,
native forward/backward wrappers, limits, dtype policy and public signature are
unchanged. This updates `WS/helpers/bc250_quant_b051.py`, not an image/site-package
or native Torch/ROCm runtime. [Verified source/results](verification.json).
Tooling/helper revision:`917b2823a6e5cddbb9b93a4a4fdc44b9fd64c2ce`.

## Measurement and implementation

The initial component profile finds~0.14–0.17ms in strict scale/input checks
versus~0.24–0.39ms whole forward in three FP32 fixtures. The trace shows two
`aten::_local_scalar_dense` reads/`hipStreamSynchronize` calls. These pre-reboot
absolute values are diagnostic, not the speedup baseline for post-reboot runs.

The first candidate reads a two-flag tensor with `tolist()`. It preserves results
and error priorities but is slower:forward ratios~0.82–0.91 versus the baseline.
Its trace uses `hipMemcpyWithStream`; an overly specific trace assertion expecting
one explicit stream-sync event fails because none is recorded for that blocking
copy path.89 checks finish before this assertion. The candidate is rejected and
never promoted; source, raw results and trace remain available. No numerical or
error assertion is relaxed.

The accepted candidate computes finite-positive scale validity and finite-input
validity on GPU, then reads their scalar conjunction once. On failure it reads
the scale flag to select the original scale error before the input error. There
is no cached/skipped data validation. Both error messages and priority are
unchanged; invalid calls may still need a second scalar read. The normal-path
trace shows two native forward/backward kernels and scalar reads/explicit stream
synchronizations2→1. Packing and `_Linear` ASTs match the original exactly.

## Paired timing and correctness

Seven short pairs initially show small/noisy improvements. A longer controlled
comparison uses15 alternating pairs of30 eager calls, one Torch CPU thread,
process affinity to four allowed CPUs, warmups and synchronization around each
block. Both modules share the same integer weight/scales and input data. Profiler
collection is separate from latency samples. Raw pair distributions are retained.

| FP32 M,N,K | Daily forward old/new latency | RCCL forward | Daily forward+backward | RCCL forward+backward |
| --- | ---: | ---: | ---: | ---: |
|8,128,256 |1.075× |1.077× |1.029× |1.030× |
|32,256,256 |1.077× |1.080× |1.033× |1.029× |
|8,512,1024 |1.145× |1.145× |1.080× |1.075× |

All15 forward pairs favor the candidate in each cohort/geometry; forward+backward
wins12–15 pairs. These are bounded helper wall-latency ratios, not model, kernel
throughput or universal speedups. They do not establish beating a resident dense
baseline. Native math, persistent packed-weight storage and backward temporaries
are unchanged; no total-memory improvement is claimed.

**333 current-boot candidate/comparison checks +142 post-update checks**, across
eight accepted gates:

-90 comparisons per cohort:bitwise old/new packing, outputs, input/bias first
  gradients for three dtypes/layouts; independent CPU references;60 scale/input
  value combinations and12 combined scale/input/bias error-priority cases; six
  paired timing fixtures and normal-path native/synchronization trace checks.
- Retained W8 recipe71 per cohort includes36 CPU pack/output/input/bias-gradient
  composites, byte/tie/zero-row/layout guards, private stream, finite differences,
  owned adapter gradients, AMP/higher-gradient policy and25 refusal cases.
- Daily5/opt-in6 liveness checks pass; post-update public launches repeat recipe71
  per cohort using the actual helper import, not the candidate directory.
-442 protected hashes/four AOT DBs/links retained, with the helper update and
  original backup deliberately accounted for.29 local BC250 tests and selected
  helper Ruff checks/format pass; no full upstream CI claim.

The user restarted `.104` while the longer test was being staged; SSH temporarily
failed. The longer job had not started and manifests remained unchanged. Runs
before boot`c6a50cfa-a368-4ba5-ab28-66250b4f1f1f` are kept separately. The accepted
long comparisons/qualification/post-update gates all use that boot. Eight live
guard states ACK and are archived. Three prior-boot guards, including the rejected
experiment, have ACK log evidence; their `/run` states were cleared by reboot.
Kernel logs match per gate in both boots and through the final current-boot batch;
no cross-boot kernel-log identity is claimed. No agent reboot/kernel change.

## Use, preservation and recovery

The helper remains explicit/not installed; no application or default Torch model
routing changes. Existing imports/API are unchanged. Native weight-only INT8
backward remains unsupported; the helper still supplies its own first input
backward. Packed INT4/NF4/full QLoRA, higher derivatives, graphs/compiled quantized
calls and broad model/quality/stability claims are not supplied. Historical Python
CPU faults remain distinct/unresolved; this is not a VM-heap/CSR/hostcall fix.

Daily native B05.2/runtimeSHA561f014e and opt-in B05.3/profileSHA446f345e,
images/ROCm/tuning/math/SDMA0/heap-off defaults, source/build/cache/boot fallbacks
and `.105` remain unchanged. No native build, ccache operation, source reset,
package installation or user-application inspection/modification/test occurs.

Promotion uses the daily exclusive update lock and an atomic helper replacement;
owner/permissions are preserved. Old helper SHA54c983733d66213070436a2ec75775fa96383f5053b5ad788d17c2b0ae75e6da
is pinned in `WS/recovery/quant-b054-before`; new SHA6832ff9e58104236d877a455fdfe90f11b3803198732497d6dd4ac77242fd319.
Identity-guarded rollback (not exercised):

```sh
WS=/home/nik/bc250work/rocm-10.2-gfx1013
doas python3 "$WS/helpers/update_dev_prefix.py" -- \
  python3 "$WS/helpers/deploy_quant_b054_on_host.py" --rollback
```

Profiling/comparison/ordinary tests use ordinary launchers and shared locks;
do not wrap them in the exclusive updater. Collect the baseline profile before
promotion (its scalar-read assertion intentionally describes the old helper).
The comparator always loads explicit preserved baseline/candidate directories.
Single-use label helpers preserve failed trials. Main reports:
`WS/logs/quant-b054-qualification-r1/final.json` and
`WS/logs/quant-b054-ordinary.json`. Local `kernel/artifacts/quant-b054` preserves
626,696-byte archiveSHA256`75a22a82c56d551f55d878d6ec67edd8241a0f2c66b9b9ac3ae8aa4969609418`,
104 raw logs/2,968,279bytes, eight tools, baseline/candidate/rejected sources,
measurements/traces, recovery and current guard records. Eight tool hashes match
local source. No own jobs/containers/armed guard remain.

**Next B05.5:** implement an explicit owned frozen-W8 `nn.Module` conversion and
versioned state/checkpoint roundtrip, qualifying outputs/input/bias/adapter first
gradients with independent references. Preserve strict validation, FP32 scales,
limits and opt-in behavior; no full-model QLoRA/quality/speed claim or user app use.
