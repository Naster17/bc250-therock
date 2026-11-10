# D48: native SDK visibility fix built and candidate-qualified

Completed 2026-11-10 on .104. Native source commit
`143f1d878296176c738274202092bb247b7b771c` adds the exact `-1` terminating selector
before the profiler SDK's invalid-token diagnostic. Global CI diagnostics remain
enabled. The known HIP sentinel import failure is repaired in this **candidate**;
the active SDK and public D44 refusal remain unchanged. .105 was not accessed.

## Build and qualification

One modified C++ unit was compiled with the retained Clang24 recipe and ccache,
then relinked against the retained inputs in image1bb4ac0e. Original host source/
build mounts were read-only; object/library/link-dependency outputs were private.
RPATH was normalized to the qualified provider layout without changing code.
The successful r2 compile was a cache hit from r1's completed compilation; relink
and packaging completed in0.641s. This is an incremental retained-input rebuild,
not an independent clean build or performance comparison. Ccache statistics/data
were preserved, with normal new entries/hits and20GiB environment limit; no reset.

Candidate SHA256 `2a975370ba4eb1c2fe39e406ff36b175f554a49d0aa19febbd326f641fcb590b`,
17,575,272B. Source SHA256
`5aa899d0c9921077fb411f9d649e6735b2381a49c97a2fe4caa79db5bb03fcc0` matches the
committed/staged source. The dynamic symbol binding/type/visibility/defined contract
for1,450 symbols, DT_NEEDED and SONAME match the old library. This is the stated
binary contract check, not an assertion about every SDK API or ABI property.

Final r3 runs23 fresh processes across daily and RCCL profiles:

| Scope | Result |
| --- | --- |
| Candidate HIP/CUDA/ROCR `-1`, each profile | Six imports pass; zero devices, CUDA lazy init remains off, CPU profiling works |
| Candidate default, each alias0 and each alias`0,-1` | Fourteen GPU processes pass, retaining selections before the terminator |
| Unchanged SDK ordinary controls | Two GPU processes pass the same bounds |
| Unchanged SDK forced preload diagnostic | Arithmetic passes; two empty GPU traces reproduce the selection-method problem |
| CPU float64 references, profiled/unprofiled exact replay and pinned copies | 227 numerical checks including controls |
| Ordinary control/candidate native GPU captures | 32 traces with kernel/copy events, positive durations and runtime correlations |

Each GPU fixture uses fixed CPU-rounded FP32 inputs for32×48×24 matmul/sine,
first input/weight gradients versus CPU float64 (atol2e-6/rtol2e-4), two exact
profiled replays, and pinned H2D/D2H copies on a nondefault stream. Every candidate
GPU output/gradient digest exactly matches its profile's ordinary old-SDK control.
Actual loaded SDK path and SHA are bound in every process; no architecture override.
No NCCL collective, full-model, AMP, higher-order, compile, physical-counter or
universal stability/performance claim follows from these small cases.

HSA returns `GPU-XX`. The pinned HSA header defines this as unsupported UUID;
six planned UUID selections were therefore unavailable, with no synthetic identity
substituted. UUID selection remains unqualified; ordinal admission is covered.

## Preserved failures and corrected experiment setup

- Build r1 compiled the unit, but linker dependency output still targeted retained
  `link.d`. Read-only protection refused the write. R2 redirects this output too;
  three local build regressions cover output redirection, CI preservation and
  overwrite refusal. R1 result's old “running” field is stale: its controller
  exited1 and the full failed log is retained, not a running job.
- Test r1 stopped on an overly strict UUID fixture assumption after the ordinary
  control's math/profile checks. The fixture now recognizes the documented
  unsupported UUID marker. Numeric/trace bounds were unchanged.
- Test r2 candidate selected through LD_PRELOAD passed arithmetic but failed the
  nonempty-trace requirement. R3 proves the unchanged SDK also loses tracing when
  preloaded, while ordinary resolution works for both old and candidate libraries.
  Selection method is implicated; internal registration timing was not instrumented.
  Candidate selection now uses a private LD_LIBRARY_PATH prefix and SONAME aliases,
  preserving normal load order. Do not deploy via SDK preloading.

All old attempts, sources, traces and logs remain. Final r3 guards ACKed and
before/after kernel logs are byte-identical in every phase. Test wall168.950s;
minimum sampled host available14,205,579,264B. No native prefix change or reboot.

## Resources and artifacts

EX cooperative board reservation + SH runtime lock cover builds/tests. CPU build
has no GPU mappings,1CPU/4GiB/no swap allowance/256PIDs,600s deadline. GPU tests
use4CPU0–3/4GiB/no swap allowance/512PIDs/256MiBshm;100ms host3GiB/disk5GiB owned-stop
reserves,60s child/420s total and independent600s recovery. No cgroup peak or host
paging-I/O measurement is claimed. Only owned containers may be stopped. Native
control service/watchdog remain active; all owned jobs/containers/guards finished.
.104 boot2774d201/r2. No unrelated app, finetune-n7, fallback, source/build tree or
cache was deleted/replaced. Local source changed only in the committed SDK unit;
the original host source/build and active SDK stayed intact.

131 local tests, Black, critical Ruff+F401 and source/docs diff checks pass.
The exported patch retains unified-diff context whitespace and is checked as patch
data. Independent replay checks source/provider bindings, visibility, raw traces,
resources/ACK and actual old/new ELF contracts. Three altered source/provider/trace
archives are refused. Offline replay does not recompute CPU numerical references;
the source-bound worker performs those comparisons. No full upstream SDK suite.

[Verified results](verified-results.json), [binary contract](abi.json),
[negative controls](negative-controls.json), [candidate package](candidate-package.json).
Local metadata archive `kernel/artifacts/sdk-d48/104-evidence-r1.tar.gz`,1,345,232B,
SHA `ef15397c5b5c49ccfa6800709759cc154f2135fa54f936a3086865846db11a4d`.
Local candidate package `kernel/artifacts/sdk-d48/sdk-d48-candidate-r1.tar.gz`,
4,040,834B, SHA `98d26859f7d57fed98ec2c5e2aa09c8121c1b55fc12b1b5470a4745a736ccf3e`.
It contains the library/source delta/license/identity, requires the recorded native
dependency cohort, and is not a standalone SDK or complete release/SBOM package.

Host tools: `~/bc250work/tools/sdk-d48/`; builds:
`evidence/sdk-d48-build-r1/r2`; tests: `evidence/sdk-d48-tests-r1/r2/r3` (actual
separate directories with these suffixes). Controllers5337/5960/6807/8382/11032
finished; exact child/CID/guard/command identities are in the retained JSON/logs.
Host metadata `evidence/sdk-d48-evidence-r1.tar.gz`; candidate archive in tools.

```sh
python3 therock/build_tools/bc250/verify_sdk_d48_exports.py \
  kernel/artifacts/sdk-d48/104-evidence-r1.tar.gz \
  --old-sdk kernel/artifacts/sdk-d48/old-librocprofiler-sdk.so.1.4.2 \
  --new-sdk kernel/artifacts/sdk-d48/librocprofiler-sdk.so.1.4.2 \
  --output /tmp/sdk-d48-replayed.json
```

**Next D49:** inspect current prefix/provider identities and preservation baseline;
prepare a coherent versioned .104 activation with the old SDK/fallback retained,
EX runtime-update lock, accurate daily/RCCL/distributed metadata, ordinary public
launcher/profile checks and rollback qualification. Keep .105 unchanged. Decide
public `-1` admission only against the qualified installed SDK identity; legacy
cohorts must retain D44's refusal. PLAN-07 dependency/license/Gloo-overlay/build-version
closure remains open, four important packages remain. Historical Python CPU
IP0x3b3e6, KFD mm/queue and .105 clocksource causes are still unknown and distinct
from this specific SDK import failure and the fixed VM-heap/D05/D21 defects.
