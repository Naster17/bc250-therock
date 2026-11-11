# PLAN-07 / D55: full isolated clean Torch rebuild

Completed 2026-11-11 on .104. Harness 3d494ef03, deadline/diagnostics fix
d6daedcf1, ccache counter fix f2690ce03.
[Receipts](artifacts.json), [replay summary](verified-results.json).
**PLAN-07 remains OPEN**: both profiles now rebuild fully from preserved
inputs, but generated-file/license provenance closure is still outstanding.
.105 was not accessed. No GPU workload ran; the boards stayed reserved-idle
under the cyan-skillfish client throughout.

## Method

`run_full_build_d55_on_host.py` runs as root on bc250 under the same board
reservation, fail-closed preflight, 12 GiB available-memory / 20 GiB disk
admission, 3 GiB host / 5 GiB disk running floors, owned-container-only stop
and per-attempt scratch removal as D54. The deadline is 7200 s (attempt 1
used 3600 s; see below).

The container is the preserved original builder image
`28ad36e62398875f0ad77112b1d0f2b1cd0b3b55560d9cc38c6908cada8caa06`,
`--rm --pull=never --network=none`, 4 CPUs (0-3), 10 GiB memory+swap,
2048 PIDs, 2 GiB tmpfs `/tmp`. Unlike D54 there is **no `--read-only`
rootfs**: the editable install writes into the image venv. Every preserved
input is still mounted read-only: the retained ROCm build prefix
`build/dist/rocm` (SDK a6dc1c88 build input, distinct from the active D49
SDK 2a975370), `rccl-b034/install-r1`, the ccache data
(`CCACHE_READONLY=1`, `CCACHE_NOSTATS=1`, zero cache growth), the D52
recipes, the toolchain and the offline six input. Only the freshly extracted
kit torch tree, its ccache config scratch and the evidence directory are
writable.

Per attempt the D52 source kit
(SHA `6867ed719c201cf7bc368800aa64c5e16a2e3a4c3c03485ad502684e2feb5723`) and
six 1.11.0
(SHA `70e8a77beed4562e7f14fe23a786b54f6296e34344c23bc42f07b15018ff98e9`) are
hash-verified before extraction. `full_build_d55.sh` sources the archived
`build-fresh-<profile>` recipe environment up to the pip command (boundary
regression-tested against `sed`), then runs the full
`python -m pip install -e . -v --no-build-isolation --no-deps` with
`PIP_NO_INDEX=1` and `offline_inputs_d54.cmake` forwarding
`PYTHON_SIX_SOURCE_DIR`. After a successful install it requires a gfx1013
offload bundle in `BatchLinearAlgebraEig.hip.o` via
`llvm-objdump --offloading`, records that object's SHA plus all
`torch/lib/*.so` / `torch/_C*.so` hashes, and writes `full-build-passed`.
The EXIT trap retains CMake cache/configure log, cgroup metrics, the 22-file
HIP hash list and the raw ccache log with hit/miss/invocation counters.

## Results

| Attempt | Status | Wall | cgroup memory peak |
| --- | --- | --- | --- |
| daily-full-1 | failed (deadline during wheel packaging) | 3602.399 s | 9,966,964,736 B |
| **daily-full-2** | **passed** | 1761.276 s | 9,418,891,264 B |
| **rccl-single-full-1** | **passed** | 1848.435 s | 9,578,196,992 B |

Attempt 1 completed all 3282 ninja steps (`libtorch_python.so` linked) and
was killed by the 3600 s deadline while pip collected license files for the
editable wheel; no compile or link failure occurred. The deadline was raised
to 7200 s and attempt 2 passed end-to-end, installing
`torch-2.13.0a0+gfx1013` offline in both profiles. All cgroup event counters
were zero (no OOM, no throttling); host minima stayed above 7.7 GiB
available memory and 56.7 GiB free disk.

Identity comparison (both profiles):

- **22/22 HIP translations byte-exact** against the D54 clean probes and
  against the preserved 20261110 source-audit retained variants
  (daily = b052, rccl-single = b053).
- `BatchLinearAlgebraEig.hip.o` SHA
  `6c3e6dc0e9731d39b09eeedf4137b2a9960a6a050e7951d1d19e9e078673d56b`,
  identical to D54 and across profiles; gfx1013 offload bundles present.
- 10 shared libraries hashed per profile. 7/10 are byte-identical across
  profiles; `libtorch_cpu.so`, `libtorch_hip.so` and `libtorch_python.so`
  differ — exactly the NCCL-affected linkage set (`USE_NCCL=1`). Recorded,
  not equated.

Comparison against the deployed D49 cohort (workspace `release_development`
matrix): the rebuilt daily `libtorch_hip.so` is byte-identical to the
deployed daily runtime (`2f2761cd…`), while daily `libtorch_cpu.so`
(`2ea92c61…` vs deployed `9586351b…`) and both rccl-single libraries differ
from their deployed counterparts. The admitted D55 references are the D54
probes and retained HIP variants (all matched); explaining the remaining
library-level deltas (embedded build metadata / generated files) belongs to
the PLAN-07 generated-file closure. No whole-runtime byte-equivalence is
claimed.

Preserved ccache reuse (read-only, zero growth):

| Profile | Hits | Misses | Invocations |
| --- | --- | --- | --- |
| daily | 3023 | 241 | 3291 |
| rccl-single | 3015 | 251 | 3293 |

The high hit rate shows the kit sources and preserved compiler fingerprints
reproduce the original build's cache keys. Wall times are admission costs
only, not performance claims.

## ccache counter finding

ccache 4.9.1 in read-only/no-stats mode omits the `Result:` summary lines
the first wrapper grepped for, so attempt 1/2 count files read 0 despite a
populated log. Live probes established the actual markers: hits log
`Succeeded getting cached result`, misses log `Read-only mode; running real
compiler`. The wrapper now counts those and retains the raw log plus
`ccache --show-config` output. Daily counts above were derived from the
retained raw log; rccl-single counts come from the corrected counters.
The running attempt-2 script was never overwritten in place: the fix was
staged as `.new` and renamed into place only after the attempt finished.

## Evidence and verification

30 files transferred to `kernel/artifacts/source-d55/` (both passed
attempts), every file SHA-verified against the host; no mismatches.
Attempt-1 evidence remains on the host at
`evidence/source-d55/bc250-d55-daily-full-1/` as the deadline-failure
receipt. Local checks: 170 `bc250*_test.py` tests OK, black unchanged on the
D55 Python files, ruff unavailable on the workstation (py_compile used).

## Next

PLAN-07 generated-file/license provenance closure from the now-proven
rebuild path (wheel license collection observed working in `probe.log`),
then PLAN-08 release assembly. No destructive source reset, no ccache reset,
.105 untouched.