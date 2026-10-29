# B07.6: native Search/PerfDb and public replay

Completed2026-10-29 on .104, tools `1eac70aa1`. Existing private plugin/source
`3616b2f9f09`, installed libraries `8389197c1`, Torch `a524980` unchanged.
Only a probe was built. No MIOpen source repair or tuning override was necessary.

## Existing machinery and genuine records

`ConvHipImplicitGemmGroupFwdXdlops::Search` already calls `GenericSearch`.
`PerfConfigBaseCK` serializes the selected kernel ID; `FindSolutionImpl` validates
loaded IDs before preparing a solution. Public CompileSolution disables enforced
search while retaining PerfDb lookup. These existing paths cover the required
native gfx1013 private-plugin experiment without a new environment knob or an
architecture-wide heuristic.

The public `miopenFindConvolutionForwardAlgorithm` searches only solver137,
with `MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemmGroupFwdXdlops`,
`MIOPEN_FIND_MODE=NORMAL`, `MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE`,
`MIOPEN_TUNING_ITERATIONS=5` and `MIOPEN_TUNING_TIME_MS_MAX=1000`.
Only two retained packed NCHW FP32 fixtures are searched: batch2,32×34,
C/K16 per group, stride/dilation1; groups1/filter3×3 and groups3/filter1×1.
Private user DB/cache directories are distinct for daily and rccl-single profiles.

Each profile generates two genuine PerfDb and two FindDb records. Public solution
queries confirm solver137; selected outputs are checked against CPU double loops.
Both profiles select block256 Default for3×3. The1×1 search selects block256
Filter1x1Stride1Pad0 in daily and Filter1x1Pad0 in RCCL; these near-performing
variants need not produce identical tuning choices. Records are not manually
selected, rewritten to a desired kernel or copied into the active native prefix.

## Fresh-process public replay

Each profile runs two fresh processes with opposite preparation/ABBA or BAAB
orders. Both paths enter public `miopenConvolutionForwardImmediate`: tuned137
versus retained HIP-direct218. Each fixture uses four warmups/path followed by
three four-window cycles,32 calls/window. No cached-only entry is substituted.
HIP-event stream intervals include full layout transformation, public dispatch,
host feeding and possible idle; synchronized wall intervals are retained.
These are not isolated kernel timings or whole-model measurements.

| Groups | Filter | Tuned public DL μs | Public HIP-direct μs | HIP-direct/DL, all four processes |
| ---: | --- | ---: | ---: | ---: |
| 1 | 3×3 | 62.42–62.79 | 68.15–68.32 | 1.085–1.092× |
| 3 | 1×1 | 37.93–38.36 | 42.94–43.06 | 1.119–1.135× |

The two fixture advantages survive public dispatch. No generalized shape,
convergence, training-speed or automatic Torch solver-selection claim follows.
The four losing B07.5 fixtures are not silently promoted to gains.

Separate verbose, untimed executions prove that fresh CompileSolution loads the
recorded IDs from PerfDb, performs no search, and runs actual selected DL kernels
plus HIP-direct `MIOpenConvUni`. All original text DB hashes remain unchanged
through normal replay. A deliberately malformed **copy** replaces each kernel ID
with an invalid ID: validation rejects it, the existing block8 Default fallback
runs correctly, no search occurs, and the genuine DB stays untouched. This tests
invalid-ID fallback, not arbitrary corrupt-file robustness or cross-version ABI.

Search wall times include first-use preparation: about1323–1330ms for3×3 and
32.6–32.8ms for1×1 in these runs. Private user caches are isolated, but installed
system databases, libraries and OS caches remain; this is not machine-cold startup.
Native search-reported profiling times are retained separately from warm public
replay intervals and must not be substituted for them.

## Evidence, limits and preservation

**148 CPU-reference checkpoints**, unchanged N2.18 tolerance
`2e-5 + 5e-5*abs(reference)`, max absolute error2.35537e-6; immutable inputs and
256-byte input/output/workspace guards pass.3172 explicit Immediate calls include
warm/replay/diagnostic/post-search calls; native search's internal evaluations
are excluded from that count. Search checks its selected result; numerical
coverage of every admitted ID remains the separate B07.3/B07.5 evidence.
Private7+7/core7/CPU ONNX4/daily5/RCCL6 = **36 acceptance checks**, six ACK600s
guards. Exact native Torch backward/retained consumers pass; no plugin backward
support added.29 local tests, critical Ruff/Black/clang-format/Python compilation/
shell/diff checks pass; full upstream CI not run. No failed GPU batch, source
repair or tolerance/reference relaxation.

Probe-only build: fourCPU/OMP4/jobs1/RAM10GiB/mem+swap12GiB, existing ccache20GiB
without reset, read-only sources/inputs/prefix, no build GPU/network,600s bound.
3701 protected hashes/35 links/four AOT DBs and136 workstation backing files match.
Same boot `e327c5de-46b8-4e05-94ad-d999a12a06e7`, kernel7.1.5-bc250rocm-r2;
kernel logs unchanged. No owned jobs/containers/armed guards at cleanup. Ordinary
free12,898,476,032→12,890,648,576B; nothing deleted. No .105/user-app access or
active-prefix/profile/default change. Historical unresolved Python CPU fault
remains distinct from fixed VM-heap corruption.

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`:

```sh
python3 helpers/build_ck_plugin_b076_on_host.py --attempt r1
python3 helpers/qualify_ck_plugin_b076_on_host.py --attempt r1 --build-attempt r1
python3 helpers/export_ck_plugin_b076_on_host.py
```

These accepted destinations already exist: preserve them. New attempts need new
versioned inputs/build/qualification/export destinations. Shared lock, resource
preflight and independent recovery guard remain required; active-prefix changes
require the separate exclusive update procedure. BuildPID30906/qualificationPID31175/
exportPID32467 exited0. `WS/logs/ck-plugin-b076-qualification-r1/final.json` and
`WS/logs/ck-plugin-b076-r1-controller.log`; fixtures/build/export at
`WS/tmp/ck-plugin-b076/{inputs-r1,build-r1,export-r1}`. Original and malformed
DBs, separate stdout/stderr/environment files and guards are retained in logs.
Local evidence `kernel/artifacts/ck-plugin-b076/evidence.tar.gz`: 1,143,590B,
SHA256 `13d5aabfaf5dcf053af67f8d3f7c62514a09b90104798e8c4d9cf4c6dffaa2e6`,
132 raw files/5,263,691B; sources/probe/DB hashes verified.

B07.7: prepare an isolated relocatable opt-in DL plugin artifact with exact native MIOpen/ABI/source/license identity and genuine example tuning records; qualify identity refusals and two build-free extractions (including spaces) before any installation. Keep installed defaults/.105/user applications unchanged; do not broaden production admission.
