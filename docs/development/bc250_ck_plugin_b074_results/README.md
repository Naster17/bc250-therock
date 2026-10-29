# B07.4: private DL versus retained HIP-direct

Completed 2026-10-29 on .104. Tools `e11969041`; private libraries source
`3616b2f9f09`, installed libraries `8389197c1`, Torch `a524980` unchanged.
No numerical code or production admission changed. **The public DL default loses
on every recorded fixture. No installation or default change is justified.**

## Method and attribution

Both paths use the same public `miopenConvolutionForwardImmediate` API, handle,
nonblocking stream, descriptors, immutable non-dyadic FP32 inputs and guarded
buffers. Forced IDs are **137**, `ConvHipImplicitGemmGroupFwdXdlops`, and **218**,
`ConvHipDirectFwd`; there is no automatic solver choice in the comparison.
An independent verbose run proves actual native DL `Default` block8
`DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<8,16,4,2,Default,1>` plus
NCHW layout transposes versus HIP-direct `MIOpenConvUni`. The historical solver
name does not establish XDLops instructions. Verbose results are excluded from
reported timing aggregates.

Six overlapping admitted packed NCHW shapes, batch2, stride/dilation1 and symmetric
padding, run in daily and rccl-single profiles. Each has two fresh subprocesses
with distinct private user caches/databases and opposite solver preparation,
first-execution and ABBA/BAAB timing orders. Four warmups/solver precede three
four-window cycles,32 calls/window. Reported values are HIP-event stream intervals
per call: they include the complete public route, layout transforms, host launch
feeding and possible stream idle, **not isolated kernel-only compute time**.
Synchronized wall intervals are retained separately. Inputs, output and required
scratch are preallocated before warm timing; CPU checking happens outside windows.
No hardware-clock normalization or whole-model timing claim.

| Input H×W | Groups | Filter | C/K per group | DL μs (first daily pass) | HIP-direct μs | HIP-direct speed advantage, all four timed runs |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 9×11 | 1 | 3×3 | 4/8 | 28.47 | 21.13 | 1.35–1.37× |
| 9×11 | 3 | 1×1 | 4/8 | 22.47 | 15.69 | 1.43–1.46× |
| 9×11 | 3 | 3×3 | 4/8 | 30.43 | 21.59 | 1.41–1.41× |
| 32×34 | 1 | 3×3 | 16/16 | 93.59 | 68.01 | 1.37–1.38× |
| 32×34 | 3 | 1×1 | 16/16 | 48.54 | 43.03 | 1.13–1.14× |
| 32×34 | 3 | 3×3 | 16/16 | 197.21 | 77.15 | 2.55–2.57× |

The stable direction across both preparation orders/profiles supports this narrow
classification. It does not exclude wins on other shapes/layouts or with other
private DL IDs. The current public DL default always chooses the block8 Default
variant; this batch does not time the other five IDs individually. NCHW transforms
are visibly part of that route, so an arithmetic-only comparison would omit costs
paid by this API call.

## First use, correctness and preservation

Workspace query, compile/prepare and first synchronized execution costs are stored
separately per solver and fixture in raw JSON. Fresh private user caches are not a
machine-wide cold state: installed system kernel databases, OS caches, shared
libraries and earlier fixture preparation remain available. DL preparation spans0.188–1326ms, versus52–724ms for HIP-direct.
The first shape prepares DL in about1.3s in DL-first runs but67ms in direct-first
runs, showing why these are not independent machine-cold measurements. First
synchronized execution spans0.060–5.258ms for DL and0.035–0.139ms for HIP-direct.
Exact values and preparation order are retained rather than generalized as model
startup costs. Later shapes may reuse prepared native code and must not be
described as independent cold compilations.

Independent CPU double convolution loops check every first execution, post-warmup
output and final output of each timing window. The unchanged N2.18 tolerance is
`2e-5 + 5e-5*abs(reference)`; max absolute error across timed runs is
`2.35537e-6`. Each check verifies immutable input/weight bytes and256-byte guards
on inputs/output/workspace. Six subprocesses (four timed and two diagnostic) give
576 such correctness checkpoints,14,184 total convolution calls; intermediate
outputs inside a32-call timing window are not copied individually.4/profile
including exact native Torch backward plus core7/CPU ONNX4/daily5/RCCL6 =
**30 acceptance checks, six acknowledged600s recovery guards**.29 local tests,
critical Ruff, Black, clang-format, Python compilation, shell syntax and diff
checks pass. Full upstream CI was not run.

The probe alone was compiled using existing ccache20GiB/no reset, fourCPU/OMP4,
RAM10GiB/memory+swap12GiB, one compile job,600s bound, read-only source/inputs/prefix,
no GPU/network in build. Existing B07.3 plugin reused unchanged.3684 protected
hashes,35 links/four AOT databases and136 workstation backing paths remain intact.
Same boot `e327c5de-46b8-4e05-94ad-d999a12a06e7`, kernel7.1.5-bc250rocm-r2;
kernel logs unchanged. No owned jobs/containers/armed guards at cleanup. No .105,
user-application access or active-prefix changes. Initial ordinary free12,926,078,976B;
cleanup12,914,167,808B; nothing deleted. Initial staging mkdir lacked permissions;
only the newly owned staging directory was created/chowned by root before build.
No failed GPU/correctness batch or tolerance relaxation. Historical unresolved
Python CPU fault remains distinct from already fixed VM-heap corruption.

## Replay and evidence

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`.

```sh
python3 helpers/build_ck_plugin_b074_on_host.py --attempt r1
python3 helpers/qualify_ck_plugin_b074_on_host.py --attempt r1 --build-attempt r1
python3 helpers/export_ck_plugin_b074_on_host.py
```

These exact attempts already exist and must be preserved. New attempts require
new versioned build/qualification/export destinations and matching validator path;
do not overwrite accepted artifacts. Use the recorded shared lock/resource preflight
and independent recovery guard. Active-prefix changes still require the exclusive
update procedure; none occurred here.

Build PID25065, qualification PID25392, export PID26699 all exited0. Logs are
`WS/logs/ck-plugin-b074-build-r1/`, `WS/logs/ck-plugin-b074-qualification-r1/`,
`WS/logs/ck-plugin-b074-r1-{build-controller,controller}.log` and
`WS/logs/ck-plugin-b074-export-controller.log`. Inputs/build/export:
`WS/tmp/ck-plugin-b074/{inputs-r1,build-r1,export-r1}`.
Local `kernel/artifacts/ck-plugin-b074/evidence.tar.gz`: 1,432,289B,
SHA256 `630a0525288253423c46e976bc3fe9c2b6009ccead3bb103be1d155c39ebd32b`; 102 raw files,
8,858,022B. Source/probe hashes and retained verification match.

B07.5: compare all six private per-ID DL variants on these retained fixtures, separating invoker-only and full layout-conversion costs; determine whether a CPU-reference-qualified performance configuration can beat HIP-direct before any opt-in packaging. Keep installed defaults, .105 and user applications unchanged.
