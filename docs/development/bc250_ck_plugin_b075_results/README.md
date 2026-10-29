# B07.5: six private DL configurations and layout costs

Completed2026-10-29 on .104, tools `45bac0baa`. Existing private plugin/source
`3616b2f9f09`, installed libraries `8389197c1`, Torch `a524980` unchanged.
Only the probe was built. **Two moderate fixtures benefit from a different private
DL configuration; the other four still favor HIP-direct. No default changed.**

## Matched comparison

Retains B07.4's six shapes, batch2, FP32, stride/dilation1 and groups1/3. Logical
input/weight values are identical across packed NCHW and NHWC, with independent
CPU double convolution references and unchanged N2.18 tolerance
`2e-5 + 5e-5*abs(reference)`. There are two applicable Default IDs on3x3 and six
IDs on1x1 (Default, Filter1x1Pad0 and Filter1x1Stride1Pad0; block8/256).

Each private ID is explicitly passed to API5 `ck_impl_fwd_get_solution`; MIOpen
prepares its owning invoker, then the temporary solution is released before use.
HIP-direct solver218 is compiled through the public API and its actual registered
invoker retrieved using the same network configuration. Both NCHW paths therefore
enter through cached invokers. This removes public lookup/validation overhead
from one side only being compared with a cached entry on the other side.

NCHW DL invokers include all per-call layout transposes. The NHWC measurements
use inputs/weights already packed before timing and no transform scratch. They
show the route without per-call transposes; they are not isolated arithmetic
kernel timings, nor an end-to-end advantage for applications starting in NCHW.
The difference between separate NCHW/NHWC measurements is not a separately
instrumented transpose duration. NCHW scratch11008–863232B; NHWC scratch0.

Two opposite-order fresh processes/profile (daily and rccl-single), six
32-call warm timing windows/path after four warmups; NCHW uses three ABBA/BAAB
cycles. NHWC runs six windows/ID, reversing ID order in the second process.
HIP events capture the complete stream interval including host feeding/idle and
private runtime guards; synchronized wall intervals are retained separately.
Each applicable ID has CPU-reference and guard checks after first execution,
warmup and every timing window. Whole intermediate windows are not copied.

Selected IDs below come from the first daily process; the same IDs retain their
win/loss direction in the other three processes. This is bounded fixture selection,
not a general shape heuristic. Times are first-daily medians in microseconds;
ratio is HIP-direct/DL across all four processes (>1 favors DL).

| Input | Groups | Filter | Selected family/block | Full NCHW DL μs | Cached HIP-direct μs | Prepacked NHWC DL μs | HIP-direct/DL range |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 9×11 | 1 | 3×3 | Default/8 | 28.05 | 21.00 | 14.60 | 0.748–0.750 |
| 9×11 | 3 | 1×1 | 1x1Stride1Pad0/8 | 20.35 | 15.50 | 5.61 | 0.760–0.762 |
| 9×11 | 3 | 3×3 | Default/8 | 29.97 | 21.47 | 15.23 | 0.716–0.717 |
| 32×34 | 1 | 3×3 | Default/256 | 62.19 | 67.92 | 45.76 | 1.091–1.097 |
| 32×34 | 3 | 1×1 | 1x1Pad0/256 | 37.04 | 42.35 | 16.40 | 1.138–1.144 |
| 32×34 | 3 | 3×3 | Default/256 | 138.87 | 76.22 | 117.97 | 0.549–0.551 |

The two moderate wins are1.091–1.097× and1.138–1.144×. The B07.4 public default
remains block8 Default and does not automatically select these configurations.
No whole-model/training speedup or new backward capability is established.

Public Immediate218/137 anchor windows are retained separately from matched
invoker comparisons. One small grouped1x1 RCCL reverse-order public-DL anchor
varies25.5–31.2μs, so these unpaired diagnostics must not be promoted to reliable
public-route speedups. The next task explicitly pairs the public paths. Two
separate verbose runs execute correctness only (no reported timing) and prove
all six actual DL ID strings and HIP-direct `MIOpenConvUni`. Native solver source
already implements `Search` via `GenericSearch` and kernel-ID PerfDb configuration;
inspect/use that machinery before inventing another override.

## Acceptance, preservation and replay

Four timed processes give80 NCHW paired ID comparisons plus80 NHWC ID measurements;
two diagnostic processes complete **2184 CPU-reference checkpoints**,52008 total
convolution calls. Maximum absolute error2.35537e-6; immutable input/weight bytes
and256-byte input/output/workspace guards pass. Private4+4/core7/CPU ONNX4/daily5/
RCCL6 = **30 checks**, six ACK600s recovery guards, plus29 local unit tests and
critical Ruff/Black/clang-format/Python/shell/diff checks. Full upstream CI not run.
Exact native Torch backward runs after each private batch; this is not plugin
backward support. No tolerance/reference relaxation or failed GPU batch.

Build uses fourCPU/OMP4/jobs1/RAM10GiB/mem+swap12GiB, existing ccache20GiB/no reset,
ro sources/inputs/prefix, no build GPU/network and600s bound. Nonfatal compiler
warnings concern ignored event-destruction status in the probe destructor.
3692 protected hashes/35 links/four AOT DBs and136 workstation backing files match.
Same boot `e327c5de-46b8-4e05-94ad-d999a12a06e7`, kernel7.1.5-bc250rocm-r2;
kernel logs unchanged. No owned jobs/containers/armed guards at cleanup. Ordinary
free12,911,759,360→12,901,400,576B. No deletion, .105/user-app access, install,
active-prefix/profile/default change. Historical Python CPU fault remains
unresolved and distinct from fixed VM-heap corruption.

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`:

```sh
python3 helpers/build_ck_plugin_b075_on_host.py --attempt r1
python3 helpers/qualify_ck_plugin_b075_on_host.py --attempt r1 --build-attempt r1
python3 helpers/export_ck_plugin_b075_on_host.py
```

These destinations already exist: preserve them. For a new attempt use new
versioned inputs/build/qualification/export destinations and matching validator
paths. Shared runtime lock, resource preflight and independent guard are required.
No active-prefix update occurred; such updates still require the exclusive lock.
BuildPID27923/qualificationPID28197/exportPID29498 exited0. Logs:
`WS/logs/ck-plugin-b075-qualification-r1/final.json`, corresponding
`WS/logs/ck-plugin-b075-r1-controller.log`, build-r1 and export-controller logs.
Fixtures at `WS/tmp/ck-plugin-b075/{inputs-r1,build-r1,export-r1}`.
Local `kernel/artifacts/ck-plugin-b075/evidence.tar.gz`: 1,497,520B,
SHA256 `4e2d4d39fd075519ff852c2a740caf6a80bbbf69f5e164b374a99777f6c0e703`,
102 raw files/6,874,663B; sources/probe hashes verified.

B07.6: audit and qualify existing solver137 Search/PerfDb selection and fresh-process public-Immediate replay in isolated databases, using the two beneficial retained configurations and paired HIP-direct references/timings. Do not invent an architecture-wide heuristic or install/admit the candidate. Preserve defaults/.105/user applications.
