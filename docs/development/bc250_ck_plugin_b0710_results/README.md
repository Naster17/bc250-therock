# B07.10: isolated FP16 MIOpen DL forward facade

Completed2026-10-29 on .104. Tools `af4704f69`, libraries source `742ee1e2c0e`;
installed native libraries `8389197c1` and Torch `a524980` unchanged.

The candidate adds an explicitly compile-gated FP16 path to the matching private
MIOpen API5 facade. Installed `bc250-ck` remains the B07.7 FP32-only artifact.
This is an isolated qualification; no active prefix, registration, native archive,
solver whitelist, runtime default or image was changed.

## Implementation and boundary

The existing upstream dispatcher already supports half inputs, weights and output.
B07.9 established its FP32 accumulator and native scalar DL implementation; this
batch reuses that code and the installed NHWGC factory archive. It needs no extra
factory object. The source change is confined to `bc250_dl_private.cpp`, under
`MIOPEN_BC250_DL_FP16_PRIVATE`; the unflagged facade retains FP32-only admission.

All tensor/request dtypes must agree. Packed 2D forward NCHW/NHWC, alpha1/beta0
and existing geometry/int32-span restrictions remain. BF16, casts, mixed types,
TF32, determinism, bias, transpose and unsupported shapes are refused. The runtime
checks exact descriptors, scalar values, null pointers, overflow, output aliasing,
workspace size/alignment and workspace aliasing before dispatch. FP16 pointers
must be16-byte aligned, conservatively covering the qualified vector operations;
FP32 retains its four-byte contract. Descriptor byte sizes determine extents.

Type strings omit dtype. The global list is a deduplicated union of nine IDs;
per-problem lists retain six FP32 and nine FP16 IDs. Same-handle public calls
alternate half/float/half/float with independent references and guard regions to
exercise dtype separation in cached invokers. All66 required API exports plus the forward-list export remain, including
explicit unsupported solver groups; the matching GetCastType shim stays hidden.

## Tests and limits

Explicit public MIOpen solver137 and direct per-ID invokers run on native
`gfx1013:xnack-`, on daily and RCCL-single profiles. Two layouts, default/private
streams, groups1/3, batch2, five geometries/two patterns and an additional C512
accumulation fixture are covered. Every half ID executes. Dyadic outputs equal
the final half-rounded FP64 reference exactly, including256.25 in the accumulator
discriminator. Non-dyadic CPU double references use actual stored half values and
the predeclared B07.9 bound:

`0.5*binary16_spacing(reference) + 2e-5 + 5e-5*abs(reference)`

Spacing has floor2^-24. FP32 retains N2.18's residual bound. Input/weight bytes
must be unchanged and256-byte probe guards intact. No tolerance was relaxed after
execution. Pointer/descriptor/scalar/overflow/workspace negative tests fail before
GPU access. A fresh process without the private path refuses all32 public half
requests per profile.

The plugin supports forward only. Four ordinary Torch FP16 convolution fixtures
per profile verify output and all first gradients through retained native
backward solvers; they do not establish plugin backward support. Installed FP32
public/per-ID/native-backward and consumer tests remain separate regression gates.
No new speed claim, default performance heuristic, overlapping-stream, graph
capture, arbitrary-shape, NaN/Inf/subnormal/overflow or full upstream-suite claim.

## Reproduction and evidence

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`:

```sh
python3 helpers/build_ck_plugin_b0710_on_host.py --attempt r2
python3 helpers/qualify_ck_plugin_b0710_on_host.py --attempt r1 --build-attempt r2
python3 helpers/export_ck_plugin_b0710_on_host.py
```

Before a build, populate a new immutable `tmp/ck-plugin-b0710/inputs-rN` with the
matching facade, upstream `ck_grouped_conv_fwd_impl.cpp`, retained b073 probe,
new half/public probes and `primitives_n216_common.hpp`. Preserve existing attempt
directories. Build uses one compiler job, four CPU/OMP threads,10GiB RAM/12GiB
memory+swap, ccache20GiB without resetting it, read-only source/SDK/inputs,
no build GPU/network and600s phase bound. Qualification uses SH runtime lock,
preflight and eight ACK watchdog guards; it never needs an active-prefix update.

R1 built the candidate and retained probe but failed compiling the new test's
assignment to MIOpen's deterministic attribute wrapper. The test now uses the
public attribute setter. R2 builds fully; first GPU qualification is the recorded
accepted run. Failed inputs/logs/partial outputs are preserved.

The historical Python CPU fault and B07.7 KFD-return uncertainty remain distinct
from fixed VM-heap corruption. This batch does not claim a kernel root-cause fix.

Across daily/RCCL, the candidate passes724 FP16 references (656 per-ID,64 public,
four mixed-handle),580 retained FP32 references,368 FP16 API refusals and954
FP16 runtime refusals.64 absent-path refusals are separate. Max per-ID half
absolute error0.00186229. Installed FP32 adds586 native references; eight half and
eight installed-float Torch output/all-first-gradient fixtures pass. Private12+12,
installed10+10, core7, CPU ONNX4, daily5 and RCCL6 = **66 checks/eight ACK guards**.
29 local tests, Black/critical Ruff/clang-format/pycompile/shell/diff checks pass.
An initial local export audit incorrectly equated66 required loader symbols with
all dynamic symbols; the exact CK export set is66 required plus the forward-list
API. No required symbol is missing; upstream helpers/GetCastType remain hidden.

3849 protected hashes,36 links, four AOT databases and136 workstation backing
files unchanged. Same e327c5de-46b8-4e05-94ad-d999a12a06e7 boot/r2 kernel;
kernel logs unchanged through cleanup, no KFD recurrence. No reboot/.105/user-app
access, no owned running jobs/containers/armed guards. Free12,592,922,624→
12,572,831,744B, nothing deleted. Build1PID19855 exits1, build2PID20834,
qualificationPID21771/exportPID23540 exit0.

Evidence: `WS/logs/ck-plugin-b0710-qualification-r1/final.json`, controller
`WS/logs/ck-plugin-b0710-r1-controller.log`; `WS/tmp/ck-plugin-b0710/`
`{inputs-r1,inputs-r2,build-r1,build-r2,export-r1}`. Local
`kernel/artifacts/ck-plugin-b0710/evidence.tar.gz`: 2,837,154B,
SHA256 `f54cb396f04ead909da37448521017d77186e90058b3c39c6cf3a925ba5a741a`;
180 raw files/14,936,817B. Source/helper/input/binary identities verified locally.

Next B07.11: bounded paired FP16 public137 versus retained HIP-direct timing and
native Search/PerfDb replay on representative owned fixtures. Use correctness and
identity gates before any versioned opt-in packaging; preserve installed FP32
and ordinary defaults. A slower candidate is a valid measured outcome.
