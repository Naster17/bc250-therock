# B07.9: FP16 DL factory and accumulation qualification

Completed2026-10-29 on .104, tools `0e8a28146`. Source libraries `3616b2f9f09`,
installed native libraries `8389197c1`, Torch `a524980` unchanged. **Isolated factory
proof only:** the installed `bc250-ck` artifact remains FP32-only. No new plugin,
production admission, default, source-header or native archive update.

## Existing source and actual build availability

N2.18 qualified FP32 scalar convolution, not FP16 convolution. The existing CK
factory already declares FP16 input/weight/output instances for group-interleaved
`NHWGC/GKYXC/NHWGK` and group-major `GNHWC/GKYXC/GNHWK`. Each uses the three scalar
DL configurations instantiated for Default,1x1Pad0 and1x1Stride1Pad0: nine IDs per
layout. Unlike the FP32 factory's duplicate transfer-parameter IDs, the FP16 third
configuration has K1=2 and a distinct type string. All nine are explicitly tested.

The template rows use **F32 accumulation** with F16 inputs/output. Exact staged
headers match the pinned tracked source. gfx1013 is excluded from the packed-dot
macro branch; scalar half values convert to float, and packed-half fallback also
accumulates float products. A native GPU feature shader reports dot2=0/mac_f32=1/
wave_size32; the consumer checks native gfx1013:xnack- and no XDL admission. This
is source/macro plus actual-operation evidence, not a complete ISA disassembly.

The installed static archive contains the interleaved FP16 factory but omits the
group-major FP16 definition. R3's actual link failure and the independent defined-
symbol audit prove that availability gap. The source already exists: R4 compiles
that one unchanged group-major factory into a private object and links it with the
unchanged installed archive. No replacement archive is installed and no arithmetic
implementation is duplicated or patched. The missing installed symbol remains a
recorded build-availability limitation for callers wanting that factory directly.
Without explicit `DL_KERNELS`, both consumer factories enumerate zero instances.

## Bounded numerical contract

Two layouts × default/private streams × groups1/3; batch2. Five geometry cases have
dyadic and quantized non-dyadic values. The additional accumulation fixture uses
512 input channels and one exact loss-of-small-increments pattern.88 fixtures per
profile, with admission evaluated for every raw instance.

| H×W | C/K per group | Filter | Stride/dilation/padding |
| --- | --- | --- | --- |
| 5×7 | 4/4 | 1×1 | 1/1/0 |
| 9×11 | 4/8 | 1×1 | 2/1/0 |
| 9×10 | 4/132 | 3×3 | 2/1/1 |
| 8×9 | 8/8 | 3×3 | 1/2/2 |
| 7×9 | 3/5 | 3×3 | 1/1/1 |
| 1×1 | 512/4 | 1×1 | 1/1/0; accumulation fixture |

Independent CPU double loops consume the **actual stored half inputs** and actual
layout strides. Dyadic/accumulation outputs must exactly equal the final half-rounded
CPU result. Non-dyadic error against FP64 is bounded by
`0.5*binary16_spacing(reference) + 2e-5 + 5e-5*abs(reference)`, with spacing floored
at2^-24. This new-type contract was fixed before the first GPU run; it adds the
output quantization allowance to the retained N2.18 FP32 residual contract. No
post-result reference/tolerance relaxation. Max absolute error0.0018510521;
max error versus half-rounded FP64 reference0.0009765625. It is not universal
correct rounding, NaN/Inf/subnormal/overflow or arbitrary cancellation coverage.

The exact accumulation case sums256 ones plus256 terms of1/1024: FP64/final-half
result256.25, while an explicit sequential half accumulator produces256. Every
admitted instance returns256.25.144 such executions are a subset of656 total, not
additional calls. This distinguishes that sequential half model; source pinning
establishes the actual float accumulator type, not an inference ruling out every
possible alternative reduction tree.

Both profiles execute328 admitted instances and464 refused arguments each:
**656 GPU CPU-reference comparisons,928 admission refusals**. Every ID executes
in each layout; refused specialization/vector-tail combinations are not launched
and leave output bytes unchanged.256-byte input/output guards and immutable input/
weight bytes pass; required workspace is zero. Streams are synchronized at reference
checks; overlapping streams, graph capture and arbitrary layout holes are unqualified.

## Regression, builds and preservation

Existing Torch FP16 convolution output/all-first-gradient fixtures pass8 times
across daily/RCCL, using the pre-existing native MIOpen routes. Installed FP32
`bc250-ck` remains unchanged:586 retained native convolution references and8 Torch
output/all-first-gradient fixtures pass, including verified-process public137.
Factory7+7/installed10+10/core7/CPU ONNX4/daily5/RCCL6 = **56 checks**, eight ACK600s
guards;29 local tests/critical Ruff/Black/clang-format/pycompile/shell/diff pass.
No plugin FP16 dispatch/backward or performance claim follows from this batch.

R1 fails before GPU on a probe-only HIP `half` name collision and missing architecture
include; R2 exposes that header's required CK base include order. R3 builds the
no-DL executable, then finds the missing group-major symbol. R4 links the isolated
upstream object and both probes; the first GPU qualification passes in full.
All failed sources/logs/partial outputs remain. One compiler job/fourCPU/OMP4,
RAM10GiB/memory+swap12GiB, ccache20GiB/no reset, read-only sources/SDK/inputs,
no build GPU/network and600s phase bound. No full Torch/library build or cache reset.

3832 protected hashes/36 links/four AOT databases/136 workstation backing paths
unchanged. Same boot `e327c5de-46b8-4e05-94ad-d999a12a06e7`, kernel7.1.5-bc250rocm-r2;
kernel logs unchanged through cleanup, no KFD warning recurrence. The earlier
KFD-return uncertainty and historical Python CPU fault remain distinct from fixed
VM-heap corruption. No reboot/.105/user-app access or owned jobs/containers/armed
guards at cleanup. Ordinary free12,625,784,832→12,595,290,112B; nothing deleted.

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`; exact accepted commands:

```sh
python3 helpers/build_ck_factory_b079_r4_on_host.py --attempt r4
python3 helpers/audit_ck_factory_b079_archive_on_host.py --build-attempt r4
python3 helpers/qualify_ck_factory_b079_on_host.py --attempt r1 --build-attempt r4
python3 helpers/export_ck_factory_b079_on_host.py
```

Staged r4 build helper/script aliases preserve the earlier versions; they match
the committed unsuffixed build helper/script. Existing attempt destinations must
be preserved. BuildPIDs13558/14091/14381 exit1 before GPU; R4PID15015/qualificationPID15696/
exportPID17462 exit0. Artifacts at `WS/tmp/ck-plugin-b079/{inputs-r1..r4,build-r1..r4,export-r1}`;
source/archive audit JSONs, `WS/logs/ck-plugin-b079-qualification-r1/final.json`
and corresponding `WS/logs/ck-plugin-b079-r1-controller.log` retain exact commands.
Local `kernel/artifacts/ck-plugin-b079/evidence.tar.gz`: 3,683,360B,
SHA256 `8bfc6d1a4ce85413684170bf7c0c39ddd13982379901f6619f5cdc5531809351`,
177 raw files/18,466,165B. Sources/staged headers, both final
binaries, supplemental object and installed-option preservation verified.

B07.10 planned: build an isolated matching-MIOpen FP16-capable DL forward facade using the qualified NHWGC factory, dtype-aware scalar/descriptor/alignment/overlap/workspace contracts and explicit public-solver CPU references. Keep registered B07.7 FP32 artifact, native archive, defaults/.105/user applications unchanged until separately qualified.
