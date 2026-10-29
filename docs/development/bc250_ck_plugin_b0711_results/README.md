# B07.11: FP16 public tuning, paired timing and HIP-direct accumulation repair

Completed2026-10-30 on .104. Tools `2d8f1dae0`, libraries source `96e5a7dfc43`.
Installed native libraries `8389197c1`, Torch `a524980`, registered FP32
`bc250-ck`, images and ordinary defaults are unchanged. The new MIOpen library
and B07.10 FP16 plugin are **isolated candidates**, not installed upgrades.

## Precision defect found before timing qualification

The original public HIP-direct solver218 misses the fixed FP32-accumulation
contract on the first small non-dyadic FP16 fixture. The preserved reproduction
reports N2/C4/K8/H9/W11/groups1/filter3, output index1:

- actual0.8427734375;
- FP64 reference from actual stored half operands0.8424274250864983;
- absolute error0.0003460124135017395 exceeds bound0.0003062619962543249.

The matching source explicitly overrides `FLOAT_ACCUM` to FP16, retaining legacy
behavior for this kernel. This explains a precision limitation under our stricter
contract; it is not a buffer-corruption finding or proof that all old Torch
training results are invalid. B07.10's small Torch gradient tests used their
recorded looser dtype tolerances and did not establish this larger precision scope.

The source repair leaves the existing `float_types.h` FP32 accumulator enabled
for **native gfx1013 forward only**. Other devices and backward retain existing
behavior. Scalar conversions therefore multiply/accumulate as float and round
only on half output. No CK arithmetic, admission whitelist or Linux kernel change.
The original failure remains in qualification attempts1/2; no tolerance change.

A separate copy of the original MIOpen build reuses its qualified host objects.
The successful native-fix-r2 executes the exact existing Ninja recipes for just
`batch_120.cpp.hpp`, its C++ object and the shared-library link. This regenerates
the embedded kernel source in a real MIOpen library, without interposition or a
mock backend. Fresh processes prove its actual loaded path (loader trace and
Torch `/proc/self/maps`) and hash before executing CPU references/gradients.
This is a resource-only relink, not a clean full MIOpen rebuild or complete ABI
qualification for deployment.

## Native tuning and paired measurements

Six fixed packed NCHW fixtures: batch2, H/W9×11 with C4/K8 per group and H/W32×34
with C16/K16, each with groups/filter1/3,3/1,3/3. The public CK solver137 uses the
unchanged B07.10 candidate; public HIP-direct218 uses the isolated precision
repair. Both paths must meet the same fixed B07.9 half-quantization plus N2.18
residual bound:

`0.5*binary16_spacing(reference) + 2e-5 + 5e-5*abs(reference)`

Spacing has floor2^-24. CPU double loops use actual quantized half operands;
256-byte guards and immutable input/weight bytes remain mandatory.

Native public Find/Search (five iterations/1000ms maximum) generates genuine
FP16 PerfDb and FindDb records in private directories. All selected IDs belong
to the nine qualified half configurations. Fresh opposite-order processes replay
those records through public Immediate calls. Three ABBA cycles/32 repeats per
window, six windows per solver, with warmup and CPU reference checkpoints outside
timing. Report medians of stream-event intervals and wall time. NCHW measurements
include layout transforms and host issuance gaps; they are not pure kernel time,
training throughput or a comparison against legacy lower-precision arithmetic.

Untimed verbose replay proves exact selected-ID loads without search or database
mutation. Invalid IDs in separate copies are rejected and fall back to the default
block8 CK configuration, with the same references; originals remain unchanged.

| H×W | C/K per group | Groups | Filter | HIP-direct/CK stream-time ratio, four processes |
| --- | --- | --- | --- | --- |
| 9×11 | 4/8 | 1 | 3×3 | 0.5973–0.6060 |
| 9×11 | 4/8 | 3 | 1×1 | 0.7103–0.7309 |
| 9×11 | 4/8 | 3 | 3×3 | 0.4998–0.5147 |
| 32×34 | 16/16 | 1 | 3×3 | 0.9983–1.0029 |
| 32×34 | 16/16 | 3 | 1×1 | 1.1706–1.1782 |
| 32×34 | 16/16 | 3 | 3×3 | 0.5830–0.5857 |

Ratios above1 favor CK. The moderate groups3/filter1 fixture benefits1.1706–1.1782×;
the moderate groups1/filter3 fixture is effectively tied. HIP-direct wins the
other four. Search selected three distinct IDs across the six cases.


No speedup threshold is an acceptance gate. Four of six cases favor repaired
HIP-direct, one is around parity, and one benefits from tuned CK in this bounded
sample. This supports keeping explicit selection; it does not justify globally
forcing CK or predict any particular model's training speed.

## Regression, failed attempts and preservation

**80 top-level checks/10 final ACK guards** pass. There are444 independent CPU
checkpoints and9516 explicit Immediate calls in the tuning/replay/timing groups;
internal native-search evaluations are excluded.12 genuine PerfDb and12 FindDb
records across the two profiles. Max absolute error0.0019515566527843475 under
the unchanged bound.29 local tests, Black/critical Ruff/clang-format, Python/shell
syntax and diff checks pass. Three earlier failed GPU guards also ACK safely.


The final timing groups also run the retained B07.10 suite in fresh processes
mapped to the rebuilt MIOpen:24 nested checks,724 half/580 float references and
eight Torch FP16 output/all-first-gradient fixtures. Separate unchanged-native
candidate gates retain another724 half/580 float references and eight such Torch
fixtures. Installed FP32 adds586 native references and eight FP32 Torch convolution
output/all-first-gradient fixtures. Plugin backward remains unsupported; native
backward solvers execute those gradients.

Buildr1's memory preflight refuses before compilation; the board is freed by the
user. Prober2 builds, qualificationr1 fails numerically; prober3 adds structured
solver/shape/error diagnostics, qualificationr2 reproduces218. Native-fix-r1's
normal Ninja dry run shows only pending CMake regeneration, hiding a396-edge
rebuild caused by newer HIP-version header timestamps. Our container is stopped
early; that copy/logs remain. Native-fix-r2 uses the three exact resource/link
recipes from a fresh original-build copy. Qualificationr3 refuses before GPU on
an omitted private-loader dependency directory; r4 supplies the existing SDK
sysdeps directory and passes. These failures are preserved, not silently retried
in place. The failed native build script is recovered and checked against its
recorded SHA; all original source/build/cache trees are retained.

Builds use fourCPU/OMP4, one compiler job,10GiB RAM/12GiB memory+swap, ccache20GiB
without reset, read-only sources/SDK/inputs, no build GPU/network and600s phase
bounds. GPU tests use the existing SH runtime lock and watchdog gates; no active
prefix update. Full source builds/other devices/arbitrary shapes/overlapping
streams/graph capture/exceptional floating values and full upstream CI remain
unqualified. The source repair is not yet active in ordinary user launches.

3864 protected hashes,36 links, four FFT AOT databases and136 workstation backing
files unchanged. Boot e327c5de-46b8-4e05-94ad-d999a12a06e7/r2 kernel unchanged;
kernel logs unchanged through cleanup, no KFD recurrence. No reboot/.105/user-app
access/deletion, no running owned jobs/containers/armed guards;13 guards ACK in all.
Free20,477,861,888→12,493,869,056B; two independent MIOpen build copies
account for most new storage. BuildPIDs23839refused/24088success/25122success;
qualPIDs24520/25960/29708exit1 and30214exit0; native26573stopped/29211success;
export846exit0. Failed copies remain remote, with failed native script SHA-verified
in the local artifact directory.


The historical Python CPU fault and B07.7 KFD-return uncertainty remain distinct
from fixed VM-heap corruption. No kernel root-cause fix is claimed.

## Reproduction and next action

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`:

```sh
python3 helpers/build_ck_plugin_b0711_on_host.py --attempt r3
python3 helpers/build_miopen_b0711_fix_on_host.py --attempt r2
python3 helpers/qualify_ck_plugin_b0711_on_host.py --attempt r4 --build-attempt r3
python3 helpers/export_ck_plugin_b0711_on_host.py --attempt r4
```

Populate fresh `inputs-rN` with the new probe and retained common header, and
`native-inputs-r1` with the patched kernel source before building. Existing attempt
directories must be preserved. Build/qualification reports retain exact commands,
resource settings, hashes, failures and recovery-guard identities.

Accepted reports: `WS/logs/ck-plugin-b0711-qualification-r4/final.json` and
`WS/logs/ck-plugin-b0711-r4-controller.log`; `WS/tmp/ck-plugin-b0711/` contains
`inputs-r1..r3`, `build-r2,r3`, `native-inputs-r1`, `native-fix-r1,r2` and
`export-r4`. Local `kernel/artifacts/ck-plugin-b0711/evidence.tar.gz`:
66,256,969B SHA256 `0ce6d10f1c369e3ed68389b48562c76f79b633b4dc6474ec025d31102c7ed8d2`;
336 raw files/520,315,914B. Exact source/helper/input/binary/library
hashes verified; the full patched kernel source is found byte-for-byte in the
rebuilt library's embedded resource. Fresh mapped-library tests pin that library.


B07.12 planned: isolated relocatable FP32/FP16 CK artifact with the matching repaired MIOpen, exact ABI/runtime/source/license identity and genuine example records; two build-free relocated extracts across daily/RCCL with CPU references, gradients and refusal checks before installation. Preserve installed FP32 artifact/native defaults/.105/user applications. Ordinary native precision-repair rollout remains pending separate compatibility qualification.

