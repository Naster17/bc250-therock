# B07.3: private DL scalar, argument and per-kernel contracts

Completed2026-10-29 on .104. Source `3616b2f9f09` over private B07.2 `2976e7d`;
tools `7519e73b8`. Installed native libraries remain `8389197c1`. The candidate
is isolated; production admission, native prefix, Torch and profiles are unchanged.

## Changes

The private facade accepts exact alpha1/beta0 only. Non-default values, including
near1 doubles, NaN and infinity, are refused rather than silently ignored by the
DL math. It also restricts plain forward convolution (no transpose/bias), validates
group/channel/output geometry, and refuses descriptors exceeding the int32 DL
span contract before calling the instance factory. This adds refusals, not general
alpha/beta arithmetic support.

Solution creation checks the requested kernel ID. The owning invoker captures
its descriptors/workspace contract and checks runtime scalars, descriptor/cast
changes, null/alignment/range conditions, output/input overlaps, and required
workspace size/alignment/tensor overlaps before upstream dispatch. Runtime
violations throw `miopenStatusBadParm`; failed API construction leaves outputs
unchanged. These virtual-address checks do not discover allocation capacity or
physical aliases through separate VMM mappings; callers still own valid allocations.

Existing upstream forward/invoker math is reused. The private six-ID filter and
all66 API5 exports remain. No production factory, default whitelist or installed
library change. The exact hidden GetCastType shim remains local to the matching
private ABI. This is not cross-version C++ ABI portability.

## Numerical and refusal evidence

Each profile tests80 fixtures: two packed layouts (NCHW/NHWC), groups1/3,
five geometries, default/private streams and dyadic/non-dyadic values. Batch2;
shapes include1x1 stride1/2,3x3 stride2,3x3 dilation2, C3/K5 and K132 tile tails.
Every listed, admitted ID is explicitly passed to `ck_impl_fwd_get_solution`;
its invoker is prepared through MIOpen and called after the temporary problem
copy and returned solution have been destroyed. This qualifies that recorded
ownership boundary, not arbitrary concurrency or lifetime use.

| Kernel variant | Calls/profile |
| --- | ---: |
| Default, block8 | 80 |
| Default, block256 | 80 |
| Filter1x1Pad0, block8 | 32 |
| Filter1x1Pad0, block256 | 32 |
| Filter1x1Stride1Pad0, block8 | 16 |
| Filter1x1Stride1Pad0, block256 | 16 |

256 direct GPU calls/profile, **512 total**, match independent CPU double loops.
Dyadic cases are exact. Non-dyadic tolerance remains N2.18's
`2e-5 + 5e-5*abs(reference)`; max absolute error `1.07042e-6`. No reference or
tolerance relaxation. Inputs and256-byte guards remain intact.

**244 API/descriptor refusals** cover scalar variants, transpose, wrong output
extent, oversized metadata and unknown kernel IDs. **492 runtime refusals** cover
changed scalars, descriptors, null/misaligned pointers, exact/partial output aliases,
missing/short/misaligned/overlapping workspace. Refused calls preserve complete
output/scratch bytes and immutable inputs. Both layouts exercise runtime guards;
workspace-specific cases apply where layout transforms require scratch.

Retained public Immediate solver137 executes64 more CPU-reference GPU calls;
64 fresh no-private-path cases refuse it,304 B07.2 contracts and eight Torch
convolution output/all-first-gradient fixtures pass. Ordinary Torch may choose
other retained solvers; its gradients do not establish plugin backward support.

Private9+9/core7/CPU ONNX4/daily5/RCCL6 = **40 final checks**, six accepted600s
recovery guards. Initial r1 also passed40, then source review identified and
explicitly excluded transpose mode. Final r2 repeats the affected full batch;
12 guards total ACK. Initial source/validator and both builds are retained.
No failed GPU batch or numerical relaxation. A setup import-path error happened
before the initial preflight/build and was corrected without running a GPU test.

## Preservation and limits

The task stages new source in versioned `inputs-r1/r2`, preserving B07.2's protected
helpers and candidates. Build mounts use read-only source/prefix/inputs, no GPU
or network, four CPUs/OMP4, one compiler job, RAM10GiB/memory+swap12GiB, existing
ccache20GiB/no reset and600s bound.3676 protected hashes/35 links/four AOT DB paths
are unchanged;136 retained workstation backing paths match.29 local tests,
critical Ruff/Black/new-source clang-format/Python compilation/shell/diff pass;
full upstream CI not run.

The board was already on a new boot at entry:
`e327c5de-46b8-4e05-94ad-d999a12a06e7`, kernel7.1.5-bc250rocm-r2. No agent reboot
or cross-boot continuity claim. Accepted kernel logs remain unchanged through
cleanup; no owned jobs/containers/armed guards. Entry ordinary free13,329,420,288B,
cleanup12,928,606,208B. No cleanup/deletion, install, prefix/default/profile change,
.105 contact or user application access. Historical Python CPU fault remains
unresolved/distinct from fixed VM-heap corruption.

Still unqualified: other shapes/types/layouts, non-default scalar arithmetic,
physical alias detection, concurrent/graph use, plugin backward and performance.
The candidate must remain private until the next measured decision.

## Next and artifacts

**B07.4:** bounded paired private DL versus existing MIOpen HIP-direct timings,
with retained CPU references and explicit solver attribution. Separate warm/cold
costs; classify wins/losses before deciding useful opt-in packaging/admission.
There is no performance-improvement claim from this correctness batch.

`WS=/home/nik/bc250work/rocm-10.2-gfx1013`. Candidate
`WS/tmp/ck-plugin-b073/build-r2/libMIOpenCKGroupedConv_gfx1013.so`; versioned input
sources and r1 validator are preserved. Buildr1PID11617/qualr1PID12209/
buildr2PID13602/qualr2PID14039/evidence export all exit0. Final
`WS/logs/ck-plugin-b073-qualification-r2/final.json`, controller
`WS/logs/ck-plugin-b073-r2-controller.log`, export `WS/tmp/ck-plugin-b073/export-r1`.

Local `kernel/artifacts/ck-plugin-b073`:211 raw files/16,842,346 bytes, source/
binary/hash/guard/solver logs, independent export/visibility and preservation
checks. Archive3,284,485 bytes, SHA256
`e0038bf797e4846a91baa4efb81fc8b9c912badb2ba0152151bedc4565725b2b`.
