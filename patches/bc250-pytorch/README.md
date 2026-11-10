# BC-250 PyTorch patches

D46 (2026-11-10) adds a separately verified complete export at
[`../bc250-pytorch-complete`](../bc250-pytorch-complete/README.md). It closes the
historical three-file gap for the qualified native source and current source head.
This old incremental directory remains unchanged/incomplete historical evidence;
do not combine its patches with the complete series. Full dependency/bundle/build
reconstruction remains PLAN-07 work. Older gap statements below describe this
historical series, not missing files in the current Git repository.

Patch0010 is a three-commit incremental mbox on Torch `da45369`: native sparse
triangular pivot marking (`a434d03`/`dc8fea0`) and test qualification `2b449c4`.
Only libtorch_hip.so changes; the old profiler-order runtime is retained.
[Build, source/trace evidence and recovery](../../docs/development/bc250_sparse_n213_results/README.md).
This does not close the historical SDPA reconstruction gap below.


Latest addition0009 is a three-commit incremental mbox on qualified Torch4af4a15:
native CPU-first ROCm profiler fixb823a1b plus test-only fixture corrections
throughda45369. Only libtorch_cpu.so changes in the separate installed candidate;
Kineto/SDK pinned sources are unchanged. [Build/qualification/recovery](../../docs/development/bc250_profiler_order_results/README.md).
This does not close the historical SDPA reconstruction gap documented below.


Current A09 source: `4af4a1508a6ec0b7b2b29f5e0115f8477c4d1bb5`.
Patch0008 adds seven Kineto/SDK profiler qualification tests on71a0426; native
profiler source was already present. Enable it with the tracked
`build_tools/bc250/build_torch_a09.sh` helper (USE_KINETO=1). Preserve the documented
read-only shared third-party mount and original source/build. Full Git plus
incremental bundle retained; older three-file SDPA reconstruction gap remains.
[Scope, build and evidence](../../docs/development/bc250_profiler_a09_results/README.md).

Current corrected A06/A07 source: `71a0426566bc9b952d9aa9b41f4a9d79ce46e0e8`.
Patch0007 is incremental onba6cbcd and includes low-precision full-operation
parity correction, D256 tiled/hybrid attention and tests. DSOs unchanged. Full Git
and incremental bundle retained; older SDPA three-file reconstruction gap remains.
[Scope/evidence](../../docs/development/bc250_a06_a07_results/README.md).

Historical initial A06 follows.

Current A06 source: `ba6cbcd2fafac1a6503f1f580033fbf72ad2a4f9`.
Patch0006 is incremental on6ed3d12 and adds only the opt-in gated-delta
preparation/composition module and its tests. Existing DSOs are unchanged.
[Scope/evidence](../../docs/development/bc250_gated_delta_a06_results/README.md).
The older three-file SDPA reconstruction gap remains.

Current sparse/lint follow-up source: `6ed3d12b184a27d21c0fa41d43269ff6bd17f1e0`.
Patch0005 is incremental on A05 source6181d00. It adds native first-order sparse
autograd and a source-only lint option. The older SDPA reconstruction gap remains.
[Scope and evidence](../../docs/development/bc250_sparse_followup_results/README.md).

Qualified A05 source: `6181d00b5beb8cc9fd6800da7a19c8abf756b30d`. Patch0004
is incremental on the full A04 source8cdfcc8. It adds only the opt-in Python
`torch.nn.bc250_training_ops` module; installed DSOs remain unchanged.
[API, qualification and deployment](../../docs/development/bc250_training_a05_results/README.md).
The historical SDPA reconstruction gap described below remains.

Qualified A04 source: `8cdfcc8e0feb3689a1ec1d292539af083b13ea94`.
Patch0003 is an incremental export based on full patched Torch
`4ee65717d69e9f51021272e3f37d0f342f68fc08`; it adds native scalar
hipSPARSELt integration. The old0001/0002 reconstruction does **not** include
three later SDPA source files. Preserve that known gap: this directory alone
does not reconstruct the current tree. Full local Git sources and the A04
incremental Git bundle on .104 retain the actual history.

Build/codegen only through `python -m pip install -e . -v --no-build-isolation`.
For raw CUDA-source updates, refresh the candidate CMake configuration so the
existing toolchain performs HIPify inside that build; retain the prior cache.
Do not invoke standalone codegen. See the [A04 report](../../docs/development/bc250_torch_a04_results/README.md).

The following baseline description is historical.

Baseline: `c779020750d7dc99a39ea8c104ef8df1fc9d0755` (ROCm PyTorch 2.13).
Local patched source: `4f13bf86476252e3d57b17d1c72f8512814400e9`.

Apply the ordered patches with `git am` to a clean checkout of that baseline when
reconstructing the source. Preserve existing work before replacing any checkout.
Export the patched tree with `build_tools/export_source_archive.py`; the Torch
build helper verifies the new exported `GIT_REVISION`.
`git am` can create a different commit hash through committer metadata; in that
case set `BC250_TORCH_SOURCE_REVISION` to the reconstructed checkout's exact HEAD
when building. The default pin is the local patched source above.

The patch fixes padding coordinates in channels-last average-pooling backward.
`build_tools/bc250/validate_operator_coverage.py` compares forward values and
nonuniform backward gradients against CPU, including FP32/FP16/BF16, padded
channels-last tensors, ceil mode and both padding-count policies. The developer
rebuild uses `pip install -e . -v --no-build-isolation`; original release copies
and source archives are retained.

The second patch permits explicit gfx1013 hipBLASLt selection after a real
zero-workspace heuristic succeeds. Default rocBLAS remains unchanged. The current pip-only build procedure above supersedes the historical standalone
HIPify instruction; a stale CMake configure can otherwise leave HIP sources stale.
