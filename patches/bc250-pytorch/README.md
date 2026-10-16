# BC-250 PyTorch patches

Current A05 source: `6181d00b5beb8cc9fd6800da7a19c8abf756b30d`. Patch0004
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
