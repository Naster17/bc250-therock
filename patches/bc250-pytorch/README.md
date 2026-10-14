# BC-250 PyTorch patch

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
zero-workspace heuristic succeeds. Default rocBLAS remains unchanged. Incremental
CUDA-source updates must run `tools/amd_build/build_amd.py --out-of-place-only`
before editable pip; a cached CMake configure otherwise leaves HIP sources stale.
