#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Called in the existing build image under the exclusive runtime lock.
set -euo pipefail
export ROCM_PATH=/workspace/build/dist/rocm HIP_PATH=/workspace/build/dist/rocm
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:/opt/torch-venv/bin:$PATH"
export CCACHE_CONFIGPATH=/workspace/cache/torch-config/ccache.conf
export CCACHE_MAXSIZE=20G
cd /workspace/pytorch-work
python tools/amd_build/build_amd.py --out-of-place-only
# Resolve any CMake regeneration first: Ninja's initial dry-run can otherwise
# stop at regeneration and hide the actual compilation work from the guard.
ninja -C build -j1 build.ninja
ninja -C build -n torch_hip > /workspace/logs/sdpa-torch-build-plan.log
# Fail instead of starting an unrelated full rebuild. The intended work is one
# SDP-utils translation unit and the existing libtorch_hip link.
python - <<'PY'
from pathlib import Path
lines = Path('/workspace/logs/sdpa-torch-build-plan.log').read_text().splitlines()
actions = [line for line in lines if line.startswith('[')]
print('\n'.join(actions), flush=True)
assert not any('Re-running CMake' in line for line in actions), 'Unresolved regeneration'
assert len(actions) <= 5, 'Unexpected broad rebuild: inspect plan before proceeding'
PY
ninja -C build -j12 torch_hip
ccache --show-stats
