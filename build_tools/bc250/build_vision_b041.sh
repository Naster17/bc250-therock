#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G
# Separate config, shared retained cache; never reset counters or objects.
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache/vision-b041-config --local-path /workspace/cache/data --no-reset-stats)"
export ROCM_PATH=/workspace/build/dist/rocm HIP_PATH=/workspace/build/dist/rocm
export HIP_CLANG_PATH=/workspace/build/dist/rocm/lib/llvm/bin
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:$ROCM_PATH/lib/rocm_sysdeps/lib"
export PYTHONPATH=/workspace/pytorch-work
export PYTORCH_ROCM_ARCH=gfx1013 FORCE_CUDA=1
export PYTORCH_NVCC="ccache $ROCM_PATH/bin/amdclang++"
export MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export BUILD_VERSION=0.28.0+gfx1013 PYTORCH_VERSION=2.13.0a0+gfx1013
export TORCHVISION_USE_NVJPEG=0
export CC=gcc CXX=g++
cd /workspace/vision-b041/source
test "$(git rev-parse HEAD)" = 8fb87713a24951e639c494b0f2a8a81b5f8e33a6
python -c 'import torch; assert str(torch.__version__)=="2.13.0a0+gfx1013" and torch.version.hip; assert torch._C._GLIBCXX_USE_CXX11_ABI'
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir /workspace/vision-b041/wheels --verbose
ccache --show-stats
dpkg-query -W libpng-dev libjpeg-dev libwebp-dev patchelf
