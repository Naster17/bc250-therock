#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache/audio-b042-config --local-path /workspace/cache/data --no-reset-stats)"
export ROCM_PATH=/workspace/build/dist/rocm HIP_PATH=/workspace/build/dist/rocm
export HIP_CLANG_PATH="$ROCM_PATH/lib/llvm/bin"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:$ROCM_PATH/lib/rocm_sysdeps/lib"
export PYTHONPATH=/workspace/pytorch-work
export PYTORCH_ROCM_ARCH=gfx1013 USE_ROCM=1 USE_CUDA=0
export PYTORCH_NVCC="ccache $ROCM_PATH/bin/amdclang++"
export MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export BUILD_VERSION=2.11.0+gfx1013 PYTORCH_VERSION=2.13.0a0+gfx1013
export BUILD_RNNT=1 BUILD_ALIGN=1 BUILD_CUDA_CTC_DECODER=0
export CC=gcc CXX=g++
cd /workspace/audio-b042/source
python -c 'import torch; assert str(torch.__version__)=="2.13.0a0+gfx1013" and torch.version.hip; assert torch._C._GLIBCXX_USE_CXX11_ABI'
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir "${AUDIO_WHEEL_DIR:-/workspace/audio-b042/wheels}" --verbose
ccache --show-stats
