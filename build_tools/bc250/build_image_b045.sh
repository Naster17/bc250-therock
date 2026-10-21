#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache/codec-b043-config --local-path /workspace/cache/data --no-reset-stats)"
export ROCM_PATH=/workspace/build/dist/rocm HIP_PATH=/workspace/build/dist/rocm
export CMAKE_PREFIX_PATH="$ROCM_PATH:$ROCM_PATH/lib/cmake"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export HIP_CLANG_PATH="$ROCM_PATH/lib/llvm/bin" PYTORCH_ROCM_ARCH=gfx1013
export LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib:/workspace/build/dist/rocm/lib/rocm_sysdeps/lib
export PYTHONPATH=/workspace/pytorch-work
export BUILD_VERSION=0.17.0+gfx1013 ENABLE_CUDA=OFF TORCHCODEC_BUILD_IMAGE=OFF
export TORCHCODEC_BUILD_JPEG=ON TORCHCODEC_BUILD_PNG=ON TORCHCODEC_BUILD_WEBP=ON TORCHCODEC_BUILD_NVJPEG=OFF
export CMAKE_BUILD_PARALLEL_LEVEL=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CMAKE_ARGS='-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache'
# Private on-board development wheel against distro FFmpeg; not a distributable
# release artifact. Keep distro copyright/license files; N8 packaging needs its
# own dependency/license disposition before publication.
export I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1
cd /workspace/image-b045/source
python -c 'import torch; assert str(torch.__version__)=="2.13.0a0+gfx1013"; assert torch._C._GLIBCXX_USE_CXX11_ABI'
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir "$CODEC_WHEEL_DIR" --verbose
pkg-config --modversion libavcodec libavformat libavutil libswresample libswscale
ffmpeg -version
dpkg-query -W > /workspace/image-b045/build-packages.txt
python -m pip list --format=json > /workspace/image-b045/build-python-packages.json
ccache --show-stats
