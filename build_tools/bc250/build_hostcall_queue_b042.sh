#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/audio-b042/ccache-probe-config --local-path /workspace/cache/data --no-reset-stats)"
export ROCM_PATH=/workspace/build/dist/rocm HIP_PATH=/workspace/build/dist/rocm
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:$ROCM_PATH/lib/rocm_sysdeps/lib"
ccache "$ROCM_PATH/bin/amdclang++" -x hip --rocm-path="$ROCM_PATH" --offload-arch=gfx1013 -O2 /workspace/helpers/hostcall_queue_b042.hip -L"$ROCM_PATH/lib" -lamdhip64 -Wl,-rpath,"$ROCM_PATH/lib" -o /workspace/audio-b042/hostcall_queue_b042
