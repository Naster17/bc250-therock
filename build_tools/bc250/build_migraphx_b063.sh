#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
prefix=/workspace/build/dist/rocm
export CCACHE_MAXSIZE=20G
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache/migraphx-b063 --local-path /workspace/cache/data --no-reset-stats)"
export PATH="$prefix/bin:$prefix/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib:$prefix/lib/rocm_sysdeps/lib"
export ROCM_PATH="$prefix" HIP_PATH="$prefix" HIP_CLANG_PATH="$prefix/lib/llvm/bin"
export TMPDIR=/workspace/task/tmp
mkdir -p "$TMPDIR"
ccache --show-stats > /workspace/logs/ccache-before.txt
for dpp in 1 0; do
    ccache "$prefix/bin/hipcc" --offload-arch=gfx1013 -std=c++17 -Werror=c++20-extensions -O2 -g0 \
      -DMIGRAPHX_NGLOBAL=128 -DMIGRAPHX_NLOCAL=128 -DMIGRAPHX_WAVEFRONTSIZE=32 \
      -DMIGRAPHX_HAS_DPP="$dpp" -I/workspace/task/source/src/targets/gpu/kernels/include -I/workspace/task/source \
      /workspace/helpers/probe_migraphx_b063.hip \
      -Wl,-rpath,"$prefix/lib" -o /workspace/task/tmp/probe-dpp"$dpp"
    ccache "$prefix/bin/hipcc" --offload-arch=gfx1013 --offload-device-only -c \
      -std=c++17 -Werror=c++20-extensions -O2 -g0 -DMIGRAPHX_NGLOBAL=128 -DMIGRAPHX_NLOCAL=128 \
      -DMIGRAPHX_WAVEFRONTSIZE=32 -DMIGRAPHX_HAS_DPP="$dpp" \
      -I/workspace/task/source/src/targets/gpu/kernels/include -I/workspace/task/source \
      /workspace/helpers/probe_migraphx_b063.hip -o /workspace/task/tmp/device-dpp"$dpp".o
    "$prefix/lib/llvm/bin/clang-offload-bundler" --unbundle --type=o \
      --targets=hipv4-amdgcn-amd-amdhsa--gfx1013 \
      --input=/workspace/task/tmp/device-dpp"$dpp".o --output=/workspace/task/tmp/device-dpp"$dpp".elf
    "$prefix/lib/llvm/bin/llvm-objdump" -d /workspace/task/tmp/device-dpp"$dpp".elf > /workspace/logs/device-dpp"$dpp".asm
    "$prefix/lib/llvm/bin/llvm-readelf" --notes /workspace/task/tmp/device-dpp"$dpp".elf > /workspace/logs/device-dpp"$dpp"-notes.txt
done
ccache --show-stats > /workspace/logs/ccache-after.txt
# An unavailable DPP API must be rejected, while the header itself compiles above.
printf '#include <migraphx/kernels/reduce.hpp>\nusing unsupported = migraphx::reduce::subwave<32>;\n' > "$TMPDIR/unsupported.hip"
set +e
"$prefix/bin/hipcc" --offload-arch=gfx1013 -std=c++17 -Werror=c++20-extensions -c \
  -DMIGRAPHX_NGLOBAL=128 -DMIGRAPHX_NLOCAL=128 -DMIGRAPHX_WAVEFRONTSIZE=32 \
  -DMIGRAPHX_HAS_DPP=0 -I/workspace/task/source/src/targets/gpu/kernels/include \
  "$TMPDIR/unsupported.hip" -o "$TMPDIR/unsupported.o" > /workspace/logs/unsupported.log 2>&1
code=$?
set -e
printf '%s\n' "$code" > /workspace/logs/unsupported.exit
test "$code" -ne 0
python3 /workspace/helpers/admit_migraphx_b063.py --source /workspace/task/source --output /workspace/logs/features.json --scratch "$TMPDIR"
ccache "$prefix/lib/llvm/bin/clang++" -D__HIP_PLATFORM_AMD__ -std=c++17 -O2 -g0 \
  -I"$prefix/include" /workspace/helpers/probe_migraphx_jit_b063.cpp \
  -L"$prefix/lib" -Wl,-rpath,"$prefix/lib" -lhiprtc -lamdhip64 \
  -o "$TMPDIR/probe-jit"
