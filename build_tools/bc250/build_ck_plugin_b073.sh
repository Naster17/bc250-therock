#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G OMP_NUM_THREADS=4
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
prefix=/workspace/build/dist/rocm
ck=/workspace/build/math-libs/composable_kernel/stage
mi=/workspace/source/rocm-libraries/projects/miopen
cfg=/workspace/build/ml-libs/MIOpen/build
mkdir /workspace/output/overlay
cp -a "$cfg/include" /workspace/output/overlay/
# Only the candidate's header-local CK invoker templates need this feature.
# Never change the original configured header or the host MIOpen library.
python3 - <<'PY'
from pathlib import Path
p=Path('/workspace/output/overlay/include/miopen/config.h')
s=p.read_text();assert s.count('#define MIOPEN_USE_COMPOSABLEKERNEL 0')==1
p.write_text(s.replace('#define MIOPEN_USE_COMPOSABLEKERNEL 0','#define MIOPEN_USE_COMPOSABLEKERNEL 1'))
PY
ccache "$prefix/bin/amdclang++" -x hip -O2 -DNDEBUG -std=c++20 --offload-arch=gfx1013 \
    -DMIOPEN_BETA_API=1 -fPIC -fvisibility=hidden -DCK_IMPL_BUILDING_DLL -DDL_KERNELS -DMIOPEN_BC250_DL_PRIVATE \
    -I/workspace/output/overlay/include -I"$mi/include" -I"$mi/src/include" \
    -I"$ck/include" -I"$ck/include/ck" -I"$prefix/include" \
    -I"$cfg/../stage/include" -I/workspace/build/third-party/sysdeps/linux/sqlite3/build/dist/lib/rocm_sysdeps/include -I/workspace/build/third-party/nlohmann-json/dist/include -I/workspace/build/base/half/stage/include -shared /workspace/inputs/bc250_dl_private.cpp \
    -I"$mi/src/ck_impl" -x none "$ck/lib/libdevice_conv_operations.a" \
    -L"$prefix/lib" -lMIOpen -Wl,--exclude-libs,ALL -Wl,-z,defs \
    -o /workspace/output/libMIOpenCKGroupedConv_gfx1013.so \
    > /workspace/output/build-plugin.log 2>&1
ccache --show-stats

bash /workspace/helpers/build_ck_plugin_probe_b073.sh
