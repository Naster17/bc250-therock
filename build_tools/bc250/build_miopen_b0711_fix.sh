#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G OMP_NUM_THREADS=4
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
cd /workspace/build/ml-libs/MIOpen/build
# Reuse qualified host objects. A normal Ninja graph also considers unrelated
# HIP-version header timestamps stale. Execute only the exact existing recipes
# for the embedded source batch, its object and the shared-library relink.
python3 - <<'PY'
from pathlib import Path
import json,subprocess
root=Path.cwd()
assert 'MIOpenConvDirUniHip.cpp' in (root/'inlined_kernels/batch_120_files.txt').read_text()
targets=['inlined_kernels/batch_120.cpp.hpp','src/CMakeFiles/MIOpen.dir/__/inlined_kernels/batch_120.cpp.o','lib/libMIOpen.so.1.0']
recipes=[]
for target in targets:
    lines=subprocess.check_output(['ninja','-t','commands','-s',target],text=True).strip().splitlines()
    assert len(lines)==1,(target,lines)
    recipes.append(dict(target=target,command=lines[0]))
Path('/workspace/output/target-recipes.json').write_text(json.dumps(recipes,indent=2)+'\n')
with Path('/workspace/output/build-native.log').open('w') as log:
    for row in recipes:
        log.write(row['target']+'\n');log.flush()
        subprocess.run(['bash','-c',row['command']],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
PY
ccache --show-stats
