#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
cd /workspace/pytorch-work
# The out-of-place HIP generation step is required before every Torch rebuild.
python tools/amd_build/build_amd.py --out-of-place-only
# Uses ccache20GiB, MAX_JOBS12/link1, gfx1013, unchanged safe feature flags,
# USE_KINETO=1 and the qualified pip editable/no-isolation build.
export BC250_A09_FRESH=0
exec bash /workspace/helpers/build_torch_a09.sh
