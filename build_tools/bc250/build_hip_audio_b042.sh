#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export CCACHE_MAXSIZE=20G
eval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/audio-b042/ccache-runtime-config --local-path /workspace/cache/data --no-reset-stats)"
python /workspace/helpers/build_hip_audio_b042.py
ccache --show-stats
