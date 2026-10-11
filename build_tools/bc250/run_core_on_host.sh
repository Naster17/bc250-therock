#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run as root on Alpine after transferring a committed source snapshot.
set -eu

workspace="${1:?usage: run_core_on_host.sh <workspace-directory>}"
image="${BC250_BUILD_IMAGE:-localhost/bc250-rocm-build:10.2-ccache}"
build_jobs="${BC250_BUILD_JOBS:-16}"
flang_jobs="${BC250_FLANG_JOBS:-4}"
test "$(id -u)" -eq 0
test -s "$workspace/source/BC250_SOURCE_REVISIONS"
test "$(cat /sys/class/watchdog/watchdog0/state)" = active

# The independent image build can still be downloading dependencies.
attempt=0
until podman image exists "$image"; do
    attempt=$((attempt + 1))
    if test "$attempt" -ge 240; then
        echo "Build image did not become available; inspect logs/build-image.log" >&2
        exit 1
    fi
    sleep 15
done

podman image inspect "$image" > "$workspace/logs/build-image-inspect.json"
podman run --rm --network none "$image" sh -c \
    'cmake --version; ninja --version; patchelf --version; dpkg-query -W; pip freeze' \
    > "$workspace/logs/build-environment.txt"

# Build containers get no GPU devices. Keep source code read-only.
mkdir -p "$workspace/cache"
rm -f "$workspace/logs/core-build.exit"
set +e
podman run --rm --name bc250-rocm-core-build --network host \
    -v "$workspace/source:/workspace/source:ro" \
    -v "$workspace/build:/workspace/build:rw" \
    -v "$workspace/cache:/workspace/cache:rw" \
    -e CMAKE_BUILD_PARALLEL_LEVEL="$build_jobs" \
    -e BC250_FLANG_JOBS="$flang_jobs" \
    -e BC250_CCACHE_MAXSIZE="${BC250_CCACHE_MAXSIZE:-20G}" \
    "$image" bash /workspace/source/build_tools/bc250/build_core.sh \
    /workspace/source /workspace/build
status=$?
set -e
echo "$status" > "$workspace/logs/core-build.exit"
exit "$status"
