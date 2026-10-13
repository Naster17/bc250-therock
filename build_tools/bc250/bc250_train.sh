#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Use the frozen consumers belonging to the current reviewed runtime.
set -eu
consumers=/opt/bc250-stack/current/python-consumers
if ! test -f "$consumers/training.env"; then
    echo 'bc250-train: the selected runtime has no reviewed training environment' >&2
    exit 2
fi
exec /usr/local/bin/bc250-torch \
    --mount-ro "$consumers" --env-file "$consumers/training.env" "$@"
