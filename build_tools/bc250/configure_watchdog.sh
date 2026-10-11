#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Configure Alpine's existing OpenRC service for the BC-250 hardware watchdog.
# Run on the host as root before GPU testing; never inside a container.
set -eu

test "$(id -u)" -eq 0 || {
    echo "Run as root on the Alpine host" >&2
    exit 1
}
test "$(cat /sys/class/watchdog/watchdog0/identity)" = "SP5100 TCO timer"
test -c /dev/watchdog0
test -f /etc/init.d/watchdog

if ! test -f /etc/conf.d/watchdog.bc250-before-10.2; then
    cp -p /etc/conf.d/watchdog /etc/conf.d/watchdog.bc250-before-10.2
fi

# Ten-second keepalives leave ample scheduling margin during compiler links.
cat > /etc/conf.d/watchdog <<'EOF'
WATCHDOG_OPTS="-t 10 -T 120"
WATCHDOG_DEV="/dev/watchdog0"
EOF
rc-service watchdog restart
rc-update add watchdog default
test "$(cat /sys/class/watchdog/watchdog0/state)" = active
test "$(cat /sys/class/watchdog/watchdog0/timeout)" = 120
echo "BC250 hardware watchdog active: interval=10s timeout=120s"
