#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed inspection of the default local Docker daemon."""
from pathlib import Path
import shutil
import subprocess


def docker_containers(socket=Path("/var/run/docker.sock")):
    if not socket.exists():
        return []
    assert shutil.which("docker"), "Docker socket exists but its CLI is unavailable"
    result = subprocess.run(
        ["docker", "--host", "unix://" + str(socket), "ps", "-q"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        "Cannot establish local Docker state: " + result.stderr
    )
    return sorted(result.stdout.split())
