#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Extend the qualified application guard to the local default Docker daemon."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


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


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    directory = Path(config["evidence"])
    assert not directory.exists(), "Use a fresh guard directory"
    source = Path("/home/nik/bc250work/tools/lora-two-host/guard.py")
    spec = importlib.util.spec_from_file_location("qualified_lora_guard", source)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    original = guard.preflight
    observed = {"before": None, "after": None}

    def preflight():
        original()
        observed["before"] = docker_containers()
        assert not observed[
            "before"
        ], "Unrelated Docker containers are running: " + repr(observed["before"])

    guard.preflight = preflight
    error = None
    try:
        guard.main()
    except BaseException as exc:
        error = exc
    finally:
        try:
            observed["after"] = docker_containers()
        except Exception as exc:
            observed["error"] = repr(exc)
            if error is None:
                error = exc
        final = directory / "final.json"
        if final.exists():
            record = json.loads(final.read_text())
            record["docker_containers"] = observed
            if observed["after"] or observed.get("error"):
                record["status"] = "failed"
            final.write_text(json.dumps(record, indent=2) + "\n")
    if error is not None:
        raise error
    assert not observed["after"], "Unrelated Docker workload started during the test"


if __name__ == "__main__":
    main()
