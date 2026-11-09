#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Ephemeral UID1000 source, pinned D35/D41 fixtures only; parent owns lifetime/locks."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
from artifact_d35 import load_manifest, digest
from capacity_plan_d42 import descriptor
from capacity_plan_d41 import CONFIG, CONTROL, parameters
from reference_transport_d40 import ReferenceServer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", choices=["control", "large"], required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert os.geteuid() == 1000 and os.uname().nodename == "bc250"
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64,) * 2)
    config = CONTROL if a.shape == "control" else CONFIG
    proof = descriptor(a.shape)
    assert a.sha256 == proof["manifest_sha256"]
    manifest = load_manifest(
        a.reference, a.sha256, config, parameters(config), check_sources=False
    )
    assert manifest == proof["manifest"]
    server = ReferenceServer(
        ("192.168.1.104", 29842),
        a.reference.parent,
        a.reference.read_bytes(),
        ["192.168.1.105"],
    )
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    result = dict(
        status="ready",
        pid=os.getpid(),
        uid=os.geteuid(),
        endpoint="http://192.168.1.104:29842",
        allowed_clients=["192.168.1.105"],
        manifest_sha256=digest(a.reference),
        source_blobs_reverified=len(manifest["tensors"]) * 2,
        sources=manifest["sources"],
        address_space_limit_bytes=256 * 1024**2,
        fd_limit=64,
        directory_listing=False,
        writes_allowed=False,
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    try:
        while running:
            server.handle_request()
    finally:
        server.server_close()
        result.update(status="stopped", transport=server.stats)
        a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
