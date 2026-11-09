#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D39 GPU audit with an explicit manifest-only remote reference reader."""
import argparse
from pathlib import Path
import sys
import artifact_d35 as artifact
import distributed_d35_growth as base
import distributed_d39_units as previous
from reference_transport_d40 import RemoteStore

previous_sources = previous.sources


def sources():
    result = previous_sources()
    result.update(
        {
            name: artifact.digest(Path(__file__).with_name(name))
            for name in [
                "reference_transport_d40.py",
                "distributed_d40_transport.py",
                "serve_reference_d40.py",
            ]
        }
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument("--transport", choices=["local", "remote"], required=True)
    p.add_argument("--endpoint")
    args, remaining = p.parse_known_args()
    path = Path(remaining[remaining.index("--reference") + 1])
    pin = remaining[remaining.index("--reference-sha256") + 1]
    remote = None
    if args.transport == "remote":
        assert args.endpoint == "http://192.168.1.104:29840"
        # The descriptor directory has only the SHA-pinned manifest. Never open
        # the original local tensor directory or fall back if a request fails.
        assert sorted(x.name for x in path.parent.iterdir()) == ["manifest.json"]
        remote = RemoteStore(args.endpoint, path, pin)
        artifact.verify_blob = remote.verify
        base.read_blob = remote.read
    else:
        assert args.endpoint is None
    original = base.atomic_json

    def publish(path, record):
        record["reference_transport"] = dict(
            kind=args.transport, stats=remote.stats if remote else None
        )
        original(path, record)

    base.atomic_json = publish
    previous.sources = sources
    sys.argv = [sys.argv[0], "--audit", "reference", *remaining]
    previous.main()


if __name__ == "__main__":
    main()
