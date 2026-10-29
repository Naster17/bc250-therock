#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fetch source-pinned ORT archives into an offline CMake URL mirror."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    selected = json.loads(args.manifest.read_text())["selected_for_candidate"]
    args.output.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        name, spec = item
        url = spec["url"]
        target = args.output / url.removeprefix("https://")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            partial = target.with_name(target.name + ".partial")
            assert not partial.exists(), partial
            with urlopen(url, timeout=120) as response, partial.open("xb") as stream:
                while data := response.read(1024 * 1024):
                    stream.write(data)
            assert (
                hashlib.sha1(partial.read_bytes()).hexdigest() == spec["upstream_sha1"]
            ), name
            partial.rename(target)
        data = target.read_bytes()
        assert hashlib.sha1(data).hexdigest() == spec["upstream_sha1"], name
        result = dict(
            spec,
            relative_path=str(target.relative_to(args.output)),
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
        )
        print("VERIFIED", name, len(data), flush=True)
        return name, result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(fetch, selected.items()))
    report = dict(status="fetched-sha1-and-sha256-verified", dependencies=results)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
