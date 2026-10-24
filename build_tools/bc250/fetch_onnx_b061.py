#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fetch the exact reviewed ONNX candidate wheels; preserve and verify existing files."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads(Path(__file__).with_name("onnx_b061_packages.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, item in pins.items():
        target = args.output / item["filename"]
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"], (
                target
            )
        else:
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                data = response.read()
            assert (
                len(data) == item["bytes"]
                and hashlib.sha256(data).hexdigest() == item["sha256"]
            ), name
            with target.open("xb") as stream:
                stream.write(data)
        print(name, item["version"], item["sha256"])


if __name__ == "__main__":
    main()
