#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run retained facade and Torch gradients in a fresh process using the rebuilt MIOpen."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
    meta = json.loads((ws / "tmp/ck-plugin-b0711/native-fix-r2/final.json").read_text())
    library = Path(meta["library"]).resolve()
    assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
    assert str(library) in Path("/proc/self/maps").read_text()
    assert hashlib.sha256(library.read_bytes()).hexdigest() == meta["library_sha256"]
    sys.argv = [
        str(ws / "helpers/validate_ck_plugin_b0710.py"),
        "--build-attempt",
        "r2",
        "--output",
        str(args.output),
    ]
    runpy.run_path(sys.argv[0], run_name="__main__")
    value = json.loads(args.output.read_text())
    assert value["status"] == "passed" and len(value["checks"]) == 12
    value["mapped_native_miopen"] = str(library)
    value["mapped_native_miopen_sha256"] = meta["library_sha256"]
    args.output.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
