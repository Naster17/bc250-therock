#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify a retained post-HIPify reference overlay; does not invoke code generation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

WORK = Path("/home/nik/bc250work")
ROOT = WORK / "evidence/source-d46-r1"


def main():
    result = json.loads((ROOT / "result.json").read_text())
    destination = ROOT / "generated-replay"
    destination.mkdir(exist_ok=False)
    expected = [
        item
        for item in result["profiles"]["daily"]["source_files"]
        if not item["matches_native"]
    ]
    for item in expected:
        target = destination / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "reconstructed-delta" / item["path"], target)
    patch = WORK / "tools/source-d46/hipify-output.diff"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=destination, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=destination, check=True)
    for item in expected:
        assert (
            hashlib.sha256((destination / item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    proof = dict(
        status="passed",
        files=len(expected),
        patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
        scope="Reference overlay reproduces five observed generated files; not an independent HIPify execution",
    )
    (ROOT / "generated-replay.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
