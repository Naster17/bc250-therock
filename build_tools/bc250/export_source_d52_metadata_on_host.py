#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export bounded D52 evidence without duplicating image/source/proof payloads."""
import json
from pathlib import Path
import tarfile

WORK = Path("/home/nik/bc250work")


def main():
    target = WORK / "evidence/source-d52-evidence-r1.tar.gz"
    assert not target.exists()
    image = json.loads((WORK / "evidence/source-d52-image-r1/result.json").read_text())
    assembly = json.loads(
        (WORK / "evidence/source-d52-assembly-r2/result.json").read_text()
    )
    package = json.loads(
        (WORK / "evidence/source-d52-package-r1/result.json").read_text()
    )
    assert image["status"] == assembly["status"] == package["status"] == "passed"
    with tarfile.open(target, "w:gz") as output:
        for folder in [
            "source-d52-image-r1",
            "source-d52-assembly-r1",
            "source-d52-assembly-r2",
            "source-d52-inputs-r1",
            "source-d52-inputs-r2",
            "source-d52-package-r1",
        ]:
            root = WORK / "evidence" / folder
            for path in root.iterdir():
                if path.is_file() and path.suffix in (".json", ".log"):
                    output.add(path, arcname="evidence/" + folder + "/" + path.name)
        for path in (WORK / "tools/source-d52").rglob("*.py"):
            output.add(
                path,
                arcname="tools/" + str(path.relative_to(WORK / "tools/source-d52")),
            )
    print(target)


if __name__ == "__main__":
    main()
