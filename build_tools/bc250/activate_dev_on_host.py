#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Select the mutable Priority 1 developer stack after its bounded GPU gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid
import xml.etree.ElementTree as et


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument(
        "--image", default="localhost/bc250-stack:10.2-gfx1013-priority1-candidate"
    )
    args = parser.parse_args()
    assert os.geteuid() == 0, "Run through doas on the host"
    ws = args.workspace.resolve(strict=True)
    required = [
        ("logs/library-expansion/lt-priority1.json", 868),
        ("logs/library-expansion/ck-priority1.json", 247),
        ("logs/priority1-dnn/lt.json", 48),
        ("logs/priority1-dnn/hip.json", 18),
        ("logs/priority1-dnn/miopen.json", 18),
        ("logs/priority1-dnn/fusion.json", 16),
    ]
    for filename, count in required:
        report = json.loads((ws / filename).read_text())
        assert report["status"] == "passed", filename
        assert (
            sum(
                item["executed_tests"]
                for item in report["checks"]
                if not item.get("name", "").startswith("compile-")
            )
            == count
        ), filename
    torch = json.loads((ws / "logs/priority1-torch-final.json").read_text())
    assert torch["status"] == "passed" and len(torch["checks"]) == 64
    assert len(torch["hipblaslt_dispatch"]) == 12 and all(
        torch["hipblaslt_dispatch"].values()
    )
    units = et.parse(ws / "logs/priority1-miopen-units.xml").getroot()
    assert int(units.attrib["tests"]) > 0
    assert all(
        int(units.attrib.get(key, 0)) == 0 for key in ("failures", "errors", "disabled")
    )
    assert all(case.find("skipped") is None for case in units.iter("testcase"))
    image = subprocess.check_output(
        ["podman", "image", "inspect", "--format", "{{.Id}}", args.image], text=True
    ).strip()
    stack = Path("/opt/bc250-stack")
    previous = (stack / "current").resolve(strict=True)
    runtime = json.loads((previous / "runtime.json").read_text())
    dev = stack / "dev/priority1"
    dev.mkdir(parents=True, exist_ok=True)
    cache = ws / "cache/dev-user"
    cache.mkdir(parents=True, exist_ok=True)
    os.chown(cache, runtime["uid"], runtime["gid"])
    for name, target in [
        ("rocm", ws / "build/dist/rocm"),
        ("torch", ws / "pytorch-work"),
        ("cache", cache),
        ("python-consumers", previous / "python-consumers"),
    ]:
        path = dev / name
        if not path.is_symlink():
            assert not path.exists(), path
            path.symlink_to(target.resolve(strict=True), target_is_directory=True)
    runtime["previous_runtime_revisions"] = {
        key: runtime[key]
        for key in ("therock_revision", "rocm_libraries_revision", "torch_revision")
    }
    runtime.update(
        release="dev-priority1",
        image=image,
        profile="gfx1013-priority1-dev",
        therock_revision=(ws / "source/GIT_REVISION").read_text().strip(),
        rocm_libraries_revision=(ws / "source/rocm-libraries/GIT_REVISION")
        .read_text()
        .strip(),
        torch_revision=(ws / "pytorch-work/GIT_REVISION").read_text().strip(),
        source_state="mutable worktree; SOURCE_PRIORITY1.json records the final snapshot",
        validated=True,
        mutable=True,
        qualification="bounded Priority 1 GPU checks",
        validation_workspace=str(ws),
        limitations=[
            "SDMA disabled; experimental VM heap remains disabled",
            "Scalar backends have documented API/type/operator subsets",
            "No MFMA/WMMA; distributed and advanced hardware profiling unqualified",
            "Mutable developer prefix and editable Torch; no long stability qualification",
        ],
    )
    (dev / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    recovery = stack / "before-dev-current"
    if not recovery.exists():
        recovery.symlink_to(previous, target_is_directory=True)
    temporary = stack / ("current-dev-" + uuid.uuid4().hex)
    temporary.symlink_to(dev, target_is_directory=True)
    os.replace(temporary, stack / "current")
    print(f"BC250_DEV_ACTIVE {dev} image={image}")


if __name__ == "__main__":
    main()
