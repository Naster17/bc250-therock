#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bundle only missing CPU image codecs into a new wheel; retain raw build."""

import hashlib
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--attempt", choices=("r3",), default="r3")
args = parser.parse_args()
root = Path("/workspace/vision-b041")
repair = root / ("wheel-repair-" + args.attempt)
wheel = next((root / "wheels").glob("torchvision-*.whl"))
assert not repair.exists()
subprocess.run(
    [
        sys.executable,
        "-m",
        "wheel",
        "unpack",
        str(wheel),
        "--dest",
        str(repair),
    ],
    check=True,
)
stage = next(repair.iterdir())
image = next((stage / "torchvision").glob("image*.so"))
libs = stage / "torchvision/codec-libs"
libs.mkdir()
licenses = libs / "LICENSES"
licenses.mkdir()
ldd = subprocess.check_output(["ldd", str(image)], text=True)
selected = {}
for line in ldd.splitlines():
    pieces = line.split()
    if (
        len(pieces) >= 3
        and pieces[1] == "=>"
        and pieces[0].startswith(
            ("libjpeg.so", "libpng16.so", "libwebp.so", "libsharpyuv.so")
        )
    ):
        source = Path(pieces[2])
        assert source.is_file(), line
        shutil.copy2(source, libs / pieces[0])
        selected[pieces[0]] = dict(
            source=str(source),
            sha256=hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
        )
        # DT_RUNPATH on image.so is not inherited by indirect dependencies.
        # Give every bundled codec its own relative lookup, including WebP's
        # dependency on libsharpyuv. Keep source DSOs and old wheels untouched.
        subprocess.run(
            ["patchelf", "--set-rpath", "$ORIGIN", str(libs / pieces[0])],
            check=True,
        )
assert all(
    any(n.startswith(prefix) for n in selected)
    for prefix in ("libjpeg.so", "libpng16.so", "libwebp.so")
), ldd
for directory in Path("/usr/share/doc").iterdir():
    if (
        directory.name.startswith(("libjpeg", "libpng", "libwebp"))
        and (directory / "copyright").is_file()
    ):
        shutil.copyfile(
            directory / "copyright", licenses / (directory.name + ".copyright")
        )
assert list(licenses.iterdir())
destination = root / ("repaired-wheels-" + args.attempt)
destination.mkdir()
subprocess.run(
    ["patchelf", "--set-rpath", "$ORIGIN/codec-libs", str(image)], check=True
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "wheel",
        "pack",
        str(stage),
        "--dest-dir",
        str(destination),
    ],
    check=True,
)
result = dict(
    status="passed",
    raw_wheel=str(wheel),
    raw_wheel_sha256=hashlib.file_digest(wheel.open("rb"), "sha256").hexdigest(),
    codecs=selected,
    original_ldd=ldd,
    repaired_wheel=str(next(destination.glob("*.whl"))),
)
(root / ("wheel-repair-" + args.attempt + ".json")).write_text(
    json.dumps(result, indent=2) + "\n"
)
