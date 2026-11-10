#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only CPU tool/Python metadata probe; never imports Torch or generates code."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tools = {}
    for name, path in dict(
        gcc="/usr/bin/gcc",
        gxx="/usr/bin/g++",
        hip_clang="/workspace/build/dist/rocm/lib/llvm/bin/clang++",
        cmake="/usr/local/therock-tools/bin/cmake",
        ninja="/opt/torch-venv/bin/ninja",
        ccache="/usr/bin/ccache",
        python=sys.executable,
    ).items():
        p = Path(path)
        assert p.exists(), path
        run = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=15
        )
        assert run.returncode == 0, (path, run.stderr)
        tools[name] = dict(
            path=path,
            resolved=str(p.resolve()),
            sha256=sha(p),
            version=(run.stdout + run.stderr).strip(),
        )
    packages = []
    for dist in metadata.distributions():
        record_files = [
            f
            for f in (dist.files or [])
            if f.name == "RECORD" and ".dist-info" in str(f)
        ]
        packages.append(
            dict(
                name=dist.metadata["Name"],
                version=dist.version,
                record_sha256=[sha(Path(dist.locate_file(f))) for f in record_files],
            )
        )
    selected = {}
    for name in [
        "setuptools",
        "ninja",
        "numpy",
        "packaging",
        "pyyaml",
        "requests",
        "six",
        "typing-extensions",
        "pip",
        "wheel",
        "cmake",
    ]:
        try:
            selected[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            selected[name] = None
    result = dict(
        status="passed",
        python_version=sys.version,
        tools=tools,
        packages=sorted(packages, key=lambda p: p["name"].lower()),
        build_packages=selected,
        cmake_on_path=shutil.which("cmake"),
        cmake_on_path_sha256=sha(Path(shutil.which("cmake"))),
        torch_imported="torch" in sys.modules,
        gpu_nodes_exposed=any(
            Path(p).exists() for p in ["/dev/kfd", "/dev/dri/renderD128"]
        ),
    )
    assert not result["torch_imported"] and not result["gpu_nodes_exposed"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
