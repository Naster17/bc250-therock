#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Back up paired Lt artifacts and build under the exclusive prefix lock."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from deploy_torch_a04_on_host import sha

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")


def main() -> None:
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="initial")
    args = parser.parse_args()
    assert args.attempt.isalnum()
    recovery = WS / "recovery/dense-int8-a08-before"
    recovery.mkdir(exist_ok=True)
    logs = WS / ("logs/dense-int8-a08-build-" + args.attempt)
    logs.mkdir()
    source = (
        WS
        / "source/rocm-libraries/projects/hipblaslt/library/src/amd_detail/gfx1013_scalar.cpp"
    )
    library = WS / "build/dist/rocm/lib/libhipblaslt.so.1.5"
    pack = WS / "build/dist/rocm/.kpack/blas_lib_gfx1013.kpack"
    before_path = recovery / "before.json"
    if before_path.exists():
        before = json.loads(before_path.read_text())
        for p, n in [
            (source, "gfx1013_scalar.cpp"),
            (library, "libhipblaslt.so.1.5"),
            (pack, "blas_lib_gfx1013.kpack"),
        ]:
            assert sha(recovery / n) == before[str(p)], n
    else:
        before = {str(p): sha(p) for p in [source, library, pack]}
        for p, n in [
            (source, "gfx1013_scalar.cpp"),
            (library, "libhipblaslt.so.1.5"),
            (pack, "blas_lib_gfx1013.kpack"),
        ]:
            shutil.copy2(p, recovery / n)
        before_path.write_text(json.dumps(before, indent=2) + "\n")
    shutil.copyfile(WS / "helpers/gfx1013_scalar_a08.cpp", source)
    command = [
        "podman",
        "run",
        "--rm",
        "--name",
        "bc250-dense-int8-a08-build",
        "--network",
        "none",
        "-v",
        f"{WS}:/workspace:rw",
        "-v",
        f"{WS}/source:/workspace/source:ro",
        "-v",
        f"{WS}/helpers:/workspace/helpers:ro",
        "localhost/bc250-rocm-build:10.2-ccache",
        "bash",
        "/workspace/helpers/build_lt_a08.sh",
    ]
    try:
        with (logs / "build.log").open("w") as output:
            subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)
        report = dict(
            status="built_not_gpu_qualified",
            before=before,
            after={str(p): sha(p) for p in [source, library, pack]},
            recovery=str(recovery),
        )
    except BaseException:
        shutil.copy2(recovery / "gfx1013_scalar.cpp", source)
        shutil.copy2(recovery / "libhipblaslt.so.1.5", library)
        shutil.copy2(recovery / "blas_lib_gfx1013.kpack", pack)
        raise
    (logs / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
