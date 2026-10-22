#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the isolated native single-device RCCL reference executable."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = Path("/home/nik/bc250work/rocm-10.2-gfx1013/rccl-b034")
    report = dict(status="running", checks=[], world_size=1, native_arch="gfx1013")
    try:
        env = dict(os.environ)
        env.update(
            LD_LIBRARY_PATH=str(candidate / "install-r1/lib")
            + ":"
            + env.get("LD_LIBRARY_PATH", ""),
            NCCL_SOCKET_IFNAME="lo",
            NCCL_IB_DISABLE="1",
            NCCL_P2P_DISABLE="1",
            NCCL_CUMEM_ENABLE="0",
            NCCL_CUMEM_HOST_ENABLE="0",
            NCCL_DEBUG="INFO",
        )
        log = args.output.parent / "native-readiness.log"
        with log.open("w") as stream:
            result = subprocess.run(
                [str(candidate / "readiness-r1")],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=150,
            )
        report["returncode"] = result.returncode
        report["checks"] = [
            line.removeprefix("PASS ")
            for line in log.read_text().splitlines()
            if line.startswith("PASS ")
        ]
        assert result.returncode == 0 and len(report["checks"]) == 550
        assert len(set(report["checks"])) == 550
        libraries = [
            line.removeprefix("LIBRARY ")
            for line in log.read_text().splitlines()
            if line.startswith("LIBRARY ")
        ]
        assert (
            len(libraries) == 1
            and Path(libraries[0]).resolve()
            == (candidate / "install-r1/lib/librccl.so").resolve()
        )
        report["loaded_library"] = libraries[0]
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
