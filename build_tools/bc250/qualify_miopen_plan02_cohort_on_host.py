#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise inactive coherent optional/native identities without changing live users."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path[:0] = [
    "/home/nik/bc250work/tools/plan01",
    "/home/nik/bc250work/rocm-10.2-gfx1013/helpers",
]
from qualify_rccl_b038_on_host import gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", required=True)
    p.add_argument("--cohort-root", type=Path, required=True)
    p.add_argument("--public", action="store_true")
    p.add_argument("--native-only", action="store_true")
    a = p.parse_args()
    root = a.cohort_root
    os.environ["PATH"] = str(root / "test-tools") + ":" + os.environ["PATH"]
    ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
    tools = Path("/home/nik/bc250work/tools/plan02")
    stage = json.loads((root / "stage.json").read_text())
    runtime = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    out = Path("/home/nik/bc250work/evidence") / ("plan02-cohort-" + a.attempt)
    out.mkdir()
    r = dict(status="running", gates=[], stage=stage)
    try:
        for profile in ["daily", "rccl-single"]:
            for route, launcher, script, extra, count in [
                (
                    "direct",
                    "bc250_torch.py",
                    tools / "validate_miopen_plan02.py",
                    ["--library-sha", stage["library_sha256"]],
                    1,
                ),
                (
                    "backward",
                    "bc250_torch.py",
                    tools / "validate_miopen_plan02_backward.py",
                    [
                        "--library",
                        "/workspace/build/dist/rocm/lib/libMIOpen.so.1.0",
                        "--sha",
                        stage["library_sha256"],
                        "--worker",
                    ],
                    12,
                ),
                (
                    "ck",
                    "bc250_ck.py",
                    tools / "validate_miopen_plan02_consumers.py",
                    ["--route", "ck", "--sha", stage["library_sha256"]],
                    3,
                ),
                (
                    "migraphx",
                    "bc250_migraphx.py",
                    tools / "validate_miopen_plan02_consumers.py",
                    ["--route", "migraphx", "--sha", stage["library_sha256"]],
                    3,
                ),
                (
                    "ort",
                    "bc250_ort.py",
                    ws / "helpers/validate_onnxruntime_b0612.py",
                    [
                        "--fixtures",
                        str(ws / "tmp/onnxruntime-b0612/generation-r1/models"),
                    ],
                    30,
                ),
            ]:
                if a.native_only and route not in {"direct", "backward"}:
                    continue
                label = "plan02-cohort-" + a.attempt + "-" + profile + "-" + route
                scratch = out / (profile + "-" + route)
                scratch.mkdir()
                os.chown(scratch, 1000, 1000)
                selected = str(root / "tools" / launcher)
                if a.public:
                    selected = (
                        "/usr/local/bin/"
                        + {
                            "bc250_torch.py": "bc250-torch",
                            "bc250_ck.py": "bc250-ck",
                            "bc250_migraphx.py": "bc250-migraphx",
                            "bc250_ort.py": "bc250-ort",
                        }[launcher]
                    )
                profile_args = ["--runtime-profile", profile]
                if a.public and os.uname().nodename == "bc500":
                    assert a.native_only, "Target has no optional SDK installation"
                    if profile == "daily":
                        # The preserved deployment launcher predates profile CLI.
                        profile_args = []
                    else:
                        selected = (
                            "/opt/bc250-stack/tools/distributed-d09/bc250_torch.py"
                        )
                cmd = [
                    selected,
                    *profile_args,
                    "--shm-size",
                    "256m",
                    "--env",
                    "MIOPEN_USER_DB_PATH=" + str(scratch / "db"),
                    "--env",
                    "MIOPEN_CUSTOM_CACHE_DIR=" + str(scratch / "cache"),
                    str(script),
                    *extra,
                ]
                r["gates"].append(gate(label, cmd, count, runtime))
                (out / "progress.json").write_text(json.dumps(r, indent=2) + "\n")
        r["status"] = "passed"
    except BaseException as e:
        r.update(status="failed", error=repr(e))
        raise
    finally:
        (out / "final.json").write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
