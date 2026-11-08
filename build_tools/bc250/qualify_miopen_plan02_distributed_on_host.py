#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retain public two-host RCCL sums and CPU-reference DDP after native rollout."""
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
    p.add_argument("--port", type=int, required=True)
    a = p.parse_args()
    assert os.uname().nodename in {"bc250", "bc500"}
    rank = 0 if os.uname().nodename == "bc250" else 1
    work = Path("/home/nik/bc250work")
    base = Path("/opt/bc250-stack")
    cohort = (base / "native-cohort").resolve(strict=True)
    stage = json.loads((cohort / "stage.json").read_text())
    d01 = json.loads((base / "distributed/d01.json").read_text())
    assert Path(d01["release"]).resolve() == cohort / "profiles/rccl-single"
    assert d01["runtime_sha256"] == stage["profiles"]["rccl-single"]
    out = work / "evidence" / ("plan02-distributed-" + a.attempt)
    out.mkdir()
    os.environ["PATH"] = str(cohort / "test-tools") + ":" + os.environ["PATH"]
    runtime = json.loads((base / "current/runtime.json").read_text())
    report = dict(status="running", gates=[], rank=rank, stage=stage)
    try:
        cmd = [
            "/usr/local/bin/bc250-distributed",
            "--node-rank",
            str(rank),
            "--master-port",
            str(a.port),
            str(work / "tools/plan02/distributed_d01_rank.py"),
            "--rank",
            str(rank),
            "--backend",
            "nccl",
        ]
        report["gates"].append(gate("plan02-distributed-" + a.attempt, cmd, 8, runtime))
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (out / "final.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
