#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Accept the new registry through ordinary launchers; restore old profile on failure."""

import json
import os
from pathlib import Path
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS


def main():
    baseline = json.loads(
        (WS / "logs/quant-b053-qualification-r1/final.json").read_text()
    )
    record = json.loads((WS / "logs/quant-b053-deployment.json").read_text())
    assert (
        baseline["status"] == "passed"
        and record["status"] == "selected-awaiting-ordinary-acceptance"
    )
    manifest = json.loads(
        Path("/opt/bc250-stack/profiles/rccl-single/runtime.json").read_text()
    )
    links = dict(baseline["links"])
    links["/opt/bc250-stack/profiles/rccl-single"] = record["new_target"]
    report = dict(
        status="running",
        gates=[],
        protected_sha256=baseline["protected_sha256"],
        links=links,
        aot_caches=baseline["aot_caches"],
        boot_id=baseline["boot_id"],
        profile_sha256=record["new_profile_sha256"],
        daily_runtime_sha256=record["daily_runtime_sha256"],
    )
    try:
        for name, launcher, arguments, count in (
            (
                "layouts",
                "bc250-torch",
                [
                    "--runtime-profile",
                    "rccl-single",
                    "validate_quant_b052.py",
                    "--nccl-available",
                ],
                220,
            ),
            (
                "train",
                "bc250-train",
                [
                    "--runtime-profile",
                    "rccl-single",
                    "validate_rccl_b038.py",
                    "--consumers",
                ],
                6,
            ),
            (
                "rccl",
                "bc250-train",
                [
                    "--runtime-profile",
                    "rccl-single",
                    "validate_rccl_b035.py",
                    "--packaged",
                ],
                64,
            ),
            (
                "guards",
                "bc250-torch",
                [
                    "--runtime-profile",
                    "rccl-single",
                    "validate_rccl_b037.py",
                    "--packaged",
                ],
                76,
            ),
            (
                "daily",
                "bc250-train",
                ["validate_rccl_b038.py", "--daily", "--consumers"],
                5,
            ),
        ):
            command = [
                "/usr/local/bin/" + launcher,
                *[
                    str(WS / "helpers" / a) if a.endswith(".py") else a
                    for a in arguments
                ],
            ]
            report["gates"].append(
                gate("quant-b053-installed-" + name, command, count, manifest)
            )
        for name, digest in report["protected_sha256"].items():
            assert sha(Path(name)) == digest, name
        assert all(os.readlink(p) == v for p, v in links.items())
        assert (
            sha(Path("/opt/bc250-stack/current/runtime.json"))
            == record["daily_runtime_sha256"]
        )
        assert (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            == report["boot_id"]
        )
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        subprocess.run(
            [
                sys.executable,
                str(WS / "helpers/update_dev_prefix.py"),
                "--",
                sys.executable,
                str(WS / "helpers/deploy_quant_b053_on_host.py"),
                "--rollback",
            ],
            check=True,
        )
        raise
    finally:
        atomic_json(WS / "logs/quant-b053-ordinary.json", report)
    print("QUANT_B053_ORDINARY_PASS", flush=True)


if __name__ == "__main__":
    main()
