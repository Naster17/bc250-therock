#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Accept public launchers after activation, with automatic identity-guarded rollback."""

import json
import os
from pathlib import Path
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS


def main():
    daily = Path("/opt/bc250-stack/current").resolve(strict=True)
    record = json.loads((WS / "logs/quant-b052-deployment.json").read_text())
    assert record["status"] == "activated-awaiting-ordinary-acceptance"
    baseline = json.loads(
        (WS / "logs/quant-b052-qualification-r1/final.json").read_text()
    )
    protected = dict(baseline["protected_sha256"])
    alias = "/opt/bc250-stack/current/runtime.json"
    assert protected[alias] == record["old_runtime_sha256"]
    protected[str(WS / "recovery/quant-b052-before/runtime.json")] = protected[alias]
    protected[alias] = record["runtime_sha256"]
    links = dict(baseline["links"])
    links["/opt/bc250-stack/current/torch"] = record["new_torch_target"]
    report = dict(
        status="running",
        gates=[],
        protected_sha256=protected,
        links=links,
        aot_caches=baseline["aot_caches"],
        boot_id=baseline["boot_id"],
        runtime_sha256=record["runtime_sha256"],
        torch_target=record["new_torch_target"],
    )
    manifest = json.loads((daily / "runtime.json").read_text())
    try:
        for name, launcher, args, count in (
            ("native", "bc250-torch", ["validate_quant_b052.py"], 220),
            (
                "train",
                "bc250-train",
                ["validate_rccl_b038.py", "--daily", "--consumers"],
                5,
            ),
            (
                "rccl",
                "bc250-train",
                [
                    "--runtime-profile",
                    "rccl-single",
                    "validate_rccl_b038.py",
                    "--consumers",
                ],
                6,
            ),
        ):
            command = [
                "/usr/local/bin/" + launcher,
                *[str(WS / "helpers" / a) if a.endswith(".py") else a for a in args],
            ]
            report["gates"].append(
                gate("quant-b052-installed-" + name, command, count, manifest)
            )
        for name, digest in protected.items():
            assert sha(Path(name)) == digest, name
        assert all(os.readlink(p) == v for p, v in links.items())
        assert (
            sha(Path("/opt/bc250-stack/profiles/rccl-single/runtime.json"))
            == record["opt_in_runtime_sha256"]
        )
        assert (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            == baseline["boot_id"]
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
                str(WS / "helpers/deploy_quant_b052_on_host.py"),
                "--rollback",
            ],
            check=True,
        )
        raise
    finally:
        atomic_json(WS / "logs/quant-b052-ordinary.json", report)
    print("QUANT_B052_ORDINARY_PASS", flush=True)


if __name__ == "__main__":
    main()
