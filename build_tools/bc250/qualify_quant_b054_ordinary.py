#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify actual helper imports through both ordinary runtime selections."""

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
        (WS / "logs/quant-b054-qualification-r1/final.json").read_text()
    )
    selected = json.loads((WS / "logs/quant-b054-deployment.json").read_text())
    protected = dict(baseline["protected_sha256"])
    target = str(WS / "helpers/bc250_quant_b051.py")
    assert protected[target] == selected["original_sha256"]
    protected[str(Path(selected["recovery"]) / "bc250_quant_b051.py")] = selected[
        "original_sha256"
    ]
    protected[target] = selected["candidate_sha256"]
    report = dict(
        status="running",
        gates=[],
        protected_sha256=protected,
        links=baseline["links"],
        aot_caches=baseline["aot_caches"],
        boot_id=baseline["boot_id"],
    )
    manifest = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    try:
        for name, arguments in (
            ("daily", []),
            ("rccl", ["--runtime-profile", "rccl-single"]),
        ):
            command = [
                "/usr/local/bin/bc250-torch",
                *arguments,
                str(WS / "helpers/validate_quant_b051.py"),
            ]
            report["gates"].append(
                gate("quant-b054-updated-" + name, command, 71, manifest)
            )
        for name, digest in protected.items():
            assert sha(Path(name)) == digest, name
        assert all(os.readlink(p) == v for p, v in baseline["links"].items())
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
                str(WS / "helpers/deploy_quant_b054_on_host.py"),
                "--rollback",
            ],
            check=True,
        )
        raise
    finally:
        atomic_json(WS / "logs/quant-b054-ordinary.json", report)
    print("QUANT_B054_ORDINARY_PASS", flush=True)


if __name__ == "__main__":
    main()
