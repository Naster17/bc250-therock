#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify isolated strict W8 optimization against protected runtime cohorts."""

import json
import os
from pathlib import Path
import subprocess

from bc250_torch import runtime_lock
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    preflight("quant-b054-qualification")
    baseline = json.loads((WS / "logs/quant-b053-ordinary.json").read_text())
    assert baseline["status"] == "passed"
    protected = dict(baseline["protected_sha256"])
    for p in [
        *(WS / "helpers").glob("*quant_b054*.py"),
        WS / "helpers/b054-baseline/bc250_quant_b051.py",
        WS / "helpers/b054-candidate/bc250_quant_b051.py",
    ]:
        protected[str(p)] = sha(p)
    assert (
        sha(WS / "helpers/b054-baseline/bc250_quant_b051.py")
        == protected[str(WS / "helpers/bc250_quant_b051.py")]
    )
    logs = WS / "logs/quant-b054-qualification-r1"
    logs.mkdir()
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    # The longer paired comparison completed after the intervening board reboot.
    prior = WS / "logs/quant-b054-compare-r3"
    result = json.loads((prior / "result.json").read_text())
    assert result["status"] == "passed" and len(result["checks"]) == 90
    assert json.loads((prior / "exit.json").read_text()) == dict(
        returncode=0, kernel_log_unchanged=True, running_containers=[]
    )
    guard = json.loads(Path("/run/quant-b054-compare-r3/state.json").read_text())
    assert guard["status"] == "acknowledged" and guard["boot_id"] == boot
    report = dict(
        status="running",
        gates=[dict(label="quant-b054-compare-r3", checks=90, guard=guard)],
        protected_sha256=protected,
        links=baseline["links"],
        aot_caches=baseline["aot_caches"],
        boot_id=boot,
        baseline="quant-b053-ordinary",
        candidate_sha256=sha(WS / "helpers/b054-candidate/bc250_quant_b051.py"),
    )
    manifest = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    try:
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            for name, digest in protected.items():
                assert sha(Path(name)) == digest, name
            for name, launcher, args, count in (
                ("recipe-daily", "bc250-torch", ["validate_quant_b054_recipe.py"], 71),
                (
                    "recipe-rccl",
                    "bc250-torch",
                    [
                        "--runtime-profile",
                        "rccl-single",
                        "validate_quant_b054_recipe.py",
                    ],
                    71,
                ),
                (
                    "compare-rccl",
                    "bc250-torch",
                    [
                        "--runtime-profile",
                        "rccl-single",
                        "compare_quant_b054.py",
                        "--nccl-available",
                        "--pairs",
                        "15",
                        "--calls",
                        "30",
                    ],
                    90,
                ),
                (
                    "daily",
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
                    *[
                        str(WS / "helpers" / a) if a.endswith(".py") else a
                        for a in args
                    ],
                ]
                report["gates"].append(
                    gate("quant-b054-r1-" + name, command, count, manifest)
                )
                atomic_json(logs / "progress.json", report)
                print("PASS", name, count, flush=True)
            report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        after = subprocess.check_output(["dmesg"])
        (logs / "kernel-after.log").write_bytes(after)
        report["changed_hashes"] = [
            p for p, d in protected.items() if sha(Path(p)) != d
        ]
        report["links_unchanged"] = all(
            os.readlink(p) == v for p, v in baseline["links"].items()
        )
        report["kernel_log_unchanged_entire_batch"] = before == after
        report["boot_unchanged"] = (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip() == boot
        )
        report["running_containers"] = subprocess.check_output(
            ["podman", "ps", "-q"], text=True
        ).split()
        if (
            report["changed_hashes"]
            or not report["links_unchanged"]
            or not report["kernel_log_unchanged_entire_batch"]
            or not report["boot_unchanged"]
            or report["running_containers"]
        ):
            report["status"] = "failed"
        atomic_json(logs / "final.json", report)
    assert report["status"] == "passed"
    print("QUANT_B054_PASS", flush=True)


if __name__ == "__main__":
    main()
