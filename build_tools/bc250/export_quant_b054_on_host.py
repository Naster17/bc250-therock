#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export W8 measurements, accepted helper, rejected experiment and recovery records."""

import ast
import json
import os
from pathlib import Path
import subprocess
import tarfile

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS

SOURCES = (
    "bc250_quant_b051.py",
    "profile_quant_b054.py",
    "compare_quant_b054.py",
    "validate_quant_b054_recipe.py",
    "qualify_quant_b054_on_host.py",
    "deploy_quant_b054_on_host.py",
    "qualify_quant_b054_ordinary.py",
    "export_quant_b054_on_host.py",
)


def main():
    assert os.geteuid() == 0
    qualified = json.loads(
        (WS / "logs/quant-b054-qualification-r1/final.json").read_text()
    )
    ordinary = json.loads((WS / "logs/quant-b054-ordinary.json").read_text())
    assert qualified["status"] == ordinary["status"] == "passed"
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    for name, digest in ordinary["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    before = (WS / "logs/quant-b054-compare-r3/kernel-before.log").read_bytes()
    assert subprocess.check_output(["dmesg"]) == before
    current = [
        dict(
            label=g["label"], status=g["guard"]["status"], boot_id=g["guard"]["boot_id"]
        )
        for g in [*qualified["gates"], *ordinary["gates"]]
    ]
    assert len(current) == 8 and all(
        g["status"] == "acknowledged" and g["boot_id"] == ordinary["boot_id"]
        for g in current
    )
    historical = []
    for label in (
        "quant-b054-profile-r1",
        "quant-b054-compare-r1",
        "quant-b054-compare-r2",
    ):
        logs = WS / "logs" / label
        assert "GUARD_ACKNOWLEDGED" in (logs / "guard.log").read_text()
        assert (logs / "kernel-before.log").read_bytes() == (
            logs / "kernel-after.log"
        ).read_bytes()
        historical.append(
            dict(
                label=label,
                status="acknowledged-log-evidence",
                boot_id="5c04afbb-ff2f-4f4b-9eac-8238bb9288fc",
                live_state_cleared_by_user_reboot=True,
            )
        )
    old = (WS / "helpers/b054-baseline/bc250_quant_b051.py").read_text()
    new = (WS / "helpers/bc250_quant_b051.py").read_text()
    for name in ("pack", "_Linear"):
        original = next(
            n for n in ast.parse(old).body if getattr(n, "name", None) == name
        )
        active = next(
            n for n in ast.parse(new).body if getattr(n, "name", None) == name
        )
        assert ast.dump(original, include_attributes=False) == ast.dump(
            active, include_attributes=False
        )
    daily_compare = json.loads(
        (WS / "logs/quant-b054-compare-r3/result.json").read_text()
    )
    rccl_compare = json.loads(
        (WS / "logs/quant-b054-r1-compare-rccl/result.json").read_text()
    )
    destination = WS / "quant-b054"
    destination.mkdir()
    report = dict(
        status="passed",
        candidate_checks=sum(g["checks"] for g in qualified["gates"]),
        ordinary_checks=sum(g["checks"] for g in ordinary["gates"]),
        current_guards=current,
        historical_guards=historical,
        protected_paths=len(ordinary["protected_sha256"]),
        links=ordinary["links"],
        aot_caches=ordinary["aot_caches"],
        helper_sha256=sha(WS / "helpers/bc250_quant_b051.py"),
        original_helper_sha256=sha(WS / "helpers/b054-baseline/bc250_quant_b051.py"),
        source_sha256={n: sha(WS / "helpers" / n) for n in SOURCES},
        daily_runtime_sha256=sha(Path("/opt/bc250-stack/current/runtime.json")),
        profile_sha256=sha(Path("/opt/bc250-stack/profiles/rccl-single/runtime.json")),
        boot_id=ordinary["boot_id"],
        user_confirmed_restart=True,
        native_prefix_changed=False,
        native_build_required=False,
        host105_contacted=False,
        running_jobs=[],
        daily_timings=daily_compare["timings"],
        rccl_timings=rccl_compare["timings"],
        daily_trace=daily_compare["traces"],
        rccl_trace=rccl_compare["traces"],
        rollback="Original helper retained; identity-guarded helper restore not exercised",
    )
    assert report["candidate_checks"] == 333 and report["ordinary_checks"] == 142
    atomic_json(destination / "verification.json", report)
    archive = destination / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted((WS / "logs").glob("quant-b054-*")):
            tar.add(p, arcname="logs/" + p.name)
        for n in SOURCES:
            tar.add(WS / "helpers" / n, arcname="helpers/" + n)
        for n in ("b054-baseline", "b054-candidate", "b054-rejected-list"):
            tar.add(WS / "helpers" / n, arcname="versions/" + n)
        for p in sorted((WS / "helpers").glob("compare_quant_b054-r*.py")):
            tar.add(p, arcname="versions/" + p.name)
        tar.add(WS / "recovery/quant-b054-before", arcname="recovery")
        for g in current:
            tar.add(Path("/run") / g["label"], arcname="guards/" + g["label"])
        tar.add(destination / "verification.json", arcname="verification.json")
    atomic_json(
        destination / "archive.json",
        dict(sha256=sha(archive), bytes=archive.stat().st_size),
    )
    for p in destination.iterdir():
        os.chown(p, 1000, 1000)
    print("QUANT_B054_EXPORT_PASS", flush=True)


if __name__ == "__main__":
    main()
