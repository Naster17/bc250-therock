#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Atomically select the qualified matched RCCL profile under the daily EX lock."""

import argparse
import json
import os
from pathlib import Path
import shutil

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_quant_b053_on_host import PROFILE
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

REGISTRY = Path("/opt/bc250-stack/profiles/rccl-single")
RECOVERY = WS / "recovery/quant-b053-before"


def select(target):
    temporary = REGISTRY.with_name("rccl-single.b053-new")
    assert not temporary.exists() and not temporary.is_symlink()
    temporary.symlink_to(target)
    os.replace(temporary, REGISTRY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    preflight("quant-b053-deploy")
    if args.rollback:
        record = json.loads((WS / "logs/quant-b053-deployment.json").read_text())
        assert REGISTRY.resolve(strict=True) == PROFILE
        assert sha(PROFILE / "runtime.json") == record["new_profile_sha256"]
        assert (
            sha(Path(record["old_target"]) / "runtime.json")
            == record["old_profile_sha256"]
        )
        select(record["old_target"])
        atomic_json(
            WS / "logs/quant-b053-rollback.json",
            dict(status="restored-old-profile", target=record["old_target"]),
        )
        return
    qualified = json.loads(
        (WS / "logs/quant-b053-qualification-r1/final.json").read_text()
    )
    assert (
        qualified["status"] == "passed"
        and sum(g["checks"] for g in qualified["gates"]) == 1732
    )
    for name, digest in qualified["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    assert all(os.readlink(p) == v for p, v in qualified["links"].items())
    old_target = os.readlink(REGISTRY)
    shutil.copy2(REGISTRY / "runtime.json", RECOVERY / "profile-runtime.json")
    (RECOVERY / "registry-target.txt").write_text(old_target + "\n")
    record = dict(
        status="selected-awaiting-ordinary-acceptance",
        old_target=old_target,
        new_target=str(PROFILE),
        old_profile_sha256=sha(REGISTRY / "runtime.json"),
        new_profile_sha256=sha(PROFILE / "runtime.json"),
        daily_runtime_sha256=sha(Path("/opt/bc250-stack/current/runtime.json")),
    )
    select(str(PROFILE))
    atomic_json(WS / "logs/quant-b053-deployment.json", record)
    print("QUANT_B053_SELECTED", flush=True)


if __name__ == "__main__":
    main()
