#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Select qualified W8 image in both manifests under the daily exclusive lock."""

import argparse
import json
import os
from pathlib import Path
import shutil

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight
from stage_quant_b056_on_host import ACTIVE

RECORD = WS / "logs/quant-b056-deployment.json"


def replace(source, target):
    target = target.resolve(strict=True)
    temporary = target.with_name("runtime.quant-b056-new.json")
    assert not temporary.exists()
    info = target.stat()
    shutil.copy2(source, temporary)
    os.chown(temporary, info.st_uid, info.st_gid)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--accept", action="store_true")
    args = parser.parse_args()
    assert not (args.rollback and args.accept)
    preflight("quant-b056-deploy")
    assert os.geteuid() == 0
    stage = json.loads((WS / "logs/quant-b056-stage-r3/final.json").read_text())
    recovery = Path(stage["recovery"])
    if args.rollback or args.accept:
        record = json.loads(RECORD.read_text())
        for name, active in ACTIVE.items():
            assert (
                sha(active / "runtime.json") == record["installed_runtime_sha256"][name]
            )
            assert (
                sha(recovery / (name + ".json"))
                == stage["candidates"][name]["previous_runtime_sha256"]
            )
        if args.rollback:
            for name, active in ACTIVE.items():
                replace(recovery / (name + ".json"), active / "runtime.json")
            record.update(status="rolled-back", rollback_used=True)
        else:
            result = json.loads(
                (WS / "logs/quant-b056-installed-r1/final.json").read_text()
            )
            assert (
                result["status"] == "passed"
                and sum(g["checks"] for g in result["gates"]) == 808
            )
            for path, digest in result["protected_sha256"].items():
                assert sha(Path(path)) == digest, path
            record.update(status="passed", installed_checks=808)
        atomic_json(RECORD, record)
        print("QUANT_B056_" + ("ROLLBACK" if args.rollback else "ACCEPT") + "_PASS")
        return
    assert not RECORD.exists()
    qualified = json.loads((WS / "logs/quant-b056-candidate-r1/final.json").read_text())
    assert (
        qualified["status"] == "passed"
        and sum(g["checks"] for g in qualified["gates"]) == 808
    )
    for path, digest in qualified["protected_sha256"].items():
        assert sha(Path(path)) == digest, path
    assert all(os.readlink(p) == v for p, v in qualified["links"].items())
    record = dict(
        status="installing",
        installed_runtime_sha256={},
        recovery=str(recovery),
        image=stage["image"],
        previous_image=stage["previous_image"],
        candidate_checks=808,
        rollback_used=False,
        native_libraries_changed=False,
    )
    # Validate both generations before touching either manifest. Preserve each
    # old manifest; on a partial update, restore only our successfully changed one.
    for name, active in ACTIVE.items():
        assert (
            sha(active / "runtime.json")
            == sha(recovery / (name + ".json"))
            == stage["candidates"][name]["previous_runtime_sha256"]
        )
        assert (
            sha(Path(stage["candidates"][name]["release"]) / "runtime.json")
            == stage["candidates"][name]["runtime_sha256"]
        )
    atomic_json(RECORD, record)
    changed = []
    try:
        for name, active in ACTIVE.items():
            replace(
                Path(stage["candidates"][name]["release"]) / "runtime.json",
                active / "runtime.json",
            )
            changed.append(name)
            record["installed_runtime_sha256"][name] = sha(active / "runtime.json")
            atomic_json(RECORD, record)
        record["status"] = "installed-awaiting-acceptance"
    except BaseException as error:
        for name in reversed(changed):
            assert (
                sha(ACTIVE[name] / "runtime.json")
                == stage["candidates"][name]["runtime_sha256"]
            )
            replace(recovery / (name + ".json"), ACTIVE[name] / "runtime.json")
        record.update(status="failed-restored", rollback_used=True, error=repr(error))
        raise
    finally:
        atomic_json(RECORD, record)
    print("QUANT_B056_INSTALLED", flush=True)


if __name__ == "__main__":
    main()
