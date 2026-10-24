#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Update only the explicitly imported W8 helper under the daily EX lock."""

import argparse
import json
import os
from pathlib import Path
import shutil

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

TARGET = WS / "helpers/bc250_quant_b051.py"
RECOVERY = WS / "recovery/quant-b054-before"


def replace(source, uid, gid):
    temporary = TARGET.with_name("bc250_quant_b051.b054-new.py")
    assert not temporary.exists()
    shutil.copy2(source, temporary)
    os.chown(temporary, uid, gid)
    os.replace(temporary, TARGET)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    preflight("quant-b054-deploy")
    if args.rollback:
        record = json.loads((WS / "logs/quant-b054-deployment.json").read_text())
        assert sha(TARGET) == record["candidate_sha256"]
        assert sha(RECOVERY / TARGET.name) == record["original_sha256"]
        replace(RECOVERY / TARGET.name, record["uid"], record["gid"])
        atomic_json(
            WS / "logs/quant-b054-rollback.json",
            dict(status="restored-original-helper"),
        )
        return
    qualified = json.loads(
        (WS / "logs/quant-b054-qualification-r1/final.json").read_text()
    )
    assert (
        qualified["status"] == "passed"
        and sum(g["checks"] for g in qualified["gates"]) == 333
    )
    for name, digest in qualified["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    assert all(os.readlink(p) == v for p, v in qualified["links"].items())
    RECOVERY.mkdir()
    shutil.copy2(TARGET, RECOVERY / TARGET.name)
    info = TARGET.stat()
    original = sha(TARGET)
    candidate = WS / "helpers/b054-candidate" / TARGET.name
    assert sha(candidate) == qualified["candidate_sha256"]
    replace(candidate, info.st_uid, info.st_gid)
    atomic_json(
        WS / "logs/quant-b054-deployment.json",
        dict(
            status="updated-experimental-helper",
            original_sha256=original,
            candidate_sha256=sha(TARGET),
            uid=info.st_uid,
            gid=info.st_gid,
            recovery=str(RECOVERY),
            native_prefix_changed=False,
        ),
    )
    print("QUANT_B054_HELPER_UPDATED", flush=True)


if __name__ == "__main__":
    main()
