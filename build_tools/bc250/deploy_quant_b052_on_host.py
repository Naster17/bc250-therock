#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Activate the qualified NCCL-off Torch fix under the daily exclusive lock."""

import argparse
import json
import os
from pathlib import Path
import shutil

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

RECOVERY = WS / "recovery/quant-b052-before"


def switch(release, target):
    temporary = release / "torch.b052-new"
    assert not temporary.exists() and not temporary.is_symlink()
    temporary.symlink_to(target)
    os.replace(temporary, release / "torch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    preflight("quant-b052-deploy")
    release = Path("/opt/bc250-stack/current").resolve(strict=True)
    if args.rollback:
        record = json.loads((WS / "logs/quant-b052-deployment.json").read_text())
        assert str((release / "torch").resolve()) == record["new_torch_target"]
        assert sha(release / "runtime.json") == record["runtime_sha256"]
        assert sha(RECOVERY / "runtime.json") == record["old_runtime_sha256"]
        switch(release, record["old_torch_target"])
        atomic_json(
            release / "runtime.json",
            json.loads((RECOVERY / "runtime.json").read_text()),
        )
        assert sha(release / "runtime.json") == record["old_runtime_sha256"]
        atomic_json(
            WS / "logs/quant-b052-rollback.json",
            dict(status="restored", old_torch_target=record["old_torch_target"]),
        )
        return
    qualified = json.loads(
        (WS / "logs/quant-b052-qualification-r1/final.json").read_text()
    )
    assert (
        qualified["status"] == "passed"
        and sum(g["checks"] for g in qualified["gates"]) == 947
    )
    for name, digest in qualified["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    assert all(os.readlink(p) == v for p, v in qualified["links"].items())
    RECOVERY.mkdir()
    old_target = os.readlink(release / "torch")
    shutil.copy2(release / "runtime.json", RECOVERY / "runtime.json")
    old_sha = sha(release / "runtime.json")
    candidate = WS / "pytorch-quant-b052"
    manifest = json.loads((release / "runtime.json").read_text())
    manifest["torch_revision"] = qualified["source"]["torch_revision"]
    manifest["source_state"] += (
        "; B05.2 native weight-only INT8 FP32 activation/scale layout normalization; NCCL-off/Gloo-on preserved"
    )
    manifest["quant_b052"] = dict(
        status="candidate qualified947/11; installed acceptance recorded separately",
        source=qualified["source"],
        native_sha256=qualified["native_sha256"],
        qualification=str(WS / "logs/quant-b052-qualification-r1/final.json"),
        deployment=str(WS / "logs/quant-b052-ordinary.json"),
        recovery=str(RECOVERY),
        scope="Native forward layout correctness only; native backward/INT4 unsupported; isolated RCCL profile still old/unfixed",
    )
    switch(release, str(candidate))
    try:
        atomic_json(release / "runtime.json", manifest)
    except BaseException:
        switch(release, old_target)
        raise
    atomic_json(
        WS / "logs/quant-b052-deployment.json",
        dict(
            status="activated-awaiting-ordinary-acceptance",
            old_torch_target=old_target,
            new_torch_target=str(candidate),
            old_runtime_sha256=old_sha,
            runtime_sha256=sha(release / "runtime.json"),
            opt_in_runtime_sha256=sha(
                Path("/opt/bc250-stack/profiles/rccl-single/runtime.json")
            ),
        ),
    )
    print("QUANT_B052_ACTIVATED", flush=True)


if __name__ == "__main__":
    main()
