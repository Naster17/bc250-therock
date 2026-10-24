#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage the optional W8 wheel/image and two isolated runtime candidates."""

import json
import os
from pathlib import Path
import shutil
import subprocess

from bc250_torch import runtime_lock
from build_quant_b056 import build
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

ROOT = WS / "quant-b056-r3"
ACTIVE = {
    "daily": Path("/opt/bc250-stack/current"),
    "rccl": Path("/opt/bc250-stack/profiles/rccl-single"),
}


def main():
    preflight("quant-b056-stage")
    assert os.geteuid() == 0
    assert shutil.disk_usage(WS).free > 512 * 1024**2
    baseline = json.loads(
        (WS / "logs/quant-b055-qualification-r1/final.json").read_text()
    )
    assert baseline["status"] == "passed"
    ROOT.mkdir()
    logs = WS / "logs/quant-b056-stage-r3"
    logs.mkdir()
    with runtime_lock(ACTIVE["daily"] / ".runtime.lock"):
        for name, digest in baseline["protected_sha256"].items():
            assert sha(Path(name)) == digest, name
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        manifests = {
            n: json.loads((p / "runtime.json").read_text()) for n, p in ACTIVE.items()
        }
        image = manifests["daily"]["image"]
        assert (
            image
            == manifests["rccl"]["image"]
            == "506f3b262fa5fe8c29b9babec07b03078dd6f85492a9ac2cffc65955297e4349"
        )
        context = ROOT / "context"
        wheel = build(WS / "helpers", context)
        containerfile = context / "Containerfile"
        containerfile.write_text(
            "FROM "
            + image
            + "\nCOPY "
            + wheel.name
            + " /tmp/"
            + wheel.name
            + "\nRUN /opt/torch-venv/bin/python -m pip install --no-index --no-deps --no-cache-dir /tmp/"
            + wheel.name
            + "\n"
        )
        with (logs / "image.log").open("w") as stream:
            subprocess.run(
                [
                    "podman",
                    "build",
                    "--pull=never",
                    "--network=none",
                    "--memory=1g",
                    "--cpu-period=100000",
                    "--cpu-quota=200000",
                    "--tag",
                    "localhost/bc250-stack:quant-b056-r3",
                    str(context),
                ],
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=180,
            )

        def inspect(name):
            return json.loads(
                subprocess.check_output(["podman", "image", "inspect", name], text=True)
            )[0]

        old, new = inspect(image), inspect("localhost/bc250-stack:quant-b056-r3")
        layers = old["RootFS"]["Layers"]
        assert new["RootFS"]["Layers"][: len(layers)] == layers
        recovery = WS / "recovery/quant-b056-before"
        recovery.mkdir()
        candidates = {}
        for name, active in ACTIVE.items():
            shutil.copy2(active / "runtime.json", recovery / (name + ".json"))
            candidate = ROOT / ("candidate-" + name)
            candidate.mkdir()
            for entry in active.resolve(strict=True).iterdir():
                if entry.name != "runtime.json":
                    (candidate / entry.name).symlink_to(entry)
            manifest = manifests[name]
            manifest["image"] = new["Id"].removeprefix("sha256:")
            manifest["quant_b056"] = dict(
                version="0.1.0",
                distribution="bc250-quant",
                wheel_sha256=sha(wheel),
                scope="Explicit bounded gfx1013 frozen W8 module, eager first derivatives; no automatic conversion",
            )
            atomic_json(candidate / "runtime.json", manifest)
            candidates[name] = dict(
                release=str(candidate),
                runtime_sha256=sha(candidate / "runtime.json"),
                previous_runtime_sha256=sha(recovery / (name + ".json")),
            )
        for name, digest in baseline["protected_sha256"].items():
            assert sha(Path(name)) == digest, name
        atomic_json(
            logs / "final.json",
            dict(
                status="packaged-not-qualified",
                image=new["Id"].removeprefix("sha256:"),
                previous_image=image,
                base_layers_preserved=True,
                wheel_sha256=sha(wheel),
                wheel=str(wheel),
                candidates=candidates,
                recovery=str(recovery),
            ),
        )
    print("QUANT_B056_STAGE_PASS", flush=True)


if __name__ == "__main__":
    main()
