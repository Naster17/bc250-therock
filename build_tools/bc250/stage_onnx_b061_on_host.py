#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage the optional ONNX export/CPU-runtime image and two isolated runtime candidates."""

import json
import os
from pathlib import Path
import shutil
import subprocess

from bc250_torch import runtime_lock
from stage_quant_b056_on_host import ACTIVE
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

ROOT = WS / "onnx-b061"


def main():
    preflight("onnx-b061-stage")
    assert os.geteuid() == 0
    assert shutil.disk_usage(WS).free > 512 * 1024**2
    baseline = json.loads((WS / "logs/quant-b056-installed-r3/final.json").read_text())
    assert baseline["status"] == "passed"
    ROOT.mkdir()
    logs = WS / "logs/onnx-b061-stage-r1"
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
            == "e2821cf8507467b22323272439d636f7c0918e27b8aa2790a5f524b10165a184"
        )
        context = ROOT / "context"
        context.mkdir()
        packages = json.loads((WS / "helpers/onnx_b061_packages.json").read_text())
        wheels = context / "wheels"
        wheels.mkdir()
        for item in packages.values():
            source = WS / "onnx-b061-input/wheels" / item["filename"]
            assert sha(source) == item["sha256"]
            shutil.copy2(source, wheels / source.name)
        (context / "requirements.txt").write_text(
            "".join(
                name
                + "=="
                + item["version"]
                + " --hash=sha256:"
                + item["sha256"]
                + "\n"
                for name, item in packages.items()
            )
        )
        (context / "Containerfile").write_text(
            "FROM " + image + "\nCOPY wheels /tmp/b061-wheels\n"
            "COPY requirements.txt /tmp/b061-requirements.txt\n"
            "RUN /opt/torch-venv/bin/python -m pip install --no-index --no-deps --no-cache-dir --require-hashes --find-links=/tmp/b061-wheels -r /tmp/b061-requirements.txt\n"
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
                    "localhost/bc250-stack:onnx-b061-r1",
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

        old, new = inspect(image), inspect("localhost/bc250-stack:onnx-b061-r1")
        layers = old["RootFS"]["Layers"]
        assert new["RootFS"]["Layers"][: len(layers)] == layers
        recovery = WS / "recovery/onnx-b061-before"
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
            manifest["onnx_b061"] = dict(
                packages={n: p["version"] for n, p in packages.items()},
                package_manifest_sha256=sha(WS / "helpers/onnx_b061_packages.json"),
                scope="Explicit modern ONNX export and CPUExecutionProvider inference; no GPU ONNX provider",
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
                packages=packages,
                context=str(context),
                candidates=candidates,
                recovery=str(recovery),
            ),
        )
    print("ONNX_B061_STAGE_PASS", flush=True)


if __name__ == "__main__":
    main()
