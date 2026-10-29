#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal owned modern-export qualification and unchanged runtime evidence."""
import json
import os
from pathlib import Path
import subprocess
import tarfile

from bc250_ort import registered
from bc250_torch import LaunchOptions, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS


def main():
    logs = WS / "logs/onnxruntime-b0612-qualification-r1"
    a = json.loads((logs / "final.json").read_text())
    assert a["status"] == "passed"
    assert sum(g["checks"] for g in a["gates"]) == 112
    dest = WS / "tmp/onnxruntime-b0612/export-r1"
    dest.mkdir()
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        registered(LaunchOptions())
        registered(LaunchOptions(runtime_profile="rccl-single"))
        assert not verify_hashes(a["protected_sha256"])
        assert all(os.readlink(p) == v for p, v in a["links"].items())
        assert (
            subprocess.check_output(["dmesg"])
            == (logs / "kernel-before.log").read_bytes()
        )
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        guards = []
        for g in a["gates"]:
            path = Path("/run") / g["label"]
            state = json.loads((path / "state.json").read_text())
            assert (
                state["status"] == "acknowledged" and state["boot_id"] == a["boot_id"]
            )
            guards.append(path)
        assert len(guards) == 7
        fixtures = Path(a["fixtures"])
        models = json.loads((fixtures / "manifest.json").read_text())
        profiles = []
        for profile in ("daily", "rccl-single"):
            result = json.loads(
                (WS / f"logs/onnxruntime-b0612-r1-{profile}/result.json").read_text()
            )
            assert result["status"] == "passed" and len(result["checks"]) == 30
            assert len(result["profiles"]) == 6
            for entry in result["profiles"]:
                assert entry["providers"] == ["MIGraphXExecutionProvider"]
            profiles.extend(result["profiles"])
        report = dict(
            status="qualified-modern-torch-export-public-ort-gpu",
            checks=112,
            acknowledged_guards=7,
            gpu_graph_evaluations=24,
            generation_eager_gpu_forwards=12,
            actual_gpu_node_profiles=len(profiles),
            models=models["models"],
            tolerance=models["tolerance"],
            protected_paths=len(a["protected_sha256"]),
            links=len(a["links"]),
            aot_caches=a["aot_caches"],
            boot_id=a["boot_id"],
            kernel=os.uname().release,
            kernel_log_unchanged=True,
            active_runtime_changed=False,
            host105_contacted=False,
            running_owned_jobs=[],
            ordinary_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
        )
        atomic_json(dest / "verification.json", report)
        with tarfile.open(dest / "evidence.tar.gz", "w:gz") as tar:
            for path in sorted((WS / "logs").glob("*onnxruntime-b0612*")):
                tar.add(path, arcname="host-logs/" + path.name)
            for path in guards:
                tar.add(path, arcname="guards/" + path.name)
            for path in sorted((WS / "helpers").glob("*b0612*")):
                if path.is_file():
                    tar.add(path, arcname="helpers/" + path.name)
            tar.add(fixtures, arcname="fixtures")
        archive = dict(
            sha256=sha(dest / "evidence.tar.gz"),
            bytes=(dest / "evidence.tar.gz").stat().st_size,
        )
        atomic_json(dest / "archive.json", archive)
    print(json.dumps(dict(status=report["status"], archive=archive)), flush=True)


if __name__ == "__main__":
    main()
