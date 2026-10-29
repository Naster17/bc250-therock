#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal matching ORT/MIGraphX candidate evidence; do not install it."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile

from bc250_migraphx import registered
from bc250_torch import LaunchOptions, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qualification", required=True)
    args = p.parse_args()
    assert args.qualification in ("r1", "r2", "r3", "r4")
    base = WS / "tmp/onnxruntime-b0610"
    logs = WS / ("logs/onnxruntime-b0610-qualification-" + args.qualification)
    q = json.loads((logs / "final.json").read_text())
    assert q["status"] == "passed" and sum(g["checks"] for g in q["gates"]) == 118
    dest = base / ("export-" + args.qualification)
    dest.mkdir()
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        registration, _ = registered(LaunchOptions())
        assert not verify_hashes(q["protected_sha256"])
        assert all(os.readlink(p) == v for p, v in q["links"].items())
        assert (
            subprocess.check_output(["dmesg"])
            == (logs / "kernel-before.log").read_bytes()
        )
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        source = json.loads((base / "source-r2.json").read_text())
        assert not verify_hashes(
            {str(base / "source-r2" / n): d for n, d in source["files_sha256"].items()}
        )
        candidate = Path(q["ort_candidate"])
        identity = {
            str(p.relative_to(candidate)): sha(p)
            for p in (candidate / "onnxruntime").rglob("*")
            if p.is_file()
        }
        guards = {}
        for path in Path("/run").glob("onnxruntime-b0610-*"):
            if not (path / "state.json").exists():
                continue
            state = json.loads((path / "state.json").read_text())
            assert (
                state["status"] == "acknowledged" and state["boot_id"] == q["boot_id"]
            )
            guards[path.name] = path
        report = dict(
            status="qualified-isolated-ort-migraphx-candidate",
            checks=118,
            accepted_guards=len(q["gates"]),
            all_acknowledged_guards=len(guards),
            candidate=str(candidate),
            candidate_identity_sha256=identity,
            source_revision=source["repair"]["revision"],
            source_files_verified=len(source["files_sha256"]),
            protected_paths=len(q["protected_sha256"]),
            links=len(q["links"]),
            aot_caches=q["aot_caches"],
            registration=registration,
            installed=False,
            runtime_changed=False,
            host105_contacted=False,
            gpu_graph_evaluations=36,
            import_orders=["ort", "torch", "migraphx"],
            boot_id=q["boot_id"],
            kernel=os.uname().release,
            kernel_log_unchanged=True,
            running_owned_jobs=[],
            ordinary_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
        )
        atomic_json(dest / "verification.json", report)
        with tarfile.open(dest / "evidence.tar.gz", "w:gz") as tar:
            for path in sorted((WS / "logs").glob("*onnxruntime-b0610*")):
                tar.add(path, arcname="host-logs/" + path.name)
            for pattern in ("configure-*", "build-*", "smoke-*"):
                for folder in sorted(base.glob(pattern)):
                    for path in sorted(folder.iterdir()):
                        if path.is_file():
                            tar.add(
                                path, arcname="build/" + folder.name + "/" + path.name
                            )
            for name, path in guards.items():
                tar.add(path, arcname="guards/" + name)
            for path in sorted((WS / "helpers").glob("*onnxruntime*b0610*")):
                if path.is_file():
                    tar.add(path, arcname="helpers/" + path.name)
            tar.add(base / "mirror/manifest.json", arcname="dependencies.json")
            tar.add(base / "source-r2.json", arcname="source-patched.json")
            for name in ("CMakeCache.txt", "build.ninja"):
                tar.add(candidate / name, arcname="candidate/" + name)
        atomic_json(
            dest / "archive.json",
            dict(
                sha256=sha(dest / "evidence.tar.gz"),
                bytes=(dest / "evidence.tar.gz").stat().st_size,
            ),
        )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
