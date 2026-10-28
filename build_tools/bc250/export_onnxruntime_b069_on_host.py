#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal source admission, isolated wrapper repair evidence and retained runtime gates."""
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
    assert os.geteuid() == 0
    logs = WS / "logs/onnxruntime-b069-qualification-r2"
    q = json.loads((logs / "final.json").read_text())
    assert (
        q["status"] == "passed"
        and len(q["gates"]) == 6
        and sum(g["checks"] for g in q["gates"]) == 38
    )
    inputs = WS / "tmp/onnxruntime-b069/inputs"
    staged = json.loads((WS / "logs/onnxruntime-b069-source-stage.json").read_text())
    audit = json.loads((inputs / "source-audit.json").read_text())
    built = WS / "tmp/onnxruntime-b069/probe-r4"
    build = json.loads((built / "final.json").read_text())
    assert staged["source_files_verified"] == 11379 and len(audit["checks"]) == 12
    dest = WS / "tmp/onnxruntime-b069/export-r1"
    dest.mkdir()
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        registered(LaunchOptions())
        registered(LaunchOptions(runtime_profile="rccl-single"))
        assert not verify_hashes(q["protected_sha256"])
        assert all(os.readlink(p) == v for p, v in q["links"].items())
        assert all(sha(inputs / n) == d for n, d in staged["input_sha256"].items())
        assert (
            subprocess.check_output(["dmesg"])
            == (logs / "kernel-before.log").read_bytes()
        )
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        guards = {}
        for path in Path("/run").glob("onnxruntime-b069-r*"):
            if not path.is_dir() or not (path / "state.json").exists():
                continue
            state = json.loads((path / "state.json").read_text())
            assert (
                state["status"] == "acknowledged" and state["boot_id"] == q["boot_id"]
            )
            guards[path.name] = path
        assert len(guards) == 10
        report = dict(
            status="audit-and-isolated-sdk-boundary-qualified",
            guarded_checks=38,
            source_audit_checks=12,
            c_api_symbols_linked=168,
            cpu_header_runs=2,
            accepted_guards=6,
            total_acknowledged_guards=len(guards),
            gpu_graph_evaluations=16,
            earlier_fixture_gpu_evaluations=16,
            ort_source_revision=audit["revision"],
            ort_source_files_verified=11379,
            ort_source_archive_sha256=audit["source_archive_sha256"],
            ort_source_archive_bytes=audit["source_archive_bytes"],
            ort_shallow_checkout=audit["shallow_repository"],
            uninitialized_gitlinks=audit["uninitialized_gitlinks"],
            migraphx_source_repair="134ee25f6",
            installed_migraphx_source="4797a61125bbec318005969f8497dbd50849cd6e",
            overlay_header_sha256=sha(built / "include/migraphx/migraphx.hpp"),
            native_protected_paths=len(q["protected_sha256"]),
            links=len(q["links"]),
            aot_caches=q["aot_caches"],
            active_runtime_changed=False,
            ort_provider_built=False,
            ort_provider_installed=False,
            kernel_log_unchanged=True,
            boot_id=q["boot_id"],
            kernel=os.uname().release,
            ordinary_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
            provisional_ort_build_disk_admission=staged["disk_plan"],
            host105_contacted=False,
            running_owned_jobs=[],
        )
        atomic_json(dest / "verification.json", report)
        with tarfile.open(dest / "evidence.tar.gz", "w:gz") as tar:
            for path in sorted((WS / "logs").glob("*onnxruntime-b069*")):
                tar.add(path, arcname="host-logs/" + path.name)
            for folder in sorted((WS / "tmp/onnxruntime-b069").glob("probe-r*")):
                for path in sorted(folder.iterdir()):
                    if path.is_file():
                        tar.add(path, arcname="build/" + folder.name + "/" + path.name)
                for relative in (
                    "include/migraphx/migraphx.hpp",
                    "build/admission_probe",
                    "build/header_cpp17",
                    "build/header_cpp20",
                    "build/CMakeCache.txt",
                ):
                    path = folder / relative
                    if path.is_file():
                        tar.add(path, arcname="build/" + folder.name + "/" + relative)
            for name, path in guards.items():
                tar.add(path, arcname="guards/" + name)
            for path in sorted((WS / "helpers").glob("*onnxruntime*b069*")):
                if path.is_file():
                    tar.add(path, arcname="helpers/" + path.name)
            for path in sorted((WS / "helpers").glob("migraphx-b069*")):
                if path.is_file():
                    tar.add(path, arcname="helpers/" + path.name)
            for name in (
                "source-audit.json",
                "dependencies.json",
                "candidate-plan.json",
            ):
                tar.add(inputs / name, arcname="inputs/" + name)
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
