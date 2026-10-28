#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify the complete pinned source snapshot and provisional ORT disk admission."""
import hashlib
import json
import os
from pathlib import Path
import tarfile

from bc250_migraphx import registered
from bc250_torch import LaunchOptions, runtime_lock
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS


def main():
    assert os.geteuid() == 0
    inputs = WS / "tmp/onnxruntime-b069/inputs"
    audit = json.loads((inputs / "source-audit.json").read_text())
    plan = json.loads((inputs / "candidate-plan.json").read_text())
    archive = inputs / "onnxruntime-source.tar.gz"
    assert (
        audit["status"] == "source-audit-passed"
        and audit["revision"] == "402979ac1bef89dcd55add41eaa7277c407599eb"
    )
    assert sha(archive) == audit["source_archive_sha256"]
    expected = set(audit["source_files_sha256"]) | set(audit["source_symlinks"])
    seen = set()
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            assert member.name in expected and member.name not in seen, member.name
            if member.isfile():
                with tar.extractfile(member) as stream:
                    assert (
                        hashlib.file_digest(stream, "sha256").hexdigest()
                        == audit["source_files_sha256"][member.name]
                    )
            else:
                assert (
                    member.issym()
                    and member.linkname == audit["source_symlinks"][member.name]
                )
            seen.add(member.name)
    assert seen == expected
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        record, _ = registered(LaunchOptions())
        fs = os.statvfs(WS)
        available = fs.f_bavail * fs.f_frsize
        assert available > plan["disk_admission_ordinary_bytes"]
        assert (
            available - plan["growth_cap_bytes"] > plan["rescue_floor_ordinary_bytes"]
        )
        receipt = dict(
            status="source-staged-budget-admitted-not-ORT-built",
            source_revision=audit["revision"],
            source_files_verified=len(audit["source_files_sha256"]),
            source_archive_sha256=audit["source_archive_sha256"],
            input_sha256={p.name: sha(p) for p in inputs.iterdir() if p.is_file()},
            ordinary_available_bytes=available,
            disk_plan=plan,
            installed_registration=record,
            ort_provider_built=False,
            installed_runtime_changed=False,
            host105_contacted=False,
        )
        atomic_json(WS / "logs/onnxruntime-b069-source-stage.json", receipt)
    print("ONNXRUNTIME_B069_SOURCE_STAGE_PASS", len(seen), available, flush=True)


if __name__ == "__main__":
    main()
