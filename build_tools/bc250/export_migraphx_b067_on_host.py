#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal relocated SDK evidence while retaining the distributable archive separately."""
import argparse
import re
import json
import os
from pathlib import Path
import subprocess
import tarfile

from bc250_torch import runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from package_migraphx_b067_on_host import STORAGE as DEFAULT_STORAGE
from qualify_training_n33_on_host import WS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    args = parser.parse_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    storage = args.storage
    logs = WS / "logs"
    qualified = json.loads(
        (
            logs / ("migraphx-b067-qualification-" + args.attempt) / "final.json"
        ).read_text()
    )
    assert (
        qualified["status"] == "passed"
        and sum(g["checks"] for g in qualified["gates"]) == 134
    )
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        assert not verify_hashes(qualified["protected_sha256"])
        assert (
            subprocess.check_output(["dmesg"])
            == (
                logs
                / ("migraphx-b067-qualification-" + args.attempt)
                / "kernel-before.log"
            ).read_bytes()
        )
        for label in qualified["gates"]:
            state = json.loads(
                (Path("/run") / label["label"] / "state.json").read_text()
            )
            assert (
                state["status"] == "acknowledged"
                and state["boot_id"] == qualified["boot_id"]
            )
        source = json.loads((storage / "source-r5.json").read_text())
        for name, digest in source["source_sha256"].items():
            assert sha(WS / "tmp/migraphx-b067/source-r5" / name) == digest, name

        def free(path):
            fs = os.statvfs(path)
            return dict(
                ordinary_available_bytes=fs.f_bavail * fs.f_frsize,
                free_including_reserved_bytes=fs.f_bfree * fs.f_frsize,
            )

        report = dict(
            status="qualified-isolated-relocatable-sdk",
            checks=134,
            gates=8,
            acknowledged_guards=8,
            source_revision=source["revision"],
            source_files_verified=len(source["source_sha256"]),
            protected_paths=len(qualified["protected_sha256"]),
            links=len(qualified["links"]),
            boot_id=qualified["boot_id"],
            root_space=free(WS),
            durable_space=free(storage),
            runtime_changed=False,
            ort_gpu_provider=False,
            installed=False,
            package=qualified["package"],
            external_preservation=qualified.get("external_preservation"),
            storage=str(storage),
            source_archive_sha256=source["archive_sha256"],
            host105_contacted=False,
        )
        dest = storage / ("export-" + args.attempt)
        dest.mkdir()
        atomic_json(dest / "verification.json", report)
        archive = dest / "evidence.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for path in sorted(logs.glob("*migraphx-b067*")):
                tar.add(path, arcname="host-logs/" + path.name)
            for pattern in ("logs-*", "consumer-*"):
                for folder in sorted(storage.glob(pattern)):
                    for path in sorted(folder.iterdir()):
                        if path.is_file():
                            tar.add(
                                path,
                                arcname="build-logs/" + folder.name + "/" + path.name,
                            )
            transfer = WS / "tmp/migraphx-b063/memory-transfer"
            for pattern in (
                "build-b067-*.log",
                "consumer-b067-*.log",
                "package-b067-*.log",
            ):
                for path in sorted(transfer.glob(pattern)):
                    tar.add(path, arcname="controller-logs/" + path.name)
            for directory in sorted(Path("/run").glob("migraphx-b067-*")):
                if directory.is_dir():
                    state = json.loads((directory / "state.json").read_text())
                    assert state["status"] == "acknowledged", str(directory)
                    tar.add(directory, arcname="guards/" + directory.name)
            for path in sorted((WS / "helpers").glob("*migraphx*b067*")):
                if path.is_file():
                    tar.add(path, arcname="helpers/" + path.name)
            tar.add(storage / "source-r5.json", arcname="source-r5.json")
            tar.add(dest / "verification.json", arcname="verification.json")
        atomic_json(
            dest / "archive.json",
            dict(sha256=sha(archive), bytes=archive.stat().st_size),
        )
    print("MIGRAPHX_B067_EXPORT_PASS", flush=True)


if __name__ == "__main__":
    main()
