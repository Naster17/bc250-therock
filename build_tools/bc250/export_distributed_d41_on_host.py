#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reverify D41 raw references, export metadata only, and preserve original oracles."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
from board_preflight import reservation
from artifact_d35 import load_manifest
from artifact_d41 import validate
from capacity_plan_d35 import CONFIG, PARAMETERS


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d41-r1"
    with reservation("d41-final-export"):
        records = {}
        for label in ["control", "large-oracle"]:
            phase = root / label
            final = json.loads((phase / "final.json").read_text())
            worker = json.loads((phase / "rank.json").read_text())
            state = json.loads((phase / "guard/state.json").read_text())
            ack = json.loads((phase / "guard/ack.json").read_text())
            assert (
                final["status"] == worker["status"] == "passed"
                and final["worker"] == worker
            )
            assert (
                final["kernel_log_unchanged"]
                and final["registry_unchanged"]
                and final["agent_shutdown_clean"]
            )
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
            )
            records[label] = dict(final=final, worker=worker)
        assert not (root / "control/roundtrip-scratch").exists()
        manifest_path = root / "large-oracle/reference/manifest.json"
        large = validate(
            manifest_path, records["large-oracle"]["worker"]["manifest_sha256"]
        )
        assert len(large["tensors"]) == 244
        old = work / "evidence/distributed-d35-r1/cpu-reference"
        old_worker = json.loads((old / "rank.json").read_text())
        old_path = old / "reference/manifest.json"
        load_manifest(old_path, old_worker["manifest_sha256"], CONFIG, PARAMETERS)
        for source, name in [
            (old_path, "CPU-control-manifest.json"),
            (old / "rank.json", "CPU-control-worker.json"),
            (old / "final.json", "CPU-control-final.json"),
        ]:
            (root / name).write_bytes(source.read_bytes())
        result = dict(
            status="passed",
            records=records,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            kernel=os.uname().release,
            original_D35_reference_blobs_reverified=True,
            large_reference_blobs_reverified=True,
            large_reference_payload_bytes=sum(
                e[k]["bytes"] for e in large["tensors"] for k in ["initial", "gradient"]
            ),
            selected_daily=str(Path("/opt/bc250-stack/current").resolve()),
            GPU_admitted=False,
        )
        (root / "handoff.json").write_text(json.dumps(result, indent=2) + "\n")
        archive = work / "evidence/distributed-d41-evidence-r1.tar.gz"
        assert not archive.exists()
        with tarfile.open(archive, "w:gz", compresslevel=1) as t:
            for path in [
                root,
                work / "tools/distributed-d41",
                Path("/opt/bc250-stack/distributed/d01.json"),
            ]:
                t.add(
                    path,
                    arcname=str(path).lstrip("/"),
                    filter=lambda i: (
                        None
                        if "__pycache__" in i.name or i.name.endswith(".raw")
                        else i
                    ),
                )
            for path in (work / "evidence").glob("distributed-d41*controller.log"):
                t.add(path, arcname=str(path).lstrip("/"))
        print(
            json.dumps(
                dict(
                    path=str(archive),
                    bytes=archive.stat().st_size,
                    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                )
            )
        )


if __name__ == "__main__":
    main()
