#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export D35 measurements and exact guard/source evidence without user data."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
from board_preflight import reservation
from artifact_d35 import load_manifest
from capacity_plan_d35 import CONFIG, PARAMETERS

LABELS = [
    "cpu-reference",
    "qualify-single",
    "qualify-fsdp",
    "single-a",
    "fsdp-a",
    "fsdp-b",
    "single-b",
]


def main():
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d35-r1"
    with reservation("d35-final-export"):
        records = {}
        for label in LABELS:
            phase = root / label
            final = json.loads((phase / "final.json").read_text())
            worker = json.loads((phase / "rank.json").read_text())
            assert final["status"] == worker["status"] == "passed"
            assert (
                final["worker"] == worker
                and final["kernel_log_unchanged"]
                and final["registry_unchanged"]
                and final["agent_shutdown_clean"]
            )
            state = json.loads((phase / "guard/state.json").read_text())
            ack = json.loads((phase / "guard/ack.json").read_text())
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
            )
            records[label] = dict(final=final, worker=worker)
        manifest_path = root / "cpu-reference/reference/manifest.json"
        manifest = load_manifest(
            manifest_path,
            records["cpu-reference"]["worker"]["manifest_sha256"],
            CONFIG,
            PARAMETERS,
        )
        assert len(manifest["tensors"]) == 148
        assert not (root / "cpu-reference/actual-single").exists()
        assert not (root / "cpu-reference/actual-fsdp").exists()
        links = []
        for path in Path("/sys/class/net").glob("*/speed"):
            try:
                links.append(
                    dict(
                        interface=path.parent.name,
                        megabits_per_second=int(path.read_text()),
                    )
                )
            except (ValueError, OSError):
                pass
        handoff = dict(
            status="passed",
            final_reference_blobs_reverified=True,
            network_links=links,
            records=records,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            kernel=os.uname().release,
            selected_daily=str(Path("/opt/bc250-stack/current").resolve()),
        )
        (root / "handoff.json").write_text(json.dumps(handoff, indent=2) + "\n")
        archive = work / "evidence/distributed-d35-evidence-r1.tar.gz"
        assert not archive.exists()
        with tarfile.open(archive, "w:gz", compresslevel=1) as t:
            for path in [
                root,
                work / "tools/distributed-d35",
                Path("/opt/bc250-stack/distributed/d01.json"),
            ]:
                t.add(
                    path,
                    arcname=str(path).lstrip("/"),
                    filter=lambda info: (
                        None
                        if "__pycache__" in info.name or info.name.endswith(".raw")
                        else info
                    ),
                )
            for path in (work / "evidence").glob("distributed-d35*controller.log"):
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
