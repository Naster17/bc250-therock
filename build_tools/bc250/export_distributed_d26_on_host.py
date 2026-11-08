#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export owned D26 numerical/restart metadata without tensor or RNG payloads."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import tarfile
from board_preflight import reservation

PHASES = [
    case + "-" + action
    for case in ["fp32", "bf16", "fp16", "overflow"]
    for action in ["seed", "resume"]
] + ["bert-control-seed"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="r1")
    args = parser.parse_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.revision)
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d26-r1"
    with reservation("d26-final-export"):
        records, payloads = {}, {}
        for label in PHASES:
            phase = root / label
            final = json.loads((phase / "final.json").read_text())
            worker = json.loads((phase / "rank.json").read_text())
            assert final["status"] == worker["status"] == "passed"
            assert final["kernel_log_unchanged"] and final["registry_unchanged"]
            assert final["docker_containers"] == dict(before=[], after=[])
            assert final["worker"] == worker and final["agent_shutdown_clean"]
            state = json.loads((phase / "guard/state.json").read_text())
            ack = json.loads((phase / "guard/ack.json").read_text())
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
            )
            records[label] = dict(final=final, worker=worker)
            if label.endswith("seed") and label != "bert-control-seed":
                checkpoint = phase / "trainer-output/checkpoint-2"
                for item in worker["checkpoint_files"]:
                    assert sha(checkpoint / item["name"]) == item["sha256"]
                values = {
                    str(p.relative_to(checkpoint)): dict(
                        bytes=p.stat().st_size, sha256=sha(p)
                    )
                    for folder in ["optimizer_0", "pytorch_model_fsdp_0"]
                    for p in (checkpoint / folder).iterdir()
                }
                assert len(values) == 6
                payloads[label] = values
        inventory = [
            dict(name=str(p.relative_to(work)), bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in [".pt", ".pth", ".bin", ".distcp"]
        ]
        handoff = dict(
            status="passed",
            records=records,
            payload_files=payloads,
            payload_inventory=inventory,
            original_checkpoint_files_unchanged=True,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            kernel=os.uname().release,
            registry_sha256=sha(Path("/opt/bc250-stack/distributed/d01.json")),
            running_containers=[],
        )
        serialized = json.dumps(handoff, indent=2) + "\n"
        if (root / "handoff.json").exists():
            assert (root / "handoff.json").read_text() == serialized
        else:
            (root / "handoff.json").write_text(serialized)
        archive = work / (
            "evidence/distributed-d26-evidence-" + args.revision + ".tar.gz"
        )
        assert not archive.exists()

        def keep(info):
            return (
                None
                if "/__pycache__" in info.name
                or Path(info.name).suffix in [".pt", ".pth", ".bin", ".distcp", ".gz"]
                else info
            )

        with tarfile.open(archive, "w:gz", compresslevel=1) as t:
            for path in [
                root,
                work / "tools/distributed-d26",
                Path("/opt/bc250-stack/distributed/d01.json"),
            ]:
                t.add(path, arcname=str(path).lstrip("/"), filter=keep)
            for log in (work / "evidence").glob("distributed-d26*controller.log"):
                t.add(log, arcname=str(log).lstrip("/"))
        archive.chmod(0o644)
        print(
            json.dumps(
                dict(
                    path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive)
                )
            )
        )


if __name__ == "__main__":
    main()
