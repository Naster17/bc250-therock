#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve frozen-policy references/refusals/replay and historical probe evidence."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
from board_preflight import reservation

PHASES = [
    ("r1", "frozen-seed", False),
    ("r2", "frozen-seed", True),
    *[
        ("r3", label, True)
        for label in [
            "frozen-seed",
            "frozen-resume",
            "frozen-missing-seed",
            "frozen-grad-seed",
            "frozen-late-seed",
            "frozen-peft-seed",
            "retained-control-seed",
            "frozen-policy-resume",
        ]
    ],
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d28-r3"
    with reservation("d28-final-export"):
        records = {}
        for series, label, accepted in PHASES:
            phase = work / "evidence" / ("distributed-d28-" + series) / label
            final = json.loads((phase / "final.json").read_text())
            worker = json.loads((phase / "rank.json").read_text())
            assert (
                final["status"]
                == worker["status"]
                == ("passed" if accepted else "failed")
            )
            assert final["kernel_log_unchanged"] and final["registry_unchanged"]
            assert final["docker_containers"] == dict(before=[], after=[])
            state = json.loads((phase / "guard/state.json").read_text())
            ack = json.loads((phase / "guard/ack.json").read_text())
            assert state["status"] == "acknowledged" and all(
                state[k] == ack[k] for k in ["nonce", "phase", "boot_id"]
            )
            if accepted:
                assert final["worker"] == worker and final["agent_shutdown_clean"]
            records[series + "/" + label] = dict(final=final, worker=worker)
            if "checkpoint_files" in worker:
                cp = phase / "trainer-output/checkpoint-2"
                assert all(
                    sha(cp / i["name"]) == i["sha256"]
                    for i in worker["checkpoint_files"]
                )
        cp = root / "frozen-seed/trainer-output/checkpoint-2"
        policy = json.loads((cp / "bc250-frozen-fp32.json").read_text())
        assert policy["schema"] == 1 and policy["precision"] == "FP32"
        payloads = {
            str(p.relative_to(cp)): dict(bytes=p.stat().st_size, sha256=sha(p))
            for folder in ["optimizer_0", "pytorch_model_fsdp_0"]
            for p in (cp / folder).iterdir()
        }
        assert len(payloads) == 6
        bad = (
            root
            / "frozen-policy-seed/trainer-output/checkpoint-2/bc250-frozen-fp32.json"
        )
        changed = json.loads(bad.read_text())
        wanted = dict(policy)
        if os.uname().nodename == "bc500":
            key = next(iter(wanted["peft_signature"]["layers"]))
            import copy

            wanted = copy.deepcopy(wanted)
            wanted["peft_signature"]["layers"][key]["scaling"]["default"] *= 2
        assert changed == wanted
        roots = [
            work / "evidence" / ("distributed-d28-" + s) for s in ["r1", "r2", "r3"]
        ]
        inventory = [
            dict(name=str(p.relative_to(work)), bytes=p.stat().st_size, sha256=sha(p))
            for r in roots
            for p in sorted(r.rglob("*"))
            if p.is_file() and p.suffix in [".pt", ".bin", ".distcp", ".safetensors"]
        ]
        handoff = dict(
            status="passed",
            records=records,
            payload_files=payloads,
            checkpoint_policy=policy,
            original_checkpoint_files_unchanged=True,
            rank1_only_private_policy_change_verified=True,
            payload_inventory=inventory,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            kernel=os.uname().release,
            registry_sha256=sha(Path("/opt/bc250-stack/distributed/d01.json")),
            running_containers=[],
        )
        dest = root / "handoff.json"
        assert not dest.exists()
        dest.write_text(json.dumps(handoff, indent=2) + "\n")
        archive = work / "evidence/distributed-d28-evidence-r3.tar.gz"
        assert not archive.exists()

        def keep(info):
            return (
                None
                if "/__pycache__" in info.name
                or Path(info.name).suffix
                in [".pt", ".pth", ".bin", ".distcp", ".gz", ".safetensors"]
                else info
            )

        with tarfile.open(archive, "w:gz", compresslevel=1) as t:
            for path in [
                *roots,
                work / "tools/distributed-d28",
                Path("/opt/bc250-stack/distributed/d01.json"),
            ]:
                t.add(path, arcname=str(path).lstrip("/"), filter=keep)
            for log in (work / "evidence").glob("distributed-d28*controller.log"):
                t.add(log, arcname=str(log).lstrip("/"))
        print(
            json.dumps(
                dict(
                    path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive)
                )
            )
        )


if __name__ == "__main__":
    main()
