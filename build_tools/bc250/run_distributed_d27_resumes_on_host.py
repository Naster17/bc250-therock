#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fresh replay plus rank1-only policy corruption in a preserved private copy."""
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d27-r3"
    for label in [
        "frozen-seed",
        "frozen-missing-seed",
        "frozen-grad-seed",
        "frozen-late-seed",
        "retained-control-seed",
    ]:
        assert (
            json.loads((root / label / "final.json").read_text())["status"] == "passed"
        )
    journal = []
    for index, case in enumerate(["frozen", "frozen-policy"]):
        if case == "frozen-policy":
            source = root / "frozen-seed"
            target = root / "frozen-policy-seed"
            assert not target.exists()
            shutil.copytree(source, target)
            if os.uname().nodename == "bc500":
                path = target / "trainer-output/checkpoint-2/bc250-frozen-fp32.json"
                value = json.loads(path.read_text())
                value["policy_sha256"] = "0" * 64
                path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        label = case + "-resume"
        command = [
            "python3",
            str(Path(__file__).with_name("guard_distributed_d27_on_host.py")),
            "--case",
            case,
            "--action",
            "resume",
            "--series",
            "r3",
            "--label",
            label,
            "--port",
            str(30521 + index),
        ]
        log = work / "evidence" / ("distributed-d27-r3-" + label + "-controller.log")
        with log.open("x") as stream:
            child = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            record = dict(
                case=case, label=label, pid=child.pid, command=command, log=str(log)
            )
            journal.append(record)
            (root / "resumes-sequence.json").write_text(
                json.dumps(journal, indent=2) + "\n"
            )
            record["exit"] = child.wait()
        (root / "resumes-sequence.json").write_text(
            json.dumps(journal, indent=2) + "\n"
        )
        assert record["exit"] == 0, record
        print(label, "passed", flush=True)


if __name__ == "__main__":
    main()
