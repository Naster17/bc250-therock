#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Continue the declared D34 forward/reverse mode order after the single pilot."""
import json
from pathlib import Path
import subprocess


def main():
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d34-r1"
    assert (
        json.loads((root / "qualify-single/final.json").read_text())["status"]
        == "passed"
    )
    journal = []
    for index, label in enumerate(
        ["qualify-fsdp", "single-a", "fsdp-a", "fsdp-b", "single-b"]
    ):
        command = [
            "python3",
            str(Path(__file__).with_name("guard_distributed_d34_on_host.py")),
            "--mode",
            (
                label.split("-")[-1]
                if label.startswith("qualify-")
                else label.split("-")[0]
            ),
            "--phase",
            "qualify" if label.startswith("qualify-") else "measure",
            "--label",
            label,
            "--port",
            str(31211 + index),
        ]
        log = work / "evidence" / ("distributed-d34-" + label + "-controller.log")
        with log.open("x") as stream:
            child = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            record = dict(label=label, pid=child.pid, command=command, log=str(log))
            journal.append(record)
            (root / "sequence.json").write_text(json.dumps(journal, indent=2) + "\n")
            record["exit"] = child.wait()
        (root / "sequence.json").write_text(json.dumps(journal, indent=2) + "\n")
        assert record["exit"] == 0, record
        print(label, "passed", flush=True)


if __name__ == "__main__":
    main()
