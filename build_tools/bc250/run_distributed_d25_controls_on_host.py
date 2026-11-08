#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sequential bounded FP16 overflow, clipping, compatibility and refusal controls."""
import json
from pathlib import Path
import subprocess


def main():
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d25-r3"
    assert json.loads((root / "fp16-seed/final.json").read_text())["status"] == "passed"
    journal = []
    for index, case in enumerate(
        ["overflow", "inactive", "fp32-control", "bf16-control", "late", "scale"]
    ):
        label = case + ("-refusal" if case in ["late", "scale"] else "-seed")
        command = [
            "python3",
            str(Path(__file__).with_name("guard_distributed_d25_on_host.py")),
            "--case",
            case,
            "--action",
            "seed",
            "--series",
            "r3",
            "--label",
            label,
            "--port",
            str(30353 + index),
        ]
        log = work / "evidence" / ("distributed-d25-r3-" + label + "-controller.log")
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
            (root / "controls-sequence.json").write_text(
                json.dumps(journal, indent=2) + "\n"
            )
            record["exit"] = child.wait()
        (root / "controls-sequence.json").write_text(
            json.dumps(journal, indent=2) + "\n"
        )
        assert record["exit"] == 0, record
        print(label, "passed", flush=True)


if __name__ == "__main__":
    main()
