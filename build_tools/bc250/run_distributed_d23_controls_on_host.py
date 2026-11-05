#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded BF16 inactive clipping/NaN refusal and FP32 compatibility controls."""
import json
from pathlib import Path
import subprocess


def main():
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d23-r1"
    assert json.loads((root / "bf16-seed/final.json").read_text())["status"] == "passed"
    journal = []
    for index, case in enumerate(["inactive", "fp32-control", "nan"]):
        label = case + ("-refusal" if case == "nan" else "-seed")
        command = [
            "python3",
            str(Path(__file__).with_name("guard_distributed_d23_on_host.py")),
            "--case",
            case,
            "--action",
            "seed",
            "--series",
            "r1",
            "--label",
            label,
            "--port",
            str(30133 + index),
        ]
        log = work / "evidence" / ("distributed-d23-" + label + "-controller.log")
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
