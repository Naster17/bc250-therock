#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded inactive-clipping and collective refusal phases after the seed run."""
import json
from pathlib import Path
import subprocess


def main():
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d22-r1"
    assert json.loads((root / "fp32-seed/final.json").read_text())["status"] == "passed"
    journal = []
    for index, case in enumerate(["inf", "nan", "overflow"]):
        label = case + ("-refusal-r2" if case == "inf" else "-refusal")
        command = [
            "python3",
            str(Path(__file__).with_name("guard_distributed_d22_on_host.py")),
            "--case",
            case,
            "--action",
            "seed",
            "--series",
            "r1",
            "--label",
            label,
            "--port",
            str(30127 + index),
        ]
        log = work / "evidence" / ("distributed-d22-" + label + "-controller.log")
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
            (root / "controls-r2-sequence.json").write_text(
                json.dumps(journal, indent=2) + "\n"
            )
            record["exit"] = child.wait()
        (root / "controls-r2-sequence.json").write_text(
            json.dumps(journal, indent=2) + "\n"
        )
        assert record["exit"] == 0, record
        print(label, "passed", flush=True)


if __name__ == "__main__":
    main()
