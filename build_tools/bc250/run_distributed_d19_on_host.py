#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sequential D19 actual Trainer evaluate/predict in two models and three precisions."""
import json
import os
from pathlib import Path
import subprocess


def main() -> None:
    assert os.geteuid() == 0
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d19-r2"
    root.mkdir()
    journal = []
    for index, (case, action) in enumerate(
        [("bertwide-fp32", "eval"), ("gpt2wide-fp32", "eval")]
    ):
        label = case + "-" + action
        command = [
            "python3",
            str(Path(__file__).with_name("guard_distributed_d19_on_host.py")),
            "--series",
            "r2",
            "--case",
            case,
            "--action",
            action,
            "--label",
            label,
            "--port",
            str(30041 + index),
        ]
        log = work / "evidence" / ("distributed-d19-r2-" + label + "-controller.log")
        with log.open("x") as output:
            process = subprocess.Popen(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            record = dict(label=label, pid=process.pid, log=str(log), command=command)
            journal.append(record)
            (root / "sequence.json").write_text(json.dumps(journal, indent=2) + "\n")
            record["exit"] = process.wait()
        (root / "sequence.json").write_text(json.dumps(journal, indent=2) + "\n")
        assert record["exit"] == 0, record
        print(label, "passed", flush=True)


if __name__ == "__main__":
    main()
