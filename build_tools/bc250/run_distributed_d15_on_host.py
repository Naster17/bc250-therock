#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sequential D15 actual BF16/FP16 Trainer seed and fresh restart phases."""
import json
import os
from pathlib import Path
import subprocess


def main() -> None:
    assert os.geteuid() == 0
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d15-r1"
    root.mkdir()
    journal = []
    for index, (case, action) in enumerate(
        [
            ("bf16", "seed"),
            ("bf16", "resume"),
            ("fp16", "seed"),
            ("fp16", "resume"),
        ]
    ):
        label = case + "-" + action
        command = [
            "python3",
            str(Path(__file__).with_name("guard_distributed_d15_on_host.py")),
            "--series",
            "r1",
            "--case",
            case,
            "--action",
            action,
            "--label",
            label,
            "--port",
            str(29971 + index),
        ]
        log = work / "evidence" / ("distributed-d15-r1-" + label + "-controller.log")
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
