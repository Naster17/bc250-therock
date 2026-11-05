#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sequential D17 actual Trainer evaluate/predict in two models and three precisions."""
import json
import os
from pathlib import Path
import subprocess


def main() -> None:
    assert os.geteuid() == 0
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d17-r1"
    root.mkdir()
    journal = []
    for index, (case, action) in enumerate(
        [
            (f"{model}-{precision}", "eval")
            for model in ["bert", "gpt2"]
            for precision in ["fp32", "bf16", "fp16"]
        ]
    ):
        label = case + "-" + action
        command = [
            "python3",
            str(Path(__file__).with_name("guard_distributed_d17_on_host.py")),
            "--series",
            "r1",
            "--case",
            case,
            "--action",
            action,
            "--label",
            label,
            "--port",
            str(29991 + index),
        ]
        log = work / "evidence" / ("distributed-d17-r1-" + label + "-controller.log")
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
