#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sequential D11 remaining BF16/FP16 seed and fresh native Accelerate replay phases."""
import json
import os
from pathlib import Path
import subprocess


def main() -> None:
    assert os.geteuid() == 0
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d11-r1"
    assert root.is_dir()
    journal = []
    for index, (precision, action) in enumerate(
        [(c, a) for c in ["bf16", "fp16"] for a in ["seed", "resume"]]
    ):
        label = precision + "-" + action
        command = [
            "python3",
            str(Path(__file__).with_name("guard_distributed_d11_on_host.py")),
            "--series",
            "r1",
            "--precision",
            precision,
            "--action",
            action,
            "--label",
            label,
            "--port",
            str(29933 + index),
        ]
        log = work / "evidence" / ("distributed-d11-r1-" + label + "-controller.log")
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
