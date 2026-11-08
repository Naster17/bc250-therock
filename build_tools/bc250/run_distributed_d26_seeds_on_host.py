#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run bounded GPT2 precision seeds and the retained shared-harness default."""
import json
from pathlib import Path
import subprocess


def main():
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d26-r1"
    assert json.loads((root / "fp32-seed/final.json").read_text())["status"] == "passed"
    journal = []
    for index, case in enumerate(["bf16", "fp16", "overflow", "bert-control"]):
        label = case + "-seed"
        command = [
            "python3",
            str(Path(__file__).with_name("guard_distributed_d26_on_host.py")),
            "--case",
            case,
            "--action",
            "seed",
            "--series",
            "r1",
            "--label",
            label,
            "--port",
            str(30462 + index),
        ]
        log = work / "evidence" / ("distributed-d26-r1-" + label + "-controller.log")
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
            (root / "seeds-sequence.json").write_text(
                json.dumps(journal, indent=2) + "\n"
            )
            record["exit"] = child.wait()
        (root / "seeds-sequence.json").write_text(json.dumps(journal, indent=2) + "\n")
        assert record["exit"] == 0, record
        print(label, "passed", flush=True)


if __name__ == "__main__":
    main()
