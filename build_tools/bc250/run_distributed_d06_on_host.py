#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the bounded D06 phases in opposite orders; stop at the first failed guard."""
import argparse
import json
import os
from pathlib import Path
import subprocess

PHASES = [
    (case, mode, f"{case}-{mode}-{order}")
    for case in ["small", "larger"]
    for order, modes in [
        ("a", ["single", "sync", "no-sync"]),
        ("b", ["no-sync", "sync", "single"]),
    ]
    for mode in modes
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--start-phase", choices=[v[2] for v in PHASES], default=PHASES[0][2]
    )
    a = p.parse_args()
    assert os.geteuid() == 0
    work = Path("/home/nik/bc250work")
    root = work / "evidence/distributed-d06-r1"
    root.mkdir(exist_ok=True)
    records = []
    journal = root / ("sequence-" + a.start_phase + ".json")
    assert not journal.exists()
    for index, (case, mode, label) in enumerate(PHASES):
        if index < [v[2] for v in PHASES].index(a.start_phase):
            continue
        assert not (root / label).exists()
        command = [
            "python3",
            str(Path(__file__).with_name("guard_distributed_d06_on_host.py")),
            "--case",
            case,
            "--mode",
            mode,
            "--label",
            label,
            "--port",
            str(29761 + index),
        ]
        log = work / "evidence" / f"distributed-d06-{label}-controller.log"
        with log.open("x") as output:
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            record = dict(label=label, pid=child.pid, log=str(log), command=command)
            records.append(record)
            journal.write_text(json.dumps(records, indent=2) + "\n")
            code = child.wait()
        record["exit"] = code
        journal.write_text(json.dumps(records, indent=2) + "\n")
        assert code == 0, record
        result = json.loads((root / label / "final.json").read_text())
        assert all(
            result[k]
            for k in [
                "agent_shutdown_clean",
                "kernel_log_unchanged",
                "registry_unchanged",
            ]
        )
        assert result["status"] == "passed"
        print(label, "passed", flush=True)


if __name__ == "__main__":
    main()
