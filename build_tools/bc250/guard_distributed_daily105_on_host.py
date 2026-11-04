#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root = Path("/home/nik/bc250work/distributed-d01-ordinary-r2")
    root.mkdir()
    os.chown(root, 1000, 1000)
    before = subprocess.check_output(["dmesg"])
    (root / "kernel-before.log").write_bytes(before)
    p = root / "guard"
    guard = subprocess.Popen(
        [
            "python3",
            "/home/nik/bc250work/kernel-r3-7.2.10/helpers/boot-guard.py",
            "--release",
            os.uname().release,
            "--phase",
            "distributed-ordinary",
            "--timeout",
            "600",
            "--directory",
            str(p),
        ],
        stdout=(root / "guard.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(100):
        if (p / "state.json").exists():
            break
        time.sleep(0.05)
    state = json.loads((p / "state.json").read_text())
    results = []
    try:
        for cmd in ["bc250-torch", "bc250-train"]:
            with (root / (cmd + ".log")).open("w") as log:
                subprocess.run(
                    [
                        "/usr/local/bin/" + cmd,
                        "/home/nik/bc250work/distributed_d01_daily_smoke.py",
                        "--output",
                        str(root / (cmd + ".json")),
                        *(["--consumers"] if cmd == "bc250-train" else []),
                    ],
                    check=True,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                )
            r = json.loads((root / (cmd + ".json")).read_text())
            assert r["status"] == "passed"
            results.append(r)
        assert before == subprocess.check_output(["dmesg"])
        (root / "final.json").write_text(
            json.dumps(
                dict(
                    status="passed",
                    checks=9,
                    results=results,
                    kernel_log_unchanged=True,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        (p / "ack.json").write_text(
            json.dumps({k: state[k] for k in ["boot_id", "nonce", "phase"]})
        )
        guard.wait(timeout=10)


if __name__ == "__main__":
    main()
