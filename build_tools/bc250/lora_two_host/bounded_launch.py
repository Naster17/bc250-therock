#!/usr/bin/env python3
"""Apply the documented test resource envelope to one application phase."""
import json
import os
from pathlib import Path
import sys

assert os.geteuid() == 0
config = json.loads(Path(sys.argv[1]).read_text())
tools = Path("/usr/local/bin/bc250-distributed").resolve().parent
sys.path.insert(0, str(tools))
if config["mode"] == "ddp":
    import bc250_distributed as launcher
else:
    import bc250_torch as launcher
run = launcher.run_process


def bounded(command):
    assert command[:2] == ["podman", "run"]
    command[2:2] = [
        "--label=bc250.lora=" + config["label"],
        "--cpus=4",
        "--cpuset-cpus=0-3",
        "--memory=10g",
        "--memory-swap=12g",
        "--pids-limit=512",
    ]
    return run(command)


launcher.run_process = bounded
sys.argv = [str(tools / "bc250_distributed.py"), *config["argv"]]
launcher.main()
