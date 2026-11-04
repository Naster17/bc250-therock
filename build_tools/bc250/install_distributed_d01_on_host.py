#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Register the tested two-host opt-in command under the host's exclusive control lock."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import fcntl


def main():
    assert os.geteuid() == 0
    rank = 0 if os.uname().nodename == "bc250" else 1
    release = (
        Path("/opt/bc250-stack/profiles/rccl-single").resolve()
        if rank == 0
        else Path("/opt/bc250-stack/releases/10.2-gfx1013-20261104-d01/rccl-profile")
    )
    lock = (
        Path("/opt/bc250-stack/current/.runtime.lock")
        if rank == 0
        else Path("/opt/bc250-stack/migration-control/.runtime.lock")
    )
    result = json.loads(
        Path("/home/nik/bc250work/distributed-d02-rccl-r1/final.json").read_text()
    )
    assert result["status"] == "passed" and result["kernel_log_unchanged"]
    command = Path("/usr/local/bin/bc250-distributed")
    assert not command.exists() and not command.is_symlink()
    with lock.open("a+") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tool = Path("/opt/bc250-stack/tools/distributed-d01")
        tool.mkdir(parents=True)
        for name in ["bc250_distributed.py", "bc250_torch.py"]:
            shutil.copy2(
                Path("/home/nik/bc250work/distributed-d01-tools") / name, tool / name
            )
            (tool / name).chmod(0o555)
        registry = Path("/opt/bc250-stack/distributed")
        registry.mkdir(exist_ok=True)
        record = dict(
            release=str(release),
            runtime_sha256=hashlib.sha256(
                (release / "runtime.json").read_bytes()
            ).hexdigest(),
            lock=str(lock),
            qualified_hosts=["192.168.1.104", "192.168.1.105"],
            ranks=2,
            GPUs_per_host=1,
            backend="nccl/RCCL",
            transport="eth0 Socket Ring/Simple; P2P/IB/SDMA/symmetric off",
            scope="bounded four-dtype sums and global-batch three-step DDP/SGD; sharding/scaling unqualified",
            old_daily_unchanged=True,
        )
        (registry / "d01.json").write_text(json.dumps(record, indent=2) + "\n")
        (registry / "d01.json").chmod(0o444)
        command.symlink_to(tool / "bc250_distributed.py")


if __name__ == "__main__":
    main()
