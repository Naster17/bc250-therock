#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Guard one rank of an owned two-host socket/GPU experiment."""
import argparse
import fcntl
import json
import pwd
import sys
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--tools", type=Path, required=True)
    a = p.parse_args()
    root = Path("/home/nik/bc250work/distributed-d02-rccl-r1")
    root.mkdir()
    os.chown(root, 1000, 1000)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            links = [str(x.readlink()) for x in (proc / "fd").iterdir()]
            if any("/dev/kfd" in x or "/dev/dri/render" in x for x in links):
                assert proc.stat().st_uid == 0 and (
                    proc / "comm"
                ).read_text().startswith("cyan-skillfish"), proc.name
        except OSError:
            pass
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    assert Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"
    lock = (
        Path("/opt/bc250-stack/current/.runtime.lock")
        if a.rank == 0
        else Path("/opt/bc250-stack/migration-control/.runtime.lock")
    )
    before = subprocess.check_output(["dmesg"])
    (root / "kernel-before.log").write_bytes(before)
    script = (
        "/home/nik/bc250work/rocm-10.2-gfx1013/helpers/distributed_d01_rank.py"
        if a.rank == 0
        else "/home/nik/bc250work/distributed_d01_rank.py"
    )
    sys.path.insert(0, str(a.tools))
    from bc250_torch import LaunchOptions, build_command

    release = a.release.resolve(strict=True)
    manifest = json.loads((release / "runtime.json").read_text())
    options = LaunchOptions(
        arguments=[
            script,
            "--rank",
            str(a.rank),
            "--backend",
            "nccl",
            "--port",
            "29842",
            "--output",
            str(root / "rank.json"),
        ],
        mounts=[(str(root), "rw"), (str(Path(script).parent), "ro")],
        environment={"GLOO_SOCKET_IFNAME": "eth0", "NCCL_DEBUG": "INFO"},
        runtime_profile="rccl-single",
    )
    envfile = root / "environment"
    envfile.write_text(
        "".join(k + "=" + v + "\n" for k, v in options.environment.items())
    )
    groups = os.getgrouplist(pwd.getpwuid(manifest["uid"]).pw_name, manifest["gid"])
    groups += [Path(x).stat().st_gid for x in ("/dev/kfd", "/dev/dri/renderD128")]
    command = build_command(release, manifest, root, root, options, groups, envfile)
    # Explicit experiment only: validate original single-device identity/settings,
    # then broaden socket interface on this one command, not the registered profile.
    command = [
        "NCCL_SOCKET_IFNAME=eth0" if x == "NCCL_SOCKET_IFNAME=lo" else x
        for x in command
    ]
    image_index = command.index(manifest["image"])
    command[image_index:image_index] = [
        "-e",
        "NCCL_ALGO=Ring",
        "-e",
        "NCCL_PROTO=Simple",
        "-e",
        "NCCL_NET_GDR_LEVEL=LOC",
    ]
    guarddir = root / "guard"
    guard = subprocess.Popen(
        [
            "python3",
            "/home/nik/bc250work/kernel-r3-7.2.10/helpers/boot-guard.py",
            "--release",
            os.uname().release,
            "--phase",
            "distributed-d02",
            "--timeout",
            "600",
            "--directory",
            str(guarddir),
        ],
        stdout=(root / "guard.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(100):
        if (guarddir / "state.json").exists():
            break
        time.sleep(0.05)
    state = json.loads((guarddir / "state.json").read_text())
    assert state["status"] == "armed"
    result = dict(
        status="running",
        command=command,
        rank=a.rank,
        boot_id=state["boot_id"],
        kernel=os.uname().release,
    )
    try:
        with lock.open("a+") as lockfile:
            fcntl.flock(lockfile, fcntl.LOCK_SH | fcntl.LOCK_NB)
            with (root / "gpu.log").open("w") as log:
                subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=240,
                )
            result["rank_result"] = json.loads((root / "rank.json").read_text())
            assert result["rank_result"]["status"] == "passed"
            result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        after = subprocess.check_output(["dmesg"])
        (root / "kernel-after.log").write_bytes(after)
        result["kernel_log_unchanged"] = before == after
        if before != after:
            result["status"] = "failed"
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        (guarddir / "ack.json").write_text(
            json.dumps({k: state[k] for k in ["boot_id", "nonce", "phase"]})
        )
        guard.wait(timeout=10)
        (root / "final.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["status"] == "passed"


if __name__ == "__main__":
    main()
