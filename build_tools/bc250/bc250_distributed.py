#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Launch one node of the qualified two-board Ethernet Torch/RCCL job."""
import argparse
import json
import os
from pathlib import Path
import pwd
import sys
import tempfile
from bc250_torch import (
    LaunchOptions,
    build_command,
    exit_status,
    run_process,
    runtime_lock,
    read_context,
    INHERITED_NAMES,
    INHERITED_PREFIXES,
)

REGISTRY = Path("/opt/bc250-stack/distributed/d01.json")


def main():
    info = REGISTRY.stat()
    assert info.st_uid == 0 and not info.st_mode & 0o022 and not REGISTRY.is_symlink()
    record = json.loads(REGISTRY.read_text())
    release = Path(record["release"])
    manifest = json.loads((release / "runtime.json").read_text())
    import hashlib

    assert (
        hashlib.sha256((release / "runtime.json").read_bytes()).hexdigest()
        == record["runtime_sha256"]
    ), "runtime changed; requalification required"
    if sys.argv[1:2] == ["--_context"]:
        assert os.geteuid() == 0
        options, cwd = read_context(Path(sys.argv[2]), manifest["uid"])
    else:
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument("--node-rank", type=int, choices=(0, 1))
        p.add_argument("--master-port", type=int, default=29861)
        p.add_argument("--info", action="store_true")
        p.add_argument("script", nargs="?", type=Path)
        p.add_argument("arguments", nargs=argparse.REMAINDER)
        a = p.parse_args()
        if a.info:
            print(json.dumps(record, indent=2))
            return
        if a.node_rank is None or a.script is None:
            p.error("--node-rank and script are required")
        assert 1024 <= a.master_port <= 65535
        script = a.script.resolve(strict=True)
        cwd = Path.cwd().resolve()
        env = {
            k: v
            for k, v in os.environ.items()
            if k in INHERITED_NAMES or k.startswith(INHERITED_PREFIXES)
        }
        env.update(GLOO_SOCKET_IFNAME="eth0")
        options = LaunchOptions(
            arguments=[
                str(Path(__file__).resolve().with_name("bc250_torchrun.py")),
                "--nnodes=2",
                "--nproc-per-node=1",
                "--node-rank=" + str(a.node_rank),
                "--master-addr=192.168.1.104",
                "--master-port=" + str(a.master_port),
                str(script),
                *a.arguments,
            ],
            mounts=[
                (str(script.parent), "rw"),
                (str(Path(__file__).resolve().parent), "ro"),
            ],
            environment=env,
            runtime_profile="rccl-single",
        )
    if os.geteuid() != 0:
        assert os.geteuid() == manifest["uid"]
        with tempfile.TemporaryDirectory(prefix="bc250-distributed-") as folder:
            path = Path(folder) / "context.json"
            path.touch(mode=0o600)
            path.write_text(json.dumps(dict(workspace=str(cwd), options=vars(options))))
            raise SystemExit(
                exit_status(
                    run_process(
                        ["doas", str(Path(__file__).resolve()), "--_context", str(path)]
                    )
                )
            )
    account = pwd.getpwuid(manifest["uid"])
    groups = os.getgrouplist(account.pw_name, manifest["gid"]) + [
        Path(x).stat().st_gid for x in ["/dev/kfd", "/dev/dri/renderD128"]
    ]
    with runtime_lock(Path(record["lock"])):
        with tempfile.TemporaryDirectory(prefix="bc250-distributed-env-") as folder:
            envfile = Path(folder) / "environment"
            envfile.write_text(
                "".join(k + "=" + v + "\n" for k, v in options.environment.items())
            )
            command = build_command(
                release, manifest, cwd, Path(account.pw_dir), options, groups, envfile
            )
            command = [
                "NCCL_SOCKET_IFNAME=eth0" if x == "NCCL_SOCKET_IFNAME=lo" else x
                for x in command
            ]
            i = command.index(manifest["image"])
            command[i:i] = [
                "-e",
                "NCCL_ALGO=Ring",
                "-e",
                "NCCL_PROTO=Simple",
                "-e",
                "NCCL_NET_GDR_LEVEL=LOC",
            ]
            raise SystemExit(exit_status(run_process(command)))


if __name__ == "__main__":
    main()
