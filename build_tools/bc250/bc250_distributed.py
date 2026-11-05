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
    parse_launcher_arguments,
    validate_options,
    check_mount,
)

REGISTRY = Path("/opt/bc250-stack/distributed/d01.json")


DISTRIBUTED_ENVIRONMENT = {"GLOO_SOCKET_IFNAME": "eth0"}
RENDEZVOUS_NAMES = {
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "PYTHON_EXEC",
    "LD_PRELOAD",
    "LD_AUDIT",
    "HSA_OVERRIDE_GFX_VERSION",
    "HSA_XNACK",
}


class ForwardOption(argparse.Action):
    """Keep shared launcher option order, including interleaved env files/overrides."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        ordered = list(getattr(namespace, "shared_options", None) or [])
        ordered.extend([option_string, values])
        namespace.shared_options = ordered


def validate_distributed_options(options: LaunchOptions, tool: Path) -> None:
    """Revalidate the two-host contract after private context handoff as well."""
    validate_options(options)
    if options.shell or options.runtime_profile != "rccl-single":
        raise ValueError(
            "Distributed launch requires Python and the pinned RCCL profile"
        )
    for name, value in options.environment.items():
        if name in RENDEZVOUS_NAMES or name.startswith(
            ("NCCL_", "TORCH_NCCL_", "TORCHELASTIC_")
        ):
            raise ValueError(f"The distributed runtime controls {name}")
        if name in DISTRIBUTED_ENVIRONMENT and value != DISTRIBUTED_ENVIRONMENT[name]:
            raise ValueError(f"The distributed runtime controls {name}")
    args = options.arguments
    fixed = [str(tool / "bc250_torchrun.py"), "--nnodes=2", "--nproc-per-node=1"]
    if (
        len(args) < 8
        or args[:3] != fixed
        or args[3] not in ["--node-rank=0", "--node-rank=1"]
    ):
        raise ValueError("Invalid distributed worker topology")
    if args[4] != "--master-addr=192.168.1.104" or not args[5].startswith(
        "--master-port="
    ):
        raise ValueError("Invalid distributed rendezvous")
    port = int(args[5].partition("=")[2])
    if not 1024 <= port <= 65535 or args[6] != str(
        tool / "bc250_distributed_worker.py"
    ):
        raise ValueError("Invalid distributed worker bootstrap")
    script = Path(args[7])
    if not script.is_absolute() or not script.is_file():
        raise ValueError("Distributed script must be an existing absolute file")
    # Do not let a hand-authored private context omit mandatory tool/script mounts.
    modes = dict(options.mounts)
    if modes.get(str(tool)) != "ro" or str(script.parent) not in modes:
        raise ValueError("Missing validated tool/script mounts")
    for source, mode in options.mounts:
        path = Path(source)
        if path != tool and (tool.is_relative_to(path) or path.is_relative_to(tool)):
            raise ValueError("Mount overlaps the distributed launcher tools")
        if path == tool and mode != "ro":
            raise ValueError("Distributed launcher tools must be read-only")


def parse_distributed_arguments(
    arguments: list[str], environment: dict[str, str], tool: Path
):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-rank", type=int, choices=(0, 1))
    parser.add_argument("--master-port", type=int, default=29861)
    parser.add_argument("--info", action="store_true")
    for option in ["--mount", "--mount-ro", "--env", "--env-file", "--shm-size"]:
        parser.add_argument(option, action=ForwardOption, metavar="VALUE")
    parser.add_argument("script", nargs="?", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    if args.info:
        return None
    if args.node_rank is None or args.script is None:
        parser.error("--node-rank and script are required")
    script = args.script.expanduser().resolve(strict=True)
    check_mount(script.parent)
    options = parse_launcher_arguments(
        list(getattr(args, "shared_options", None) or [])
        + ["--runtime-profile", "rccl-single", "--", str(script), *args.arguments],
        environment,
    )
    for source, _ in options.mounts:
        path = Path(source)
        if path == tool or tool.is_relative_to(path) or path.is_relative_to(tool):
            raise ValueError("Mount overlaps the distributed launcher tools")
    options.arguments = [
        str(tool / "bc250_torchrun.py"),
        "--nnodes=2",
        "--nproc-per-node=1",
        f"--node-rank={args.node_rank}",
        "--master-addr=192.168.1.104",
        f"--master-port={args.master_port}",
        str(tool / "bc250_distributed_worker.py"),
        str(script),
        *args.arguments,
    ]
    # A caller's explicit read-only script parent takes precedence over the automatic rw mount.
    if str(script.parent) not in dict(options.mounts):
        ancestors = [
            (Path(source), mode)
            for source, mode in options.mounts
            if script.parent.is_relative_to(Path(source))
        ]
        mode = (
            max(ancestors, key=lambda item: len(item[0].parts))[1]
            if ancestors
            else "rw"
        )
        options.mounts.append((str(script.parent), mode))
    options.mounts.append((str(tool), "ro"))
    options.environment = {**DISTRIBUTED_ENVIRONMENT, **options.environment}
    validate_distributed_options(options, tool)
    return options


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
        options = parse_distributed_arguments(
            sys.argv[1:], dict(os.environ), Path(__file__).resolve().parent
        )
        if options is None:
            print(json.dumps(record, indent=2))
            return
        cwd = Path.cwd().resolve()
    validate_distributed_options(options, Path(__file__).resolve().parent)
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
