#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run under update_dev_prefix.py's exclusive lock; never nest a launcher lock."""

import argparse
import json
import os
from pathlib import Path
import pwd
import subprocess
import tempfile
import time

from bc250_torch import LaunchOptions, build_command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument(
        "--guard-release",
        choices=("7.2.10-bc250rocm-r3", "7.1.5-bc250rocm-r2"),
        default="7.2.10-bc250rocm-r3",
        help="Explicit kernel release for the existing acknowledgement guard",
    )
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--release", type=Path, default=Path("/opt/bc250-stack/current"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--installed", action="store_true")
    ap.add_argument("--integration", action="store_true")
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--script", help="Test basename staged in WS/helpers")
    ap.add_argument("--test-arg", action="append", default=[])
    ap.add_argument("--capture-core", action="store_true")
    ap.add_argument("--torch-candidate", type=Path)
    args = ap.parse_args()
    ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
    kws = Path("/home/nik/bc250work/kernel-r3-7.2.10")
    release = args.release.resolve()
    manifest = json.loads((release / "runtime.json").read_text())
    assert os.geteuid() == 0
    assert (
        os.uname().release == args.guard_release
    ), "Guard release must match running kernel"
    assert Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"
    assert not subprocess.check_output(
        ["podman", "ps", "-q"], text=True
    ).strip(), "Preserve other workloads"
    logs = ws / "logs" / args.label
    logs.mkdir()
    os.chown(logs, manifest["uid"], manifest["gid"])
    (logs / "kernel-before.log").write_bytes(subprocess.check_output(["dmesg"]))
    guard_dir = Path("/run") / args.label
    guard_log = (logs / "guard.log").open("w")
    guard = subprocess.Popen(
        [
            "python3",
            str(kws / "helpers/boot-guard.py"),
            "--release",
            args.guard_release,
            "--phase",
            args.label,
            "--timeout",
            "600",
            "--directory",
            str(guard_dir),
        ],
        stdout=guard_log,
        stderr=subprocess.STDOUT,
    )
    for _ in range(100):
        if (guard_dir / "state.json").exists():
            break
        if guard.poll() is not None:
            guard_log.close()
            raise RuntimeError(f"Guard exited before arming; see {logs / 'guard.log'}")
        time.sleep(0.05)
    if not (guard_dir / "state.json").exists():
        guard.terminate()
        guard.wait(timeout=5)
        guard_log.close()
        raise RuntimeError(f"Guard did not arm; see {logs / 'guard.log'}")
    state = json.loads((guard_dir / "state.json").read_text())
    assert state["status"] == "armed"
    test_script = (
        "validate_tiled_sdpa_integration.py"
        if args.integration
        else "validate_tiled_sdpa.py"
    )
    options = LaunchOptions(
        arguments=[
            str(ws / "helpers" / test_script),
            "--output",
            str(logs / "result.json"),
        ]
    )
    if args.regression:
        options.arguments = [str(kws / "helpers/torch-reference.py")]
    if args.script:
        assert Path(args.script).name == args.script
        options.arguments = [
            str(ws / "helpers" / args.script),
            "--output",
            str(logs / "result.json"),
            *args.test_arg,
        ]
    if (
        not args.installed
        and not args.integration
        and not args.regression
        and not args.script
    ):
        options.arguments += [
            "--candidate",
            str(ws / "helpers/gfx1013_attention_candidate.py"),
        ]
    if args.smoke:
        options.arguments += ["--smoke"]
    # Preserve pinned consumers; do not install or modify any packages.
    consumers = release / "python-consumers"
    options.mounts = [(str(consumers), "ro")]
    options.environment = dict(
        line.split("=", 1)
        for line in (consumers / "training.env").read_text().splitlines()
        if line
    )
    # build_command mounts this explicit lexical path; runtime.env may name
    # /opt/bc250-stack/current instead of the resolved development release.
    options.environment["PYTHONPATH"] = str(consumers / "site-packages")
    options.environment["HF_HUB_OFFLINE"] = "1"
    options.environment.update(item.split("=", 1) for item in args.env)
    account = pwd.getpwuid(manifest["uid"])
    groups = os.getgrouplist(account.pw_name, manifest["gid"])
    groups += [Path(p).stat().st_gid for p in ["/dev/kfd", "/dev/dri/renderD128"]]
    code = None
    name = args.label
    try:
        with tempfile.TemporaryDirectory(prefix="bc250-sdpa-env-") as tmp:
            envfile = Path(tmp) / "env"
            envfile.write_text(
                "".join(f"{k}={v}\n" for k, v in options.environment.items())
            )
            command = build_command(
                release, manifest, ws, Path(account.pw_dir), options, groups, envfile
            )
            # Explicit test environments may override the launcher's fixed paths
            # for an isolated, uninstalled candidate; never change daily defaults.
            image_index = command.index(manifest["image"])
            for item in args.env:
                command[image_index:image_index] = ["-e", item]
                image_index += 2
            if args.torch_candidate:
                matches = [
                    i
                    for i, item in enumerate(command)
                    if item.endswith(":/workspace/pytorch-work:ro")
                ]
                assert len(matches) == 1
                command[matches[0]] = (
                    str(args.torch_candidate.resolve(strict=True))
                    + ":/workspace/pytorch-work:ro"
                )
            command[2:2] = ["--name", name]
            if args.capture_core:
                command[2:2] = ["--ulimit", "core=268435456:268435456"]
            (logs / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            with (logs / "gpu.log").open("w") as output:
                code = subprocess.call(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=540,
                )
    finally:
        # Bound cleanup to this named container; never stop another workload.
        subprocess.run(
            ["podman", "stop", "--time", "3", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        after = subprocess.check_output(["dmesg"])
        (logs / "kernel-after.log").write_bytes(after)
        (logs / "exit.json").write_text(
            json.dumps(
                {
                    "returncode": code,
                    "kernel_log_unchanged": after
                    == (logs / "kernel-before.log").read_bytes(),
                }
            )
            + "\n"
        )
        # Returning normally over SSH after container exit completes this bounded
        # deadline. Test assertions remain failures in result/exit evidence.
        assert not subprocess.check_output(
            ["podman", "ps", "-q", "--filter", f"name=^{name}$"], text=True
        ).strip()
        (guard_dir / "ack.json").write_text(
            json.dumps({k: state[k] for k in ["boot_id", "nonce", "phase"]})
        )
        guard.wait(timeout=5)
        guard_log.close()
    raise SystemExit(code if code is not None else 1)


if __name__ == "__main__":
    main()
