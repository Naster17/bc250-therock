#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the serialized user context and installed profile refusals."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from bc250_torch import LaunchOptions, RCCL_SINGLE_ENVIRONMENT
from deploy_torch_a04_on_host import atomic_json
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-worker", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert os.geteuid() == 0
    manifest = json.loads(
        Path("/opt/bc250-stack/profiles/rccl-single/runtime.json").read_text()
    )
    if args.context_worker:
        assert args.output is not None
        options = LaunchOptions(
            runtime_profile="rccl-single",
            arguments=[
                str(WS / "helpers/validate_rccl_b038.py"),
                "--output",
                str(args.output),
            ],
        )
        # Same user-owned, mode0600 context consumed after normal doas elevation.
        # Root invokes the elevated half directly; no doas-policy changes.
        with tempfile.TemporaryDirectory(
            prefix="bc250-b038-context-", dir=WS / "experiments"
        ) as folder:
            directory = Path(folder)
            os.chown(directory, manifest["uid"], manifest["gid"])
            path = directory / "context.json"
            path.touch(mode=0o600)
            path.write_text(json.dumps(dict(workspace=str(WS), options=vars(options))))
            os.chown(path, manifest["uid"], manifest["gid"])
            subprocess.run(
                ["/usr/local/bin/bc250-torch", "--_context", str(path)],
                check=True,
                timeout=500,
            )
        return
    accepted = json.loads(
        (WS / "logs/rccl-b038-qualification-r1/final.json").read_text()
    )
    assert accepted["status"] == "passed"
    refusals = []
    for name in RCCL_SINGLE_ENVIRONMENT:
        command = [
            "/usr/local/bin/bc250-torch",
            "--runtime-profile",
            "rccl-single",
            "--env",
            name + "=outside-qualified-profile",
            "-c",
            'raise RuntimeError("must not run")',
        ]
        result = subprocess.run(
            command, cwd=WS, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 2 and f"controls {name}" in result.stderr, result
        refusals.append(
            dict(name=name, returncode=result.returncode, stderr=result.stderr)
        )
    result = subprocess.run(
        ["/usr/local/bin/bc250-torch", "--runtime-profile", "unknown", "-c", "pass"],
        cwd=WS,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2 and "Unknown runtime profile" in result.stderr
    refusals.append(
        dict(name="unknown-profile", returncode=result.returncode, stderr=result.stderr)
    )
    gate_result = gate(
        "rccl-b038-r1-private-context",
        ["python3", str(Path(__file__).resolve()), "--context-worker"],
        5,
        manifest,
    )
    unsafe_update = subprocess.run(
        [
            "python3",
            str(WS / "helpers/update_dev_prefix.py"),
            "--release",
            "/opt/bc250-stack/profiles/rccl-single",
            "--",
            "python3",
            "-c",
            "raise RuntimeError('must not run')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (
        unsafe_update.returncode != 0
        and "through its daily prefix lock" in unsafe_update.stderr
    )
    refusals.append(
        dict(
            name="independent-profile-update-lock",
            returncode=unsafe_update.returncode,
            stderr=unsafe_update.stderr,
        )
    )
    lockdir = WS / "logs/rccl-b038-lock-proof"
    lockdir.mkdir()
    os.chown(lockdir, manifest["uid"], manifest["gid"])
    ready = lockdir / "ready"
    code = f"import time; from pathlib import Path; Path({str(ready)!r}).touch(); time.sleep(3)"
    with (lockdir / "launcher.log").open("w") as log:
        proc = subprocess.Popen(
            [
                "/usr/local/bin/bc250-torch",
                "--runtime-profile",
                "rccl-single",
                "-c",
                code,
            ],
            cwd=WS,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while not ready.exists():
                assert proc.poll() is None and time.monotonic() < deadline
                time.sleep(0.05)
            with Path("/opt/bc250-stack/current/.runtime.lock").open("r+") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    raise AssertionError("Profile failed to hold daily shared lock")
            assert proc.wait(timeout=15) == 0
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
    atomic_json(
        WS / "logs/rccl-b038-context.json",
        dict(
            status="passed",
            refusals=refusals,
            gate=gate_result,
            daily_lock_exclusion=True,
        ),
    )
    print("RCCL_B038_CONTEXT_PASS", flush=True)


if __name__ == "__main__":
    main()
