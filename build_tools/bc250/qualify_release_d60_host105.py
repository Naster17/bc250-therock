#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded target-board regression using packaged launcher command construction.

D60 revision: selects runtime_profile from the release manifest so the
rccl-single profile gets its rccl mount/environment via the updated helper,
and passes --nccl-available to the distributed gate for that profile.
Deployed over the migration qualifier (original backed up in D60 evidence).
"""

import argparse
import json
import os
import pwd
import subprocess
import sys
import time
from pathlib import Path

from bc250_torch import LaunchOptions, build_command
from migrate_reference_host import digest

GATES = (
    ("core", "validate_cleanup_runtime.py", [], 7),
    ("distributed", "validate_distributed_b031.py", [], 31),
    ("sparse", "validate_sparse_b021.py", [], 253),
    ("fft", "validate_fft_b011.py", [], 245),
    ("vision", "validate_vision_b041.py", ["--group", "ops"], 51),
    ("audio", "validate_audio_b042.py", ["--group", "native", "--codec-available"], 40),
    ("compiled-dense", "validate_compiler_n310.py", ["--family", "dense"], 14),
    ("compiled-cnn", "validate_compiler_n310.py", ["--family", "cnn"], 14),
    ("profiler", "validate_distributed_profiler_b031.py", [], 1),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--core-only", action="store_true")
    args = ap.parse_args()
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    release = args.release.resolve(strict=True)
    manifest = json.loads((release / "runtime.json").read_text())
    assert manifest["validated"]
    args.output.mkdir()
    # Output and fresh compile caches belong only to the stack fixture user.
    os.chown(args.output, manifest["uid"], manifest["gid"])
    account = pwd.getpwuid(manifest["uid"])
    groups = os.getgrouplist(account.pw_name, manifest["gid"])
    groups += [Path(p).stat().st_gid for p in ("/dev/kfd", "/dev/dri/renderD128")]
    report = {
        "status": "running",
        "kernel": os.uname().release,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "runtime_sha256": digest(release / "runtime.json"),
        "gates": [],
        "checks": 0,
        "scope": "One target board; bounded CPU-reference outputs/first derivatives/optimizer updates and compile/profiler checks",
    }

    def save() -> None:
        (args.output / "final.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for name, filename, extra, expected in GATES[:1] if args.core_only else GATES:
            extra = list(extra)
            if name == "distributed" and manifest.get("rccl_single_profile"):
                extra.append("--nccl-available")
            directory = args.output / name
            directory.mkdir()
            os.chown(directory, manifest["uid"], manifest["gid"])
            before = subprocess.check_output(["dmesg"])
            (directory / "kernel-before.log").write_bytes(before)
            guard_dir = directory / "guard"
            guard = subprocess.Popen(
                [
                    sys.executable,
                    "/home/nik/bc250work/kernel-r3-7.2.10/helpers/boot-guard.py",
                    "--release",
                    os.uname().release,
                    "--phase",
                    name,
                    "--timeout",
                    "600",
                    "--directory",
                    str(guard_dir),
                ],
                stdout=(directory / "guard.log").open("w"),
                stderr=subprocess.STDOUT,
            )
            for _ in range(100):
                if (guard_dir / "state.json").exists():
                    break
                time.sleep(0.05)
            state = json.loads((guard_dir / "state.json").read_text())
            assert state["status"] == "armed" and not state["dry_run"]
            options = LaunchOptions(
                runtime_profile=(
                    "rccl-single" if manifest.get("rccl_single_profile") else "daily"
                ),
                mounts=[
                    (str(release / "python-consumers"), "ro"),
                    (str(args.output), "rw"),
                ],
                arguments=[
                    str(Path(__file__).with_name(filename)),
                    *extra,
                    "--output",
                    str(directory / "result.json"),
                ],
            )
            env = directory / "environment"
            env.write_text(
                "OMP_NUM_THREADS=4\nOPENBLAS_NUM_THREADS=1\n"
                + f"PYTHONPATH={release}/python-consumers/site-packages\n"
            )
            command = build_command(
                release,
                manifest,
                Path(__file__).parent,
                Path(account.pw_dir),
                options,
                groups,
                env,
            )
            command[2:2] = ["--cpus=4", "--memory=10g", "--memory-swap=12g"]
            (directory / "command.json").write_text(
                json.dumps(command, indent=2) + "\n"
            )
            with (directory / "console.log").open("w") as stream:
                result = subprocess.run(
                    command,
                    check=False,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=540,
                )
            assert result.returncode == 0, (name, result.returncode)
            result = json.loads((directory / "result.json").read_text())
            assert result["status"] == "passed" and len(result["checks"]) == expected, (
                name,
                result,
            )
            after = subprocess.check_output(["dmesg"])
            (directory / "kernel-after.log").write_bytes(after)
            assert before == after, "New kernel messages require review: " + name
            (guard_dir / "ack.json").write_text(
                json.dumps({k: state[k] for k in ("boot_id", "nonce", "phase")})
            )
            assert guard.wait(timeout=5) == 0
            assert (
                json.loads((guard_dir / "state.json").read_text())["status"]
                == "acknowledged"
            )
            report["gates"].append(
                {
                    "name": name,
                    "checks": expected,
                    "status": "passed",
                    "dmesg_unchanged": True,
                }
            )
            report["checks"] += expected
            save()
        assert digest(release / "runtime.json") == report["runtime_sha256"]
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()
    print("MIGRATED_HOST_QUALIFICATION_PASS", report["checks"], flush=True)


if __name__ == "__main__":
    main()
