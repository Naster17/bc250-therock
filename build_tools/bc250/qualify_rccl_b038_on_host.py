#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Guard ordinary installed launcher acceptance; never hold an exclusive lock."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from board_preflight import preflight, reserved
from docker_preflight import docker_containers


@reserved
def gate(label, command, count, manifest):
    preflight(label)
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    assert Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"
    logs = WS / "logs" / label
    logs.mkdir()
    os.chown(logs, manifest["uid"], manifest["gid"])
    command = [*command, "--output", str(logs / "result.json")]
    atomic_json(logs / "command.json", command)
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    guard_dir = Path("/run") / label
    guard_log = (logs / "guard.log").open("w")
    guard = subprocess.Popen(
        [
            "python3",
            "/home/nik/bc250work/kernel-r3-7.2.10/helpers/boot-guard.py",
            "--release",
            os.uname().release,
            "--phase",
            label,
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
        assert guard.poll() is None, "Recovery guard did not arm"
        time.sleep(0.05)
    state = json.loads((guard_dir / "state.json").read_text())
    assert state["status"] == "armed"
    code = None
    try:
        with (logs / "gpu.log").open("w") as log:
            proc = subprocess.Popen(
                command,
                cwd=WS,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = proc.wait(timeout=540)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)
                raise
    finally:
        after = subprocess.check_output(["dmesg"])
        (logs / "kernel-after.log").write_bytes(after)
        running = subprocess.check_output(["podman", "ps", "-q"], text=True).split()
        docker = None
        docker_error = None
        try:
            docker = docker_containers()
        except Exception as exc:
            docker_error = repr(exc)
        atomic_json(
            logs / "exit.json",
            dict(
                returncode=code,
                kernel_log_unchanged=before == after,
                running_containers=running,
                docker_containers=docker,
                docker_error=docker_error,
            ),
        )
        # ACK only after ordinary launcher exit and no remaining container.
        # Assertions remain failures; no other workload is stopped by this test.
        if not running:
            atomic_json(
                guard_dir / "ack.json",
                {k: state[k] for k in ("boot_id", "nonce", "phase")},
            )
            guard.wait(timeout=5)
        guard_log.close()
    assert (
        code == 0
        and before == after
        and not running
        and not docker
        and not docker_error
    ), label
    result = json.loads((logs / "result.json").read_text())
    assert result["status"] == "passed" and len(result["checks"]) == count, result
    final_guard = json.loads((guard_dir / "state.json").read_text())
    assert final_guard["status"] == "acknowledged"
    return dict(label=label, checks=count, command=command, guard=final_guard)


def main():
    # These legacy batch dependencies are not required by the reusable gate.
    # Deployment hosts intentionally do not contain the historical build helpers.
    from bc250_torch import runtime_lock
    from package_rccl_b038_on_host import PROFILE
    from qualify_rccl_b037_on_host import GATES

    preflight("rccl-b038-qualification")
    logs = WS / "logs/rccl-b038-qualification-r1"
    logs.mkdir()
    package = json.loads((WS / "logs/rccl-b038-package/final.json").read_text())
    assert package["status"] == "packaged-not-qualified"
    assert sha(PROFILE / "runtime.json") == package["profile_sha256"]
    baseline = json.loads(
        (WS / "logs/rccl-b037-qualification-r2/final.json").read_text()
    )
    protected = dict(baseline["protected_sha256"])
    # Exactly these two launcher copies intentionally change. Preserve their
    # original hashes/bytes in the package recovery record, not as false drift.
    changed = [Path("/usr/local/bin/bc250-torch"), WS / "helpers/bc250_torch.py"]
    for path in changed:
        assert sha(path) == package["launcher_sha256"]
        protected[str(path)] = sha(path)
    for path in (
        PROFILE / "runtime.json",
        WS / "helpers/validate_rccl_b038.py",
        *[
            WS / "helpers" / ("validate_rccl_b" + n + ".py")
            for n in ("035", "036", "037")
        ],
    ):
        protected[str(path)] = sha(path)
    for name, digest in package["native_identity_sha256"].items():
        protected[str(PROFILE / name)] = digest
    daily = Path("/opt/bc250-stack/current").resolve(strict=True)
    before = subprocess.check_output(["dmesg"])
    report = dict(
        status="running",
        gates=[],
        protected_sha256=protected,
        intentional_launcher_changes=[str(p) for p in changed],
        prior_baseline="rccl-b037-qualification-r2",
        profile=str(PROFILE),
        profile_sha256=package["profile_sha256"],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        links=baseline["links"],
        aot_caches=baseline["aot_caches"],
    )
    manifest = json.loads((PROFILE / "runtime.json").read_text())
    try:
        # Compatible nested SH locks: public launchers acquire SH too. Updaters
        # are excluded for the complete batch; no private lock bypass is used.
        with runtime_lock(daily / ".runtime.lock"):
            for name, digest in protected.items():
                assert sha(Path(name)) == digest, name
            plan = [
                (
                    "packaged-torch",
                    "bc250-torch",
                    False,
                    "validate_rccl_b038.py",
                    [],
                    5,
                ),
                (
                    "packaged-train",
                    "bc250-train",
                    False,
                    "validate_rccl_b038.py",
                    ["--consumers"],
                    6,
                ),
            ]
            for name, script, arguments, count in GATES:
                extra = (
                    ["--packaged"]
                    if script
                    in (
                        "validate_rccl_b035.py",
                        "validate_rccl_b036.py",
                        "validate_rccl_b037.py",
                    )
                    else []
                )
                plan.append(
                    (name, "bc250-train", False, script, [*arguments, *extra], count)
                )
            plan += [
                (
                    "daily-torch",
                    "bc250-torch",
                    True,
                    "validate_rccl_b038.py",
                    ["--daily"],
                    4,
                ),
                (
                    "daily-train",
                    "bc250-train",
                    True,
                    "validate_rccl_b038.py",
                    ["--daily", "--consumers"],
                    5,
                ),
            ]
            for name, launcher, is_daily, script, arguments, count in plan:
                command = ["/usr/local/bin/" + launcher]
                if not is_daily:
                    command += ["--runtime-profile", "rccl-single"]
                command += [str(WS / "helpers" / script), *arguments]
                accepted = gate("rccl-b038-r1-" + name, command, count, manifest)
                report["gates"].append(accepted)
                atomic_json(logs / "progress.json", report)
                print("PASS", name, count, flush=True)
            report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["changed_hashes"] = [
            p for p, d in protected.items() if sha(Path(p)) != d
        ]
        report["links_unchanged"] = all(
            os.readlink(p) == target for p, target in baseline["links"].items()
        )
        report["kernel_log_unchanged_entire_batch"] = (
            subprocess.check_output(["dmesg"]) == before
        )
        report["boot_unchanged"] = (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            == report["boot_id"]
        )
        report["running_containers"] = subprocess.check_output(
            ["podman", "ps", "-q"], text=True
        ).split()
        report["daily_runtime_sha256"] = sha(daily / "runtime.json")
        report["disk_available_bytes"] = (
            os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize
        )
        if (
            report["changed_hashes"]
            or not report["links_unchanged"]
            or not report["kernel_log_unchanged_entire_batch"]
            or not report["boot_unchanged"]
            or report["running_containers"]
            or report["daily_runtime_sha256"] != package["daily_runtime_sha256"]
        ):
            report["status"] = "failed"
        atomic_json(logs / "final.json", report)
    assert report["status"] == "passed"
    print("RCCL_B038_PASS", logs, flush=True)


if __name__ == "__main__":
    main()
