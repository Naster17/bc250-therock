#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Six predeclared .104 mixed processes, independent recovery and owned-only stops."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from board_preflight import reservation, preflight

WORK = Path("/home/nik/bc250work")
TOOLS = WORK / "tools/mixed-d45"
ROOT = WORK / "evidence/mixed-d45-r1"
LAUNCHER_SHA = "a7b3a5b8db0bf178356271258da3b0dd6fb816461192e817a92a1c40d908a5cb"


def owned():
    return subprocess.check_output(
        ["podman", "ps", "-q", "--filter", "label=bc250.d45=mixed-r1"],
        text=True,
        timeout=10,
    ).split()


def resources():
    available = (
        int(
            next(
                s.split()[1]
                for s in Path("/proc/meminfo").read_text().splitlines()
                if s.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    return dict(available_bytes=available, disk_free_bytes=shutil.disk_usage(ROOT).free)


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    ROOT.mkdir(exist_ok=False)
    ROOT.chmod(0o755)
    assert not any(
        k in os.environ
        for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    )
    launcher = Path("/usr/local/bin/bc250-torch")
    assert hashlib.sha256(launcher.read_bytes()).hexdigest() == LAUNCHER_SHA
    registry = Path("/opt/bc250-stack/distributed/d01.json").read_bytes()
    result = dict(
        status="running",
        phases=[],
        kernel=os.uname().release,
        launcher_sha256=LAUNCHER_SHA,
        registry_sha256=hashlib.sha256(registry).hexdigest(),
        sources={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in TOOLS.glob("*.py")
        },
    )
    guard = None
    state = None
    with reservation("d45-mixed") as admission:
        result["admission"] = admission
        assert resources()["disk_free_bytes"] >= 5 * 1024**3
        guarddir = ROOT / "guard"
        with (ROOT / "guard.log").open("w") as log:
            guard = subprocess.Popen(
                [
                    "python3",
                    str(TOOLS / "deadline_guard_d42.py"),
                    "--release",
                    os.uname().release,
                    "--phase",
                    "d45-mixed",
                    "--timeout",
                    "600",
                    "--directory",
                    str(guarddir),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        try:
            for _ in range(100):
                if (guarddir / "state.json").exists():
                    break
                time.sleep(0.05)
            state = json.loads((guarddir / "state.json").read_text())
            assert state["status"] == "armed"
            result["boot_id"] = state["boot_id"]
            for index in range(6):
                profile = "daily" if index % 2 == 0 else "rccl-single"
                name = f"{index}-{profile}"
                folder = ROOT / name
                folder.mkdir()
                os.chown(folder, 1000, 1000)
                before = subprocess.check_output(["dmesg"])
                (folder / "kernel-before.log").write_bytes(before)
                command = [
                    "python3",
                    str(TOOLS / "mixed_d45_launch.py"),
                    str(launcher),
                    "--runtime-profile",
                    profile,
                    "--shm-size",
                    "256m",
                    str(TOOLS / "mixed_d45.py"),
                    "--seed",
                    str(4500 + index),
                    "--output",
                    str(folder / "worker.json"),
                ]
                phase = dict(
                    name=name,
                    profile=profile,
                    command=command,
                    samples=[],
                    containers=[],
                )
                result["phases"].append(phase)
                start = time.monotonic()
                cgroup = None
                with (folder / "worker.log").open("w") as log:
                    child = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    phase["child_pid"] = child.pid
                    try:
                        while child.poll() is None:
                            ids = owned()
                            phase["containers"] = sorted(
                                set(phase["containers"]) | set(ids)
                            )
                            if ids and cgroup is None:
                                assert len(ids) == 1
                                pid = subprocess.check_output(
                                    [
                                        "podman",
                                        "inspect",
                                        "--format",
                                        "{{.State.Pid}}",
                                        ids[0],
                                    ],
                                    text=True,
                                    timeout=10,
                                ).strip()
                                if pid != "0":
                                    try:
                                        cgroup = Path("/sys/fs/cgroup") / Path(
                                            "/proc", pid, "cgroup"
                                        ).read_text().split("::", 1)[1].strip().lstrip(
                                            "/"
                                        )
                                        assert (
                                            int((cgroup / "memory.max").read_text())
                                            == 4 * 1024**3
                                        )
                                        assert (
                                            int(
                                                (cgroup / "memory.swap.max").read_text()
                                            )
                                            == 0
                                        )
                                        phase["cgroup"] = str(cgroup)
                                    except FileNotFoundError:
                                        cgroup = None
                            sample = resources()
                            if cgroup is not None:
                                try:
                                    for key in (
                                        "memory.current",
                                        "memory.peak",
                                        "memory.swap.current",
                                    ):
                                        sample[key] = int((cgroup / key).read_text())
                                    sample["oom_kill"] = int(
                                        dict(
                                            line.split()
                                            for line in (cgroup / "memory.events")
                                            .read_text()
                                            .splitlines()
                                        )["oom_kill"]
                                    )
                                except FileNotFoundError:
                                    pass
                            phase["samples"].append(sample)
                            assert (
                                sample["available_bytes"] >= 3 * 1024**3
                                and sample["disk_free_bytes"] >= 5 * 1024**3
                            ), sample
                            assert (
                                sample.get("memory.swap.current", 0)
                                == sample.get("oom_kill", 0)
                                == 0
                            ), sample
                            assert time.monotonic() - start < 60, "Child deadline"
                            time.sleep(0.1)
                        phase["exit"] = child.returncode
                    finally:
                        for cid in owned():
                            subprocess.run(
                                ["podman", "stop", "--time", "5", cid],
                                check=True,
                                timeout=15,
                            )
                        if child.poll() is None:
                            child.terminate()
                        child.wait(timeout=15)
                phase["seconds"] = time.monotonic() - start
                after = subprocess.check_output(["dmesg"])
                (folder / "kernel-after.log").write_bytes(after)
                phase["kernel_unchanged"] = before == after
                assert phase["exit"] == 0 and phase["kernel_unchanged"], name
                assert phase.get("cgroup") and len(phase["containers"]) == 1
                phase["worker"] = json.loads((folder / "worker.json").read_text())
                assert phase["worker"]["status"] == "passed"
                phase["postflight"] = preflight("d45-after-" + name)
            assert (
                Path("/opt/bc250-stack/distributed/d01.json").read_bytes() == registry
            )
            assert hashlib.sha256(launcher.read_bytes()).hexdigest() == LAUNCHER_SHA
            result["status"] = "passed"
        except BaseException as error:
            result.update(status="failed", error=repr(error))
            raise
        finally:
            assert not owned(), "Owned workload remains; recovery stays armed"
            # A new foreign workload fails qualification but must not trigger a
            # recovery reboot once our own GPU work has already exited.
            try:
                result["postflight"] = preflight("d45-final")
            except Exception as error:
                result.update(status="failed", postflight_error=repr(error))
            assert state is not None
            (guarddir / "ack.json").write_text(
                json.dumps({k: state[k] for k in ("boot_id", "nonce", "phase")})
            )
            assert guard.wait(timeout=10) == 0
            result["guard"] = json.loads((guarddir / "state.json").read_text())
            (ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["status"] == "passed", result.get("postflight_error")


if __name__ == "__main__":
    main()
