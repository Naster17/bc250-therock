#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D59: qualify the D58 release by independent-path activation and rollback.

Runs as root on bc250 (.104). Activates release 10.2-gfx1013-20261111-138a66f52
through the public pointer/registry/helper mechanism, runs the ordinary
bc250-* caller gates plus identity refusals, then rolls the selection back to
the D49 cohort. The release keeps validated=true only if every gate passes;
the additive helper allowlist stays deployed (it still binds the D49 cohort).
"""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

from board_preflight import preflight

VERSION = "10.2-gfx1013-20261111-138a66f52"
BASE = Path("/opt/bc250-stack")
WORK = Path("/home/nik/bc250work")
WS = WORK / "rocm-10.2-gfx1013"
RELEASE = BASE / "releases" / VERSION
TOOLS = Path(__file__).resolve().parent
EVIDENCE = WORK / "evidence/release-d59"
FIXTURES = WS / "helpers"
BOOT_GUARD = WORK / "kernel-r3-7.2.10/helpers/boot-guard.py"
REGISTRY = BASE / "distributed/d01.json"
HELPERS = [
    Path("/usr/local/bin/bc250-torch"),
    BASE / "tools/distributed-d09/bc250_torch.py",
]
POINTERS = {
    BASE / "current": RELEASE / "profiles/daily",
    BASE / "profiles/rccl-single": RELEASE / "profiles/rccl-single",
}
SDK_SHA = "2a975370ba4eb1c2fe39e406ff36b175f554a49d0aa19febbd326f641fcb590b"
FINAL_RUNTIME = {
    "daily": "86d94f43f629b84572c2fc8639ef64beb96b32647439533f97260c7cd4e243fc",
    "rccl-single": "f183bd3f22ec55dcd1b1e10b44afb94f9edcb076b3224c34ad630f993fb0d67c",
}
D49_DAILY = BASE / "candidates/sdk-d49-r1/profiles/daily"
GIB = 1024**3
DEADLINE = 5400
GATE_TIMEOUT = 540

SENTINELS = {
    "daily": {
        "rocm/lib/librocprofiler-sdk.so.1.4.2": SDK_SHA,
        "rocm/lib/libamdhip64.so": "a1162308076a99ff5cf5e4e15ca8bff53928d8bc4e624c42033d7e15dd2d1e3b",
        "rocm/lib/libMIOpen.so": "381eac05240e33acced9472fb7c3071afe8bc4f434407e67a29ce64423b935ac",
        "torch/torch/lib/libtorch_hip.so": "2f2761cd7f259612863406ab0e85d7b1b5ce8b59a32655bf9ac7db9359d0df27",
        "torch/torch/lib/libtorch_cpu.so": "9586351bda1617acaefacacedd938435464cd92337df1ec3e61044c453478195",
    },
    "rccl-single": {
        "rocm/lib/librocprofiler-sdk.so.1.4.2": SDK_SHA,
        "rocm/lib/libamdhip64.so": "a1162308076a99ff5cf5e4e15ca8bff53928d8bc4e624c42033d7e15dd2d1e3b",
        "rocm/lib/libMIOpen.so": "381eac05240e33acced9472fb7c3071afe8bc4f434407e67a29ce64423b935ac",
        "torch/torch/lib/libtorch_hip.so": "a31cbb159d3edfb4b7e1cc59dfcc325db185c4c540e1fcaa952e09e833335533",
        "torch/torch/lib/libtorch_cpu.so": "8a1b0e1fecef7afc9309d0460b2fbd13df20b8bd667d7036c58a341a020e5346",
    },
}


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, data, mode=0o644):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
        stream.write(data)
    pending.chmod(mode)
    pending.replace(path)


def link(path, target):
    pending = path.with_name(path.name + ".d59-new")
    assert not pending.exists() and not pending.is_symlink()
    pending.symlink_to(target)
    pending.replace(path)


def available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise AssertionError("MemAvailable missing")


def floors(start):
    assert available() >= 3 * GIB, "memory floor"
    assert shutil.disk_usage(WORK).free >= 5 * GIB, "disk floor"
    assert time.monotonic() - start < DEADLINE, "deadline"


def gate(label, command, count, uid, gid, start):
    preflight(label)
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    assert Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"
    logs = EVIDENCE / label
    logs.mkdir()
    os.chown(logs, uid, gid)
    command = [*command, "--output", str(logs / "result.json")]
    atomic(logs / "command.json", (json.dumps(command, indent=2) + "\n").encode())
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    guard_dir = Path("/run") / label
    guard_log = (logs / "guard.log").open("w")
    guard = subprocess.Popen(
        [
            "python3",
            str(BOOT_GUARD),
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
                code = proc.wait(timeout=GATE_TIMEOUT)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)
                raise
    finally:
        after = subprocess.check_output(["dmesg"])
        (logs / "kernel-after.log").write_bytes(after)
        running = subprocess.check_output(["podman", "ps", "-q"], text=True).split()
        atomic(
            logs / "exit.json",
            (
                json.dumps(
                    dict(
                        returncode=code,
                        kernel_log_unchanged=before == after,
                        running_containers=running,
                    ),
                    indent=2,
                )
                + "\n"
            ).encode(),
        )
        if not running:
            atomic(
                guard_dir / "ack.json",
                (
                    json.dumps({k: state[k] for k in ("boot_id", "nonce", "phase")})
                    + "\n"
                ).encode(),
            )
            guard.wait(timeout=5)
        guard_log.close()
    assert code == 0 and before == after and not running, label
    result = json.loads((logs / "result.json").read_text())
    assert result["status"] == "passed" and len(result["checks"]) == count, (
        label,
        result,
    )
    final_guard = json.loads((guard_dir / "state.json").read_text())
    assert final_guard["status"] == "acknowledged"
    floors(start)
    return dict(label=label, checks=count, seconds=None, guard=final_guard["status"])


def plain(label, command, expect, start):
    proc = subprocess.run(command, cwd=WS, capture_output=True, text=True, timeout=180)
    record = dict(
        label=label,
        returncode=proc.returncode,
        stdout=proc.stdout[-2000:],
        stderr=proc.stderr[-2000:],
    )
    atomic(
        (EVIDENCE / (label + ".json")), (json.dumps(record, indent=2) + "\n").encode()
    )
    assert proc.returncode == expect, (label, proc.returncode, proc.stderr[-500:])
    floors(start)
    return record


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    start = time.monotonic()
    EVIDENCE.mkdir(exist_ok=False)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    historical_runtime = sha(
        BASE / "releases/10.2-gfx1013-20261013-4b000d542/runtime.json"
    )
    assert (BASE / "current").resolve() == D49_DAILY
    manifest = json.loads((D49_DAILY / "runtime.json").read_text())
    uid, gid = manifest["uid"], manifest["gid"]
    journal = dict(status="running", version=VERSION, boot_id=boot_id, gates=[])
    atomic(EVIDENCE / "result.json", (json.dumps(journal, indent=2) + "\n").encode())

    with (WORK / "evidence/.board-reservation.lock").open("a") as board, (
        BASE / "current/.runtime.lock"
    ).open("r") as runtime:
        fcntl.flock(board, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(runtime, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (
            os.fstat(runtime.fileno()).st_ino
            == (D49_DAILY / ".runtime.lock").stat().st_ino
        )
        journal["preflight"] = preflight("d59-activate")
        before = dict(
            pointers={str(p): os.readlink(p) for p in POINTERS},
            registry=str(REGISTRY),
            helpers=[],
            release_runtime={},
        )
        registry_backup = EVIDENCE / "registry.before.json"
        shutil.copy2(REGISTRY, registry_backup)
        before["registry_backup"] = str(registry_backup)
        for index, target in enumerate(HELPERS):
            assert not target.is_symlink()
            backup = EVIDENCE / f"{index}-helper.before.py"
            shutil.copy2(target, backup)
            before["helpers"].append(
                dict(
                    path=str(target),
                    backup=str(backup),
                    mode=target.stat().st_mode & 0o777,
                    sha256=sha(target),
                )
            )
        for profile in FINAL_RUNTIME:
            path = RELEASE / "profiles" / profile / "runtime.json"
            backup = EVIDENCE / f"runtime-{profile}.before.json"
            shutil.copy2(path, backup)
            before["release_runtime"][profile] = dict(
                path=str(path), backup=str(backup), sha256=sha(path)
            )
        atomic(EVIDENCE / "before.json", (json.dumps(before, indent=2) + "\n").encode())

        def restore_selection():
            for name, target in before["pointers"].items():
                link(Path(name), target)
            atomic(REGISTRY, registry_backup.read_bytes())

        try:
            for profile, digest in FINAL_RUNTIME.items():
                payload = (TOOLS / f"runtime-{profile}-final.json").read_bytes()
                assert hashlib.sha256(payload).hexdigest() == digest
                atomic(RELEASE / "profiles" / profile / "runtime.json", payload)
            record = json.loads(REGISTRY.read_text())
            record["release"] = str(RELEASE / "profiles/rccl-single")
            record["runtime_sha256"] = FINAL_RUNTIME["rccl-single"]
            atomic(REGISTRY, (json.dumps(record, indent=2) + "\n").encode())
            for path, target in POINTERS.items():
                link(path, target)
            helper_bytes = (TOOLS / "bc250_torch.py").read_bytes()
            for target in HELPERS:
                atomic(target, helper_bytes, 0o755)
            assert (BASE / "current").resolve() == RELEASE / "profiles/daily"
            for profile, digest in FINAL_RUNTIME.items():
                base = BASE / (
                    "current" if profile == "daily" else "profiles/rccl-single"
                )
                assert sha(base / "runtime.json") == digest
                assert sha(base / "rocm/lib/librocprofiler-sdk.so.1.4.2") == SDK_SHA
            journal["activated"] = str((BASE / "current").resolve())

            scratch = EVIDENCE / "checkpoint-scratch"
            scratch.mkdir()
            os.chown(scratch, uid, gid)
            gates = [
                (
                    "d59-quant-b052",
                    [
                        "/usr/local/bin/bc250-torch",
                        str(FIXTURES / "validate_quant_b052.py"),
                    ],
                    220,
                ),
                (
                    "d59-rccl-daily",
                    [
                        "/usr/local/bin/bc250-train",
                        str(FIXTURES / "validate_rccl_b038.py"),
                        "--daily",
                        "--consumers",
                    ],
                    5,
                ),
                (
                    "d59-rccl-single",
                    [
                        "/usr/local/bin/bc250-train",
                        "--runtime-profile",
                        "rccl-single",
                        str(FIXTURES / "validate_rccl_b038.py"),
                        "--consumers",
                    ],
                    6,
                ),
                *[
                    (
                        f"d59-compiler-{family}",
                        [
                            "/usr/local/bin/bc250-torch",
                            str(FIXTURES / "validate_compiler_n310.py"),
                            "--family",
                            family,
                        ],
                        14,
                    )
                    for family in ("dense", "gated", "cnn", "batchnorm")
                ],
                (
                    "d59-distributed-b031",
                    [
                        "/usr/local/bin/bc250-torch",
                        str(FIXTURES / "validate_distributed_b031.py"),
                    ],
                    31,
                ),
                (
                    "d59-checkpoint-owned",
                    [
                        "/usr/local/bin/bc250-torch",
                        "--mount",
                        str(scratch),
                        str(FIXTURES / "validate_checkpoint_release_d59.py"),
                        "--scratch-root",
                        str(scratch),
                    ],
                    8,
                ),
            ]
            for label, command, count in gates:
                t0 = time.monotonic()
                entry = gate(label, command, count, uid, gid, start)
                entry["seconds"] = round(time.monotonic() - t0, 3)
                journal["gates"].append(entry)
                atomic(
                    EVIDENCE / "result.json",
                    (json.dumps(journal, indent=2) + "\n").encode(),
                )

            neg = plain(
                "d59-negative-visibility",
                [
                    "/usr/local/bin/bc250-torch",
                    "--env",
                    "HIP_VISIBLE_DEVICES=-1",
                    "-c",
                    "import torch; assert not torch.cuda.is_available(); "
                    "print('negative-visibility-cpu-ok')",
                ],
                0,
                start,
            )
            assert "negative-visibility-cpu-ok" in neg["stdout"]
            journal["negative_visibility"] = "admitted-on-bound-release"
            plain(
                "d59-refusal-rccl-env",
                [
                    "/usr/local/bin/bc250-torch",
                    "--runtime-profile",
                    "rccl-single",
                    "--env",
                    "NCCL_P2P_DISABLE=0",
                    "-c",
                    "pass",
                ],
                2,
                start,
            )
            plain(
                "d59-refusal-unknown-profile",
                [
                    "/usr/local/bin/bc250-torch",
                    "--runtime-profile",
                    "bogus",
                    "-c",
                    "pass",
                ],
                2,
                start,
            )
            journal["refusals"] = ["rccl-env-override", "unknown-profile"]

            for profile, files in SENTINELS.items():
                base = BASE / "candidates/sdk-d49-r1/profiles" / profile
                for rel, digest in files.items():
                    assert sha(base / rel) == digest, (profile, rel)
            assert (
                Path("/proc/sys/kernel/random/boot_id").read_text().strip() == boot_id
            )
            assert (
                sha(BASE / "releases/10.2-gfx1013-20261013-4b000d542/runtime.json")
                == historical_runtime
            )
            journal["protected_checks"] = "passed"

            restore_selection()
            assert (BASE / "current").resolve() == D49_DAILY
            journal["rollback"] = "selection-restored-to-d49"
            journal["helper_note"] = (
                "additive allowlist helper stays deployed; binds D49 and D59 cohorts"
            )
            journal["release_validated"] = True
            journal["status"] = "passed"
        except BaseException as error:
            journal.update(status="failed", error=repr(error))
            try:
                restore_selection()
                for item in before["helpers"]:
                    atomic(
                        Path(item["path"]),
                        Path(item["backup"]).read_bytes(),
                        item["mode"],
                    )
                for profile, info in before["release_runtime"].items():
                    atomic(Path(info["path"]), Path(info["backup"]).read_bytes())
                journal["restored_after_failure"] = True
            except BaseException as restore_error:
                journal["restore_error"] = repr(restore_error)
            raise
        finally:
            journal["seconds"] = round(time.monotonic() - start, 3)
            journal["current_after"] = str((BASE / "current").resolve())
            atomic(
                EVIDENCE / "result.json",
                (json.dumps(journal, indent=2) + "\n").encode(),
            )
    # The D49 cohort is mutable, so its launcher takes the SH runtime lock;
    # run the post-rollback smoke only after the EX hold is released.
    if journal["status"] == "passed":
        post = plain(
            "d59-post-rollback-smoke",
            [
                "/usr/local/bin/bc250-torch",
                "-c",
                "import torch; print('rollback-ok', torch.__version__)",
            ],
            0,
            start,
        )
        assert "rollback-ok" in post["stdout"]
        journal["rollback_smoke"] = "passed"
        atomic(
            EVIDENCE / "result.json",
            (json.dumps(journal, indent=2) + "\n").encode(),
        )
    print(json.dumps(journal, indent=2))


if __name__ == "__main__":
    main()
