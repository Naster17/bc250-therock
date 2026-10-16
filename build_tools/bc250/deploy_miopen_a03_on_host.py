#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deploy only A03 MIOpen under update_dev_prefix.py's exclusive runtime lock."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sqlite3

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
PREFIX = WS / "build/dist/rocm"
RELEASE = Path("/opt/bc250-stack/current").resolve()
LOGS = WS / "logs/miopen-a03-deployment-r2"
RECOVERY = WS / "recovery/miopen-a03-before"
LIB = "libMIOpen.so.1.0"
PACKAGE = WS / "build/bc250-a03-db-package-r3"
DB = PREFIX / "share/miopen/db"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_copy(source, target):
    temp = target.with_name(target.name + ".a03-new")
    assert not temp.exists()
    shutil.copy2(source, temp)
    os.replace(temp, target)


def atomic_json(path, data):
    temp = path.with_name(path.name + ".a03-new")
    assert not temp.exists()
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temp, path)


def check_gate(label, count=None):
    folder = WS / "logs" / label
    assert json.loads((folder / "exit.json").read_text()) == dict(
        returncode=0, kernel_log_unchanged=True
    )
    assert "GUARD_ACKNOWLEDGED" in (folder / "guard.log").read_text()
    if count is not None:
        result = json.loads((folder / "result.json").read_text())
        assert result["status"] == "passed"
        assert len(result["results"]) == count
        assert all(r["status"] == "passed" for r in result["results"])


def main():
    assert os.geteuid() == 0
    # update_dev_prefix holds the exclusive lock across staging, gates and rollback.
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    assert Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"
    LOGS.mkdir()
    source = json.loads((WS / "helpers/miopen-a03-source.json").read_text())
    for rel, expected in source["sources_sha256"].items():
        assert sha(WS / "source" / rel) == expected, rel
    for name, expected in source["helper_sha256"].items():
        assert sha(WS / "helpers" / name) == expected, name
    for label, count in source["candidate_gates"].items():
        check_gate(label, count)
    warning = "CK grouped conv library not found"
    assert warning in (WS / "logs/miopen-a03-handle-before/gpu.log").read_text()
    assert warning not in (WS / "logs/miopen-a03-handle-candidate/gpu.log").read_text()
    assert (
        warning in (WS / "logs/miopen-a03-handle-explicit-missing/gpu.log").read_text()
    )
    for label in ["miopen-a03-final-replay", "miopen-a03-final-tune-replay"]:
        log = (WS / "logs" / label / "gpu.log").read_text()
        assert warning not in log
        assert not any(
            "Warning" in line and "File is unreadable" in line
            for line in log.splitlines()
        )
    assert (
        "Perf Db: record loaded: ConvHipDirectFwd"
        in (WS / "logs/miopen-a03-final-tune-replay/gpu.log").read_text()
    )
    for db in (WS / "cache/a03-final-replay/kernels").rglob("*.ukdb"):
        with sqlite3.connect("file:" + str(db) + "?mode=ro", uri=True) as c:
            assert c.execute("select count(*) from kern_db").fetchone() == (0,)
    assert sha(PREFIX / "lib" / LIB) == sha(RECOVERY / LIB)
    bundle = json.loads((PACKAGE / "manifest.json").read_text())
    assert (
        bundle["source"]["rocm_libraries_revision"] == source["rocm_libraries_revision"]
    )
    for name, expected in bundle["files"].items():
        assert sha(PACKAGE / "db" / name) == expected
        assert not (DB / name).exists(), "Never overwrite existing system DBs"
    assert not (DB / "BC250_A03_MANIFEST.json").exists()
    protected = [
        PREFIX / "lib/libhsa-runtime64.so.1.21.0",
        PREFIX / "lib/libamdhip64.so",
        PREFIX / "lib/librocblas.so",
        PREFIX / "lib/libhipblaslt.so",
        PREFIX / ".kpack/blas_lib_gfx1013.kpack",
        WS / "pytorch-work/torch/lib/libtorch_hip.so",
        Path("/opt/bc250-stack/previous/rocm/lib/libhsa-runtime64.so.1.21.0"),
        Path("/opt/bc250-stack/previous/rocm/lib/libMIOpen.so.1.0"),
    ]
    protected += [DB / name for name in bundle["shared_heuristics"]]
    frozen = {str(p): sha(p) for p in protected}
    old_runtime = (RELEASE / "runtime.json").read_bytes()
    backup_metadata = RECOVERY / "runtime.before-a03.json"
    assert not backup_metadata.exists()
    backup_metadata.write_bytes(old_runtime)
    report = dict(
        status="staging",
        source=source,
        protected_before=frozen,
        installed_gates=[],
        database_package=bundle,
        before_library_sha256=sha(PREFIX / "lib" / LIB),
    )

    def save():
        atomic_json(LOGS / "result.json", report)

    def run(label, command, timeout):
        report["phase"] = label
        save()
        with (LOGS / (label + ".log")).open("w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )

    changed = False
    created_db_files = []
    try:
        run(
            "stage",
            [
                "podman",
                "run",
                "--rm",
                "--name",
                "bc250-a03-stage",
                "--network",
                "none",
                "-v",
                f"{WS}:/workspace:rw",
                "-v",
                f"{WS}/source:/workspace/source:ro",
                "localhost/bc250-rocm-build:10.2-ccache",
                "bash",
                "-c",
                "set -eu; cmake --install /workspace/build/ml-libs/MIOpen/build "
                "--prefix /workspace/build/bc250-a03-stage/miopen --component runtime",
            ],
            180,
        )
        candidate = WS / "build/bc250-a03-stage/miopen/lib" / LIB
        assert sha(candidate) != report["before_library_sha256"]
        changed = True
        atomic_copy(candidate, PREFIX / "lib" / LIB)
        for name in bundle["files"]:
            target = DB / name
            atomic_copy(PACKAGE / "db" / name, target)
            created_db_files.append(target)
        atomic_copy(PACKAGE / "manifest.json", DB / "BC250_A03_MANIFEST.json")
        created_db_files.append(DB / "BC250_A03_MANIFEST.json")
        runtime = json.loads(old_runtime)
        runtime["previous_a03_rocm_libraries_revision"] = runtime[
            "rocm_libraries_revision"
        ]
        runtime["rocm_libraries_revision"] = source["rocm_libraries_revision"]
        runtime["effective_therock_port_revision"] = source["therock_port_revision"]
        runtime[
            "source_state"
        ] += "; A03 capability-aware optional CK probing and bounded native FindDb/PerfDb/kernel database package"
        runtime["a03_miopen"] = dict(
            status="installed gates pending",
            source_manifest=str(WS / "SOURCE_MIOPEN_A03.json"),
            library_sha256=sha(PREFIX / "lib" / LIB),
        )
        atomic_json(RELEASE / "runtime.json", runtime)
        for label, script, testargs, count, fresh in [
            (
                "miopen-a03-replay-installed",
                "validate_miopen_a03.py",
                ["--phase", "replay"],
                15,
                True,
            ),
            (
                "miopen-a03-tune-installed",
                "validate_miopen_a03.py",
                ["--phase", "replay", "--profile", "tune", "--direction", "fwd"],
                1,
                True,
            ),
            (
                "miopen-a03-unseen-installed",
                "validate_miopen_a03.py",
                ["--profile", "unseen"],
                3,
                True,
            ),
            (
                "miopen-a03-conv-installed",
                "validate_miopen_a01_torch.py",
                [],
                None,
                False,
            ),
            (
                "miopen-a03-a02-smoke-installed",
                "validate_miopen_a02.py",
                ["--smoke"],
                3,
                False,
            ),
            (
                "miopen-a03-core-installed",
                "validate_cleanup_runtime.py",
                [],
                None,
                False,
            ),
        ]:
            command = [
                sys.executable,
                str(WS / "helpers/validate_miopen_a01_on_host.py"),
                "--label",
                label,
                "--script",
                script,
            ]
            command += ["--test-arg=" + a for a in testargs]
            if fresh:
                cache = WS / "cache" / label
                for sub in ["db", "kernels"]:
                    folder = cache / sub
                    folder.mkdir(parents=True)
                    os.chown(folder, runtime["uid"], runtime["gid"])
                command += [
                    "--env=MIOPEN_USER_DB_PATH=" + str(cache / "db"),
                    "--env=MIOPEN_CUSTOM_CACHE_DIR=" + str(cache / "kernels"),
                    "--env=MIOPEN_ENABLE_LOGGING=1",
                    "--env=MIOPEN_LOG_LEVEL=6",
                ]
            run(label, command, 570)
            check_gate(label, count)
            folder = WS / "logs" / label
            text = (folder / "gpu.log").read_text()
            assert "CK grouped conv library not found" not in text
            assert not any(
                "Warning" in line and "File is unreadable" in line
                for line in text.splitlines()
            )
            if script == "validate_miopen_a01_torch.py":
                rows = json.loads((folder / "result.json").read_text())["results"]
                assert len(rows) == 30 and all(r["status"] == "passed" for r in rows)
            if fresh and "unseen" not in label:
                assert not list(
                    (cache / "db").glob("*.txt")
                ), "Replay must not populate user FindDb/PerfDb"
                for db in (cache / "kernels").rglob("*.ukdb"):
                    with sqlite3.connect("file:" + str(db) + "?mode=ro", uri=True) as c:
                        assert c.execute("select count(*) from kern_db").fetchone() == (
                            0,
                        ), "System KDB must satisfy replay"
            if "tune-installed" in label:
                assert "Perf Db: record loaded: ConvHipDirectFwd" in text
            if "unseen" in label:
                assert len(list((cache / "db").glob("*.ufdb.txt"))) == 1
                assert (cache / "db/gfx1013_20.HIP.3_6_2_.ufdb.txt").read_text().count(
                    "\n"
                ) == 3
            report["installed_gates"].append(label)
            save()
        assert {str(p): sha(p) for p in protected} == frozen
        for name, expected in bundle["files"].items():
            assert sha(DB / name) == expected
        runtime["a03_miopen"].update(
            status="bounded installed GPU/reference gates passed",
            installed_gates=report["installed_gates"],
            limitations="Only listed convolution geometries/types/solutions and one searched performance record are qualified; optional CK plugin remains absent; no universal tuning/speedup claim.",
            database_files=bundle["files"],
        )
        atomic_json(RELEASE / "runtime.json", runtime)
        report.update(
            status="passed",
            after_library_sha256=sha(PREFIX / "lib" / LIB),
            protected_unchanged=True,
            running_agent_jobs=[],
        )
        atomic_json(WS / "SOURCE_MIOPEN_A03.json", report)
        atomic_json(RELEASE / "SOURCE_MIOPEN_A03.json", report)
        save()
    except BaseException as e:
        if changed:
            atomic_copy(RECOVERY / LIB, PREFIX / "lib" / LIB)
            for target in created_db_files:
                target.unlink()
            atomic_copy(backup_metadata, RELEASE / "runtime.json")
        report.update(
            status="failed_rolled_back" if changed else "failed_before_deployment",
            error=str(e),
        )
        save()
        raise


if __name__ == "__main__":
    main()
