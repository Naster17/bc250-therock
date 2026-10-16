#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deploy only A02 MIOpen under update_dev_prefix.py's exclusive runtime lock."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
PREFIX = WS / "build/dist/rocm"
RELEASE = Path("/opt/bc250-stack/current").resolve()
LOGS = WS / "logs/miopen-a02-deployment"
RECOVERY = WS / "recovery/miopen-a02-before"
LIB = "libMIOpen.so.1.0"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_copy(source, target):
    temp = target.with_name(target.name + ".a02-new")
    assert not temp.exists()
    shutil.copy2(source, temp)
    os.replace(temp, target)


def atomic_json(path, data):
    temp = path.with_name(path.name + ".a02-new")
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
    source = json.loads((WS / "helpers/miopen-a02-source.json").read_text())
    for rel, expected in source["sources_sha256"].items():
        assert sha(WS / "source" / rel) == expected, rel
    for name, expected in source["helper_sha256"].items():
        assert sha(WS / "helpers" / name) == expected, name
    for label, count in source["candidate_gates"].items():
        check_gate(label, count)
    assert sha(PREFIX / "lib" / LIB) == sha(RECOVERY / LIB)
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
    frozen = {str(p): sha(p) for p in protected}
    old_runtime = (RELEASE / "runtime.json").read_bytes()
    backup_metadata = RECOVERY / "runtime.before-a02.json"
    assert not backup_metadata.exists()
    backup_metadata.write_bytes(old_runtime)
    report = dict(
        status="staging",
        source=source,
        protected_before=frozen,
        installed_gates=[],
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
    try:
        run(
            "stage",
            [
                "podman",
                "run",
                "--rm",
                "--name",
                "bc250-a02-stage",
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
                "--prefix /workspace/build/bc250-a02-stage/miopen --component runtime",
            ],
            180,
        )
        candidate = WS / "build/bc250-a02-stage/miopen/lib" / LIB
        assert sha(candidate) != report["before_library_sha256"]
        changed = True
        atomic_copy(candidate, PREFIX / "lib" / LIB)
        runtime = json.loads(old_runtime)
        runtime["previous_a02_rocm_libraries_revision"] = runtime[
            "rocm_libraries_revision"
        ]
        runtime["rocm_libraries_revision"] = source["rocm_libraries_revision"]
        runtime["effective_therock_port_revision"] = source["therock_port_revision"]
        runtime[
            "source_state"
        ] += "; A02 native legacy CK reductions and BF16 compute-precision epilogues"
        runtime["a02_miopen"] = dict(
            status="installed gates pending",
            source_manifest=str(WS / "SOURCE_MIOPEN_A02.json"),
            library_sha256=sha(PREFIX / "lib" / LIB),
        )
        atomic_json(RELEASE / "runtime.json", runtime)
        for label, script, testargs, count, fresh in [
            (
                "miopen-a02-reduction-installed",
                "validate_miopen_a02.py",
                ["--suite", "reduction"],
                57,
                True,
            ),
            (
                "miopen-a02-rnn-installed",
                "validate_miopen_a02.py",
                ["--suite", "rnn"],
                12,
                False,
            ),
            (
                "miopen-a02-conv-installed",
                "validate_miopen_a01_torch.py",
                [],
                None,
                False,
            ),
            (
                "miopen-a02-events-installed",
                "validate_hip_event_a01.py",
                ["--expect-pass"],
                None,
                False,
            ),
            (
                "miopen-a02-core-installed",
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
                cache.mkdir()
                os.chown(cache, runtime["uid"], runtime["gid"])
                command += ["--env=MIOPEN_CUSTOM_CACHE_DIR=" + str(cache)]
            run(label, command, 570)
            check_gate(label, count)
            if script == "validate_miopen_a01_torch.py":
                rows = json.loads((WS / "logs" / label / "result.json").read_text())[
                    "results"
                ]
                assert len(rows) == 30 and all(r["status"] == "passed" for r in rows)
            if script == "validate_hip_event_a01.py":
                assert (
                    json.loads((WS / "logs" / label / "result.json").read_text())[
                        "unstable_or_nonpositive"
                    ]
                    == 0
                )
            report["installed_gates"].append(label)
            save()
        assert {str(p): sha(p) for p in protected} == frozen
        runtime["a02_miopen"].update(
            status="bounded installed GPU/reference gates passed",
            installed_gates=report["installed_gates"],
            limitations="FP32/FP16/BF16 reductions with FP32 compute and FP32/FP16 recurrent networks in recorded fixtures only; BF16 RNN, all dtype/NaN/index contracts and universal performance unqualified",
        )
        atomic_json(RELEASE / "runtime.json", runtime)
        report.update(
            status="passed",
            after_library_sha256=sha(PREFIX / "lib" / LIB),
            protected_unchanged=True,
            running_agent_jobs=[],
        )
        atomic_json(WS / "SOURCE_MIOPEN_A02.json", report)
        atomic_json(RELEASE / "SOURCE_MIOPEN_A02.json", report)
        save()
    except BaseException as e:
        if changed:
            atomic_copy(RECOVERY / LIB, PREFIX / "lib" / LIB)
            atomic_copy(backup_metadata, RELEASE / "runtime.json")
        report.update(
            status="failed_rolled_back" if changed else "failed_before_deployment",
            error=str(e),
        )
        save()
        raise


if __name__ == "__main__":
    main()
