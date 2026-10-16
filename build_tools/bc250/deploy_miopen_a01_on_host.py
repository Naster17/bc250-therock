#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage and deploy qualified A01 DSOs under update_dev_prefix.py's exclusive lock."""

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
LOGS = WS / "logs/miopen-a01-deployment"
RECOVERY = WS / "recovery/miopen-a01-before"
LIBS = ["libhsa-runtime64.so.1.21.0", "libMIOpen.so.1.0"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_copy(source, target):
    temp = target.with_name(target.name + ".a01-new")
    assert not temp.exists()
    shutil.copy2(source, temp)
    os.replace(temp, target)


def atomic_json(path, data):
    temp = path.with_suffix(".a01-new")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temp, path)


def main():
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    assert Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"
    LOGS.mkdir()
    source = json.loads((WS / "helpers/miopen-a01-source.json").read_text())
    for rel, expected in source["sources_sha256"].items():
        assert sha(WS / "source" / rel) == expected, rel
    gates = [
        "miopen-a01-rocr-events-candidate-r3",
        "miopen-a01-rocr-direct-candidate",
        "miopen-a01-event-reference-candidate",
        "miopen-a01-combined-direct-candidate-r2",
        "miopen-a01-torch-candidate",
    ]
    for label in gates:
        folder = WS / "logs" / label
        assert json.loads((folder / "exit.json").read_text()) == dict(
            returncode=0, kernel_log_unchanged=True
        )
        assert "GUARD_ACKNOWLEDGED" in (folder / "guard.log").read_text()
    assert (
        json.loads((WS / "logs" / gates[0] / "result.json").read_text())[
            "unstable_or_nonpositive"
        ]
        == 0
    )
    for label, n in [(gates[1], 15), (gates[3], 15), (gates[4], 30)]:
        rows = json.loads((WS / "logs" / label / "result.json").read_text())["results"]
        assert len(rows) == n and all(r["status"] == "passed" for r in rows)
    assert (
        len(json.loads((WS / "logs" / gates[2] / "result.json").read_text())["samples"])
        == 6
    )
    protected = [
        PREFIX / "lib/libamdhip64.so",
        PREFIX / "lib/librocblas.so",
        PREFIX / "lib/libhipblaslt.so",
        PREFIX / ".kpack/blas_lib_gfx1013.kpack",
        WS / "pytorch-work/torch/lib/libtorch_hip.so",
        Path("/opt/bc250-stack/previous/rocm/lib/libhsa-runtime64.so.1.21.0"),
        Path("/opt/bc250-stack/previous/rocm/lib/libMIOpen.so.1.0"),
    ]
    frozen = {str(p): sha(p) for p in protected}
    for name in LIBS:
        assert sha(PREFIX / "lib" / name) == sha(
            RECOVERY / name
        ), "Preserve the original rollback binary"
    old_runtime = (RELEASE / "runtime.json").read_bytes()
    assert not (RECOVERY / "runtime.before-a01.json").exists()
    (RECOVERY / "runtime.before-a01.json").write_bytes(old_runtime)
    report = dict(
        status="staging", source=source, protected_before=frozen, installed_gates=[]
    )

    def save():
        atomic_json(LOGS / "result.json", report)

    def run(label, args, timeout):
        report["phase"] = label
        save()
        with (LOGS / (label + ".log")).open("w") as log:
            subprocess.run(
                args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout
            )

    save()
    changed = False
    try:
        # Rebuild the restored qualified HIP source so a later normal build cannot
        # accidentally install the discarded diagnostic object. Install only the
        # two changed components into an isolated staging tree, preserving RPATHs.
        script = """set -euo pipefail
export CCACHE_MAXSIZE=20G CMAKE_BUILD_PARALLEL_LEVEL=16
source_dir=/workspace/source
build_dir=/workspace/build
eval "$(python3 "$source_dir/build_tools/setup_ccache.py" --dir /workspace/cache --local-path /workspace/cache/data --no-reset-stats)"
cmake --build "$build_dir/core/clr/build" --target amdhip64 --parallel 16
cmake --install "$build_dir/core/ROCR-Runtime/build" --prefix "$build_dir/bc250-a01-stage/rocr" --component binary
cmake --install "$build_dir/ml-libs/MIOpen/build" --prefix "$build_dir/bc250-a01-stage/miopen" --component runtime
ccache --show-stats
"""
        run(
            "stage",
            [
                "podman",
                "run",
                "--rm",
                "--name",
                "bc250-a01-stage",
                "--network",
                "none",
                "-v",
                f"{WS}:/workspace:rw",
                "-v",
                f"{WS}/source:/workspace/source:ro",
                "localhost/bc250-rocm-build:10.2-ccache",
                "bash",
                "-c",
                script,
            ],
            600,
        )
        report["before_library_sha256"] = {
            name: sha(PREFIX / "lib" / name) for name in LIBS
        }
        changed = True
        for name, component in zip(LIBS, ["rocr", "miopen"]):
            candidate = WS / "build/bc250-a01-stage" / component / "lib" / name
            assert sha(candidate) != report["before_library_sha256"][name]
            atomic_copy(candidate, PREFIX / "lib" / name)
        runtime = json.loads(old_runtime)
        runtime["previous_a01_runtime_revisions"] = {
            k: runtime[k]
            for k in [
                "rocm_systems_revision",
                "rocm_libraries_revision",
                "therock_revision",
            ]
        }
        runtime.update(
            rocm_systems_revision=source["rocm_systems_revision"],
            rocm_libraries_revision=source["rocm_libraries_revision"],
        )
        runtime[
            "source_state"
        ] += "; A01 native gfx1013 ROCr dispatch clock epoch and MIOpen dilated GEMM workspace query correction"
        runtime["a01_miopen"] = {
            "status": "candidate qualified; installed gates pending",
            "source_manifest": str(WS / "SOURCE_MIOPEN_A01.json"),
            "library_sha256": {name: sha(PREFIX / "lib" / name) for name in LIBS},
        }
        atomic_json(RELEASE / "runtime.json", runtime)
        for label, script_name, testargs, fresh in [
            (
                "miopen-a01-events-installed",
                "validate_hip_event_a01.py",
                ["--expect-pass"],
                False,
            ),
            (
                "miopen-a01-event-reference-installed",
                "validate_gpu_clock_a01.py",
                ["--reference"],
                False,
            ),
            (
                "miopen-a01-direct-installed",
                "validate_miopen_a01.py",
                ["--extended", "--expect-pass"],
                True,
            ),
            ("miopen-a01-torch-installed", "validate_miopen_a01_torch.py", [], True),
            ("miopen-a01-core-installed", "validate_cleanup_runtime.py", [], False),
        ]:
            command = [
                sys.executable,
                str(WS / "helpers/validate_miopen_a01_on_host.py"),
                "--label",
                label,
                "--script",
                script_name,
            ]
            command += ["--test-arg=" + a for a in testargs]
            if fresh:
                db = WS / "cache/a01-direct-fresh" / label
                db.mkdir()
                os.chown(db, runtime["uid"], runtime["gid"])
                command += ["--env=MIOPEN_USER_DB_PATH=" + str(db)]
            run(label, command, 570)
            exit = json.loads((WS / "logs" / label / "exit.json").read_text())
            assert exit == dict(returncode=0, kernel_log_unchanged=True)
            assert (
                "GUARD_ACKNOWLEDGED" in (WS / "logs" / label / "guard.log").read_text()
            )
            report["installed_gates"].append(label)
            save()
        assert {str(p): sha(p) for p in protected} == frozen
        runtime["a01_miopen"]["status"] = "bounded installed GPU/reference gates passed"
        runtime["a01_miopen"]["installed_gates"] = report["installed_gates"]
        runtime["a01_miopen"][
            "limitations"
        ] = "Native gfx1013 positive epoch mismatch on this boot; absolute CPU/GPU trace alignment, opposite epoch offsets, reset/reboot and universal convolution/performance unqualified; A02 legacy CK reduction separate"
        atomic_json(RELEASE / "runtime.json", runtime)
        report.update(
            status="passed",
            after_library_sha256=runtime["a01_miopen"]["library_sha256"],
            protected_unchanged=True,
            running_agent_jobs=[],
        )
        atomic_json(WS / "SOURCE_MIOPEN_A01.json", report)
        atomic_json(RELEASE / "SOURCE_MIOPEN_A01.json", report)
        save()
    except BaseException as e:
        if changed:
            for name in LIBS:
                atomic_copy(RECOVERY / name, PREFIX / "lib" / name)
            (RELEASE / "runtime.json").write_bytes(old_runtime)
        report.update(
            status="failed_rolled_back" if changed else "failed_before_deployment",
            error=str(e),
        )
        save()
        raise


if __name__ == "__main__":
    main()
