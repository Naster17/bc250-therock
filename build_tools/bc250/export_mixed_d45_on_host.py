#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Archive owned mixed checks and a narrow read-only SDK build-config audit."""
import hashlib
import json
import os
from pathlib import Path
import tarfile

from board_preflight import reservation

WORK = Path("/home/nik/bc250work")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    root = WORK / "evidence/mixed-d45-r1"
    with reservation("d45-export") as admission:
        result = json.loads((root / "result.json").read_text())
        assert (
            result["status"] == "passed" and result["guard"]["status"] == "acknowledged"
        )
        assert (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            == result["boot_id"]
        )
        assert sha(Path("/usr/local/bin/bc250-torch")) == result["launcher_sha256"]
        build = WORK / "rocm-10.2-gfx1013/build/profiler/rocprofiler-sdk/build"
        sdk = Path("/opt/bc250-stack/current/rocm/lib/librocprofiler-sdk.so").resolve()
        config = (build / "CMakeCache.txt").read_text()
        ninja = (build / "build.ninja").read_text()
        audit = dict(
            retained_build=str(build),
            runtime_sdk=str(sdk),
            runtime_sdk_bytes=sdk.stat().st_size,
            runtime_sdk_sha256=sha(sdk),
            ci_configuration=[
                s for s in config.splitlines() if s.startswith("ROCPROFILER_BUILD_CI:")
            ],
            compiler_ci_definition_present="-DROCPROFILER_CI=1" in ninja,
            build_artifacts=[
                dict(path=str(p), sha256=sha(p))
                for p in build.rglob(sdk.name)
                if p.is_file() and not p.is_symlink()
            ],
            note="Build config is retained evidence; byte equality with runtime is required before attributing it to the active DSO.",
        )
        (root / "sdk-triage.json").write_text(json.dumps(audit, indent=2) + "\n")
        (root / "export.json").write_text(
            json.dumps(
                dict(status="passed", preflight=admission, boot_id=result["boot_id"]),
                indent=2,
            )
            + "\n"
        )
        archive = WORK / "evidence/mixed-d45-evidence-r1.tar.gz"
        assert not archive.exists()
        with tarfile.open(archive, "w:gz") as output:
            for name in [
                "tools/mixed-d45",
                "evidence/mixed-d45-r1",
                "evidence/mixed-d45-controller.log",
            ]:
                output.add(WORK / name, arcname=name)
        archive.chmod(0o644)
        print(
            json.dumps(
                dict(
                    path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive)
                )
            )
        )


if __name__ == "__main__":
    main()
