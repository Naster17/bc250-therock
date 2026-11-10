#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only bounded source replay/installed-source audit; native prefix read-only."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

from audit_elf_d46 import compare

WORK = Path("/home/nik/bc250work")
TOOLS = WORK / "tools/source-d46"
ROOT = WORK / "evidence/source-d46-r1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert os.uname().nodename == "bc250"
    daily = Path("/opt/bc250-stack/current").resolve()
    with (daily / ".runtime.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        ROOT.mkdir(exist_ok=False)
        manifest = json.loads((TOOLS / "files.json").read_text())
        subset = ROOT / "reconstructed-delta"
        subset.mkdir()
        with tarfile.open(TOOLS / "torch-baseline-subset.tar.gz") as archive:
            archive.extractall(subset, filter="data")
        for item in manifest:
            path = subset / item["path"]
            assert (sha(path) if path.exists() else None) == item["baseline_sha256"]
        applied = []
        for patch, field in [
            ("0001-qualified-native-torch.patch", "native_sha256"),
            ("0002-agent-exit-lifetime.patch", "source_head_sha256"),
        ]:
            for action in ("--check", "--apply"):
                subprocess.run(
                    ["git", "apply", action, str(TOOLS / patch)],
                    cwd=subset,
                    check=True,
                    capture_output=True,
                )
            for item in manifest:
                path = subset / item["path"]
                assert (sha(path) if path.exists() else None) == item[field], (
                    patch,
                    item["path"],
                )
            applied.append(
                dict(
                    patch=patch, sha256=sha(TOOLS / patch), files_verified=len(manifest)
                )
            )
        profiles = {}
        for name, prefix in [
            ("daily", daily),
            (
                "rccl-single",
                Path(
                    "/opt/bc250-stack/candidates/miopen-plan02-r4/profiles/rccl-single"
                ).resolve(),
            ),
        ]:
            runtime = json.loads((prefix / "runtime.json").read_text())
            source = (prefix / "torch").resolve()
            files = []
            for item in manifest:
                path = source / item["path"]
                actual = sha(path) if path.exists() else None
                files.append(
                    dict(
                        path=item["path"],
                        sha256=actual,
                        matches_native=actual == item["native_sha256"],
                    )
                )
            config = source / "build/CMakeCache.txt"
            selected = [
                line
                for line in config.read_text().splitlines()
                if line.startswith(
                    ("USE_NCCL:", "USE_DISTRIBUTED:", "USE_ROCM:", "USE_KINETO:")
                )
            ]
            profiles[name] = dict(
                prefix=str(prefix),
                torch_source=str(source),
                runtime_sha256=sha(prefix / "runtime.json"),
                revisions={
                    k: runtime.get(k)
                    for k in (
                        "torch_revision",
                        "rocm_libraries_revision",
                        "rocm_systems_revision",
                        "triton_revision",
                    )
                },
                source_files=files,
                build_options=selected,
            )
        build = WORK / "rocm-10.2-gfx1013/build/profiler/rocprofiler-sdk/build"
        elf = compare(
            (build / "lib/librocprofiler-sdk.so.1.4.2").read_bytes(),
            (daily / "rocm/lib/librocprofiler-sdk.so").read_bytes(),
        )
        config = (build / "CMakeCache.txt").read_text()
        elf["ci_configuration"] = [
            line
            for line in config.splitlines()
            if line.startswith("ROCPROFILER_BUILD_CI:")
        ]
        elf["compiler_ci_definition_present"] = (
            "-DROCPROFILER_CI=1" in (build / "build.ninja").read_text()
        )
        sdk_files = {}
        for relative in (
            "source/lib/rocprofiler-sdk/agent.cpp",
            "source/lib/common/logging.hpp",
            "cmake/rocprofiler_build_settings.cmake",
            "cmake/rocprofiler_options.cmake",
        ):
            path = (
                WORK
                / "rocm-10.2-gfx1013/source/rocm-systems/projects/rocprofiler-sdk"
                / relative
            )
            sdk_files[relative] = sha(path)
        assert Path("/opt/bc250-stack/current").resolve() == daily
        result = dict(
            status="passed",
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            applied=applied,
            source_delta_files=len(manifest),
            profiles=profiles,
            sdk_elf=elf,
            sdk_source_sha256=sdk_files,
            scope="35-file focused host replay; full Git tree replay separately on workstation; no native build or GPU execution",
        )
        (ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    status=result["status"],
                    source_delta_files=len(manifest),
                    profile_mismatches={
                        k: [
                            i["path"]
                            for i in v["source_files"]
                            if not i["matches_native"]
                        ]
                        for k, v in profiles.items()
                    },
                    sdk_rpath_only_exact_replay=elf["rpath_only_exact_replay"],
                )
            )
        )


if __name__ == "__main__":
    main()
