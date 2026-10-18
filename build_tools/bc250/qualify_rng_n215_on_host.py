#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only installed-runtime RNG qualification under the exclusive lock."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, check_gate, sha

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
LOGS = WS / "logs/rng-n215-qualification-r2"


def main() -> None:
    assert os.geteuid() == 0
    LOGS.mkdir()
    baseline = json.loads(
        (WS / "logs/fft-n214-qualification-r2/final.json").read_text()
    )
    protected = dict(baseline["protected_sha256"])
    candidate = Path(baseline["torch_target"])
    prefix = Path(baseline["rocm_target"])
    header_source = json.loads((WS / "helpers/header-source.json").read_text())
    recovery = WS / "recovery/rng-n215-source-before"
    recovery.mkdir()
    for name, digests in header_source["files"].items():
        path = (
            WS / "source/rocm-libraries/projects/rocrand/library/include/rocrand" / name
        )
        staged = WS / "helpers" / name
        assert sha(path) == digests["before_sha256"], path
        assert sha(staged) == digests["after_sha256"], staged
        shutil.copy2(path, recovery / name)
        stat = path.stat()
        temporary = path.with_name(name + ".n215-new")
        assert not temporary.exists()
        temporary.write_bytes(staged.read_bytes())
        os.chown(temporary, stat.st_uid, stat.st_gid)
        temporary.chmod(stat.st_mode & 0o777)
        os.replace(temporary, path)
        protected[str(path)] = digests["after_sha256"]
        protected[str(recovery / name)] = digests["before_sha256"]
        installed = prefix / "include/rocrand" / name
        protected[str(installed)] = sha(installed)
    for name in ("librocrand.so", "libhiprand.so"):
        path = prefix / "lib" / name
        protected[str(path)] = sha(path)
    cache_paths = [Path(p) for p in baseline["aot_caches"]]
    for p, digest in protected.items():
        assert sha(Path(p)) == digest, p
    links = {
        str(p): os.readlink(p)
        for p in [
            Path("/opt/bc250-stack/current"),
            Path("/opt/bc250-stack/previous"),
            Path("/opt/bc250-stack/current/torch"),
            Path("/opt/bc250-stack/current/rocm"),
        ]
    }
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    before = subprocess.check_output(["dmesg"])
    (LOGS / "kernel-before.log").write_bytes(before)
    source = {
        p.name: sha(p)
        for p in [
            Path(__file__),
            WS / "helpers/validate_rng_n215.py",
            WS / "helpers/validate_rng_n215_native.py",
            WS / "helpers/validate_priority2_math.py",
            WS / "helpers/validate_cleanup_runtime.py",
            WS / "helpers/validate_torch_a04_on_host.py",
            WS / "helpers/update_dev_prefix.py",
            WS / "helpers/header-source.json",
            WS / "helpers/rocrand.h",
            WS / "helpers/rocrand.hpp",
        ]
    }
    report = dict(
        status="running",
        protected_sha256=protected,
        links=links,
        source_sha256=source,
        boot_id=boot,
        gates=[],
        aot_caches=[str(p) for p in cache_paths],
        native_torch_revision=(candidate / "GIT_REVISION").read_text().strip(),
        runtime_sha256=baseline["runtime_sha256"],
        torch_target=str(candidate),
        rocm_target=str(prefix),
        prefix_changed=False,
        native_build_required=False,
        source_header_update=header_source,
        source_header_recovery=str(recovery),
    )
    atomic_json(LOGS / "before.json", report)
    prior = json.loads((WS / "logs/rng-n215-qualification/final.json").read_text())
    for name in (
        "validate_rng_n215.py",
        "validate_priority2_math.py",
        "validate_cleanup_runtime.py",
        "validate_torch_a04_on_host.py",
        "update_dev_prefix.py",
    ):
        assert prior["source_sha256"][name] == sha(WS / "helpers" / name), name
    report["retained_gates"] = prior["gates"]
    assert [g["checks"] for g in report["retained_gates"]] == [80, 22, 20, 1]
    for gate in report["retained_gates"]:
        check_gate(gate["label"])
    specs = [
        ("native-r2", "validate_rng_n215_native.py", [], 68),
        ("legacy", "validate_priority2_math.py", ["--group", "rng"], 28),
        ("core", "validate_cleanup_runtime.py", [], 7),
    ]
    try:
        for group, script, arguments, count in specs:
            label = "rng-n215-" + group
            command = [
                sys.executable,
                str(WS / "helpers/validate_torch_a04_on_host.py"),
                "--label",
                label,
                "--script",
                script,
            ]
            command += ["--test-arg=" + a for a in arguments]
            with (LOGS / (label + ".log")).open("w") as output:
                subprocess.run(
                    command,
                    check=True,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=570,
                )
            check_gate(label)
            result = json.loads((WS / "logs" / label / "result.json").read_text())
            rows = result.get("checks", result.get("results", []))
            assert result["status"] == "passed" and len(rows) == count, (label, result)
            guard = json.loads((Path("/run") / label / "state.json").read_text())
            assert guard["status"] == "acknowledged" and guard["timeout_seconds"] == 600
            report["gates"].append(
                dict(label=label, checks=count, command=command, guard=guard)
            )
            atomic_json(LOGS / "progress.json", report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["source_hashes_unchanged"] = all(
            sha(WS / "helpers" / name) == digest for name, digest in source.items()
        )
        report["changed_hashes"] = [
            p for p, digest in protected.items() if sha(Path(p)) != digest
        ]
        report["links_unchanged"] = all(
            os.readlink(p) == target for p, target in links.items()
        )
        report["kernel_log_unchanged_entire_batch"] = (
            subprocess.check_output(["dmesg"]) == before
        )
        report["kernel"] = subprocess.check_output(["uname", "-r"], text=True).strip()
        report["disk_available_bytes"] = (
            os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize
        )
        report["running_containers"] = subprocess.check_output(
            ["podman", "ps", "-q"], text=True
        ).split()
        assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == boot
        if (
            report["changed_hashes"]
            or not report["source_hashes_unchanged"]
            or report["running_containers"]
            or not report["links_unchanged"]
            or not report["kernel_log_unchanged_entire_batch"]
        ):
            report["status"] = "failed"
        atomic_json(LOGS / "final.json", report)
    assert report["status"] == "passed"
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "protected_sha256"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
