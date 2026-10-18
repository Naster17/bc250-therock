#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only installed-runtime FFT qualification under the exclusive lock."""

import json
import os
from pathlib import Path
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, check_gate, sha

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
LOGS = WS / "logs/fft-n214-qualification-r2"


def main() -> None:
    assert os.geteuid() == 0
    LOGS.mkdir()
    baseline = json.loads(
        (WS / "logs/sparse-n213-qualification/final.json").read_text()
    )
    protected = dict(baseline["protected_sha256"])
    candidate = Path(baseline["torch_target"])
    protected.update(
        {str(candidate / p): d for p, d in baseline["candidate_sha256"].items()}
    )
    protected["/opt/bc250-stack/current/runtime.json"] = baseline["runtime_sha256"]
    prefix = Path(baseline["rocm_target"])
    fft_paths = [prefix / "lib/libhipfft.so", prefix / "lib/librocfft.so"]
    fft_paths += list((prefix / ".kpack").glob("*fft*"))
    cache_paths = sorted((WS / "build/math-libs/rocFFT").rglob("*.db"))
    assert len(cache_paths) == 4, cache_paths
    fft_paths += cache_paths
    for p in fft_paths:
        protected[str(p)] = sha(p)
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
            WS / "helpers/validate_fft_n214.py",
            WS / "helpers/validate_fft_n214_native.py",
            WS / "helpers/validate_priority2_math.py",
            WS / "helpers/validate_cleanup_runtime.py",
            WS / "helpers/validate_torch_a04_on_host.py",
            WS / "helpers/update_dev_prefix.py",
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
    )
    atomic_json(LOGS / "before.json", report)
    specs = [
        ("complex", "validate_fft_n214.py", ["--group", "complex"], 96),
        ("real", "validate_fft_n214.py", ["--group", "real"], 72),
        ("contracts", "validate_fft_n214.py", ["--group", "contracts"], 3),
        ("native", "validate_fft_n214_native.py", [], 16),
        ("legacy", "validate_priority2_math.py", ["--group", "fft"], 30),
        ("core", "validate_cleanup_runtime.py", [], 7),
    ]
    try:
        for group, script, arguments, count in specs:
            label = "fft-n214-r2-" + group
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
            report["gates"].append(dict(label=label, checks=count, command=command))
            atomic_json(LOGS / "progress.json", report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
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
