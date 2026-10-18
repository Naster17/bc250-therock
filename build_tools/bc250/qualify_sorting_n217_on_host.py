#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Targeted primitive builds and guarded installed-runtime qualification."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, check_gate, sha

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="r1")
    args = parser.parse_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 2 * 1024**3
    logs = WS / ("logs/sorting-n217-qualification-" + args.attempt)
    logs.mkdir()
    baseline = json.loads(
        (WS / "logs/primitives-n216-qualification-r1/final.json").read_text()
    )
    protected = dict(baseline["protected_sha256"])
    protected.update(baseline["binary_sha256"])
    prefix = Path(baseline["rocm_target"])
    for name in (
        "rocprim/device/device_radix_sort.hpp",
        "rocprim/device/device_merge_sort.hpp",
        "rocprim/device/device_select.hpp",
        "rocprim/type_traits.hpp",
        "hipcub/backend/rocprim/device/device_radix_sort.hpp",
        "hipcub/backend/rocprim/device/device_merge_sort.hpp",
        "hipcub/backend/rocprim/device/device_select.hpp",
        "thrust/sort.h",
        "thrust/copy.h",
    ):
        path = prefix / "include" / name
        protected[str(path)] = sha(path)
    common = WS / "helpers/primitives_n216_common.hpp"
    assert sha(common) == baseline["source_sha256"][common.name]
    protected[str(common)] = sha(common)
    for p, digest in protected.items():
        assert sha(Path(p)) == digest, p
    source = {
        name: sha(WS / "helpers" / name)
        for name in (
            Path(__file__).name,
            "sorting_n217.cpp",
            "primitives_n216_common.hpp",
            "build_sorting_n217.sh",
            "validate_sorting_n217.py",
            "validate_cleanup_runtime.py",
            "validate_sorting_n217_torch.py",
            "validate_torch_a04_on_host.py",
            "update_dev_prefix.py",
        )
    }
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    report = dict(
        status="running",
        protected_sha256=protected,
        source_sha256=source,
        links=baseline["links"],
        boot_id=boot,
        gates=[],
        torch_target=baseline["torch_target"],
        rocm_target=str(prefix),
        runtime_sha256=baseline["runtime_sha256"],
        aot_caches=baseline["aot_caches"],
        prefix_changed=False,
        library_or_torch_build_required=False,
    )
    atomic_json(logs / "before.json", report)
    name = "bc250-sorting-n217-build-" + args.attempt
    output = WS / ("build/bc250-coverage/sorting-n217-" + args.attempt)
    output.mkdir()
    command = ["podman", "run", "--rm", "--name", name, "--network", "none"]
    for src, dst, mode in (
        (WS / "source", "/workspace/source", "ro"),
        (WS / "helpers", "/workspace/helpers", "ro"),
        (WS / "build", "/workspace/build", "ro"),
        (WS / "cache", "/workspace/cache", "rw"),
        (output, "/workspace/output", "rw"),
    ):
        command += ["-v", f"{src}:{dst}:{mode}"]
    command += [
        "localhost/bc250-rocm-build:10.2-ccache",
        "bash",
        "/workspace/helpers/build_sorting_n217.sh",
    ]
    report["build_command"] = command
    try:
        with (logs / "build.log").open("w") as log:
            subprocess.run(
                command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1200
            )
        report["binary_sha256"] = {
            str(output / n): sha(output / n) for n in ("rocprim", "hipcub", "thrust")
        }
        atomic_json(logs / "progress.json", report)
        for group, script, arguments, count in (
            (
                "rocprim",
                "validate_sorting_n217.py",
                ["--library", "rocprim", "--attempt", args.attempt],
                656,
            ),
            (
                "hipcub",
                "validate_sorting_n217.py",
                ["--library", "hipcub", "--attempt", args.attempt],
                656,
            ),
            (
                "thrust",
                "validate_sorting_n217.py",
                ["--library", "thrust", "--attempt", args.attempt],
                648,
            ),
            ("legacy", "validate_sorting_n217.py", ["--library", "legacy"], 3),
            ("torch", "validate_sorting_n217_torch.py", [], 62),
            ("core", "validate_cleanup_runtime.py", [], 7),
        ):
            label = "sorting-n217-" + args.attempt + "-" + group
            command = [
                sys.executable,
                str(WS / "helpers/validate_torch_a04_on_host.py"),
                "--label",
                label,
                "--script",
                script,
            ]
            command += ["--test-arg=" + a for a in arguments]
            with (logs / (label + ".log")).open("w") as log:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=570,
                )
            check_gate(label)
            result = json.loads((WS / "logs" / label / "result.json").read_text())
            assert result["status"] == "passed" and len(result["checks"]) == count
            guard = json.loads((Path("/run") / label / "state.json").read_text())
            assert guard["status"] == "acknowledged" and guard["timeout_seconds"] == 600
            report["gates"].append(
                dict(label=label, checks=count, command=command, guard=guard)
            )
            atomic_json(logs / "progress.json", report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        subprocess.run(
            ["podman", "stop", "--time", "3", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        report["changed_hashes"] = [
            p for p, digest in protected.items() if sha(Path(p)) != digest
        ]
        report["source_hashes_unchanged"] = all(
            sha(WS / "helpers" / n) == d for n, d in source.items()
        )
        report["links_unchanged"] = all(
            os.readlink(p) == target for p, target in report["links"].items()
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
            or not report["links_unchanged"]
            or not report["kernel_log_unchanged_entire_batch"]
            or report["running_containers"]
        ):
            report["status"] = "failed"
        atomic_json(logs / "final.json", report)
    assert report["status"] == "passed"
    print("SORTING_N217_PASS", logs)


if __name__ == "__main__":
    main()
