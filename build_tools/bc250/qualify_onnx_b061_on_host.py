#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Guard ONNX candidate and ordinary installed package acceptance."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from bc250_torch import runtime_lock
from deploy_torch_a04_on_host import atomic_json, sha, check_gate
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight
from stage_quant_b056_on_host import ACTIVE


def cases():
    for cohort in ("daily", "rccl"):
        for group, count in (("audit", 4), ("export", 33), ("order", 2)):
            yield (
                cohort + "-" + group,
                cohort,
                "validate_onnx_b061.py",
                ["--group", group],
                count,
            )
        yield (
            cohort + "-consumers",
            cohort,
            "validate_rccl_b038.py",
            (["--daily"] if cohort == "daily" else []) + ["--consumers"],
            (5 if cohort == "daily" else 6),
        )
    yield "w8", "daily", "validate_quant_b056.py", ["--group", "recipe"], 71
    yield "core", "daily", "validate_cleanup_runtime.py", [], 7
    yield "compiled", "daily", "validate_compiler_n310.py", ["--family", "dense"], 14


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    phase = "installed" if args.installed else "candidate"
    preflight("onnx-b061-" + phase)
    stage = json.loads((WS / "logs/onnx-b061-stage-r1/final.json").read_text())
    assert stage["status"] == "packaged-not-qualified"
    baseline = json.loads((WS / "logs/quant-b056-installed-r3/final.json").read_text())
    assert baseline["status"] == "passed"
    protected = dict(baseline["protected_sha256"])
    if args.installed:
        deployed = json.loads((WS / "logs/onnx-b061-deployment.json").read_text())
        assert deployed["status"] == "installed-awaiting-acceptance"
        for name, active in ACTIVE.items():
            path = str(active / "runtime.json")
            # The prior baseline uses resolved RCCL manifest names.
            for old in list(protected):
                if Path(old).resolve() == (active / "runtime.json").resolve():
                    assert (
                        protected[old]
                        == stage["candidates"][name]["previous_runtime_sha256"]
                    )
                    protected[old] = deployed["installed_runtime_sha256"][name]
            protected[str(Path(stage["recovery"]) / (name + ".json"))] = stage[
                "candidates"
            ][name]["previous_runtime_sha256"]
    for path in (WS / "helpers").glob("*onnx_b061*"):
        protected[str(path)] = sha(path)
    for item in stage["packages"].values():
        protected[str(Path(stage["context"]) / "wheels" / item["filename"])] = item[
            "sha256"
        ]
    for name in ("Containerfile", "requirements.txt"):
        path = Path(stage["context"]) / name
        protected[str(path)] = sha(path)
    for item in stage["candidates"].values():
        protected[str(Path(item["release"]) / "runtime.json")] = item["runtime_sha256"]
    logs = WS / ("logs/onnx-b061-" + phase + "-r1")
    logs.mkdir()
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    report = dict(
        status="running",
        gates=[],
        protected_sha256=protected,
        links=baseline["links"],
        aot_caches=baseline["aot_caches"],
        boot_id=boot,
        phase=phase,
    )
    manifest = json.loads((ACTIVE["daily"] / "runtime.json").read_text())
    try:
        with runtime_lock(ACTIVE["daily"] / ".runtime.lock"):
            for name, digest in protected.items():
                assert sha(Path(name)) == digest, name
            for name, cohort, script, arguments, count in cases():
                label = "onnx-b061-" + phase + "-r1-" + name
                if args.installed:
                    command = [
                        "/usr/local/bin/"
                        + (
                            "bc250-train"
                            if script
                            not in ("validate_onnx_b061.py", "validate_quant_b056.py")
                            else "bc250-torch"
                        )
                    ]
                    if cohort == "rccl":
                        command += ["--runtime-profile", "rccl-single"]
                    command += [str(WS / "helpers" / script), *arguments]
                    record = gate(label, command, count, manifest)
                else:
                    command = [
                        sys.executable,
                        str(WS / "helpers/validate_torch_a04_on_host.py"),
                        "--guard-release",
                        os.uname().release,
                        "--label",
                        label,
                        "--release",
                        stage["candidates"][cohort]["release"],
                        "--script",
                        script,
                    ]
                    if cohort == "rccl":
                        command += ["--runtime-profile", "rccl-single"]
                    command += ["--test-arg=" + a for a in arguments]
                    with (logs / (label + ".log")).open("w") as stream:
                        subprocess.run(
                            command,
                            check=True,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            timeout=570,
                        )
                    check_gate(label)
                    result = json.loads(
                        (WS / "logs" / label / "result.json").read_text()
                    )
                    assert (
                        result["status"] == "passed" and len(result["checks"]) == count
                    ), result
                    guard = json.loads(
                        (Path("/run") / label / "state.json").read_text()
                    )
                    assert (
                        guard["status"] == "acknowledged" and guard["boot_id"] == boot
                    )
                    record = dict(
                        label=label, command=command, checks=count, guard=guard
                    )
                report["gates"].append(record)
                atomic_json(logs / "progress.json", report)
                print("PASS", name, count, flush=True)
            assert sum(g["checks"] for g in report["gates"]) == 181
            report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        after = subprocess.check_output(["dmesg"])
        (logs / "kernel-after.log").write_bytes(after)
        report["changed_hashes"] = [
            name for name, digest in protected.items() if sha(Path(name)) != digest
        ]
        report["links_unchanged"] = all(
            os.readlink(p) == v for p, v in baseline["links"].items()
        )
        report["kernel_log_unchanged_entire_batch"] = before == after
        report["boot_unchanged"] = (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip() == boot
        )
        report["running_containers"] = subprocess.check_output(
            ["podman", "ps", "-q"], text=True
        ).split()
        if (
            report["changed_hashes"]
            or not report["links_unchanged"]
            or not report["kernel_log_unchanged_entire_batch"]
            or not report["boot_unchanged"]
            or report["running_containers"]
        ):
            report["status"] = "failed"
        atomic_json(logs / "final.json", report)
    assert report["status"] == "passed"
    print("ONNX_B061_" + phase.upper() + "_PASS", flush=True)


if __name__ == "__main__":
    main()
