#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Guard optional W8 candidate and ordinary installed package acceptance."""

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
        for group, count in (
            ("imports", 4),
            ("module", 32),
            ("state", 36),
            ("resume", 3),
            ("recipe", 71),
        ):
            yield (
                cohort + "-" + group,
                cohort,
                "validate_quant_b056.py",
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
    for name, script, count in (
        ("core", "validate_cleanup_runtime.py", 7),
        ("fft", "validate_fft_b011.py", 245),
        ("sparse", "validate_sparse_b021.py", 253),
    ):
        yield name, "daily", script, [], count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    phase = "installed" if args.installed else "candidate"
    preflight("quant-b056-" + phase)
    stage = json.loads((WS / "logs/quant-b056-stage-r3/final.json").read_text())
    assert stage["status"] == "packaged-not-qualified"
    baseline = json.loads(
        (WS / "logs/quant-b055-qualification-r1/final.json").read_text()
    )
    assert baseline["status"] == "passed"
    protected = dict(baseline["protected_sha256"])
    if args.installed:
        deployed = json.loads((WS / "logs/quant-b056-deployment.json").read_text())
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
    for path in (WS / "helpers").glob("*quant_b056*.py"):
        protected[str(path)] = sha(path)
    protected[stage["wheel"]] = stage["wheel_sha256"]
    for item in stage["candidates"].values():
        protected[str(Path(item["release"]) / "runtime.json")] = item["runtime_sha256"]
    logs = WS / ("logs/quant-b056-" + phase + "-r1")
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
                label = "quant-b056-" + phase + "-r1-" + name
                if args.installed:
                    command = ["/usr/local/bin/bc250-torch"]
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
            assert sum(g["checks"] for g in report["gates"]) == 808
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
    print("QUANT_B056_" + phase.upper() + "_PASS", flush=True)


if __name__ == "__main__":
    main()
