#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage and qualify the isolated native Torch layout fix under the daily EX lock."""

import json
import os
from pathlib import Path
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, sha, check_gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

GATES = (
    ("layouts", "validate_quant_b052.py", [], 220),
    ("recipe", "validate_quant_b051.py", [], 71),
    ("core", "validate_cleanup_runtime.py", [], 7),
    ("sparse", "validate_sparse_b021.py", [], 253),
    ("fft", "validate_fft_b011.py", [], 245),
    ("compiled-dense", "validate_compiler_n310.py", ["--family", "dense"], 14),
    ("compiled-cnn", "validate_compiler_n310.py", ["--family", "cnn"], 14),
    ("profiler", "validate_distributed_profiler_b031.py", [], 1),
    ("gloo", "validate_distributed_b031.py", [], 31),
    ("vision", "validate_vision_b041.py", ["--group", "ops"], 51),
    ("audio", "validate_audio_b042.py", ["--group", "native", "--codec-available"], 40),
)


def main():
    preflight("quant-b052-qualification")
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    baseline = json.loads(
        (WS / "logs/quant-b051-qualification-r1/final.json").read_text()
    )
    built = json.loads((WS / "logs/quant-b052-build/final.json").read_text())
    assert baseline["status"] == "passed" and built["status"] == "built-not-qualified"
    candidate = Path(built["candidate"])
    protected = dict(baseline["protected_sha256"])
    for name, digest in built["native_sha256"].items():
        protected[str(candidate / "torch/lib" / name)] = digest
    for name, digest in built["post_hipify_source_sha256"].items():
        protected[str(candidate / name)] = digest
    protected[str(candidate / "GIT_REVISION")] = sha(candidate / "GIT_REVISION")
    protected[str(candidate / "aten/src/ATen/native/hip/int8mm.hip")] = built[
        "generated_hip_sha256"
    ]
    protected[str(candidate / "build/CMakeCache.txt")] = sha(
        candidate / "build/CMakeCache.txt"
    )
    for name in (
        "build_quant_b052_on_host.py",
        "build_torch_b052.sh",
        "seal_quant_b052_build.py",
        "qualify_quant_b052_on_host.py",
        "validate_quant_b052.py",
    ):
        protected[str(WS / "helpers" / name)] = sha(WS / "helpers" / name)
    daily = Path("/opt/bc250-stack/current").resolve(strict=True)
    release = WS / "quant-b052/candidate-release-r1"
    release.mkdir(parents=True)
    for p in daily.iterdir():
        if p.name not in ("runtime.json", "torch"):
            (release / p.name).symlink_to(p)
    (release / "torch").symlink_to(candidate)
    manifest = json.loads((daily / "runtime.json").read_text())
    manifest["torch_revision"] = built["source"]["torch_revision"]
    manifest["quant_b052"] = dict(
        status="native candidate; qualification outcome below",
        source=built["source"],
        build_evidence=str(WS / "logs/quant-b052-build/final.json"),
        qualification_evidence=str(WS / "logs/quant-b052-qualification-r1/final.json"),
        scope="isolated layout correction; daily and opt-in unchanged; native backward/INT4 remain unsupported",
    )
    atomic_json(release / "runtime.json", manifest)
    protected[str(release / "runtime.json")] = sha(release / "runtime.json")
    logs = WS / "logs/quant-b052-qualification-r1"
    logs.mkdir()
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    report = dict(
        status="running",
        gates=[],
        protected_sha256=protected,
        links=baseline["links"],
        aot_caches=baseline["aot_caches"],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        candidate_release=str(release),
        candidate_runtime_sha256=sha(release / "runtime.json"),
        source=built["source"],
        native_sha256=built["native_sha256"],
        default_changed=False,
        host105_contacted=False,
    )
    atomic_json(logs / "before.json", report)
    try:
        for name, digest in protected.items():
            assert sha(Path(name)) == digest, name
        for name, script, args, count in GATES:
            label = "quant-b052-r1-" + name
            command = [
                sys.executable,
                str(WS / "helpers/validate_torch_a04_on_host.py"),
                "--guard-release",
                os.uname().release,
                "--release",
                str(release),
                "--label",
                label,
                "--script",
                script,
                *["--test-arg=" + a for a in args],
            ]
            with (logs / (name + ".log")).open("w") as stream:
                subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=570,
                )
            check_gate(label)
            result = json.loads((WS / "logs" / label / "result.json").read_text())
            assert result["status"] == "passed" and len(result["checks"]) == count
            guard = json.loads((Path("/run") / label / "state.json").read_text())
            assert guard["status"] == "acknowledged"
            report["gates"].append(
                dict(label=label, checks=count, guard=guard, command=command)
            )
            atomic_json(logs / "progress.json", report)
            print("PASS", name, count, flush=True)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        after = subprocess.check_output(["dmesg"])
        (logs / "kernel-after.log").write_bytes(after)
        report["changed_hashes"] = [
            p for p, d in protected.items() if sha(Path(p)) != d
        ]
        report["links_unchanged"] = all(
            os.readlink(p) == v for p, v in baseline["links"].items()
        )
        report["kernel_log_unchanged_entire_batch"] = before == after
        report["boot_unchanged"] = (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            == report["boot_id"]
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
    print("QUANT_B052_PASS", flush=True)


if __name__ == "__main__":
    main()
