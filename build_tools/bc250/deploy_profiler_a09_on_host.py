#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Activate a qualified Kineto build reversibly under the exclusive runtime lock."""
import json
import os
from pathlib import Path
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, check_gate, sha, switch

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
RELEASE = Path("/opt/bc250-stack/current").resolve()
CANDIDATE = WS / "pytorch-profiler-a09"


def main() -> None:
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    source = json.loads((WS / "helpers/profiler-a09-source.json").read_text())
    qualified = json.loads(
        (WS / "logs/profiler-a09-candidate-regressions/result.json").read_text()
    )
    assert qualified["status"] == "passed" and len(qualified["passed_gates"]) == 9
    for label in qualified["passed_gates"]:
        check_gate(label)
    for name, digest in source["candidate_library_sha256"].items():
        assert sha(CANDIDATE / "torch/lib" / name) == digest, name
    assert source["candidate_library_sha256"] == qualified["candidate_library_sha256"]
    for name, digest in source["helper_sha256"].items():
        assert sha(WS / "helpers" / name) == digest, name
    assert (CANDIDATE / "GIT_REVISION").read_text().strip() == source["torch_revision"]
    assert (RELEASE / "torch").resolve() == WS / "pytorch-sparse-followup"
    old_target = os.readlink(RELEASE / "torch")
    old_runtime = (RELEASE / "runtime.json").read_bytes()
    prior = json.loads((WS / "SOURCE_A06_A07.json").read_text())
    protected = list(prior["protected_before"]) + list(prior["after_sha256"])
    protected += [
        str(RELEASE / "rocm/lib/librocprofiler-sdk.so"),
        str(RELEASE / "rocm/lib/librocprofiler-register.so"),
        str(WS / "pytorch-sparse-followup/GIT_REVISION"),
    ]
    frozen = {p: sha(Path(p)) for p in protected}
    assert all(frozen[p] == digest for p, digest in prior["protected_before"].items())
    assert all(frozen[p] == digest for p, digest in prior["after_sha256"].items())
    recovery = WS / "recovery/profiler-a09-before"
    logs = WS / "logs/profiler-a09-deployment"
    recovery.mkdir()
    logs.mkdir()
    (recovery / "runtime.json").write_bytes(old_runtime)
    (recovery / "torch-link.txt").write_text(old_target + "\n")
    report = dict(
        status="staging",
        source=source,
        old_torch_target=old_target,
        new_torch_target=str(CANDIDATE),
        protected_before=frozen,
        installed_gates=[],
    )
    changed = False
    try:
        switch(CANDIDATE)
        changed = True
        runtime = json.loads(old_runtime)
        runtime["previous_profiler_a09_torch_revision"] = runtime["torch_revision"]
        runtime["torch_revision"] = source["torch_revision"]
        runtime["effective_therock_port_revision"] = source["therock_port_revision"]
        runtime[
            "source_state"
        ] += "; Kineto enabled with native rocprofiler-sdk tracing, no architecture override"
        runtime["profiler_a09"] = dict(
            status="installed qualification pending",
            backend="rocprofiler-sdk",
            source_manifest=str(WS / "SOURCE_PROFILER_A09.json"),
            library_sha256=source["candidate_library_sha256"],
        )
        atomic_json(RELEASE / "runtime.json", runtime)
        for name, script, count in [
            ("api", "validate_profiler_a09.py", 7),
            ("core", "validate_cleanup_runtime.py", None),
            ("model", "validate_a06_a07_model.py", 3),
            ("cli", "validate_a06_a07_cli.py", 2),
        ]:
            label = "profiler-a09-installed-" + name
            report["phase"] = label
            atomic_json(logs / "result.json", report)
            with (logs / (name + ".log")).open("w") as output:
                subprocess.run(
                    [
                        sys.executable,
                        str(WS / "helpers/validate_torch_a04_on_host.py"),
                        "--label",
                        label,
                        "--script",
                        script,
                    ],
                    check=True,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=570,
                )
            check_gate(label)
            result = json.loads((WS / "logs" / label / "result.json").read_text())
            assert result["status"] == "passed"
            if count is not None:
                assert len(result["results"]) == count
            report["installed_gates"].append(label)
        assert {p: sha(Path(p)) for p in protected} == frozen
        for name, digest in source["candidate_library_sha256"].items():
            assert sha(CANDIDATE / "torch/lib" / name) == digest
        runtime["profiler_a09"].update(
            status="bounded installed trace/reference/lifecycle and model/resume gates passed",
            installed_gates=report["installed_gates"],
            limitations="No hardware counters, broad concurrency, compile/graph profiling, long trace or universal overhead qualification",
        )
        atomic_json(RELEASE / "runtime.json", runtime)
        report.update(status="passed", protected_unchanged=True, running_agent_jobs=[])
        for path in [
            WS / "SOURCE_PROFILER_A09.json",
            RELEASE / "SOURCE_PROFILER_A09.json",
            logs / "result.json",
        ]:
            atomic_json(path, report)
    except BaseException as error:
        if changed:
            switch(old_target)
            temporary = RELEASE / "runtime.json.a09-rollback"
            assert not temporary.exists()
            temporary.write_bytes(old_runtime)
            os.replace(temporary, RELEASE / "runtime.json")
        report.update(
            status="failed_rolled_back" if changed else "failed_before_switch",
            error=str(error),
        )
        atomic_json(logs / "result.json", report)
        raise


if __name__ == "__main__":
    main()
