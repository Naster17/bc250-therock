#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install opt-in A06 Python module under the existing exclusive runtime lock."""
import json
import os
from pathlib import Path
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, check_gate, sha

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
RELEASE = Path("/opt/bc250-stack/current").resolve()
TORCH = (RELEASE / "torch").resolve()
TARGET = TORCH / "torch/nn/bc250_gated_delta.py"
RECOVERY = WS / "recovery/gated-delta-a06-before"
LOGS = WS / "logs/gated-delta-a06-deployment"


def main() -> None:
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    source = json.loads((WS / "helpers/gated-delta-a06-source.json").read_text())
    for label, count in source["candidate_gates"].items():
        check_gate(label, count)
        result = json.loads((WS / "logs" / label / "result.json").read_text())
        assert result["status"] == "passed", label
        hashes = (
            [row["source_sha256"] for row in result["results"]]
            if count
            else [result["source_sha256"]]
        )
        assert hashes and all(
            digest == source["module_sha256"] for digest in hashes
        ), label
    for name, digest in source["helper_sha256"].items():
        assert sha(WS / "helpers" / name) == digest, name
    assert TORCH == WS / "pytorch-sparse-followup"
    assert not TARGET.exists()
    assert sha(WS / "helpers/bc250_gated_delta.py") == source["module_sha256"]
    old_runtime = (RELEASE / "runtime.json").read_bytes()
    old_stamp = (TORCH / "GIT_REVISION").read_bytes()
    assert old_stamp.decode().strip() == source["parent_torch_revision"]
    protected = [
        Path(p)
        for p in json.loads((WS / "SOURCE_SPARSE_FOLLOWUP.json").read_text())[
            "protected_before"
        ]
    ]
    protected += list((TORCH / "torch/lib").glob("*.so"))
    protected += [TORCH / "torch/nn/bc250_training_ops.py"]
    frozen = {str(p): sha(p) for p in protected}
    RECOVERY.mkdir()
    LOGS.mkdir()
    (RECOVERY / "runtime.json").write_bytes(old_runtime)
    (RECOVERY / "GIT_REVISION").write_bytes(old_stamp)
    report = dict(
        status="staging",
        source=source,
        target=str(TARGET),
        protected_before=frozen,
        installed_gates=[],
    )
    changed = False
    try:
        temporary = TARGET.with_suffix(".a06-new")
        assert not temporary.exists()
        temporary.write_bytes((WS / "helpers/bc250_gated_delta.py").read_bytes())
        os.chmod(temporary, 0o644)
        os.replace(temporary, TARGET)
        changed = True
        runtime = json.loads(old_runtime)
        runtime["previous_a06_torch_revision"] = runtime["torch_revision"]
        runtime["torch_revision"] = source["torch_revision"]
        runtime["effective_therock_port_revision"] = source["therock_port_revision"]
        runtime[
            "source_state"
        ] += "; A06 opt-in native gated-delta preparation/backward, ATen solve/scan"
        runtime["a06_gated_delta"] = dict(
            status="installed gates pending",
            module="torch.nn.bc250_gated_delta",
            module_sha256=source["module_sha256"],
            source_manifest=str(WS / "SOURCE_GATED_DELTA_A06.json"),
        )
        atomic_json(RELEASE / "runtime.json", runtime)
        for suffix, script, arguments, count in [
            ("api", "validate_gated_delta_a06.py", ["--installed"], 40),
            ("profile", "benchmark_gated_delta_a06.py", [], None),
            ("training", "validate_training_a05.py", ["--installed"], 69),
            ("core", "validate_cleanup_runtime.py", [], None),
        ]:
            label = "gated-delta-a06-" + suffix + "-installed"
            report["phase"] = label
            atomic_json(LOGS / "result.json", report)
            command = [
                sys.executable,
                str(WS / "helpers/validate_torch_a04_on_host.py"),
                "--label",
                label,
                "--script",
                script,
            ]
            command += ["--test-arg=" + arg for arg in arguments]
            if suffix == "profile":
                command += ["--env", "BC250_A06_INSTALLED=1"]
            with (LOGS / (label + ".log")).open("w") as log:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=570,
                )
            check_gate(label, count)
            result = json.loads((WS / "logs" / label / "result.json").read_text())
            assert result["status"] == "passed"
            if suffix == "api":
                assert all(
                    row["module"]
                    == "/workspace/pytorch-work/torch/nn/bc250_gated_delta.py"
                    and row["source_sha256"] == source["module_sha256"]
                    for row in result["results"]
                )
            if suffix == "profile":
                assert result["source_sha256"] == source["module_sha256"]
            report["installed_gates"].append(label)
        assert {str(p): sha(p) for p in protected} == frozen
        (TORCH / "GIT_REVISION").write_text(source["torch_revision"] + "\n")
        runtime["a06_gated_delta"].update(
            status="bounded installed references and paired timings passed",
            installed_gates=report["installed_gates"],
            limitations="Opt-in eager first-order preparation only; ATen triangular solve and scan remain. No automatic trainer hook, FLA package, packed/cache/higher-order/compile/graph or universal speedup qualification.",
        )
        atomic_json(RELEASE / "runtime.json", runtime)
        report.update(status="passed", protected_unchanged=True, running_agent_jobs=[])
        for path in [
            WS / "SOURCE_GATED_DELTA_A06.json",
            RELEASE / "SOURCE_GATED_DELTA_A06.json",
            LOGS / "result.json",
        ]:
            atomic_json(path, report)
    except BaseException as error:
        if changed:
            TARGET.unlink()
            (TORCH / "GIT_REVISION").write_bytes(old_stamp)
            temporary = RELEASE / "runtime.json.a06-rollback"
            temporary.write_bytes(old_runtime)
            os.replace(temporary, RELEASE / "runtime.json")
        report.update(
            status="failed_rolled_back" if changed else "failed_before_deployment",
            error=str(error),
        )
        atomic_json(LOGS / "result.json", report)
        raise


if __name__ == "__main__":
    main()
