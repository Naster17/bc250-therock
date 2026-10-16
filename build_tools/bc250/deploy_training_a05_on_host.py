#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deploy the opt-in Python module under update_dev_prefix's exclusive lock."""
import json
import os
from pathlib import Path
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, check_gate, sha

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
RELEASE = Path("/opt/bc250-stack/current").resolve()
TORCH = (RELEASE / "torch").resolve()
TARGET = TORCH / "torch/nn/bc250_training_ops.py"
LOGS = WS / "logs/training-a05-deployment"
RECOVERY = WS / "recovery/training-a05-before"


def main() -> None:
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    check_gate("training-a05-candidate-final", 69)
    source = WS / "helpers/bc250_training_ops.py"
    source_hash = sha(source)
    candidate = json.loads((WS / "logs/training-a05-candidate-final/result.json").read_text())
    assert all(row["source_sha256"] == source_hash for row in candidate["results"])
    assert TORCH == WS / "pytorch-a04"
    assert not TARGET.exists()
    RECOVERY.mkdir()
    LOGS.mkdir()
    old_runtime = (RELEASE / "runtime.json").read_bytes()
    (RECOVERY / "runtime.json").write_bytes(old_runtime)
    old_stamp = (TORCH / "GIT_REVISION").read_bytes()
    (RECOVERY / "GIT_REVISION").write_bytes(old_stamp)
    a04 = json.loads((WS / "SOURCE_TORCH_A04.json").read_text())
    protected = [Path(p) for p in a04["protected_before"]]
    protected += list((TORCH / "torch/lib").glob("*.so"))
    protected += [Path("/home/nik/finetune-n7/replica") / name
                  for name in ["train.py", "bc250_ops.py"]]
    protected += [RELEASE / "python-consumers/training.env"]
    frozen = {str(p): sha(p) for p in protected}
    revision = (WS / "helpers/GIT_REVISION_A05").read_text().strip()
    report = dict(status="staging", torch_revision=revision,
                  before_torch_revision=old_stamp.decode().strip(), source_sha256=source_hash,
                  target=str(TARGET), protected_before=frozen, installed_gates=[],
                  helper_sha256={name: sha(WS / "helpers" / name) for name in
                                 ["deploy_training_a05_on_host.py", "test_training_a05.py", "validate_training_a05.py"]})
    changed = False
    try:
        temporary = TARGET.with_suffix(".a05-new")
        assert not temporary.exists()
        temporary.write_bytes(source.read_bytes())
        os.chmod(temporary, 0o644)
        os.replace(temporary, TARGET)
        changed = True
        runtime = json.loads(old_runtime)
        runtime["previous_a05_torch_revision"] = runtime["torch_revision"]
        runtime["torch_revision"] = revision
        runtime["source_state"] += "; A05 opt-in Python causal-conv all gradients/chunked-loss API"
        runtime["a05_training_ops"] = dict(status="installed gates pending",
                                           module="torch.nn.bc250_training_ops",
                                           source_sha256=source_hash,
                                           source_manifest=str(WS / "SOURCE_TRAINING_A05.json"))
        atomic_json(RELEASE / "runtime.json", runtime)
        for suffix, script, arguments, count in [
            ("api", "validate_training_a05.py", ["--installed"], 69),
            ("core", "validate_cleanup_runtime.py", [], None),
            ("sparse", "validate_torch_sparse_a04.py", [], 20),
        ]:
            label = "training-a05-" + suffix + "-installed"
            report["phase"] = label
            atomic_json(LOGS / "result.json", report)
            command = [sys.executable, str(WS / "helpers/validate_torch_a04_on_host.py"),
                       "--label", label, "--script", script]
            command += ["--test-arg=" + arg for arg in arguments]
            with (LOGS / (label + ".log")).open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=570)
            check_gate(label, count)
            result = json.loads((WS / "logs" / label / "result.json").read_text())
            assert result["status"] == "passed"
            if suffix == "api":
                assert all(row["module"] == "/workspace/pytorch-work/torch/nn/bc250_training_ops.py"
                           and row["source_sha256"] == source_hash for row in result["results"])
            report["installed_gates"].append(label)
        assert {str(p): sha(p) for p in protected} == frozen
        (TORCH / "GIT_REVISION").write_text(revision + "\n")
        runtime["a05_training_ops"].update(status="bounded installed GPU/reference gates passed",
                                          installed_gates=report["installed_gates"],
                                          limitations="Opt-in eager first-order only; no automatic model hooks, FLA, compile/graph or sustained performance qualification")
        atomic_json(RELEASE / "runtime.json", runtime)
        report.update(status="passed", protected_unchanged=True, running_agent_jobs=[])
        for p in [WS / "SOURCE_TRAINING_A05.json", RELEASE / "SOURCE_TRAINING_A05.json", LOGS / "result.json"]:
            atomic_json(p, report)
    except BaseException as error:
        if changed:
            TARGET.unlink()
            (TORCH / "GIT_REVISION").write_bytes(old_stamp)
            temporary = RELEASE / "runtime.json.a05-rollback"
            temporary.write_bytes(old_runtime)
            os.replace(temporary, RELEASE / "runtime.json")
        report.update(status="failed_rolled_back" if changed else "failed_before_deployment", error=str(error))
        atomic_json(LOGS / "result.json", report)
        raise


if __name__ == "__main__":
    main()
