#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reversible sparse-autograd candidate switch under exclusive runtime lock."""
import json
import os
from pathlib import Path
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, check_gate, sha, switch

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
RELEASE = Path("/opt/bc250-stack/current").resolve()
CANDIDATE = WS / "pytorch-sparse-followup"
RECOVERY = WS / "recovery/sparse-followup-before"
LOGS = WS / "logs/sparse-followup-deployment"


def main() -> None:
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    source = json.loads((WS / "helpers/sparse-followup-source.json").read_text())
    for label, count in source["candidate_gates"].items():
        check_gate(label, count)
    for name, digest in source["candidate_library_sha256"].items():
        assert sha(CANDIDATE / "torch/lib" / name) == digest, name
    for name, digest in source["helper_sha256"].items():
        assert sha(WS / "helpers" / name) == digest, name
    assert (RELEASE / "torch").resolve() == WS / "pytorch-a04"
    assert (CANDIDATE / "GIT_REVISION").read_text().strip() == source["torch_revision"]
    old_target = os.readlink(RELEASE / "torch")
    old_runtime = (RELEASE / "runtime.json").read_bytes()
    protected = [Path(p) for p in json.loads((WS / "SOURCE_TRAINING_A05.json").read_text())["protected_before"]]
    protected += [WS / "pytorch-a04/torch/nn/bc250_training_ops.py"]
    frozen = {str(p): sha(p) for p in protected}
    RECOVERY.mkdir(); LOGS.mkdir()
    (RECOVERY / "runtime.json").write_bytes(old_runtime)
    (RECOVERY / "torch-link.txt").write_text(old_target + "\n")
    report = dict(status="staging", source=source, protected_before=frozen,
                  old_torch_target=old_target, new_torch_target=str(CANDIDATE), installed_gates=[])
    changed = False
    try:
        switch(CANDIDATE)
        changed = True
        runtime = json.loads(old_runtime)
        runtime["previous_sparse_followup_torch_revision"] = runtime["torch_revision"]
        runtime["torch_revision"] = source["torch_revision"]
        runtime["effective_therock_port_revision"] = source["therock_port_revision"]
        runtime["a05_training_ops"]["source_sha256"] = sha(CANDIDATE / "torch/nn/bc250_training_ops.py")
        runtime["source_state"] += "; native structured sparse first-order backward and retained-slot conversion gradients"
        runtime["sparse_autograd_followup"] = dict(status="installed gates pending",
                                                  source_manifest=str(WS / "SOURCE_SPARSE_FOLLOWUP.json"))
        atomic_json(RELEASE / "runtime.json", runtime)
        for suffix, script, arguments, count in [
            ("grad", "validate_sparse_followup.py", [], 38),
            ("forward", "validate_torch_sparse_a04.py", [], 20),
            ("training", "validate_training_a05.py", ["--installed"], 69),
            ("core", "validate_cleanup_runtime.py", [], None),
            ("conv", "validate_miopen_a01_torch.py", [], 30),
            ("bench", "validate_sparse_followup.py", ["--benchmark"], 12),
        ]:
            label = "sparse-followup-" + suffix + "-installed"
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
            report["installed_gates"].append(label)
        assert {str(p): sha(p) for p in protected} == frozen
        runtime["sparse_autograd_followup"].update(
            status="bounded installed backward/forward/reference and paired eager timing gates passed",
            installed_gates=report["installed_gates"],
            library_sha256=source["candidate_library_sha256"],
            limitations="Floating first-order retained-slot gradients; dense GPU backward intermediates; no integer/higher-order/compile/graph or universal speedup claim")
        atomic_json(RELEASE / "runtime.json", runtime)
        report.update(status="passed", protected_unchanged=True, running_agent_jobs=[])
        for path in [WS / "SOURCE_SPARSE_FOLLOWUP.json", RELEASE / "SOURCE_SPARSE_FOLLOWUP.json", LOGS / "result.json"]:
            atomic_json(path, report)
    except BaseException as error:
        if changed:
            switch(old_target)
            temporary = RELEASE / "runtime.json.sparse-rollback"
            temporary.write_bytes(old_runtime)
            os.replace(temporary, RELEASE / "runtime.json")
        report.update(status="failed_rolled_back" if changed else "failed_before_switch", error=str(error))
        atomic_json(LOGS / "result.json", report)
        raise


if __name__ == "__main__":
    main()
