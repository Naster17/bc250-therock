#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Atomic Python/app deployment with rollback under exclusive prefix lock."""
import json
import os
from pathlib import Path
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, check_gate, sha

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
RELEASE = Path("/opt/bc250-stack/current").resolve()
TORCH = (RELEASE / "torch").resolve()
RECOVERY = WS / "recovery/a06-a07-before-r2"
LOGS = WS / "logs/a06-a07-deployment-r2"


def main() -> None:
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    source = json.loads((WS / "helpers/a06-a07-source.json").read_text())
    for label in source["candidate_gates"]:
        check_gate(label)
        assert (
            json.loads((WS / "logs" / label / "result.json").read_text())["status"]
            == "passed"
        )
    for name, digest in source["helper_sha256"].items():
        assert sha(WS / "helpers" / name) == digest, name
    assert TORCH == WS / "pytorch-sparse-followup"
    targets = {
        TORCH / "torch/nn/bc250_gated_delta.py": "bc250_gated_delta_qualified.py",
        TORCH / "torch/nn/attention/_gfx1013.py": "gfx1013_attention_d256_hybrid.py",
        Path("/home/nik/finetune-n7/replica/train.py"): "a07_app_train.py",
        Path("/home/nik/finetune-n7/replica/bc250_ops.py"): "a07_app_ops.py",
        Path("/home/nik/finetune-n7/scripts/train_bc250.py"): "a07_app_launcher.py",
    }
    for path, expected in source["before_sha256"].items():
        assert sha(Path(path)) == expected, path
    protected = [
        Path(p)
        for p in json.loads((WS / "SOURCE_GATED_DELTA_A06.json").read_text())[
            "protected_before"
        ]
        if Path(p) not in targets
    ]
    frozen = {str(p): sha(p) for p in protected}
    old = {p: p.read_bytes() for p in targets}
    old_runtime = (RELEASE / "runtime.json").read_bytes()
    old_revision = (TORCH / "GIT_REVISION").read_bytes()
    RECOVERY.mkdir()
    LOGS.mkdir()
    for path, data in old.items():
        name = (
            "gated_delta.py"
            if path == TORCH / "torch/nn/bc250_gated_delta.py"
            else (
                "attention.py"
                if path == TORCH / "torch/nn/attention/_gfx1013.py"
                else "launcher.py" if path.name == "train_bc250.py" else path.name
            )
        )
        (RECOVERY / name).write_bytes(data)
    (RECOVERY / "runtime.json").write_bytes(old_runtime)
    (RECOVERY / "GIT_REVISION").write_bytes(old_revision)
    report = dict(
        status="staging",
        source=source,
        protected_before=frozen,
        installed_gates=[],
        targets={str(p): name for p, name in targets.items()},
    )
    changed = []

    def replace(path: Path, data: bytes) -> None:
        temporary = path.with_name(path.name + ".a06-a07-new")
        assert not temporary.exists()
        temporary.write_bytes(data)
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.chown(temporary, path.stat().st_uid, path.stat().st_gid)
        os.replace(temporary, path)

    try:
        for path, name in targets.items():
            replace(path, (WS / "helpers" / name).read_bytes())
            changed.append(path)
        runtime = json.loads(old_runtime)
        runtime["previous_a06_a07_torch_revision"] = runtime["torch_revision"]
        runtime["torch_revision"] = source["torch_revision"]
        runtime["effective_therock_port_revision"] = source["therock_port_revision"]
        runtime["a06_gated_delta"].update(
            status="corrected installed gates pending",
            module_sha256=sha(TORCH / "torch/nn/bc250_gated_delta.py"),
            precision_policy="Reference-ordered GPU preparation/scan for low precision and autocast; native FP32/direct preparation retained",
        )
        runtime["a07_attention"] = dict(
            status="installed gates pending",
            module_sha256=sha(TORCH / "torch/nn/attention/_gfx1013.py"),
            policy="D<=256 explicit. BF16 D>128 training uses math forward/native tiled backward; quadratic forward temporaries, linear saved state. Math default.",
        )
        runtime[
            "source_state"
        ] += "; A06 precision correction, A07 D256 attention and opt-in trainer backend"
        atomic_json(RELEASE / "runtime.json", runtime)
        for label, script, args, count in [
            ("a06-installed-corrected", "validate_a06_a07.py", ["--component=a06"], 42),
            ("a07-installed-d256", "validate_a06_a07.py", ["--component=a07"], 36),
            ("a06-installed-autocast", "validate_gated_autocast_a06.py", [], 4),
            ("a06-a07-installed-model", "validate_a06_a07_model.py", [], 3),
            ("a06-a07-installed-cli", "validate_a06_a07_cli.py", [], 2),
            (
                "a06-a07-installed-training",
                "validate_training_a05.py",
                ["--installed"],
                69,
            ),
            (
                "a06-a07-installed-attention-regression",
                "validate_tiled_sdpa_integration.py",
                [],
                None,
            ),
            ("a06-a07-installed-core", "validate_cleanup_runtime.py", [], None),
        ]:
            label += "-r2"
            report["phase"] = label
            atomic_json(LOGS / "result.json", report)
            cmd = [
                sys.executable,
                str(WS / "helpers/validate_torch_a04_on_host.py"),
                "--label",
                label,
                "--script",
                script,
            ] + ["--test-arg=" + a for a in args]
            with (LOGS / (label + ".log")).open("w") as log:
                subprocess.run(
                    cmd, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=570
                )
            check_gate(label)
            result = json.loads((WS / "logs" / label / "result.json").read_text())
            assert result["status"] == "passed", label
            if count is not None:
                assert len(result["results"]) == count, label
            report["installed_gates"].append(label)
        assert {str(p): sha(p) for p in protected} == frozen
        for path, name in targets.items():
            assert sha(path) == source["helper_sha256"][name], str(path)
        (TORCH / "GIT_REVISION").write_text(source["torch_revision"] + "\n")
        runtime["a06_gated_delta"][
            "status"
        ] = "precision-corrected bounded installed gates passed"
        runtime["a07_attention"][
            "status"
        ] = "bounded installed D256/operator/LoRA/CLI gates passed"
        runtime["a06_a07_source_manifest"] = str(WS / "SOURCE_A06_A07.json")
        atomic_json(RELEASE / "runtime.json", runtime)
        report.update(
            status="passed",
            protected_unchanged=True,
            after_sha256={str(p): sha(p) for p in targets},
            running_agent_jobs=[],
        )
        for p in [
            WS / "SOURCE_A06_A07.json",
            RELEASE / "SOURCE_A06_A07.json",
            LOGS / "result.json",
        ]:
            atomic_json(p, report)
    except BaseException as error:
        for path in reversed(changed):
            replace(path, old[path])
        replace(RELEASE / "runtime.json", old_runtime)
        replace(TORCH / "GIT_REVISION", old_revision)
        report.update(status="failed_rolled_back", error=str(error))
        atomic_json(LOGS / "result.json", report)
        raise


if __name__ == "__main__":
    main()
