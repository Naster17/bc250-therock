#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Switch a qualified Torch candidate under the exclusive prefix-update lock."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
PREFIX = WS / "build/dist/rocm"
RELEASE = Path("/opt/bc250-stack/current").resolve()
CANDIDATE = WS / "pytorch-a04"
RECOVERY = WS / "recovery/torch-a04-before"
LOGS = WS / "logs/torch-a04-deployment"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".a04-new")
    assert not temporary.exists()
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def switch(target):
    temporary = RELEASE / "torch.a04-new"
    assert not temporary.is_symlink() and not temporary.exists()
    temporary.symlink_to(target)
    os.replace(temporary, RELEASE / "torch")


def check_gate(label, count=None):
    folder = WS / "logs" / label
    assert json.loads((folder / "exit.json").read_text()) == dict(
        returncode=0, kernel_log_unchanged=True
    ), label
    assert "GUARD_ACKNOWLEDGED" in (folder / "guard.log").read_text(), label
    if count is not None:
        result = json.loads((folder / "result.json").read_text())
        assert result["status"] == "passed", label
        assert len(result["results"]) == count, label
        assert all(row["status"] == "passed" for row in result["results"]), label


def main():
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    check_gate("torch-a04-candidate-r3", 20)
    source = json.loads((WS / "helpers/torch-a04-source.json").read_text())
    assert (RELEASE / "torch").is_symlink()
    old_target = os.readlink(RELEASE / "torch")
    assert (RELEASE / "torch").resolve() == WS / "pytorch-work"
    for name, digest in source["candidate_library_sha256"].items():
        assert sha(CANDIDATE / "torch/lib" / name) == digest, name
    for name, digest in source["helper_sha256"].items():
        assert sha(WS / "helpers" / name) == digest, name
    assert (CANDIDATE / "GIT_REVISION").read_text().strip() == source["torch_revision"]
    cache = (CANDIDATE / "build/CMakeCache.txt").read_text()
    assert "hipsparselt_DIR:PATH=/workspace/build/dist/rocm/lib/cmake/hipsparselt" in cache
    protected = [
        PREFIX / "lib" / name for name in [
            "libhsa-runtime64.so.1.21.0", "libamdhip64.so", "libMIOpen.so.1.0",
            "librocblas.so", "libhipblaslt.so", "libhipsparselt.so",
        ]
    ]
    protected += [PREFIX / ".kpack/blas_lib_gfx1013.kpack"]
    protected += [p for p in (PREFIX / "share/miopen/db").iterdir() if p.is_file()]
    protected += [
        WS / "pytorch-work/torch/lib/libtorch_hip.so",
        Path("/opt/bc250-stack/previous/rocm/lib/libhsa-runtime64.so.1.21.0"),
        Path("/opt/bc250-stack/previous/rocm/lib/libMIOpen.so.1.0"),
        Path("/opt/bc250-stack/previous/torch/torch/lib/libtorch_hip.so"),
    ]
    frozen = {str(p): sha(p) for p in protected}
    LOGS.mkdir()
    old_runtime = (RELEASE / "runtime.json").read_bytes()
    backup = RECOVERY / "runtime.before-a04.json"
    assert not backup.exists()
    backup.write_bytes(old_runtime)
    (RECOVERY / "torch-link.before-a04.txt").write_text(old_target + "\n")
    report = dict(status="staging", source=source, protected_before=frozen,
                  old_torch_target=old_target, new_torch_target=str(CANDIDATE),
                  installed_gates=[])
    changed = False
    try:
        switch(CANDIDATE)
        changed = True
        runtime = json.loads(old_runtime)
        runtime["previous_a04_torch_revision"] = runtime["torch_revision"]
        runtime["torch_revision"] = source["torch_revision"]
        runtime["source_state"] += "; A04 native Torch hipSPARSELt integration"
        runtime["a04_torch_hipsparselt"] = dict(
            status="installed gates pending",
            source_manifest=str(WS / "SOURCE_TORCH_A04.json"),
            limitations="Native scalar same-dtype FP32/FP16/BF16/INT8 only; no sparse hardware speedup, automatic dense replacement, or sustained qualification",
        )
        atomic_json(RELEASE / "runtime.json", runtime)
        for suffix, script, testargs, count in [
            ("sparse", "validate_torch_sparse_a04.py", [], 20),
            ("conv", "validate_miopen_a01_torch.py", [], 30),
            ("core", "validate_cleanup_runtime.py", [], None),
            ("attention", "validate_tiled_sdpa.py", ["--smoke"], None),
        ]:
            label = "torch-a04-" + suffix + "-installed"
            command = [sys.executable, str(WS / "helpers/validate_torch_a04_on_host.py"),
                       "--label", label, "--script", script]
            command += ["--test-arg=" + arg for arg in testargs]
            report["phase"] = label
            atomic_json(LOGS / "result.json", report)
            with (LOGS / (label + ".log")).open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=570)
            check_gate(label, count)
            result = json.loads((WS / "logs" / label / "result.json").read_text())
            assert result["status"] == "passed", label
            text = (WS / "logs" / label / "gpu.log").read_text()
            assert "CK grouped conv library not found" not in text
            assert not any("Warning" in line and "File is unreadable" in line
                           for line in text.splitlines())
            report["installed_gates"].append(label)
        assert {str(p): sha(p) for p in protected} == frozen
        runtime["a04_torch_hipsparselt"].update(
            status="bounded installed GPU/reference gates passed",
            installed_gates=report["installed_gates"],
            library_sha256=source["candidate_library_sha256"],
        )
        atomic_json(RELEASE / "runtime.json", runtime)
        report.update(status="passed", protected_unchanged=True, running_agent_jobs=[])
        atomic_json(WS / "SOURCE_TORCH_A04.json", report)
        atomic_json(RELEASE / "SOURCE_TORCH_A04.json", report)
        atomic_json(LOGS / "result.json", report)
    except BaseException as error:
        if changed:
            switch(old_target)
            temporary = RELEASE / "runtime.json.a04-rollback"
            temporary.write_bytes(old_runtime)
            os.replace(temporary, RELEASE / "runtime.json")
        report.update(status="failed_rolled_back" if changed else "failed_before_switch",
                      error=str(error))
        atomic_json(LOGS / "result.json", report)
        raise


if __name__ == "__main__":
    main()
