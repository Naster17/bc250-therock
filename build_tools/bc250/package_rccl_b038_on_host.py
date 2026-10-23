#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install the explicit rank-one RCCL profile under the daily exclusive lock."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from bc250_torch import RCCL_SINGLE_ENVIRONMENT
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

PROFILE = Path("/opt/bc250-stack/dev/rccl-single-b038")
REGISTRY = Path("/opt/bc250-stack/profiles/rccl-single")


def rollback():
    record = json.loads((WS / "logs/rccl-b038-package/final.json").read_text())
    recovery = Path(record["recovery"])
    assert REGISTRY.resolve(strict=True) == PROFILE
    assert sha(PROFILE / "runtime.json") == record["profile_sha256"]
    baseline = json.loads(
        (WS / "logs/rccl-b037-qualification-r2/final.json").read_text()
    )
    for destination, backup in (
        (Path("/usr/local/bin/bc250-torch"), recovery / "bc250-torch"),
        (WS / "helpers/bc250_torch.py", recovery / "bc250_torch.py"),
    ):
        assert sha(destination) == record["launcher_sha256"]
        assert sha(backup) == baseline["protected_sha256"][str(destination)]
    for destination, backup in (
        (Path("/usr/local/bin/bc250-torch"), recovery / "bc250-torch"),
        (WS / "helpers/bc250_torch.py", recovery / "bc250_torch.py"),
    ):
        temporary = destination.with_name(destination.name + ".b038-rollback")
        assert not temporary.exists()
        shutil.copy2(backup, temporary)
        os.replace(temporary, destination)
    REGISTRY.unlink()
    atomic_json(
        WS / "logs/rccl-b038-rollback.json",
        {"status": "restored-launcher-and-unregistered-profile"},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    assert os.geteuid() == 0
    preflight("rccl-b038-package")
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    if args.rollback:
        rollback()
        return
    baseline = json.loads(
        (WS / "logs/rccl-b037-qualification-r2/final.json").read_text()
    )
    assert baseline["status"] == "passed"
    for name, digest in baseline["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    daily = Path("/opt/bc250-stack/current").resolve(strict=True)
    assert (
        sha(daily / "runtime.json")
        == "32fd45595989e09597dbb0fcfc8992595e8861c0581a49827fcf965e35a865e0"
    )
    assert not PROFILE.exists() and not REGISTRY.exists() and not REGISTRY.is_symlink()
    recovery = WS / "recovery/rccl-b038-before"
    recovery.mkdir()
    launcher = Path("/usr/local/bin/bc250-torch")
    shutil.copy2(launcher, recovery / "bc250-torch")
    shutil.copy2(WS / "helpers/bc250_torch.py", recovery / "bc250_torch.py")
    shutil.copy2(daily / "runtime.json", recovery / "runtime.json")
    PROFILE.mkdir()
    for name in ("rocm", "cache", "python-consumers", "rocblas-gemm-n32.csv"):
        (PROFILE / name).symlink_to(daily / name)
    (PROFILE / "torch").symlink_to(WS / "pytorch-rccl-b037")
    library = PROFILE / "rccl/lib/librccl.so.1.0"
    library.parent.mkdir(parents=True)
    original = WS / "rccl-b034/install-r1/lib/librccl.so.1.0"
    assert (
        sha(original)
        == "ca790dc73b76a94a7dc97538608cd94b789e73cc18d756b0631111c18230583e"
    )
    shutil.copy2(original, library)
    library.chmod(0o644)
    for name in ("librccl.so", "librccl.so.1"):
        (library.parent / name).symlink_to(library.name)
    built = json.loads((WS / "logs/rccl-b037-build-r2/final.json").read_text())
    identities = {"rccl/lib/librccl.so.1.0": sha(library)}
    for name, digest in built["native_sha256"].items():
        path = "torch/torch/lib/" + name
        assert sha(PROFILE / path) == digest, path
        identities[path] = digest
    identities["torch/GIT_REVISION"] = sha(PROFILE / "torch/GIT_REVISION")
    manifest = json.loads(
        (WS / "rccl-b037/candidate-release-r2/runtime.json").read_text()
    )
    manifest.update(
        release="dev-rccl-single-b038",
        profile="gfx1013-rccl-single-opt-in",
        qualification="B03.7 bounded native qualification; B03.8 packaging outcome in rccl_single_profile.packaged_evidence",
        limitations=[
            "Opt-in one physical GPU/rank one only; no multi-device/transport/scaling proof",
            "Filtered 375-object RCCL; numeric FP8 and symmetric device API unsupported",
            "No real offload payload, broad graph/concurrency or speed qualification",
            "Shares mutable daily ROCm, cache and consumers; hold daily runtime lock",
            "SDMA and experimental VM heap remain disabled",
        ],
        rccl_single_profile=dict(
            environment=RCCL_SINGLE_ENVIRONMENT,
            identity_sha256=identities,
            native_evidence=str(WS / "logs/rccl-b037-qualification-r2/final.json"),
            packaged_evidence=str(WS / "logs/rccl-b038-qualification-r1/final.json"),
            lock_release=str(daily),
        ),
    )
    manifest["rccl_b035"]["status"] = "native-qualified B03.7; packaged B03.8 opt-in"
    manifest["rccl_b035"]["rccl"] = str(PROFILE / "rccl")
    atomic_json(PROFILE / "runtime.json", manifest)
    REGISTRY.parent.mkdir(exist_ok=True)
    REGISTRY.symlink_to(PROFILE)
    source = WS / "helpers/bc250_torch.b038-next.py"
    temporary = launcher.with_name("bc250-torch.b038-next")
    shutil.copyfile(source, temporary)
    temporary.chmod(0o755)
    os.replace(temporary, launcher)
    shutil.copyfile(source, WS / "helpers/bc250_torch.py")
    logs = WS / "logs/rccl-b038-package"
    logs.mkdir()
    atomic_json(
        logs / "final.json",
        dict(
            status="packaged-not-qualified",
            profile=str(PROFILE),
            registry=str(REGISTRY),
            profile_sha256=sha(PROFILE / "runtime.json"),
            launcher_sha256=sha(launcher),
            daily_runtime_sha256=sha(daily / "runtime.json"),
            native_identity_sha256=identities,
            recovery=str(recovery),
        ),
    )


if __name__ == "__main__":
    main()
