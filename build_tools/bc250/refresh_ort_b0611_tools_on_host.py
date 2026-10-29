#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve the initial ORT wrapper/records and select its caller-context repair."""
from pathlib import Path
import json, shutil, os
from bc250_torch import runtime_lock
from bc250_ort import REGISTRY, NAME, BASE, check_package
from bc250_migraphx import sha
from deploy_torch_a04_on_host import atomic_json
from qualify_training_n33_on_host import WS

with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock"), exclusive=True):
    record = json.loads(REGISTRY.read_text())
    version = BASE / (NAME + ".json")
    assert version.read_bytes() == REGISTRY.read_bytes()
    check_package(Path(record["package"]), record)
    tools = Path("/opt/bc250-stack/tools/ort-b0611")
    target = tools / "bc250_ort.py"
    assert sha(target) == record["launcher_sha256"]["bc250_ort.py"]
    recovery = WS / "recovery/ort-b0611-before-context"
    recovery.mkdir()
    for path in (target, REGISTRY, version):
        shutil.copy2(path, recovery / path.name)
    temporary = tools / "bc250_ort.context-new.py"
    shutil.copy2(WS / "helpers/bc250_ort.py", temporary)
    temporary.chmod(0o755)
    os.replace(temporary, target)
    old = record["launcher_sha256"]["bc250_ort.py"]
    record["launcher_sha256"]["bc250_ort.py"] = sha(target)
    atomic_json(version, record)
    version.chmod(0o644)
    atomic_json(REGISTRY, record)
    REGISTRY.chmod(0o644)
    atomic_json(
        WS / "logs/onnxruntime-b0611-context-refresh.json",
        dict(
            status="passed",
            previous_sha256=old,
            new_sha256=sha(target),
            recovery=str(recovery),
            archive_unchanged=True,
            runtime_changed=False,
        ),
    )
    print("ORT_B0611_CONTEXT_REFRESH_PASS")
