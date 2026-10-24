#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit isolated dependency resolution; do not build or install a full backend."""

from pathlib import Path
import subprocess
import tarfile

from bc250_torch import runtime_lock
from deploy_torch_a04_on_host import atomic_json
from qualify_training_n33_on_host import WS


def main():
    root = WS / "migraphx-b062"
    deps = root / "rocm-cmake"
    deps.mkdir()
    with tarfile.open(root / "rocm-cmake.tar.gz") as tar:
        tar.extractall(deps, filter="data")
    scratch = WS / "tmp/migraphx-b062/20261024-configure"
    scratch.mkdir(parents=True)
    logs = WS / "logs/migraphx-b062-configure-r1"
    logs.mkdir()
    original = (WS / "helpers/build_migraphx_b062.sh").read_text()
    command = original[original.index("set +e\ncmake -S") :]
    command = command.replace(
        "/workspace/source/base/rocm-cmake/share/rocmcmakebuildtools/cmake",
        "/workspace/deps/share/rocmcmakebuildtools/cmake",
    )
    script = (
        '#!/bin/bash\nset -eu\nprefix=/workspace/build/dist/rocm\nexport PATH="$prefix/bin:$prefix/lib/llvm/bin:$PATH"\nexport ROCM_PATH="$prefix" HIP_PATH="$prefix"\nexport LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/llvm/lib:$prefix/lib/rocm_sysdeps/lib"\nexport CCACHE_MAXSIZE=20G\neval "$(python3 /workspace/source/build_tools/setup_ccache.py --dir /workspace/cache/migraphx-b062 --local-path /workspace/cache/data --no-reset-stats)"\n'
        + command
    )
    (scratch / "configure.sh").write_text(script)
    run = [
        "podman",
        "run",
        "--name",
        "bc250-migraphx-b062-configure-r1",
        "--network=none",
        "--cpus=4",
        "--memory=10g",
        "--memory-swap=12g",
    ]
    for src, dst, mode in (
        (WS / "source", "/workspace/source", "ro"),
        (WS / "build", "/workspace/build", "ro"),
        (WS / "cache", "/workspace/cache", "rw"),
        (scratch, "/workspace/task", "rw"),
        (root / "source", "/workspace/task/source", "ro"),
        (deps, "/workspace/deps", "ro"),
        (logs, "/workspace/logs", "rw"),
    ):
        run += ["-v", f"{src}:{dst}:{mode}"]
    run += ["fa3e8c254b6a", "bash", "/workspace/task/configure.sh"]
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        with (logs / "controller.log").open("w") as stream:
            result = subprocess.run(
                run, stdout=stream, stderr=subprocess.STDOUT, timeout=180
            )
    atomic_json(
        logs / "final.json",
        dict(
            status="audit-complete",
            container_exit=result.returncode,
            configure_exit=int((logs / "configure-exit.txt").read_text())
            if (logs / "configure-exit.txt").exists()
            else None,
            command=run,
            full_build_started=False,
        ),
    )


if __name__ == "__main__":
    main()
