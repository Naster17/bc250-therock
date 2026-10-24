#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Link the owned graph fixture and HIPRTC subprocess driver after the GPU backend."""
import json
from pathlib import Path
import subprocess

from bc250_torch import runtime_lock
from build_migraphx_b064_on_host import STORAGE
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    preflight("migraphx-b064-fixture-build")
    built = json.loads((STORAGE / "logs-build-r4/final.json").read_text())
    assert (
        built["status"] == "backend-built-not-qualified" and not built["changed_hashes"]
    )
    logs = STORAGE / "logs-fixture-r1"
    logs.mkdir()
    script = (WS / "helpers/build_migraphx_b064.sh").read_text().split("\ncmake -S")[0]
    script += "\ncmake --build /workspace/task/build-r1 --target probe_migraphx_b064 migraphx-hiprtc-driver --parallel 4 > /workspace/logs/build.log 2>&1\nccache --show-stats > /workspace/logs/ccache-after.txt\n"
    script += """
shopt -s nullglob
objects=(/workspace/task/build-r1/migraphx/lib/*.hip-amdgcn-amd-amdhsa--gfx1013)
test "${#objects[@]}" -gt 0
for object in "${objects[@]}"; do
  "$prefix/lib/llvm/bin/llvm-readelf" --notes "$object" >> /workspace/logs/device-notes.txt
  "$prefix/lib/llvm/bin/llvm-objdump" -d "$object" >> /workspace/logs/device.asm
done
"$prefix/lib/llvm/bin/clang++" --version > /workspace/logs/compiler-version.txt
"""
    (STORAGE / "finish.sh").write_text(script)
    command = [
        part.replace(
            "bc250-migraphx-b064-build-r4", "bc250-migraphx-b064-fixture-r1"
        ).replace(
            str(STORAGE / "logs-build-r4") + ":/workspace/logs",
            str(logs) + ":/workspace/logs",
        )
        for part in built["command"]
    ]
    command[-1] = "/workspace/task/finish.sh"
    report = dict(status="building-fixture", command=command)
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        try:
            with (logs / "controller.log").open("w") as stream:
                subprocess.run(
                    command,
                    check=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
            report["status"] = "fixture-built-not-qualified"
        except BaseException as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            report["changed_hashes"] = [
                p for p, d in built["protected_sha256"].items() if sha(Path(p)) != d
            ]
            atomic_json(logs / "final.json", report)
    assert not report["changed_hashes"]


if __name__ == "__main__":
    main()
