#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inspect three target-image link inputs using stdlib only and no GPU mappings."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("/home/nik/bc250work/evidence/source-d47-r1")


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    budget = json.loads((ROOT / "sdk-build-admission.json").read_text())
    recipe = json.loads((ROOT / "sdk-recipe.json").read_text())
    missing = [x["argument"] for x in recipe["inputs"] if not x["present"]]
    assert set(missing) == {
        "/usr/lib/x86_64-linux-gnu/libpthread.a",
        "/usr/lib/x86_64-linux-gnu/libdl.a",
        "/usr/lib/x86_64-linux-gnu/librt.a",
    }
    code = (
        "import hashlib,json,pathlib; paths="
        + repr(missing)
        + "; print(json.dumps({p: {'bytes': pathlib.Path(p).stat().st_size, 'sha256': hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()} for p in paths}))"
    )
    command = [
        "podman",
        "run",
        "--rm",
        "--pull=never",
        "--read-only",
        "--network=none",
        "--cap-drop=all",
        "--security-opt=no-new-privileges",
        "--cpus=1",
        "--memory=256m",
        "--memory-swap=256m",
        "--pids-limit=64",
        "--label=bc250.d47=image-inputs",
        "--entrypoint=python3",
        budget["image"],
        "-c",
        code,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    proof = dict(
        command=command,
        image=budget["image"],
        exit=completed.returncode,
        stderr=completed.stderr,
        no_gpu_mappings=True,
        no_torch_import=True,
    )
    assert completed.returncode == 0, proof
    proof["files"] = json.loads(completed.stdout)
    assert set(proof["files"]) == set(missing)
    assert recipe["compiler_present"]
    proof["status"] = "passed"
    proof["combined_link_input_count"] = len(recipe["inputs"])
    proof["recipe_admission"] = (
        "inputs-verified; candidate output-redirection and fresh resource/ownership gates still required"
    )
    (ROOT / "sdk-image-inputs.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status="passed",
                image_files=len(proof["files"]),
                combined_link_inputs=len(recipe["inputs"]),
            )
        )
    )


if __name__ == "__main__":
    main()
