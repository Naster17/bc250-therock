# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Install isolated, pinned Transformers/PEFT consumers of the validated Torch."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import sysconfig


PACKAGES = (
    "transformers==5.2.0",
    "peft==0.18.1",
    "accelerate==1.12.0",
    "huggingface-hub==1.4.1",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    parent = Path(sysconfig.get_path("purelib"))
    environment = directory / "venv"
    if not (environment / "bin/python").exists():
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    site = (
        environment
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    (site / "bc250_runtime.pth").write_text(
        f"import site; site.addsitedir({str(parent)!r})\n"
    )
    constraints = directory / "runtime-constraints.txt"
    constraints.write_text(
        "".join(
            f"{name}=={importlib.metadata.version(name)}\n"
            for name in ("torch", "triton", "numpy", "click")
        )
    )
    subprocess.run(
        [
            str(environment / "bin/python"),
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            "--constraint",
            str(constraints),
            *PACKAGES,
        ],
        check=True,
    )
    subprocess.run([str(environment / "bin/python"), "-m", "pip", "check"], check=True)
    (directory / "training.env").write_text(
        f"PYTHONPATH={site}\nHF_HUB_DISABLE_XET=1\nTOKENIZERS_PARALLELISM=false\nOMP_NUM_THREADS=8\n"
    )
    result = subprocess.check_output(
        [
            str(environment / "bin/python"),
            "-c",
            "import importlib.metadata as m,json; import torch,transformers,peft; print(json.dumps({n:m.version(n) for n in ('torch','triton','numpy','transformers','peft','accelerate','huggingface-hub','safetensors','tokenizers')}))",
        ],
        text=True,
    )
    (directory / "packages.json").write_text(
        json.dumps(json.loads(result), indent=2) + "\n"
    )
    print(f"TRAINING_ENV_READY {directory / 'training.env'}", flush=True)


if __name__ == "__main__":
    main()
