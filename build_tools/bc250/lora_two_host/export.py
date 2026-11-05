#!/usr/bin/env python3
"""Export an explicit metadata whitelist; keep private datasets/weights on hosts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert (
        os.geteuid() == 0
    ), "Recovery evidence directories require root for complete export"
    home = Path("/home/nik")
    evidence = home / "bc250work/evidence/lora-two-host"
    project = home / "bc250-projects/lora-smoke/project"
    selected = {}

    def add(path, name):
        if path.is_file():
            assert not path.is_symlink()
            selected[name] = path

    for phase in evidence.iterdir():
        if not phase.is_dir():
            continue
        final = phase / "final.json"
        assert final.exists(), ("Unfinished phase", phase)
        assert json.loads(final.read_text())["status"] == "passed", phase
        for file in phase.rglob("*"):
            if file.is_file():
                add(file, "evidence/" + str(file.relative_to(evidence)))
        output = project / "outputs/two-host-tests" / phase.name
        for pattern in [
            "audit-rank*.json",
            "gradient-rank*.json",
            "*results.json",
            "train.log",
            "metrics.jsonl",
            "run_config.json",
        ]:
            for file in output.glob(pattern):
                add(file, "application/" + str(file.relative_to(project)))
        add(
            output / "final_adapter/bc250-save-verification.json",
            "application/"
            + str(
                (output / "final_adapter/bc250-save-verification.json").relative_to(
                    project
                )
            ),
        )
    for file in evidence.glob("*.json"):
        if file.name != "export-inventory.json":
            add(file, "evidence/" + file.name)
    for pattern in ["*.py", "*.json", "run-distributed"]:
        for file in Path(__file__).parent.glob(pattern):
            add(file, "tools/" + file.name)
    for name in ["train_two_host.py", "bc250_lora_support.py", "run-distributed"]:
        add(project / name, "application/" + name)
    for file in project.glob("train_two_host.*.py"):
        add(file, "application/" + file.name)
    sites = project.parent / "env/lib/python3.12/site-packages"
    for name in [
        "transformers/trainer.py",
        "trl/trainer/sft_trainer.py",
        "accelerate/accelerator.py",
    ]:
        add(sites / name, "framework/" + name)
    inventory = {
        name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        }
        for name, path in selected.items()
    }
    manifest = evidence / "export-inventory.json"
    manifest.write_text(json.dumps(inventory, indent=2) + "\n")
    selected["export-inventory.json"] = manifest
    with tarfile.open(args.output, "x:gz") as archive:
        for name, path in sorted(selected.items()):
            assert not name.endswith(
                (".f32", ".safetensors", ".pt", ".jsonl")
            ) or name.endswith("metrics.jsonl")
            archive.add(path, arcname=name, recursive=False)
    print(
        json.dumps(
            {
                "files": len(selected),
                "bytes": args.output.stat().st_size,
                "sha256": hashlib.file_digest(
                    args.output.open("rb"), "sha256"
                ).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
