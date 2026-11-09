#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Extract small immutable qualification records from verified D32 archives."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from verify_distributed_d32_exports import read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host104", type=Path, required=True)
    p.add_argument("--host105", type=Path, required=True)
    p.add_argument("--qualified-results", type=Path, required=True)
    p.add_argument("--output-directory", type=Path, required=True)
    a = p.parse_args()
    qualified = json.loads(a.qualified_results.read_text())
    assert qualified["status"] == "passed" and qualified["groups"] == 72
    for rank, path in enumerate([a.host104, a.host105]):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == qualified["archives"][rank]["sha256"]
        handoff, _ = read(path, rank)
        source = "home/nik/bc250work/tools/distributed-d32/"
        with tarfile.open(path) as archive:
            sources = {
                n: hashlib.sha256(archive.extractfile(source + n).read()).hexdigest()
                for n in [
                    "distributed_d32_capacity.py",
                    "capacity_plan_d32.py",
                    "distributed_d05_train.py",
                ]
            }
        refs = {}
        for mode in ["single", "ddp", "fsdp"]:
            record = handoff["records"][mode + "-a"]
            w, f = record["worker"], record["final"]
            refs[mode] = dict(
                reference=w["reference"],
                checks=w["checks"],
                state=w["state"],
                global_losses=[s["global_loss"] for s in w["samples"]],
                registry_sha256=f["registry_sha256"],
                container_id=f["container_id"],
                config=w["config"],
                torch=w["torch"],
            )
        proof = dict(
            schema=1,
            qualification="D32 CPU FP64 loss/ALL gradients/clip/ALL AdamW",
            physical_rank=rank,
            archive_sha256=sha,
            source_sha256=sources,
            reference=refs,
            original_container_ids=[
                v["final"]["container_id"] for v in handoff["records"].values()
            ],
        )
        output = a.output_directory / f"distributed_d33_qualification_{rank}.json"
        assert not output.exists()
        output.write_text(json.dumps(proof, sort_keys=True, indent=2) + "\n")
        print(
            rank, hashlib.sha256(output.read_bytes()).hexdigest(), output.stat().st_size
        )


if __name__ == "__main__":
    main()
