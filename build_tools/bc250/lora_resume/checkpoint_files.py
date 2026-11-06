#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate and exchange sealed DDP common payloads without replacing rank RNGs.

Offline only: stop both training workers before exporting or preparing a copy.
The original native checkpoint is never modified by this utility.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile

MANIFEST = "bc250-l02.json"
REQUIRED = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "trainer_state.json",
    "training_args.bin",
}
MAX_BYTES = 3 * 1024**3


def file_info(path):
    assert path.is_file() and not path.is_symlink(), path
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def common_inventory(checkpoint):
    result = {}
    for file in checkpoint.iterdir():
        if file.name == MANIFEST or re.fullmatch(r"rng_state_\d+\.pth", file.name):
            continue
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", file.name), file.name
        result[file.name] = file_info(file)
    assert REQUIRED <= result.keys(), REQUIRED - result.keys()
    assert len(result) <= 64 and sum(v["bytes"] for v in result.values()) <= MAX_BYTES
    return result


def read_manifest(checkpoint):
    path = checkpoint / MANIFEST
    assert path.is_file() and not path.is_symlink()
    m = json.loads(path.read_text())
    assert m["format"] == "bc250-lora-ddp-v1" and m["world_size"] == 2
    assert set(m["ranks"]) == {"0", "1"}
    assert REQUIRED <= m["payload"].keys()
    assert len(m["payload"]) <= 64
    assert sum(v["bytes"] for v in m["payload"].values()) <= MAX_BYTES
    for name in m["payload"]:
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", name) and not name.startswith(
            "rng_state_"
        )
        assert name != MANIFEST
    state = m["ranks"]["0"]["state"]
    assert state["step"] == m["step"] > 0
    for r in m["ranks"].values():
        assert r["state"]["step"] == m["step"]
        for key in ["parameters", "optimizer", "scheduler", "scaler"]:
            assert r["state"][key] == state[key], key
    return m


def validate(checkpoint, rank):
    m = read_manifest(checkpoint)
    assert (
        common_inventory(checkpoint) == m["payload"]
    ), "Incomplete or modified common checkpoint payload"
    assert (
        file_info(checkpoint / f"rng_state_{rank}.pth")
        == m["ranks"][str(rank)]["rng_file"]
    ), "Wrong or missing rank RNG file"
    trainer = json.loads((checkpoint / "trainer_state.json").read_text())
    assert trainer["global_step"] == m["step"]
    return m


def export(checkpoint, output):
    m = validate(checkpoint, 0)
    assert not output.exists()
    with tarfile.open(output, "x:") as archive:
        for name in [MANIFEST, *sorted(m["payload"])]:
            archive.add(checkpoint / name, arcname=name, recursive=False)
    return {
        "status": "passed",
        "checkpoint_id": m["checkpoint_id"],
        "archive": file_info(output),
    }


def prepare(archive, local_checkpoint, destination, rank):
    assert not destination.exists(), destination
    local = read_manifest(local_checkpoint)
    rng = local_checkpoint / f"rng_state_{rank}.pth"
    assert file_info(rng) == local["ranks"][str(rank)]["rng_file"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage and validate every file before one atomic rename. Failed stages are
    # removed; original payload and local RNG evidence remain untouched.
    with tempfile.TemporaryDirectory(
        prefix=".l02-stage-", dir=destination.parent
    ) as temp:
        staging = Path(temp) / "checkpoint"
        staging.mkdir()
        with tarfile.open(archive, "r:") as t:
            members = t.getmembers()
            assert (
                len(members) <= 65
                and sum(x.size for x in members) <= MAX_BYTES + 1024**2
            )
            assert len({x.name for x in members}) == len(members)
            assert {x.name for x in members} == set(local["payload"]) | {MANIFEST}
            for member in members:
                assert member.isfile() and re.fullmatch(r"[A-Za-z0-9_.-]+", member.name)
                with t.extractfile(member) as source, (staging / member.name).open(
                    "xb"
                ) as target:
                    shutil.copyfileobj(source, target, 4 * 1024**2)
        assert (
            read_manifest(staging) == local
        ), "Common payload and local RNG are from different checkpoints"
        shutil.copyfile(rng, staging / rng.name)
        validate(staging, rank)
        staging.rename(destination)
    assert file_info(rng) == local["ranks"][str(rank)]["rng_file"]
    return {
        "status": "passed",
        "checkpoint_id": local["checkpoint_id"],
        "rank": rank,
        "destination": str(destination),
        "source_rng_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    e = sub.add_parser("export")
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    i = sub.add_parser("prepare")
    i.add_argument("--archive", type=Path, required=True)
    i.add_argument("--local-checkpoint", type=Path, required=True)
    i.add_argument("--destination", type=Path, required=True)
    i.add_argument("--rank", type=int, choices=[0, 1], required=True)
    v = sub.add_parser("validate")
    v.add_argument("--checkpoint", type=Path, required=True)
    v.add_argument("--rank", type=int, choices=[0, 1], required=True)
    a = parser.parse_args()
    if a.action == "export":
        result = export(a.checkpoint, a.output)
    elif a.action == "prepare":
        result = prepare(a.archive, a.local_checkpoint, a.destination, a.rank)
    else:
        m = validate(a.checkpoint, a.rank)
        result = {
            "status": "passed",
            "checkpoint_id": m["checkpoint_id"],
            "rank": a.rank,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
