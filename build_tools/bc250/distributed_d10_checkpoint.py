#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Strict local metadata/file validation for the owned fixed-world D10 checkpoint."""
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def local_part(
    commit: dict | None,
    folder: Path,
    config: dict,
    cohort: str,
    rank: int,
    world: int,
    step: int,
) -> Path:
    if not isinstance(commit, dict):
        raise ValueError("Missing checkpoint commit")
    expected = dict(
        schema=1, config=config, cohort=cohort, world=world, next_attempt=step
    )
    if any(commit.get(k) != v for k, v in expected.items()):
        raise ValueError("Checkpoint schema/config/cohort/world/step mismatch")
    parts = commit.get("parts")
    if not isinstance(parts, list) or len(parts) != world:
        raise ValueError("Incomplete checkpoint part set")
    if any(not isinstance(v, dict) for v in parts) or sorted(
        v.get("rank", -1) for v in parts
    ) != list(range(world)):
        raise ValueError("Invalid/duplicate checkpoint rank set")
    for record in parts:
        if record.get("file") != f"rank-{record['rank']}.pt":
            raise ValueError("Invalid checkpoint filename")
        if type(record.get("bytes")) is not int or record["bytes"] <= 0:
            raise ValueError("Invalid checkpoint size")
        value = record.get("sha256")
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Invalid checkpoint digest")
    record = next(v for v in parts if v["rank"] == rank)
    path = folder / record["file"]
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record["bytes"]
        or digest(path) != record["sha256"]
    ):
        raise ValueError("Local checkpoint part missing/damaged")
    return path
