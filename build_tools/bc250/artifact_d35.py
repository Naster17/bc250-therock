#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded raw CPU-reference artifacts. No pickle, global cache flush or GPU imports."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

GIB = 1024**3
SOURCES = [
    "artifact_d35.py",
    "capacity_plan_d35.py",
    "fixture_d35.py",
    "prepare_distributed_d35_reference.py",
    "distributed_d35_growth.py",
    "distributed_d32_capacity.py",
    "distributed_d33_fresh.py",
    "capacity_plan_d32.py",
    "qualification_d33.py",
    "distributed_d05_train.py",
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def admit_storage(free, parameters, *, producing):
    # Original reference: FP32 initial values + FP64 reference gradients.
    # GPU qualification additionally needs one temporary full FP32 actual-gradient spool.
    payload = parameters * (12 if producing else 4)
    if free < payload + 5 * GIB:
        raise ValueError(
            "Insufficient disk space: preserve5GiB plus bounded reference/audit payload"
        )
    return dict(
        free_before_bytes=free,
        required_payload_bytes=payload,
        minimum_free_after_bytes=5 * GIB,
    )


def safe_path(root, name):
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.endswith(".raw")
    ):
        raise ValueError("Invalid owned raw artifact name")
    path = Path(root) / name
    if path.is_symlink():
        raise ValueError("Symlink raw artifact refused")
    return path


def write_blob(root, name, data):
    path = safe_path(root, name)
    view = memoryview(data).cast("B")
    result = hashlib.sha256()
    with path.open("xb", buffering=0) as stream:
        os.chmod(path, 0o600)
        for start in range(0, len(view), 1024**2):
            part = view[start : start + 1024**2]
            while part:
                part = part[stream.write(part) :]
            result.update(view[start : start + 1024**2])
        os.fsync(stream.fileno())
        # Drop only this owned artifact's cache pages, never host-global caches.
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return dict(file=name, bytes=len(view), sha256=result.hexdigest())


def verify_blob(root, item):
    path = safe_path(root, item["file"])
    flags = os.O_RDONLY | os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        if os.fstat(fd).st_size != item["bytes"]:
            raise ValueError("Reference artifact size changed")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            found = hashlib.file_digest(stream, "sha256").hexdigest()
        if found != item["sha256"]:
            raise ValueError("Reference artifact digest changed")
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def read_blob(root, item):
    path = safe_path(root, item["file"])
    data = bytearray(item["bytes"])
    with path.open("rb", buffering=0) as stream:
        if stream.readinto(data) != len(data) or stream.read(1):
            raise ValueError("Reference artifact read length changed")
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError("Reference artifact digest changed while reading")
    return data


def load_manifest(path, wanted, config, parameters, *, check_sources=True):
    if not wanted or digest(path) != wanted:
        raise ValueError("Reference manifest digest changed")
    record = json.loads(Path(path).read_text())
    if (
        sys.byteorder != "little"
        or record["schema"] != 1
        or record["status"] != "passed"
        or record["config"] != config
        or record["parameters"] != parameters
        or record["cuda_initialized"] is not False
        or not math.isfinite(record["CPU_FP64_loss"])
    ):
        raise ValueError("Unqualified reference shape/status/platform")
    if check_sources and record["sources"] != source_hashes():
        raise ValueError("Reference training sources changed")
    names, files, elements = set(), set(), 0
    for entry in record["tensors"]:
        if (
            entry["name"] in names
            or not entry["shape"]
            or any(type(d) is not int or d <= 0 for d in entry["shape"])
        ):
            raise ValueError("Invalid reference tensor schema")
        names.add(entry["name"])
        count = math.prod(entry["shape"])
        elements += count
        for key, width in [("initial", 4), ("gradient", 8)]:
            item = entry[key]
            if item["file"] in files or item["bytes"] != count * width:
                raise ValueError("Invalid reference tensor length/alias")
            files.add(item["file"])
            verify_blob(Path(path).parent, item)
    if elements != parameters or not names:
        raise ValueError("Incomplete reference parameter coverage")
    return record
