#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D58: materialize a self-contained versioned release from the qualified cohort.

Runs as root on bc250 (.104) under the EX prefix-update lock via
helpers/update_dev_prefix.py --release /opt/bc250-stack/current.
Assembly only: never activates, never touches `current` or the sources.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

VERSION = "10.2-gfx1013-20261111-138a66f52"
STACK = Path("/opt/bc250-stack")
WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
RELEASE = STACK / "releases" / VERSION
CURRENT = STACK / "current"
GIB = 1024**3
DEADLINE = 3600
FLOOR = 5 * GIB

SOURCES = {
    "rocm": STACK / "candidates/sdk-d49-r1/rocm",
    "cache": WS / "cache/dev-user",
    "python-consumers": STACK
    / "releases/10.2-gfx1013-20261013-4b000d542/python-consumers",
    "profiles/daily/torch": WS / "pytorch-quant-b052",
    "profiles/rccl-single/torch": WS / "pytorch-quant-b053",
    "profiles/rccl-single/rccl": STACK / "dev/rccl-single-b053/rccl",
}
LIVE_PROFILES = {
    "daily": STACK / "candidates/sdk-d49-r1/profiles/daily",
    "rccl-single": STACK / "candidates/sdk-d49-r1/profiles/rccl-single",
}
GEMM_CSV = STACK / "dev/priority1/rocblas-gemm-n32.csv"
SENTINELS = {
    "daily": {
        "rocm/lib/librocprofiler-sdk.so.1.4.2": "2a975370ba4eb1c2fe39e406ff36b175f554a49d0aa19febbd326f641fcb590b",
        "rocm/lib/libamdhip64.so": "a1162308076a99ff5cf5e4e15ca8bff53928d8bc4e624c42033d7e15dd2d1e3b",
        "rocm/lib/libMIOpen.so": "381eac05240e33acced9472fb7c3071afe8bc4f434407e67a29ce64423b935ac",
        "torch/torch/lib/libtorch_hip.so": "2f2761cd7f259612863406ab0e85d7b1b5ce8b59a32655bf9ac7db9359d0df27",
        "torch/torch/lib/libtorch_cpu.so": "9586351bda1617acaefacacedd938435464cd92337df1ec3e61044c453478195",
    },
    "rccl-single": {
        "rocm/lib/librocprofiler-sdk.so.1.4.2": "2a975370ba4eb1c2fe39e406ff36b175f554a49d0aa19febbd326f641fcb590b",
        "rocm/lib/libamdhip64.so": "a1162308076a99ff5cf5e4e15ca8bff53928d8bc4e624c42033d7e15dd2d1e3b",
        "rocm/lib/libMIOpen.so": "381eac05240e33acced9472fb7c3071afe8bc4f434407e67a29ce64423b935ac",
        "torch/torch/lib/libtorch_hip.so": "a31cbb159d3edfb4b7e1cc59dfcc325db185c4c540e1fcaa952e09e833335533",
        "torch/torch/lib/libtorch_cpu.so": "8a1b0e1fecef7afc9309d0460b2fbd13df20b8bd667d7036c58a341a020e5346",
    },
}


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_stats(root):
    files = total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_file() and not p.is_symlink():
                files += 1
                total += p.stat().st_size
    return files, total


def adapt_runtime(data, version, profile, assembly):
    out = dict(data)
    out["release"] = version
    out["validated"] = False
    out["mutable"] = False
    out["assembly"] = dict(assembly, profile=profile)
    return out


def verify_sentinels(base, profile):
    for rel, expected in SENTINELS[profile].items():
        actual = sha256(base / rel)
        assert actual == expected, (profile, rel, actual)


def manifest(root, name, out_dir):
    lines = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in sorted(filenames):
            p = Path(dirpath) / fname
            if p.is_symlink() or not p.is_file():
                continue
            lines.append(f"{sha256(p)}  {p.relative_to(root).as_posix()}\n")
    lines.sort(key=lambda line: line.split("  ", 1)[1])
    target = out_dir / f"{name}.sha256"
    target.write_text("".join(lines))
    return len(lines), sha256(target)


def self_containment(root):
    violations = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                resolved = p.resolve()
                if not resolved.is_relative_to(root.resolve()):
                    violations.append(f"{p.relative_to(root)} -> {os.readlink(p)}")
    return violations


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    start = time.monotonic()
    current_before = CURRENT.resolve()
    assert current_before == LIVE_PROFILES["daily"].resolve()
    assert not RELEASE.exists(), RELEASE
    running = subprocess.run(
        ["podman", "ps", "-q"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert not running, f"running containers: {running}"

    receipt = {
        "task": "D58 release assembly",
        "version": VERSION,
        "release": str(RELEASE),
        "current_before": str(current_before),
        "sources": {k: str(v) for k, v in SOURCES.items()},
        "preflight": {"running_containers": [], "current": str(current_before)},
    }

    sizes = {}
    for name, src in SOURCES.items():
        files, total = tree_stats(src)
        sizes[name] = {"files": files, "bytes": total}
    needed = sum(s["bytes"] for s in sizes.values())
    free = shutil.disk_usage(RELEASE.parent).free
    assert free >= needed + FLOOR, (free, needed)
    receipt["source_sizes"] = sizes
    receipt["estimated_bytes"] = needed
    receipt["free_bytes_before"] = free

    for profile, live in LIVE_PROFILES.items():
        verify_sentinels(live, profile)
    receipt["source_sentinels_verified"] = True

    RELEASE.mkdir(parents=True)
    (RELEASE / "manifests").mkdir()
    assembly_meta = {
        "task": "D58",
        "date": "2026-11-11",
        "therock_revision": "138a66f52e330a21904a66ea66d226587b7f61e5",
        "source_candidate": "sdk-d49-r1",
        "note": "materialized self-contained release; qualification pending (D59)",
    }
    copied = {}
    for name, src in SOURCES.items():
        dest = RELEASE / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-a", str(src), str(dest)], check=True, timeout=1800)
        files, total = tree_stats(dest)
        copied[name] = {"files": files, "bytes": total}
        assert shutil.disk_usage(RELEASE).free >= FLOOR, name
        assert time.monotonic() - start < DEADLINE, name
    receipt["copied"] = copied

    for profile, live in LIVE_PROFILES.items():
        pdir = RELEASE / "profiles" / profile
        for item in sorted(live.iterdir()):
            if item.is_symlink() or item.name == ".runtime.lock":
                continue
            dest = pdir / item.name
            if dest.exists():
                continue
            if item.is_dir():
                subprocess.run(["cp", "-a", str(item), str(dest)], check=True)
            else:
                shutil.copy2(item, dest)
        shutil.copy2(GEMM_CSV, pdir / GEMM_CSV.name)
        for link, target in (
            ("rocm", "../../rocm"),
            ("cache", "../../cache"),
            ("python-consumers", "../../python-consumers"),
        ):
            (pdir / link).symlink_to(target)
        (pdir / ".runtime.lock").touch()
        data = json.loads((live / "runtime.json").read_text())
        adapted = adapt_runtime(data, VERSION, profile, assembly_meta)
        (pdir / "runtime.json").write_text(json.dumps(adapted, indent=2) + "\n")

    daily = json.loads((LIVE_PROFILES["daily"] / "runtime.json").read_text())
    top = {
        k: daily[k]
        for k in (
            "image",
            "uid",
            "gid",
            "therock_revision",
            "rocm_libraries_revision",
            "rocm_systems_revision",
            "torch_revision",
            "triton_revision",
            "triton_wheel_sha256",
            "sdma",
            "limitations",
        )
        if k in daily
    }
    top = {
        "release": VERSION,
        **top,
        "validated": False,
        "mutable": False,
        "profiles": ["daily", "rccl-single"],
        "assembly": assembly_meta,
    }
    (RELEASE / "runtime.json").write_text(json.dumps(top, indent=2) + "\n")

    for profile in LIVE_PROFILES:
        verify_sentinels(RELEASE / "profiles" / profile, profile)
    receipt["release_sentinels_verified"] = True

    violations = self_containment(RELEASE)
    receipt["self_containment_violations"] = violations
    assert not violations, violations

    manifests = {}
    for name in SOURCES:
        count, digest = manifest(
            RELEASE / name, name.replace("/", "-"), RELEASE / "manifests"
        )
        manifests[name] = {"files": count, "sha256": digest}
        assert time.monotonic() - start < DEADLINE, name
    receipt["manifests"] = manifests

    assert CURRENT.resolve() == current_before, "current changed"
    receipt["current_after"] = str(CURRENT.resolve())
    receipt["free_bytes_after"] = shutil.disk_usage(RELEASE).free
    receipt["seconds"] = round(time.monotonic() - start, 3)
    receipt["status"] = "assembled"
    (RELEASE / "assembly.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
