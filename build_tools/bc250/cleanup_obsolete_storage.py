#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit 2026-10-16 BC-250 cleanup. Apply only under update_dev_prefix.py.

Never use global image/container/volume prune. Retain source histories, current
incremental builds/caches, user workloads and the independently tested fallback.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
STACK = Path("/opt/bc250-stack")
LOGS = WS / "logs/cleanup-20261016"
OLD = STACK / "releases/10.2-gfx1013-20261012-a640c9296"
FALLBACK = STACK / "releases/10.2-gfx1013-20261013-4b000d542"
CANDIDATE = STACK / "candidates/priority2-20261014"
KEEP_TAGS = {
    "localhost/bc250-rocm-build:10.2",
    "localhost/bc250-rocm-build:10.2-ccache",
    "localhost/bc250-pytorch-build:2.13-gfx1013",
    "localhost/bc250-pytorch-runtime:2.13-gfx1013",
    "localhost/bc250-triton-build:gfx1013",
    "localhost/bc250-triton-runtime:gfx1013",
}


def run(args):
    return subprocess.check_output(args, text=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def available():
    s = os.statvfs("/")
    return s.f_bavail * s.f_frsize


def atomic_json(path, value):
    tmp = path.with_suffix(".new")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def containers():
    rows = json.loads(run(["podman", "ps", "-a", "--format", "json"]))
    result = []
    for row in rows:
        cid = row.get("Id") or row.get("id") or row.get("ID")
        data = json.loads(run(["podman", "inspect", cid]))[0]
        result.append(
            {
                "id": data["Id"],
                "name": data["Name"],
                "status": data["State"]["Status"],
                "image": data["Image"],
            }
        )
    return result


def fingerprints():
    result = {}
    for label, root in [
        ("current", (STACK / "current").resolve()),
        ("fallback", FALLBACK),
    ]:
        names = [
            "runtime.json",
            "rocm/lib/libMIOpen.so",
            "rocm/lib/librocblas.so",
            "rocm/lib/libhipblaslt.so",
            "rocm/.kpack/blas_lib_gfx1013.kpack",
            "torch/torch/lib/libtorch_hip.so",
            "python-consumers/training.env",
            "python-consumers/packages.json",
        ]
        for name in names:
            p = root / name
            if p.is_file():
                result[label + "/" + name] = digest(p)
    return result


def references(prefixes):
    found = []
    for pid in Path("/proc").iterdir():
        if not pid.name.isdigit():
            continue
        for name in ["cwd", "exe"]:
            try:
                target = str((pid / name).readlink())
            except OSError:
                continue
            if any(target == p or target.startswith(p + "/") for p in prefixes):
                found.append({"pid": pid.name, "kind": name, "path": target})
        try:
            maps = (pid / "maps").read_text()
        except OSError:
            continue
        for line in maps.splitlines():
            if any(p + "/" in line for p in prefixes):
                found.append({"pid": pid.name, "kind": "map", "path": line.split()[-1]})
    return found


def plan():
    current = (STACK / "current").resolve(strict=True)
    assert current == STACK / "dev/priority1"
    assert (current / "python-consumers").resolve().is_relative_to(FALLBACK)
    c = containers()
    assert not any(r["status"] == "running" for r in c), "Preserve live jobs"
    stopped = [
        r
        for r in c
        if r["name"].startswith("bc250-")
        and r["status"] in ["exited", "created", "stopped"]
    ]
    protected = [r for r in c if r not in stopped]
    raw = json.loads(run(["podman", "images", "-a", "--format", "json"]))
    images = {}
    for item in raw:
        iid = item.get("Id") or item.get("id") or item.get("ID")
        row = json.loads(run(["podman", "image", "inspect", iid]))[0]
        images[row["Id"]] = {
            "id": row["Id"],
            "tags": row.get("RepoTags") or [],
            "parent": row.get("Parent") or "",
            "size": row.get("Size"),
            "project": False,
        }
    keep = {r["image"] for r in protected}
    keep |= {
        json.loads((root / "runtime.json").read_text())["image"]
        for root in [current, FALLBACK]
    }
    for row in images.values():
        tags = row["tags"]
        if any(
            tag in KEEP_TAGS or not tag.startswith("localhost/bc250-") for tag in tags
        ):
            keep.add(row["id"])
        row["project"] = any(tag.startswith("localhost/bc250-") for tag in tags)
        if not tags:
            history = run(
                [
                    "podman",
                    "history",
                    "--no-trunc",
                    "--format",
                    "{{.CreatedBy}}",
                    row["id"],
                ]
            )
            row["project"] = bool(
                re.search(r"gfx1013|bc250|/workspace/(pytorch|triton|source)", history)
            )
            if not row["project"]:
                keep.add(row["id"])
    changed = True
    while changed:
        changed = False
        for iid in list(keep):
            parent = images.get(iid, {}).get("parent")
            if parent and parent not in keep:
                keep.add(parent)
                changed = True
    obsolete = [
        row for row in images.values() if row["project"] and row["id"] not in keep
    ]

    # Children first; --no-prune prevents implicit deletion of unlisted parents.
    def depth(row):
        seen = set()
        p = row
        while p.get("parent") and p["parent"] not in seen:
            seen.add(p["parent"])
            p = images.get(p["parent"], {})
        return len(seen)

    obsolete.sort(key=depth, reverse=True)
    payloads = [
        str(root / name)
        for root in [CANDIDATE, OLD]
        for name in ["rocm", "torch", "cache", "python-consumers"]
        if (root / name).exists()
    ]
    fixture_roots = [
        WS / "training-validation",
        WS / "logs/qwen35-cli-installed/synthetic-cli",
        WS / "logs/qwen35-cli-resume-fixed/synthetic-cli",
        WS / "logs/qwen35-resume-guards/synthetic-cli",
    ]
    generated = []
    for root in fixture_roots:
        for p in root.rglob("*"):
            if (p.is_file() or p.is_symlink()) and (
                p.suffix in [".pt", ".safetensors"] or p.name == "tokenizer.json"
            ):
                generated.append(
                    {
                        "path": str(p),
                        "symlink": str(p.readlink()) if p.is_symlink() else None,
                        "bytes": p.stat().st_size if p.exists() else 0,
                    }
                )
    refs = references([str(CANDIDATE), str(OLD)])
    assert not refs, refs
    result = {
        "created_unix": time.time(),
        "available_before_bytes": available(),
        "current": str(current),
        "fallback": str(FALLBACK),
        "previous_before": str((STACK / "previous").resolve()),
        "remove_payloads": payloads,
        "generated_fixture_files": generated,
        "remove_containers": stopped,
        "protected_containers": protected,
        "remove_images": obsolete,
        "keep_image_ids": sorted(keep),
        "images_before": images,
        "protected_sha256": fingerprints(),
        "process_references": refs,
        "user_scope": "Only explicit obsolete project payloads and synthetic fixtures; no global prune or package removal.",
    }
    atomic_json(LOGS / "plan.json", result)
    print(
        json.dumps(
            {
                "payloads": payloads,
                "generated_fixture_bytes": sum(r["bytes"] for r in generated),
                "stopped_project_containers": len(stopped),
                "obsolete_images": [(r["id"][:12], r["tags"]) for r in obsolete],
                "protected_containers": protected,
            },
            indent=2,
        )
    )


def apply():
    p = json.loads((LOGS / "plan.json").read_text())
    assert not (LOGS / "applied.json").exists(), "One-shot cleanup already completed"
    assert (STACK / "current").resolve() == Path(p["current"])
    assert (
        fingerprints() == p["protected_sha256"]
    ), "Protected runtime changed since inspection"
    assert not run(["podman", "ps", "-q"]).strip(), "Preserve newly started jobs"
    assert Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"
    check = WS / "logs/cleanup-fallback-before/exit.json"
    assert json.loads(check.read_text()) == {
        "returncode": 0,
        "kernel_log_unchanged": True,
    }
    assert json.loads((check.parent / "result.json").read_text())["status"] == "passed"
    assert not references([str(CANDIDATE), str(OLD)])
    journal = {
        "available_before_bytes": available(),
        "removed": [],
        "previous_before": p["previous_before"],
    }

    def record(row):
        journal["removed"].append(row)
        atomic_json(LOGS / "journal.json", journal)
        print(json.dumps(row), flush=True)

    # Retain historical container output without copying environments/secrets.
    logdir = LOGS / "retired-container-logs"
    logdir.mkdir()
    for row in p["remove_containers"]:
        current = json.loads(run(["podman", "inspect", row["id"]]))[0]
        assert current["Name"] == row["name"] and current["State"]["Status"] in [
            "exited",
            "created",
            "stopped",
        ]
        captured = subprocess.run(
            ["podman", "logs", "--tail", "300", row["id"]],
            capture_output=True,
            text=True,
        )
        text = captured.stdout + captured.stderr
        safe = "\n".join(
            (
                "[redacted possible credential]"
                if re.search(
                    r"(?i)password\s*[=:]|token\s*[=:]|authorization|api[_-]?key\s*[=:]|://[^/\s:@]+:[^/\s@]+@|BEGIN .*PRIVATE KEY",
                    line,
                )
                else line
            )
            for line in text.splitlines()
        )
        (logdir / (row["name"] + ".log")).write_text(safe + "\n")
        subprocess.run(
            ["podman", "rm", row["id"]], check=True, stdout=subprocess.DEVNULL
        )
        record({"kind": "container", "id": row["id"], "name": row["name"]})
    # Consolidate to the tested newer independent fallback; current is unchanged.
    assert (STACK / "previous").is_symlink() and (STACK / "previous").resolve() == OLD
    temp = STACK / ".previous-cleanup-new"
    assert not temp.exists()
    temp.symlink_to(FALLBACK)
    os.replace(temp, STACK / "previous")
    journal["previous_after"] = str((STACK / "previous").resolve())
    atomic_json(LOGS / "journal.json", journal)
    for root in [CANDIDATE, OLD]:
        assert root.is_dir() and not root.is_symlink()
        shutil.copy2(root / "runtime.json", root / "runtime.retired.json")
        for name in ["rocm", "torch", "cache", "python-consumers"]:
            target = root / name
            if str(target) not in p["remove_payloads"]:
                continue
            assert target.is_dir() and not target.is_symlink()
            assert not list(target.rglob(".git")), "Preserve unexpected source history"
            shutil.rmtree(target)
            record({"kind": "retired_payload", "path": str(target)})
        (root / "runtime.json").unlink()
        atomic_json(
            root / "RETIRED.json",
            {
                "retired_unix": time.time(),
                "reason": "Explicit obsolete-artifact cleanup; original runtime metadata and validation retained",
                "fallback": str(FALLBACK),
                "record": str(LOGS / "journal.json"),
                "payload_removed": True,
            },
        )
    for row in p["generated_fixture_files"]:
        target = Path(row["path"])
        assert any(
            target.is_relative_to(root)
            for root in [
                WS / "training-validation",
                WS / "logs/qwen35-cli-installed",
                WS / "logs/qwen35-cli-resume-fixed",
                WS / "logs/qwen35-resume-guards",
            ]
        )
        assert (
            target.suffix in [".pt", ".safetensors"] or target.name == "tokenizer.json"
        )
        if not target.is_symlink() and target.exists():
            row["sha256"] = digest(target)
        target.unlink()
        record({"kind": "synthetic_fixture", **row})
    for row in p["remove_images"]:
        assert row["id"] not in p["keep_image_ids"]
        subprocess.run(
            ["podman", "rmi", "--no-prune", row["id"]],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        record({"kind": "image", "id": row["id"], "tags": row["tags"]})
    assert (
        fingerprints() == p["protected_sha256"]
    ), "Protected runtime fingerprint changed"
    left = containers()
    for before in p["protected_containers"]:
        assert before in left, "User container metadata changed"
    assert {row["name"] for row in left} == {
        row["name"] for row in p["protected_containers"]
    }
    for iid in p["keep_image_ids"]:
        subprocess.run(["podman", "image", "exists", iid], check=True)
    journal.update(
        available_after_bytes=available(),
        recovered_bytes=available() - p["available_before_bytes"],
        protected_sha256_unchanged=True,
        user_containers_unchanged=True,
        running_agent_jobs=[],
    )
    atomic_json(LOGS / "applied.json", journal)
    print(json.dumps({k: v for k, v in journal.items() if k != "removed"}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    assert os.geteuid() == 0
    if args.apply:
        apply()
    else:
        plan()


if __name__ == "__main__":
    main()
