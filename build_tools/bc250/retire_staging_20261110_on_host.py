#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retire verified closed staging, evidence, helper and documentation copies."""

import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import shutil

from board_preflight import preflight
from cleanup_storage_20261110 import protected, sha
from external_kit_d53 import inventory, raise_walk_error

WORK = Path("/home/nik/bc250work")
EVIDENCE = WORK / "evidence"
TOOLS = WORK / "tools/storage-cleanup-20261110"
WS = WORK / "rocm-10.2-gfx1013"
TMP = WS / "tmp"
DOCS = WORK / "docs/snapshots"
WORKSPACE_LEGACY = {
    "agent-memory",
    "staging",
    "{logs,recovery,source,build}",
    "launcher-io-validation-3w7nl4_x",
    "launcher-io-validation-hb7hqctb",
    "launcher-io-validation-hleaqta2",
    "0001-use-supported-Clang-instruction-cache-builtin.patch",
    "0003-fix-clr-use-the-established-HIP-pool-backend-by-default-on-gfx1013.patch",
    "ffts-CMakeLists.txt",
    "ffts-post-hook.cmake",
    "post_hook_therock-ffts.cmake",
    "finish-expanded-new.sh",
    "finish_expanded_validation_on_host.sh",
    "launcher-fixed.py",
    "media_caps",
    "media_caps.c",
    "quant-b056-r2-stage-source.py",
    "rocdevice.cpp",
    "rocprim-arch-fixed.cpp",
    "rocprim_arch_smoke.cpp",
    "validate_rocprim_arch_on_host.sh",
}
REGISTRY_KEEP = {
    "distributed-d01",
    "distributed-d05-tool-install-r1",
    "distributed-d09-tool-install-r1",
    "distributed-d35-r1",
    "distributed-d41-r1",
    "source-d52-package-r1",
}


def checked_roots(manifest, phase):
    allowed = {
        "docs": {DOCS},
        "workspace": {WS},
        "staging": {TMP, EVIDENCE, WORK / "tools"},
    }[phase]
    roots = {Path(path): label for label, path in manifest["roots"].items()}
    assert len(roots) == len(manifest["roots"])
    for path in roots:
        assert path.parent in allowed and path.resolve() == path, path
        assert not path.name.startswith(".") and not path.is_symlink(), path
        if path.parent == WS:
            assert path.name in WORKSPACE_LEGACY
        elif path.parent == DOCS:
            assert path.name not in {"20261110-d52-final", "20261110-d53-final"}
            assert path != (DOCS.parent / "current").resolve()
        elif path.parent == EVIDENCE:
            assert path.name not in REGISTRY_KEEP
            assert not path.name.startswith("storage-cleanup-20261110")
        elif path.parent == WORK / "tools":
            assert path.name not in {"distributed-d01", "distributed-d09"}
            assert not path.name.startswith("storage-cleanup-")
            assert not path.name.startswith("source-") or path.name == "source-d46"
    return roots


def retained_references(roots, entries):
    blocked = defaultdict(list)

    def containing(path):
        return next((p for p in (path, *path.parents) if p in roots), None)

    def mark(path, reason):
        root = containing(path)
        if root is not None and reason not in blocked[root]:
            blocked[root].append(reason)

    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        target = Path(line.split()[4].replace("\\040", " "))
        mark(target, "mount:" + str(target))

    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for item in [proc / "cwd", proc / "root", *(proc / "fd").iterdir()]:
                try:
                    target = item.readlink()
                except FileNotFoundError:
                    continue
                if target.is_absolute():
                    mark(target, str(item))
            for line in (proc / "maps").read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6 and fields[5].startswith("/"):
                    mark(Path(fields[5].removesuffix(" (deleted)")), str(proc / "maps"))
        except (FileNotFoundError, ProcessLookupError):
            continue

    for base in (WORK, Path("/opt/bc250-stack"), Path("/usr/local/bin"), Path("/etc")):
        for directory, dirs, files in os.walk(base, onerror=raise_walk_error):
            parent = Path(directory)
            dirs[:] = [name for name in dirs if parent / name not in roots]
            for name in (*dirs, *files):
                path = parent / name
                if path in roots or not path.is_symlink():
                    continue
                mark(Path(os.path.normpath(parent / path.readlink())), str(path))
                try:
                    mark(path.resolve(), str(path))
                except RuntimeError:
                    continue

    by_label = {label: path for path, label in roots.items()}
    links = []
    for entry in entries:
        if entry["kind"] == "symlink":
            label, _, relative = entry["path"].partition("/")
            origin = by_label[label]
            link = origin / relative
            target = Path(os.path.normpath(link.parent / entry["target"]))
            links.append((origin, target, str(link)))
    while True:
        old = set(blocked)
        for origin, target, source in links:
            if origin in blocked:
                mark(target, source)
        if old == set(blocked):
            return dict(blocked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["docs", "staging", "workspace"])
    args = parser.parse_args()
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    phase = args.phase
    receipt = json.loads((TOOLS / f"{phase}-receipt.json").read_text())
    manifest_file = TOOLS / f"{phase}-verified-manifest.json"
    assert (
        receipt["status"] == "passed"
        and sha(manifest_file) == receipt["manifest_sha256"]
    )
    manifest = json.loads(manifest_file.read_text())
    roots = checked_roots(manifest, phase)
    output = EVIDENCE / "storage-cleanup-20261110" / f"{phase}-retirement.json"
    assert not output.exists()
    result = dict(status="running", pid=os.getpid(), deleted=[], archive=receipt)
    with (EVIDENCE / ".board-reservation.lock").open("a") as board, Path(
        "/opt/bc250-stack/current/.runtime.lock"
    ).open("r") as runtime:
        fcntl.flock(board, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(runtime, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.write_text(json.dumps(result, indent=2) + "\n")
        try:
            result["preflight"] = preflight(f"storage-cleanup-{phase}")
            before = protected()
            original = json.loads(
                (EVIDENCE / "storage-cleanup-20261110/plan.json").read_text()
            )["protected"]
            assert before["files"] == original["files"]
            assert before["links"] == original["links"]
            assert inventory(manifest["roots"], allow_fifos=True) == manifest
            retained = retained_references(roots, manifest["entries"])
            result["retained"] = {str(p): why for p, why in retained.items()}
            result["free_before"] = shutil.disk_usage(WORK).free
            for path in sorted(roots):
                if path in retained:
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                result["deleted"].append(str(path))
            assert protected() == before
            result["postflight"] = preflight(f"storage-cleanup-{phase}-final")
            result.update(
                status="passed",
                free_after=shutil.disk_usage(WORK).free,
                protected_files=len(before["files"]),
            )
        except BaseException as error:
            result.update(status="failed", error=repr(error))
            raise
        finally:
            output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "deleted": len(result["deleted"])}))


if __name__ == "__main__":
    main()
