#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audited project-only generated-artifact cleanup; run under exclusive lock.

Retains source/generated source, metadata, active builds, caches, installed DSOs,
images, boot/software fallback and user data. Never performs global prune.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
LOGS = WS / "logs/storage-refresh-20261018"
OLD_TORCH = [
    WS / n / "build" for n in ("pytorch-work", "pytorch-a04", "pytorch-sparse-followup")
]
TRITON_TESTS = [WS / "triton-work/build/bin", WS / "triton-work/build/test"]
FIXTURES = [
    WS / "logs" / n / "synthetic-cli"
    for n in (
        "a06-a07-installed-cli",
        "a06-a07-installed-cli-r3",
        "a06-a07-installed-cli-r4",
        "profiler-a09-candidate-cli",
        "profiler-a09-installed-cli",
    )
]
CONTAINERS = {
    "bc250-a04-torch-build",
    "bc250-a04-torch-build-r2",
    "bc250-a04-torch-build-r3",
    "bc250-a04-torch-build-r4",
    "bc250-sparse-followup-build",
    "bc250-sparse-followup-build-r2",
    "bc250-sparse-followup-build-r3",
    "bc250-sparse-followup-build-r4",
    "bc250-profiler-a09-build",
}


def run(command):
    return subprocess.check_output(command, text=True)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(name, data):
    p = LOGS / name
    tmp = p.with_suffix(".new")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, p)


def free():
    s = os.statvfs(WS)
    return s.f_bavail * s.f_frsize


def container_rows():
    result = []
    for line in run(["podman", "ps", "-a", "--format", "{{.ID}}"]).splitlines():
        r = json.loads(run(["podman", "inspect", line]))[0]
        result.append(
            dict(
                id=r["Id"],
                name=r["Name"],
                image=r["Image"],
                status=r["State"]["Status"],
            )
        )
    return result


def stamp(path):
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode) and not path.is_symlink(), str(path)
    return dict(
        inode=s.st_ino,
        device=s.st_dev,
        size=s.st_size,
        mtime_ns=s.st_mtime_ns,
        blocks=s.st_blocks,
    )


def refs(prefixes):
    found = []

    def matches(target):
        return any(target == p or target.startswith(p + "/") for p in prefixes)

    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        for p in [proc / "cwd", proc / "exe", *list((proc / "fd").glob("*"))]:
            try:
                target = str(p.readlink())
            except OSError:
                continue
            if matches(target):
                found.append(dict(pid=proc.name, kind=p.parent.name, path=target))
        try:
            lines = (proc / "maps").read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            target = line.split()[-1]
            if matches(target):
                found.append(dict(pid=proc.name, kind="map", path=target))
    return found


def symlink_refs(prefixes):
    roots = [Path("/opt/bc250-stack")] + [
        WS / n
        for n in (
            "pytorch-work",
            "pytorch-a04",
            "pytorch-sparse-followup",
            "pytorch-profiler-a09",
            "triton-work",
            "helpers",
            "logs",
            "training-env",
            "training-env-stable",
        )
    ]
    found = []
    for root in roots:
        for current, dirs, files in os.walk(root, followlinks=False):
            base = Path(current)
            dirs[:] = [
                d
                for d in dirs
                if base / d not in OLD_TORCH + TRITON_TESTS and d != ".git"
            ]
            for n in dirs + files:
                p = base / n
                if not p.is_symlink():
                    continue
                try:
                    target = p.resolve()
                except (OSError, RuntimeError):
                    continue
                if any(target == r or target.is_relative_to(r) for r in prefixes):
                    found.append(dict(path=str(p), target=str(target)))
    return found


def protected():
    p = json.loads((WS / "logs/solver-session-n211/final.json").read_text())[
        "protected_sha256"
    ].copy()
    for prefix in (Path("/opt/bc250-stack/current"), Path("/opt/bc250-stack/previous")):
        for name in (
            "runtime.json",
            "torch/torch/lib/libtorch_cpu.so",
            "torch/torch/lib/libtorch_hip.so",
            "python-consumers/training.env",
            "python-consumers/packages.json",
            "rocm/lib/libMIOpen.so",
            "rocm/lib/librocblas.so",
            "rocm/.kpack/blas_lib_gfx1013.kpack",
        ):
            path = prefix / name
            p[str(path)] = sha(path)
    for n in (
        "pytorch-work",
        "pytorch-a04",
        "pytorch-sparse-followup",
        "pytorch-profiler-a09",
    ):
        for path in (WS / n / "torch").glob("_C*.so"):
            p[str(path)] = sha(path)
        p[str(WS / n / "GIT_REVISION")] = sha(WS / n / "GIT_REVISION")
    path = WS / "triton-work/python/triton/_C/libtriton.so"
    p[str(path)] = sha(path)
    return p


def check_protected(data):
    for p, h in data.items():
        assert sha(Path(p)) == h, p


def images():
    return sorted(
        run(
            [
                "podman",
                "images",
                "-a",
                "--no-trunc",
                "--format",
                "{{.ID}} {{.Repository}}:{{.Tag}}",
            ]
        ).splitlines()
    )


def check_gate(label):
    logs = WS / "logs" / label
    assert json.loads((logs / "exit.json").read_text()) == dict(
        returncode=0, kernel_log_unchanged=True
    )
    r = json.loads((logs / "result.json").read_text())
    assert r["status"] == "passed" and len(r["checks"]) == 8
    g = json.loads((Path("/run") / label / "state.json").read_text())
    assert g["status"] == "acknowledged" and g["timeout_seconds"] == 600


def plan():
    LOGS.mkdir()
    assert (
        Path("/opt/bc250-stack/current/torch").resolve() == WS / "pytorch-profiler-a09"
    )
    assert Path("/opt/bc250-stack/previous").resolve() == Path(
        "/opt/bc250-stack/releases/10.2-gfx1013-20261013-4b000d542"
    )
    rows = container_rows()
    assert not any(r["status"] == "running" for r in rows)
    prefixes = OLD_TORCH + TRITON_TESTS + FIXTURES
    links = symlink_refs(prefixes)
    assert not links, links
    live = refs([str(p) for p in prefixes])
    assert not live, live
    files = []
    for root in OLD_TORCH + TRITON_TESTS + FIXTURES:
        assert root.resolve() == root and root.is_dir(), str(root)
        for p in root.rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            kind = None
            if root in OLD_TORCH and (
                p.suffix in (".o", ".a", ".so") or ".so." in p.name
            ):
                kind = "superseded-torch-compiled-output"
            elif root in TRITON_TESTS:
                with p.open("rb") as stream:
                    elf = stream.read(4) == b"\x7fELF"
                if elf or p.suffix in (".o", ".a", ".so"):
                    kind = "old-triton-test-or-tool-binary"
            elif root in FIXTURES and (
                p.suffix in (".pt", ".safetensors") or p.name == "tokenizer.json"
            ):
                kind = "synthetic-validation-weight-state-tokenizer"
            if kind:
                files.append(dict(path=str(p), kind=kind, stamp=stamp(p)))
    keep = protected()
    assert not set(r["path"] for r in files) & set(keep)
    check_protected(keep)
    data = dict(
        status="planned",
        created_unix=time.time(),
        free_before_bytes=free(),
        files=files,
        containers=[r for r in rows if r["name"] in CONTAINERS],
        protected_containers=[r for r in rows if r["name"] not in CONTAINERS],
        images_before=images(),
        protected_sha256=keep,
        links={
            str(p): str(p.readlink())
            for p in [
                Path("/opt/bc250-stack/current"),
                Path("/opt/bc250-stack/previous"),
                Path("/opt/bc250-stack/current/torch"),
                Path("/opt/bc250-stack/current/rocm"),
            ]
        },
        process_references=live,
        inbound_symlinks=links,
    )
    save("plan.json", data)
    print(
        "STORAGE_REFRESH_PLAN",
        len(files),
        "files",
        sum(r["stamp"]["blocks"] * 512 for r in files),
        "allocated bytes",
        len(data["containers"]),
        "stopped containers",
        flush=True,
    )


def apply():
    data = json.loads((LOGS / "plan.json").read_text())
    check_gate("storage-refresh-current-before")
    assert not run(["podman", "ps", "-q"]).strip()
    check_protected(data["protected_sha256"])
    assert not refs([str(p) for p in OLD_TORCH + TRITON_TESTS + FIXTURES])
    assert not symlink_refs(OLD_TORCH + TRITON_TESTS + FIXTURES)
    for row in data["files"]:
        p = Path(row["path"])
        assert p.resolve() == p and stamp(p) == row["stamp"], str(p)
    journal = []
    save("applied.json", dict(status="running", actions=journal))
    # Rootfs changes belong only to these stopped project build containers.
    tails = LOGS / "retired-container-logs"
    tails.mkdir()
    for row in data["containers"]:
        current = [r for r in container_rows() if r["id"] == row["id"]]
        assert current == [row] and row["status"] in ("exited", "created", "stopped")
        logs = subprocess.run(
            ["podman", "logs", "--tail", "100", row["id"]],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout
        logs = "\n".join(
            (
                "[REDACTED]"
                if re.search(
                    "password|secret|authorization|private.?key|access.?token",
                    line,
                    re.I,
                )
                else line
            )
            for line in logs.splitlines()
        )
        (tails / (row["name"] + ".log")).write_text(logs + "\n")
        subprocess.run(
            ["podman", "rm", row["id"]], check=True, stdout=subprocess.DEVNULL
        )
        journal.append(
            dict(
                action="removed-stopped-project-container",
                id=row["id"],
                name=row["name"],
            )
        )
        save("applied.json", dict(status="running", actions=journal))
    # No rmtree: preserve generated sources, build metadata, directory paths and links.
    for row in data["files"]:
        p = Path(row["path"])
        assert stamp(p) == row["stamp"], str(p)
        p.unlink()
        journal.append(
            dict(
                action="unlinked-generated-file",
                path=str(p),
                kind=row["kind"],
                bytes=row["stamp"]["size"],
            )
        )
    save("applied.json", dict(status="files-removed-verifying", actions=journal))
    # Mark affected build trees without altering their source or source revisions.
    for root in OLD_TORCH + [WS / "triton-work/build"]:
        (root / "GENERATED_ARTIFACTS_RETIRED_20261018.json").write_text(
            json.dumps(
                dict(
                    reason="User-authorized obsolete compiled output cleanup; sources/configuration retained, Ninja will regenerate missing targets",
                    journal=str(LOGS / "applied.json"),
                ),
                indent=2,
            )
            + "\n"
        )
    check_protected(data["protected_sha256"])
    assert images() == data["images_before"]
    rows = container_rows()
    for row in data["protected_containers"]:
        assert row in rows, row
    for p, target in data["links"].items():
        assert str(Path(p).readlink()) == target, p
    after = free()
    save(
        "applied.json",
        dict(
            status="passed",
            free_before_bytes=data["free_before_bytes"],
            free_after_bytes=after,
            recovered_bytes=after - data["free_before_bytes"],
            protected_hashes_unchanged=len(data["protected_sha256"]),
            actions=journal,
            images_unchanged=True,
            protected_containers_unchanged=True,
        ),
    )
    print(
        "STORAGE_REFRESH_REMOVAL_AND_HASHES_PASS",
        after - data["free_before_bytes"],
        "bytes recovered",
        flush=True,
    )


def main() -> None:
    assert os.geteuid() == 0
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if args.apply:
        apply()
    else:
        plan()


if __name__ == "__main__":
    main()
