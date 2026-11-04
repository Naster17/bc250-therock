#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage a matched candidate under the .105 control EX lock; never select it here."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    source = Path("/home/nik/bc250work/distributed-d01-import")
    meta = json.loads((source / "manifest.json").read_text())
    for name, row in meta["files"].items():
        assert sha(source / name) == row["sha256"], name
    base = Path("/opt/bc250-stack/current").resolve()
    assert base.name == "10.2-gfx1013-20261023-host105"
    a = json.loads((source / "104-inventory.json").read_text())["rocm"]
    b = json.loads((source / "105-inventory.json").read_text())["rocm"]
    assert a == b
    assert os.statvfs(base).f_bavail * os.statvfs(base).f_frsize > 9 * 1024**3
    dest = Path("/opt/bc250-stack/releases/10.2-gfx1013-20261104-d01")
    dest.mkdir()
    subprocess.run(
        ["podman", "load", "-i", str(source / "image.tar")], check=True, timeout=600
    )
    identity = json.loads(
        subprocess.check_output(
            ["podman", "image", "inspect", meta["image"]], text=True
        )
    )[0]["Id"]
    assert identity.removeprefix("sha256:") == meta["image"]
    # This one redundant transfer copy is retired after verified import. The image,
    # immutable originals, .104 export and workstation recovery copy are retained.
    (source / "image.tar").unlink()
    payload = dest / "payload"
    payload.mkdir()
    with tarfile.open(source / "userspace.tar.gz") as tar:
        tar.extractall(payload, filter="data")
    library = payload / "rccl/lib/librccl.so.1.0"
    for name in ("librccl.so", "librccl.so.1"):
        alias = library.parent / name
        assert sha(alias) == sha(library)
        alias.unlink()
        alias.symlink_to(library.name)
    (dest / "rocm").symlink_to(base / "rocm")
    (dest / "torch").symlink_to(payload / "daily-torch")
    (dest / "python-consumers").symlink_to(payload / "python-consumers")
    shutil.copy2(payload / "rocblas-gemm-n32.csv", dest / "rocblas-gemm-n32.csv")
    (dest / "cache").mkdir()
    os.chown(dest / "cache", 1000, 1000)
    rccl = dest / "rccl-profile"
    rccl.mkdir()
    for name, target in [
        ("rocm", base / "rocm"),
        ("torch", payload / "rccl-torch"),
        ("rccl", payload / "rccl"),
        ("python-consumers", payload / "python-consumers"),
        ("cache", dest / "cache"),
        ("rocblas-gemm-n32.csv", dest / "rocblas-gemm-n32.csv"),
    ]:
        (rccl / name).symlink_to(target)
    for path, manifest in [
        (dest, payload / "daily-runtime.json"),
        (rccl, payload / "rccl-runtime.json"),
    ]:
        data = json.loads(manifest.read_text())
        data.update(
            release=path.name,
            mutable=False,
            target_qualification="PENDING D01/D02; not selected",
            immutable_rocm_base=str(base),
            source_export=meta,
        )
        (path / "runtime.json").write_text(json.dumps(data, indent=2) + "\n")
    assert Path("/opt/bc250-stack/current").resolve() == base
    (dest / "stage.json").write_text(
        json.dumps(
            dict(
                status="staged-not-qualified-not-selected",
                source_image=identity,
                immutable_base_dependency=str(base),
                rocm_entries_matched=len(a),
                kernel_unchanged=os.uname().release,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
