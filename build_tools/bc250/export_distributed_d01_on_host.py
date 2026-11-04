#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Freeze selected daily/RCCL userspace; preserve installed runtime and kernel."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    dest = Path("/home/nik/bc250work/distributed-d01-export")
    dest.mkdir()
    daily = Path("/opt/bc250-stack/current").resolve()
    rccl = Path("/opt/bc250-stack/profiles/rccl-single").resolve()
    with (daily / ".runtime.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        with tarfile.open(dest / "userspace.tar.gz", "w:gz", dereference=True) as t:
            for name, path in [
                ("daily-torch", (daily / "torch").resolve()),
                ("rccl-torch", (rccl / "torch").resolve()),
                ("rccl", (rccl / "rccl").resolve()),
                ("python-consumers", daily / "python-consumers"),
            ]:
                t.add(path, arcname=name)
            for name, path in [
                ("daily-runtime.json", daily / "runtime.json"),
                ("rccl-runtime.json", rccl / "runtime.json"),
                ("rocblas-gemm-n32.csv", daily / "rocblas-gemm-n32.csv"),
            ]:
                t.add(path, arcname=name)
        image = json.loads((daily / "runtime.json").read_text())["image"]
        subprocess.run(
            [
                "podman",
                "save",
                "--format",
                "oci-archive",
                "-o",
                str(dest / "image.tar"),
                image,
            ],
            check=True,
            timeout=600,
        )
        (dest / "manifest.json").write_text(
            json.dumps(
                dict(
                    status="frozen-not-target-qualified",
                    image=image,
                    files={
                        p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                        for p in [dest / "userspace.tar.gz", dest / "image.tar"]
                    },
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
