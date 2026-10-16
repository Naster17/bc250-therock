#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Guarded application deployment. Run as root under update_dev_prefix.py."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
PROJECT = Path("/home/nik/finetune-n7")
OLD_TRAINER_SHA256 = "5bd3579f70f8db67c24c65dfec08e41cfd90583ce45b7a99fe804a392a1d4ade"


def atomic_copy(source: Path, destination: Path, owner) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix="." + destination.name + "-", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, owner.st_mode & 0o777)
        os.chown(temporary, owner.st_uid, owner.st_gid)
        os.replace(temporary, destination)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume-fix",
        action="store_true",
        help="Upgrade the tested intermediate deployment",
    )
    args = parser.parse_args()
    assert os.geteuid() == 0
    trainer = PROJECT / "replica/train.py"
    expected = (
        "5006581fed477a4e2b20e6eab6e9ae49ebe0037027ea2370623d0c85fd67e63e"
        if args.resume_fix
        else OLD_TRAINER_SHA256
    )
    assert (
        hashlib.sha256(trainer.read_bytes()).hexdigest() == expected
    ), "Trainer changed; review before replacing"
    backup = WS / "recovery/qwen35-training-before"
    if args.resume_fix:
        assert (
            hashlib.sha256((backup / "train.py").read_bytes()).hexdigest()
            == OLD_TRAINER_SHA256
        )
        assert (PROJECT / "replica/bc250_ops.py").read_bytes() == (
            WS / "helpers/qwen35_training_ops.py"
        ).read_bytes()
        assert not (backup / "train.native-before-resume-fix.py").exists()
        shutil.copy2(trainer, backup / "train.native-before-resume-fix.py")
    else:
        assert not (PROJECT / "replica/bc250_ops.py").exists()
        assert not (PROJECT / "BC250_TRAINING_OPS.md").exists()
        backup.mkdir(parents=True, exist_ok=False)
        shutil.copy2(trainer, backup / "train.py")
    owner = trainer.stat()
    installs = [
        (WS / "helpers/qwen35_training_ops.py", PROJECT / "replica/bc250_ops.py"),
        (WS / "helpers/qwen35-train.after.py", trainer),
        (WS / "helpers/qwen35_training_usage.md", PROJECT / "BC250_TRAINING_OPS.md"),
    ]
    for source, destination in installs:
        atomic_copy(source, destination, owner)
    record = {
        "backup": str(backup),
        "old_trainer_sha256": OLD_TRAINER_SHA256,
        "files": {
            str(dst): hashlib.sha256(dst.read_bytes()).hexdigest()
            for _, dst in installs
        },
    }
    (WS / "SOURCE_QWEN35_TRAINING_OPS.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
