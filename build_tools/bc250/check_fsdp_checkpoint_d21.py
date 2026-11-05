#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline checkpoint union guards with actual files and unchanged local state."""
from pathlib import Path
import subprocess
import sys
import tempfile


def contracts():
    tool = Path(__file__).with_name("bc250_fsdp_checkpoint.py")
    with tempfile.TemporaryDirectory(prefix="bc250-d21-checkpoint-") as temporary:
        root = Path(temporary)

        def call(*args, ok=True):
            p = subprocess.run(
                [sys.executable, str(tool), *map(str, args)],
                capture_output=True,
                text=True,
            )
            assert (p.returncode == 0) == ok, (args, p.stdout, p.stderr)

        for rank in [0, 1]:
            checkpoint = root / str(rank)
            checkpoint.mkdir()
            (checkpoint / "rng-local.pth").write_bytes(b"local-" + str(rank).encode())
            for name in ["pytorch_model_fsdp_0", "optimizer_0"]:
                d = checkpoint / name
                d.mkdir()
                (d / f"__{rank}_0.distcp").write_bytes((name + str(rank)).encode())
                if rank == 0:
                    (d / ".metadata").write_bytes(b"trusted-metadata-" + name.encode())
            call(
                "export",
                "--checkpoint",
                checkpoint,
                "--output",
                root / f"{rank}.tar.gz",
            )
        call(
            "merge",
            "--input",
            root / "0.tar.gz",
            root / "1.tar.gz",
            "--output",
            root / "union.tar.gz",
        )
        for rank in [0, 1]:
            call(
                "install",
                "--checkpoint",
                root / str(rank),
                "--archive",
                root / "union.tar.gz",
            )
            for name in ["pytorch_model_fsdp_0", "optimizer_0"]:
                assert (root / str(rank) / name / "__0_0.distcp").exists() and (
                    root / str(rank) / name / "__1_0.distcp"
                ).exists()
            assert (root / str(rank) / "rng-local.pth").read_bytes() == b"local-" + str(
                rank
            ).encode()
        call(
            "merge",
            "--input",
            root / "0.tar.gz",
            root / "0.tar.gz",
            "--output",
            root / "invalid.tar.gz",
            ok=False,
        )
        assert not (root / "invalid.tar.gz").exists()
        corrupted = root / "1/optimizer_0/__0_0.distcp"
        corrupted.write_bytes(b"conflicting-existing-state")
        call(
            "install",
            "--checkpoint",
            root / "1",
            "--archive",
            root / "union.tar.gz",
            ok=False,
        )
        assert corrupted.read_bytes() == b"conflicting-existing-state"
        call(
            "export",
            "--checkpoint",
            root / "0",
            "--output",
            root / "0.tar.gz",
            ok=False,
        )
    return [
        "two-host-DCP-payload-union-and-idempotent-install",
        "local-RNG-preserved",
        "duplicate-input-and-existing-payload-conflict-refusals",
        "archive-overwrite-refusal",
    ]


if __name__ == "__main__":
    print(contracts())
