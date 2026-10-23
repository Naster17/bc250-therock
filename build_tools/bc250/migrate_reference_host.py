#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inspect or seal a copied BC-250 release; never prune sources or reboot."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path) -> dict:
    result = {}
    for path in sorted(root.rglob("*")):
        name = str(path.relative_to(root))
        if path.is_symlink():
            result[name] = {"link": os.readlink(path)}
        elif path.is_file():
            result[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "seal"))
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = args.release.resolve(strict=True)
    if args.action == "inventory":
        result = {
            name: inventory((release / name).resolve(strict=True))
            for name in ("rocm", "torch")
        }
    else:
        assert os.geteuid() == 0
        assert str(release).startswith("/opt/bc250-stack/releases/")
        manifest = json.loads((release / "runtime.json").read_text())
        assert manifest["torch_revision"] == "2b449c4abe36b896a6efb2a68e7a42d9922d5696"
        assert (
            manifest["image"]
            == "506f3b262fa5fe8c29b9babec07b03078dd6f85492a9ac2cffc65955297e4349"
        )
        assert (
            subprocess.check_output(
                [
                    "podman",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    manifest["image"],
                ],
                text=True,
            ).strip()
            == manifest["image"]
        )
        assert (release / "rocm").is_dir() and not (release / "rocm").is_symlink()
        assert (release / "torch").is_dir() and not (release / "torch").is_symlink()
        for section in ("rocm", "torch"):
            root = release / section
            for path in root.rglob("*"):
                if path.is_symlink():
                    assert not Path(os.readlink(path)).is_absolute(), (
                        f"Non-relocatable link: {path}"
                    )
                    assert path.resolve().is_relative_to(root), f"Escaping link: {path}"
        source = manifest.copy()
        (release / "source-runtime.json").write_text(
            json.dumps(source, indent=2) + "\n"
        )
        manifest.update(
            release="10.2-gfx1013-20261023-host105",
            mutable=False,
            validation_workspace=str(release),
            migration_host="192.168.1.105",
            migration_status="copied; host-specific qualification pending",
            qualification="Source qualification retained; see migration evidence for target qualification",
        )
        manifest["limitations"] = [
            "SDMA and experimental VM heap disabled; math SDPA default",
            "Native gfx1013 subsets as documented; RCCL disabled in this daily Torch build",
            "Bounded source-host qualification; no universal or sustained target claim",
        ]
        (release / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
        cache = Path("/var/cache/bc250-stack/host105")
        cache.mkdir(parents=True, exist_ok=False)
        os.chown(cache, manifest["uid"], manifest["gid"])
        (release / "cache").symlink_to(cache)
        result = {
            "status": "sealed_not_selected",
            "release": str(release),
            "source_runtime_sha256": digest(release / "source-runtime.json"),
            "runtime_sha256": digest(release / "runtime.json"),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"status": "passed", "action": args.action, "output": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
