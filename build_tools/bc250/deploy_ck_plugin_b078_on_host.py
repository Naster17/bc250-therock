#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install/register only the qualified r2 opt-in artifact; retain data on unregister."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
from bc250_ck import (
    BASE,
    REGISTRY,
    NAME,
    ARCHIVE_SHA,
    MANIFEST_SHA,
    check_package,
    sha,
    trusted,
)
from bc250_torch import runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

TOOLS = Path("/opt/bc250-stack/tools/ck-b078")
RECORD = BASE / (NAME + ".json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("install", "register", "unregister"))
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    preflight("ck-plugin-b078-" + args.action)
    assert not args.receipt.exists()
    daily = Path("/opt/bc250-stack/current")
    with runtime_lock(daily / ".runtime.lock", exclusive=True):
        if args.action == "install":
            assert (
                os.statvfs("/opt").f_bavail * os.statvfs("/opt").f_frsize > 4 * 1024**3
            )
            baseline = json.loads(
                (WS / "logs/ck-plugin-b077-qualification-r3/final.json").read_text()
            )
            assert baseline["status"] == "passed"
            assert not verify_hashes(baseline["protected_sha256"])
            assert all(os.readlink(p) == v for p, v in baseline["links"].items())
            archive = WS / "tmp/ck-plugin-b077/package-r2" / (NAME + ".tar.gz")
            assert sha(archive) == ARCHIVE_SHA
            BASE.mkdir(exist_ok=True)
            trusted(BASE)
            target = BASE / NAME
            assert not target.exists() and not RECORD.exists() and not REGISTRY.exists()
            stage = BASE / ".stage-ck-b078"
            stage.mkdir()
            with tarfile.open(archive) as t:
                t.extractall(stage, filter="data")
            root = stage / NAME
            for item in (root, *root.rglob("*")):
                assert not item.is_symlink()
                os.chown(item, 0, 0)
                item.chmod(
                    0o755 if item.is_dir() or item.stat().st_mode & 0o111 else 0o644
                )
            check_package(root, dict(manifest_sha256=MANIFEST_SHA))
            root.rename(target)
            TOOLS.mkdir(parents=True)
            for name in ("bc250_ck.py", "bc250_torch.py", "bc250_migraphx.py"):
                shutil.copyfile(WS / "helpers" / name, TOOLS / name)
                (TOOLS / name).chmod(0o755)
            assert sha(TOOLS / "bc250_torch.py") == sha(
                Path("/usr/local/bin/bc250-torch")
            )
            launcher = Path("/usr/local/bin/bc250-ck")
            assert not launcher.exists() and not launcher.is_symlink()
            launcher.symlink_to(TOOLS / "bc250_ck.py")
            manifests = {
                "daily": daily / "runtime.json",
                "rccl-single": Path(
                    "/opt/bc250-stack/profiles/rccl-single/runtime.json"
                ),
            }
            record = dict(
                schema=1,
                package=str(target),
                archive_sha256=ARCHIVE_SHA,
                manifest_sha256=MANIFEST_SHA,
                runtime_sha256={k: sha(p) for k, p in manifests.items()},
                runtime_images={
                    k: json.loads(p.read_text())["image"] for k, p in manifests.items()
                },
                launcher_sha256={
                    name: sha(TOOLS / name)
                    for name in ("bc250_ck.py", "bc250_torch.py", "bc250_migraphx.py")
                },
                source_revision="3616b2f9f09314d87304c4e20c77592d0dea507f",
                default_admission_changed=False,
                private_solver_forced=False,
                scope="explicit FP32 packed2D forward; exact native ABI; recorded fixtures only",
            )
            atomic_json(RECORD, record)
            RECORD.chmod(0o644)
        record = json.loads(RECORD.read_text())
        check_package(Path(record["package"]), record)
        assert all(
            sha(TOOLS / name) == h for name, h in record["launcher_sha256"].items()
        )
        for profile, location in [
            ("daily", daily),
            ("rccl-single", Path("/opt/bc250-stack/profiles/rccl-single")),
        ]:
            assert sha(location / "runtime.json") == record["runtime_sha256"][profile]
        if args.action == "unregister":
            assert (
                not REGISTRY.is_symlink()
                and REGISTRY.read_bytes() == RECORD.read_bytes()
            )
            REGISTRY.unlink()
        else:
            assert not REGISTRY.exists() and not REGISTRY.is_symlink()
            atomic_json(REGISTRY, record)
            REGISTRY.chmod(0o644)
        atomic_json(
            args.receipt,
            dict(
                status="passed",
                action=args.action,
                registration=record,
                registered=REGISTRY.exists(),
                native_runtime_changed=False,
                ordinary_free_bytes=os.statvfs("/opt").f_bavail
                * os.statvfs("/opt").f_frsize,
            ),
        )
    print("CK_PLUGIN_B078_" + args.action.upper() + "_PASS", flush=True)


if __name__ == "__main__":
    main()
