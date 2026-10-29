#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install/register an opt-in ORT package or unregister it without deleting payloads."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile

from bc250_ort import BASE, NAME, REGISTRY, check_package
from bc250_migraphx import registered as sdk_registered, sha, trusted
from bc250_torch import LaunchOptions, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

TOOLS = Path("/opt/bc250-stack/tools/ort-b0611")
RECORD = BASE / (NAME + ".json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("install", "register", "unregister"))
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    assert not args.receipt.exists()
    preflight("onnxruntime-b0611-" + args.action)
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock"), exclusive=True):
        sdk_record, _ = sdk_registered(LaunchOptions())
        sdk_registered(LaunchOptions(runtime_profile="rccl-single"))
        if args.action == "install":
            assert os.statvfs(BASE).f_bavail * os.statvfs(BASE).f_frsize > 4 * 1024**3
            q = json.loads(
                (WS / "logs/onnxruntime-b0611-relocation-r1/final.json").read_text()
            )
            assert (
                q["status"] == "passed" and sum(g["checks"] for g in q["gates"]) == 68
            )
            assert not verify_hashes(q["protected_sha256"])
            packaged = json.loads(
                (WS / "tmp/onnxruntime-b0611/package-r1/final.json").read_text()
            )
            archive = Path(packaged["archive"])
            assert sha(archive) == packaged["archive_sha256"]
            target = BASE / NAME
            assert not target.exists() and not REGISTRY.exists() and not RECORD.exists()
            stage = BASE / ".stage-ort-b0611"
            stage.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(stage, filter="data")
            root = stage / NAME
            for path in (root, *root.rglob("*")):
                if path.is_symlink():
                    os.lchown(path, 0, 0)
                else:
                    os.chown(path, 0, 0)
                    path.chmod(
                        0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644
                    )
            record = dict(
                schema=1,
                package=str(target),
                index_sha256=packaged["index_sha256"],
                manifest_sha256=packaged["manifest_sha256"],
                archive_sha256=packaged["archive_sha256"],
                sdk_registration=sdk_record,
                source_revision="b81a7866e5a6eff964b9e411535b4c9198651a9f",
                automatic_torch_dispatch=False,
                autograd=False,
                qualified_dtype=["FP32"],
            )
            check_package(root, record)
            root.rename(target)
            TOOLS.mkdir(parents=True)
            record["launcher_sha256"] = {}
            for name in ("bc250_ort.py", "bc250_migraphx.py", "bc250_torch.py"):
                shutil.copyfile(WS / "helpers" / name, TOOLS / name)
                (TOOLS / name).chmod(0o755)
                record["launcher_sha256"][name] = sha(TOOLS / name)
            public = Path("/usr/local/bin/bc250-ort")
            assert not public.exists() and not public.is_symlink()
            public.symlink_to(TOOLS / "bc250_ort.py")
            atomic_json(RECORD, record)
            RECORD.chmod(0o644)
        record = json.loads(RECORD.read_text())
        trusted(RECORD)
        assert record["sdk_registration"] == sdk_record
        check_package(Path(record["package"]), record)
        for name, digest in record["launcher_sha256"].items():
            assert sha(TOOLS / name) == digest
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
                active_runtime_changed=False,
                host105_contacted=False,
            ),
        )
    print("ORT_B0611_" + args.action.upper() + "_PASS", flush=True)


if __name__ == "__main__":
    main()
