#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install/register the qualified opt-in SDK, or unregister it without deleting data."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile

from bc250_migraphx import BASE, REGISTRY, NAME, INDEX_SHA, check_sdk, sha, trusted
from bc250_torch import runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

ARCHIVE_SHA = "11f7f7ab1923132131dc0118af62fbd425e55cfd802d86fdcb57e4952583c8ba"
TOOLS = Path("/opt/bc250-stack/tools/migraphx-b068")
RECORD = BASE / (NAME + ".json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "register", "unregister"))
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    preflight("migraphx-b068-" + args.action)
    assert not args.receipt.exists()
    daily = Path("/opt/bc250-stack/current")
    with runtime_lock(daily / ".runtime.lock", exclusive=True):
        if args.action == "install":
            assert (
                os.statvfs("/opt").f_bavail * os.statvfs("/opt").f_frsize > 4 * 1024**3
            )
            archive = WS / "tmp/migraphx-b067-native/incoming/sdk.tar.gz"
            assert sha(archive) == ARCHIVE_SHA
            baseline = json.loads(
                (WS / "logs/migraphx-b067-qualification-r5/final.json").read_text()
            )
            assert baseline["status"] == "passed"
            assert not verify_hashes(baseline["protected_sha256"])
            assert all(os.readlink(p) == v for p, v in baseline["links"].items())
            sdk_manifest = json.loads(
                (
                    WS
                    / "tmp/migraphx-b067-native/package-r3/relocated-A"
                    / NAME
                    / "share/bc250-sdk/manifest.json"
                ).read_text()
            )
            assert sha(daily / "runtime.json") == sdk_manifest["runtime_sha256"]
            assert (
                json.loads((daily / "runtime.json").read_text())["image"]
                == sdk_manifest["runtime_image"]
            )
            BASE.mkdir(mode=0o755, exist_ok=True)
            trusted(BASE)
            stage = BASE / ".stage-b068"
            stage.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(stage, filter="data")
            sdk = stage / NAME
            for path in (sdk, *sdk.rglob("*")):
                if path.is_symlink():
                    os.lchown(path, 0, 0)
                else:
                    os.chown(path, 0, 0)
                    path.chmod(
                        0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644
                    )
            check_sdk(sdk)
            target = BASE / NAME
            assert not target.exists() and not RECORD.exists() and not REGISTRY.exists()
            sdk.rename(target)
            # Retain the empty staging directory as an installation receipt location.
            TOOLS.mkdir(parents=True)
            for name in ("bc250_migraphx.py", "bc250_torch.py"):
                shutil.copyfile(WS / "helpers" / name, TOOLS / name)
                (TOOLS / name).chmod(0o755)
            assert sha(TOOLS / "bc250_torch.py") == sha(
                Path("/usr/local/bin/bc250-torch")
            )
            launcher = Path("/usr/local/bin/bc250-migraphx")
            assert not launcher.exists() and not launcher.is_symlink()
            launcher.symlink_to(TOOLS / "bc250_migraphx.py")
            manifests = {
                "daily": daily / "runtime.json",
                "rccl-single": Path(
                    "/opt/bc250-stack/profiles/rccl-single/runtime.json"
                ),
            }
            record = dict(
                schema=1,
                sdk=str(target),
                archive_sha256=ARCHIVE_SHA,
                index_sha256=INDEX_SHA,
                runtime_sha256={n: sha(p) for n, p in manifests.items()},
                runtime_image=json.loads(manifests["daily"].read_text())["image"],
                launcher_sha256={
                    n: sha(TOOLS / n) for n in ("bc250_migraphx.py", "bc250_torch.py")
                },
                source_revision="4797a61125bbec318005969f8497dbd50849cd6e",
                automatic_torch_dispatch=False,
                ort_gpu_provider=False,
                autograd=False,
            )
            atomic_json(RECORD, record)
            RECORD.chmod(0o644)
        record = json.loads(RECORD.read_text())
        check_sdk(Path(record["sdk"]))
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
                ordinary_free_bytes=os.statvfs("/opt").f_bavail
                * os.statvfs("/opt").f_frsize,
            ),
        )
    print("MIGRAPHX_B068_" + args.action.upper() + "_PASS", flush=True)


if __name__ == "__main__":
    main()
