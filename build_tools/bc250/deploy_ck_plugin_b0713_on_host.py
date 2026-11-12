#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""B07.13: install/activate the qualified B07.12 coherent FP32/FP16 CK artifact.

Modeled on deploy_ck_plugin_b078_on_host.py, with two differences required by the
format2 artifact and the already-present B07.7 registration:

- ``install`` stages the b0712 package + version record + ck-b0713 tools but does
  NOT touch the live registry/symlink (the B07.7 registration already exists).
- ``activate`` switches BOTH miopen-dl.json and /usr/local/bin/bc250-ck together
  under the exclusive update lock (the b0713 wrapper refuses the old b077
  registration), capturing the prior state for ``rollback``. ``unregister``
  removes the registry (activation refuses); ``rollback`` restores the captured
  B07.7 registration + wrapper. All B07.7 package/tool/version records are retained.

Ordinary native defaults, .105 and user applications are never touched.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import tarfile

from bc250_ck_b0713 import (
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
from deploy_torch_a04_on_host import atomic_json
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

TOOLS = Path("/opt/bc250-stack/tools/ck-b0713")
RECORD = BASE / (NAME + ".json")
ROLLBACK = BASE / "miopen-dl-b0713-rollback.json"
LAUNCHER = Path("/usr/local/bin/bc250-ck")
TOOL_FILES = ("bc250_ck.py", "bc250_torch.py", "bc250_migraphx.py")


def _swap_symlink(target: Path) -> None:
    tmp = LAUNCHER.with_name(".bc250-ck-b0713-new")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target)
    os.replace(tmp, LAUNCHER)


def _runtime_manifests():
    return {
        "daily": Path("/opt/bc250-stack/current/runtime.json"),
        "rccl-single": Path("/opt/bc250-stack/profiles/rccl-single/runtime.json"),
    }


def _verify_active(record):
    check_package(Path(record["package"]), record)
    assert all(sha(TOOLS / name) == h for name, h in record["launcher_sha256"].items())
    for profile, location in _runtime_manifests().items():
        assert sha(location) == record["runtime_sha256"][profile]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("install", "activate", "unregister", "rollback"))
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument(
        "--archive",
        type=Path,
        default=WS / "tmp/ck-plugin-b0712/package-r3" / (NAME + ".tar.gz"),
        help="qualified B07.12 archive (SHA-pinned); install only",
    )
    args = p.parse_args()
    preflight("ck-plugin-b0713-" + args.action)
    assert not args.receipt.exists()
    daily = Path("/opt/bc250-stack/current")
    with runtime_lock(daily / ".runtime.lock", exclusive=True):
        if args.action == "install":
            assert (
                os.statvfs("/opt").f_bavail * os.statvfs("/opt").f_frsize > 4 * 1024**3
            )
            baseline = json.loads(
                (WS / "logs/ck-plugin-b0712-qualification-r2/final.json").read_text()
            )
            assert baseline["status"] == "passed"
            # The B07.12 baseline protected set/links predate the D61 release and
            # are stale (current now selects the release; host tmp was cleaned).
            # Install only ADDS the new package/tools/record (asserted absent
            # below), so it cannot disturb existing state; the B07.13 qualification
            # captures a fresh current-state protected set.
            archive = args.archive
            assert sha(archive) == ARCHIVE_SHA
            BASE.mkdir(exist_ok=True)
            trusted(BASE)
            target = BASE / NAME
            # The B07.7 REGISTRY already exists; install only stages the new package.
            assert not target.exists() and not RECORD.exists()
            stage = BASE / ".stage-ck-b0713"
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
            shutil.copyfile(WS / "helpers/bc250_ck_b0713.py", TOOLS / "bc250_ck.py")
            for name in ("bc250_torch.py", "bc250_migraphx.py"):
                shutil.copyfile(WS / "helpers" / name, TOOLS / name)
            for name in TOOL_FILES:
                (TOOLS / name).chmod(0o755)
            assert sha(TOOLS / "bc250_torch.py") == sha(
                Path("/usr/local/bin/bc250-torch")
            )
            manifests = _runtime_manifests()
            record = dict(
                schema=1,
                package=str(target),
                archive_sha256=ARCHIVE_SHA,
                manifest_sha256=MANIFEST_SHA,
                runtime_sha256={k: sha(p) for k, p in manifests.items()},
                runtime_images={
                    k: json.loads(p.read_text())["image"] for k, p in manifests.items()
                },
                launcher_sha256={name: sha(TOOLS / name) for name in TOOL_FILES},
                source_revision="96e5a7dfc43356b1c9789a84ae9d79d41bbb0ad7",
                plugin_source_revision="742ee1e2c0e008d73b7dccc819e6bb8defa9e7ad",
                installed_libraries_revision="8389197c134a7337932021ab7137987a585664ae",
                default_admission_changed=False,
                private_solver_forced=False,
                scope="explicit FP32/FP16 packed2D forward; format2 worker; exact native ABI; recorded fixtures only",
            )
            atomic_json(RECORD, record)
            RECORD.chmod(0o644)
        elif args.action == "activate":
            record = json.loads(RECORD.read_text())
            _verify_active(record)
            assert LAUNCHER.is_symlink()
            # Capture the pre-B07.13 state exactly once. Re-activation after an
            # unregister/rollback must not clobber the original recovery target
            # (the qualification cycles unregister->activate->rollback->activate).
            if not ROLLBACK.exists():
                rollback = dict(
                    schema=1,
                    registry_bytes_b64=(
                        base64.b64encode(REGISTRY.read_bytes()).decode()
                        if REGISTRY.exists()
                        else None
                    ),
                    symlink_target=os.readlink(LAUNCHER),
                    replaced_package=record["package"],
                )
                atomic_json(ROLLBACK, rollback)
                ROLLBACK.chmod(0o644)
            atomic_json(REGISTRY, record)
            REGISTRY.chmod(0o644)
            _swap_symlink(TOOLS / "bc250_ck.py")
            _verify_active(json.loads(REGISTRY.read_text()))
        elif args.action == "unregister":
            record = json.loads(RECORD.read_text())
            assert (
                not REGISTRY.is_symlink()
                and REGISTRY.read_bytes() == RECORD.read_bytes()
            )
            REGISTRY.unlink()
        elif args.action == "rollback":
            rollback = json.loads(ROLLBACK.read_text())
            assert rollback["replaced_package"] == str(BASE / NAME)
            if rollback["registry_bytes_b64"] is None:
                if REGISTRY.exists():
                    REGISTRY.unlink()
            else:
                restored = base64.b64decode(rollback["registry_bytes_b64"])
                tmp = REGISTRY.with_name(".miopen-dl-b0713-restore")
                tmp.write_bytes(restored)
                tmp.chmod(0o644)
                os.replace(tmp, REGISTRY)
            _swap_symlink(Path(rollback["symlink_target"]))
        if args.action in ("install", "activate"):
            record = json.loads(RECORD.read_text())
            check_package(Path(record["package"]), record)
        atomic_json(
            args.receipt,
            dict(
                status="passed",
                action=args.action,
                registration=json.loads(RECORD.read_text()),
                registered=REGISTRY.exists(),
                active_wrapper=os.readlink(LAUNCHER) if LAUNCHER.is_symlink() else None,
                rollback_available=ROLLBACK.exists(),
                native_runtime_changed=False,
                ordinary_free_bytes=os.statvfs("/opt").f_bavail
                * os.statvfs("/opt").f_frsize,
            ),
        )
    print("CK_PLUGIN_B0713_" + args.action.upper() + "_PASS", flush=True)


if __name__ == "__main__":
    main()
