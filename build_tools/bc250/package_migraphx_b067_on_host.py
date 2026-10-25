#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create an opt-in SDK archive and two independently extracted relocation roots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

from bc250_torch import runtime_lock
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

STORAGE = WS / "tmp/migraphx-b064/20261025-storage/mount/b067"
NAME = "bc250-migraphx-2.18.0.dev-gfx1013-cp312-b067"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", required=True, type=Path)
    parser.add_argument("--attempt", default="r1", choices=("r1", "r2", "r3"))
    args = parser.parse_args()
    preflight("migraphx-b067-package")
    assert args.install.parent == STORAGE and args.install.name.startswith("sdk-r")
    assert os.path.ismount(STORAGE.parent)
    assert os.statvfs(STORAGE).f_bavail * os.statvfs(STORAGE).f_frsize > 16 * 1024**3
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 768 * 1024**2
    dest = STORAGE / ("package-" + args.attempt)
    dest.mkdir()
    sdk = dest / NAME
    report = dict(
        status="packaging",
        install=str(args.install),
        runtime_changed=False,
        ort_gpu_provider=False,
    )
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        # The external build wrapper's LICENSE link was copied literally by
        # upstream install. Materialize only this known license from source;
        # reject other broken/escaping links during the manifest audit below.
        staged_license = args.install / "share/doc/migraphx/LICENSE"
        skip_license = staged_license.is_symlink()
        if skip_license:
            assert os.readlink(staged_license) == "source/LICENSE"

        def ignore(folder, names):
            return (
                {"LICENSE"}
                if skip_license and Path(folder) == staged_license.parent
                else set()
            )

        shutil.copytree(args.install, sdk, symlinks=True, ignore=ignore)
        installed_license = sdk / "share/doc/migraphx/LICENSE"
        source_license = STORAGE / "resources/licenses/bc250-migraphx-LICENSE"
        if installed_license.exists():
            assert sha(installed_license) == sha(source_license)
        else:
            shutil.copyfile(source_license, installed_license)
        # Installed libraries use lib/migraphx/lib, while upstream installs
        # the helper under prefix/bin. A verified sibling copy preserves the
        # isolated JIT route without parent-traversing symlinks on SSHFS.
        shutil.copy2(
            sdk / "bin/migraphx-hiprtc-driver",
            sdk / "lib/migraphx/lib/migraphx-hiprtc-driver",
        )
        assert sha(sdk / "bin/migraphx-hiprtc-driver") == sha(
            sdk / "lib/migraphx/lib/migraphx-hiprtc-driver"
        )
        info = sdk / "share/bc250-sdk"
        info.mkdir(parents=True)
        resources = STORAGE / "resources"
        shutil.copytree(resources / "licenses", info / "licenses")
        shutil.copytree(resources / "sources", info / "sources")
        shutil.copyfile(resources / "README.md", sdk / "README.md")
        shutil.copyfile(
            WS / "helpers/migraphx_b067_launcher.py", sdk / "bin/bc250-migraphx.py"
        )
        models = STORAGE.parent / "b065/models-r2"
        generation = STORAGE.parent / "b065/models-generation-r2.json"
        meta = json.loads(generation.read_text())
        assert meta["status"] == "passed" and len(meta["checks"]) == 12
        for name, digest in meta["sha256"].items():
            assert sha(models / name) == digest, name
        shutil.copytree(models, info / "fixtures")
        shutil.copyfile(generation, info / "fixtures.json")
        pins = json.loads((resources / "provenance.json").read_text())
        runtime = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
        manifest = dict(
            name=NAME,
            version="2.18.0.dev",
            architecture="gfx1013",
            python="CPython 3.12",
            platform="qualified glibc runtime image",
            pins=pins,
            runtime_image=runtime["image"],
            runtime_sha256=sha(Path("/opt/bc250-stack/current/runtime.json")),
            dtype_qualified=["FP32"],
            automatic_torch_dispatch=False,
            ort_gpu_provider=False,
            autograd=False,
            jit_compiler_process="lib/migraphx/lib/migraphx-hiprtc-driver (verified sibling copy)",
        )
        atomic_json(info / "manifest.json", manifest)
        hashes, links = {}, {}
        for path in sorted(sdk.rglob("*")):
            name = str(path.relative_to(sdk))
            if path.is_symlink():
                assert not Path(os.readlink(path)).is_absolute(), name
                assert path.resolve().is_relative_to(sdk.resolve()), name
                assert path.exists(), name
                links[name] = os.readlink(path)
            elif path.is_file():
                hashes[name] = sha(path)
        atomic_json(info / "sha256.json", dict(files=hashes, symlinks=links))
        archive = dest / (NAME + ".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(sdk, arcname=NAME)
        roots = []
        for name in ("relocated-A", "relocated B with spaces"):
            folder = dest / name
            folder.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(folder, filter="data")
            extracted = folder / NAME
            for name, digest in hashes.items():
                assert sha(extracted / name) == digest, name
            for name, target in links.items():
                assert os.readlink(extracted / name) == target
            roots.append(str(extracted))
        report.update(
            status="packaged-not-qualified",
            archive=str(archive),
            archive_sha256=sha(archive),
            archive_bytes=archive.stat().st_size,
            files=len(hashes),
            symlinks=len(links),
            roots=roots,
            manifest=manifest,
        )
        atomic_json(dest / "final.json", report)
    print("MIGRAPHX_B067_PACKAGE_PASS", flush=True)


if __name__ == "__main__":
    main()
