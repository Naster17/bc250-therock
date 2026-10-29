#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Package the matching ORT candidate, preserve all originals, and extract two roots."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from bc250_migraphx import registered
from bc250_torch import LaunchOptions, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

NAME = "bc250-ort-migraphx-1.31.0-gfx1013-cp312-b0611"
BASE = WS / "tmp/onnxruntime-b0611"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", required=True)
    args = p.parse_args()
    assert args.attempt in ("r1", "r2", "r3")
    preflight("onnxruntime-b0611-package-" + args.attempt)
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 8 * 1024**3
    task = BASE / ("package-" + args.attempt)
    task.mkdir(parents=True)
    pkg = task / NAME
    info = pkg / "share/bc250-ort"
    info.mkdir(parents=True)
    baseline = json.loads(
        (WS / "logs/onnxruntime-b0610-qualification-r1/final.json").read_text()
    )
    candidate = WS / "tmp/onnxruntime-b0610/task-r2/build"
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        assert baseline["status"] == "passed" and not verify_hashes(
            baseline["protected_sha256"]
        )
        registration, sdk = registered(LaunchOptions())
        shutil.copytree(
            candidate / "onnxruntime", pkg / "python/onnxruntime", symlinks=True
        )
        lib = pkg / "lib"
        lib.mkdir()
        for path in candidate.glob("libonnxruntime.so*"):
            if path.is_symlink():
                (lib / path.name).symlink_to(os.readlink(path))
            else:
                shutil.copy2(path, lib / path.name)
        shutil.copytree(WS / "tmp/onnxruntime-b0610/source-r2/include", pkg / "include")
        (pkg / "bin").mkdir()
        shutil.copyfile(WS / "helpers/ort_b0611_launcher.py", pkg / "bin/bc250-ort.py")
        metadata = pkg / "python/onnxruntime/capi/build_and_package_info.py"
        assert not metadata.exists()
        metadata.write_text(
            "# Generated package identity; no CUDA/training claim.\npackage_name = 'onnxruntime-migraphx'\n__version__ = '1.31.0+gfx1013.b0611'\n"
        )
        dist = pkg / "python/onnxruntime_migraphx-1.31.0+gfx1013.b0611.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: onnxruntime-migraphx\nVersion: 1.31.0+gfx1013.b0611\nRequires-Python: >=3.12,<3.13\nLicense: MIT\n"
        )
        (dist / "INSTALLER").write_text("bc250-versioned-opt-in-archive\n")
        (dist / "top_level.txt").write_text("onnxruntime\n")
        # Normalize only copied ELF files; keep original build binaries untouched.
        code = """
import pathlib, subprocess, json, hashlib
root=pathlib.Path('/package')
files=[p for p in root.rglob('*') if p.is_file() and not p.is_symlink() and p.open('rb').read(4)==b'\\x7fELF']
record=[]
for p in files:
 before=hashlib.sha256(p.read_bytes()).hexdigest()
 needed=subprocess.check_output(['patchelf','--print-needed',str(p)],text=True)
 old=subprocess.check_output(['patchelf','--print-rpath',str(p)],text=True).strip()
 subprocess.run(['patchelf','--set-rpath','$ORIGIN',str(p)],check=True)
 assert subprocess.check_output(['patchelf','--print-needed',str(p)],text=True)==needed
 record.append(dict(path=str(p.relative_to(root)),original_sha256=before,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),original_runpath=old,runpath='$ORIGIN',needed=needed.splitlines()))
pathlib.Path('/record/elf.json').write_text(json.dumps(dict(tool_version=subprocess.check_output(['patchelf','--version'],text=True).strip(),tool_sha256=hashlib.sha256(pathlib.Path('/usr/local/bin/patchelf').read_bytes()).hexdigest(),files=record),indent=2)+'\\n')
"""
        image = registration["runtime_image"]
        with (task / "patchelf.log").open("w") as log:
            subprocess.run(
                [
                    "podman",
                    "run",
                    "--rm",
                    "--network=none",
                    "--read-only",
                    "--cpus=2",
                    "--memory=1g",
                    "-v",
                    str(pkg) + ":/package:rw",
                    "-v",
                    str(task) + ":/record:rw",
                    image,
                    "python",
                    "-c",
                    code,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        sources = info / "sources"
        sources.mkdir()
        for source, name in [
            (
                WS / "tmp/onnxruntime-b069/inputs/onnxruntime-source.tar.gz",
                "ort-upstream-source.tar.gz",
            ),
            (
                WS / "tmp/onnxruntime-b0610/incoming/source-repair.tar.gz",
                "ort-repair-files.tar.gz",
            ),
            (WS / "helpers/onnxruntime_b0610_source_repair.json", "ort-repair.json"),
            (WS / "helpers/build_onnxruntime_b0610.sh", "build.sh"),
            (WS / "helpers/onnxruntime_b0610_compiler.cmake", "compiler.cmake"),
        ]:
            shutil.copy2(source, sources / name)
        deps = json.loads(
            (WS / "tmp/onnxruntime-b0610/mirror/manifest.json").read_text()
        )
        (sources / "dependencies").mkdir()
        for name, spec in deps["dependencies"].items():
            path = WS / "tmp/onnxruntime-b0610/mirror" / spec["relative_path"]
            assert sha(path) == spec["sha256"]
            packaged_name = name + "".join(path.suffixes)
            shutil.copy2(path, sources / "dependencies" / packaged_name)
            spec["packaged_archive"] = "dependencies/" + packaged_name
        atomic_json(sources / "dependencies.json", deps)
        shutil.copy2(pkg / "python/onnxruntime/LICENSE", pkg / "LICENSE")
        shutil.copy2(
            pkg / "python/onnxruntime/ThirdPartyNotices.txt",
            pkg / "ThirdPartyNotices.txt",
        )
        (pkg / "README.md").write_text(
            "Explicit CPython 3.12 gfx1013 ORT/MIGraphX inference package.\nRequires qualified BC-250 glibc/ROCm image and registered B06.8 MIGraphX SDK.\nFP32 static IR10/opset18 fixture scope only; no autograd/all-model/dtype/performance claim.\nUse bin/bc250-ort.py through the qualified SDK launcher; ordinary Torch defaults are unchanged.\nSource/dependency archives and licenses are retained under share/bc250-ort/sources.\n"
        )
        atomic_json(
            info / "manifest.json",
            dict(
                name=NAME,
                version="1.31.0+gfx1013.b0611",
                source_revision="b81a7866e5a6eff964b9e411535b4c9198651a9f",
                runtime_image=image,
                runtime_sha256=registration["runtime_sha256"],
                sdk_registration=registration,
                python="CPython 3.12",
                architecture="gfx1013",
                dtype_qualified=["FP32"],
                autograd=False,
                automatic_torch_dispatch=False,
                source_sha256={
                    str(p.relative_to(sources)): sha(p)
                    for p in sources.rglob("*")
                    if p.is_file()
                },
                elf_normalization=json.loads((task / "elf.json").read_text()),
            ),
        )
        files, links = {}, {}
        for path in pkg.rglob("*"):
            n = str(path.relative_to(pkg))
            if path.is_symlink():
                assert (
                    not Path(os.readlink(path)).is_absolute()
                    and path.resolve().is_relative_to(pkg.resolve())
                    and path.exists()
                )
                links[n] = os.readlink(path)
            elif path.is_file():
                files[n] = sha(path)
        atomic_json(info / "sha256.json", dict(files=files, symlinks=links))
        archive = task / (NAME + ".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(pkg, arcname=NAME)
        roots = []
        for label in ("relocated-A", "relocated B with spaces"):
            folder = task / label
            folder.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(folder, filter="data")
            root = folder / NAME
            assert not verify_hashes({str(root / n): d for n, d in files.items()})
            assert all(os.readlink(root / n) == v for n, v in links.items())
            roots.append(str(root))
        assert not verify_hashes(baseline["protected_sha256"])
        report = dict(
            status="packaged-not-qualified",
            archive=str(archive),
            archive_sha256=sha(archive),
            archive_bytes=archive.stat().st_size,
            roots=roots,
            files=len(files) + 1,
            symlinks=len(links),
            index_sha256=sha(info / "sha256.json"),
            manifest_sha256=sha(info / "manifest.json"),
            runtime_changed=False,
            host105_contacted=False,
        )
        atomic_json(task / "final.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
