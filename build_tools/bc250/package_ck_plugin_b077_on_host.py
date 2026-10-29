#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Package unchanged qualified binaries with exact ABI pins and two relocation roots."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from bc250_torch import runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

NAME = "bc250-miopen-dl-gfx1013-b077"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", choices=("r1", "r2"), required=True)
    args = parser.parse_args()
    preflight("ck-plugin-b077-package-" + args.attempt)
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 4 * 1024**3
    base = WS / "tmp/ck-plugin-b077"
    out = base / ("package-" + args.attempt)
    out.mkdir()
    baseline = json.loads(
        (WS / "logs/ck-plugin-b076-qualification-r1/final.json").read_text()
    )
    protected = dict(baseline["protected_sha256"])
    resources = base / "resources"
    for p in resources.iterdir():
        if p.is_file():
            protected[str(p)] = sha(p)
    root = out / NAME
    before = subprocess.check_output(["dmesg"])
    r = dict(
        status="packaging",
        protected_sha256=protected,
        links=baseline["links"],
        aot_caches=baseline["aot_caches"],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        runtime_changed=False,
        host105_contacted=False,
    )
    atomic_json(out / "progress.json", r)
    try:
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            assert not verify_hashes(protected)
            for folder in ("lib", "bin", "probes", "sources", "licenses", "examples"):
                (root / folder).mkdir(parents=True)

            def copy(src, dst):
                src = Path(src)
                protected[str(src)] = sha(src)
                shutil.copy2(src, root / dst)

            copy(
                WS / "tmp/ck-plugin-b073/build-r2/libMIOpenCKGroupedConv_gfx1013.so",
                "lib/libMIOpenCKGroupedConv_gfx1013.so",
            )
            for kind in ("b072", "b073"):
                copy(
                    WS / f"tmp/ck-plugin-b073/build-r2/probe-{kind}",
                    f"probes/probe-{kind}",
                )
            copy(WS / "tmp/ck-plugin-b076/build-r1/probe-b076", "probes/probe-b076")
            copy(
                WS
                / (
                    "helpers/ck_plugin_b077_entry.py"
                    if args.attempt == "r1"
                    else "helpers/ck_plugin_b077_entry_r2.py"
                ),
                "bin/bc250-ck.py",
            )
            for source, dest in [
                ("plugin-sources.tar.gz", "sources/plugin-sources.tar.gz"),
                ("provenance.json", "sources/provenance.json"),
                ("LICENSE.md", "licenses/MIOpen-LICENSE.md"),
                ("CK-LICENSE", "licenses/CK-LICENSE"),
            ]:
                copy(resources / source, dest)
            shutil.copytree(
                WS / "tmp/ck-plugin-b073/inputs-r2", root / "sources/accepted-inputs"
            )
            copy(WS / "helpers/build_ck_plugin_b073.sh", "sources/build-plugin.sh")
            copy(
                WS / "tmp/ck-plugin-b073/build-r2/overlay/include/miopen/config.h",
                "sources/private-config.h",
            )
            copy(
                WS / "build/ml-libs/MIOpen/build/include/miopen/config.h",
                "sources/native-config.h",
            )
            for profile in ("daily", "rccl"):
                path = WS / f"logs/ck-plugin-b076-r1-private-{profile}/selected-db"
                (root / "examples" / profile).mkdir()
                for p in path.glob("*.txt"):
                    copy(p, f"examples/{profile}/" + p.name)
            audit = json.loads(
                (WS / "logs/ck-plugin-b071-source-audit.json").read_text()
            )
            native = json.loads((resources / "runtime-identity-r2.json").read_text())
            (root / "README.md").write_text(
                (
                    WS
                    / (
                        "helpers/ck_plugin_b077_README.md"
                        if args.attempt == "r1"
                        else "helpers/ck_plugin_b077_README_r2.md"
                    )
                ).read_text()
            )
            files = {
                str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()
            }
            pins = dict(
                format=1,
                name=NAME,
                architecture="gfx1013:xnack-",
                api=5,
                runtime_libraries=native,
                required_symbols=audit["required_symbols"],
                files=files,
                source_revision="3616b2f9f09314d87304c4e20c77592d0dea507f",
                installed_libraries_revision="8389197c134a7337932021ab7137987a585664ae",
                sdk_requirements=dict(
                    compiler="/workspace/build/dist/rocm/bin/amdclang++",
                    ck_instance_archive_sha256=sha(
                        WS
                        / "build/math-libs/composable_kernel/stage/lib/libdevice_conv_operations.a"
                    ),
                ),
                scope="FP32 packed2D forward, alpha1/beta0 only; no backward or production admission",
            )
            atomic_json(root / "manifest.json", pins)
            for p in root.rglob("*"):
                if p.is_file():
                    p.chmod(0o555 if p.parent.name in ("bin", "probes") else 0o444)
            archive = out / (NAME + ".tar.gz")
            with tarfile.open(archive, "w:gz") as t:
                t.add(root, arcname=NAME)
            roots = []
            for folder in ("extract-a", "extract with spaces"):
                target = base / (args.attempt + "-" + folder)
                target.mkdir()
                with tarfile.open(archive) as t:
                    t.extractall(target, filter="data")
                extracted = target / NAME
                assert all(sha(extracted / n) == h for n, h in files.items())
                assert (extracted / "manifest.json").read_bytes() == (
                    root / "manifest.json"
                ).read_bytes()
                roots.append(str(extracted))
            r.update(
                status="packaged-not-qualified",
                archive=str(archive),
                archive_sha256=sha(archive),
                archive_bytes=archive.stat().st_size,
                roots=roots,
                payload_files=len(files) + 1,
                payload_bytes=sum(
                    p.stat().st_size for p in root.rglob("*") if p.is_file()
                ),
            )
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        r.update(
            changed_hashes=verify_hashes(protected),
            links_unchanged=all(
                os.readlink(p) == v for p, v in baseline["links"].items()
            ),
            kernel_log_unchanged=before == subprocess.check_output(["dmesg"]),
        )
        if (
            r["changed_hashes"]
            or not r["links_unchanged"]
            or not r["kernel_log_unchanged"]
        ):
            r["status"] = "failed"
        atomic_json(out / "final.json", r)
    assert r["status"] == "packaged-not-qualified"
    print("CK_PLUGIN_B077_PACKAGE_PASS", flush=True)


if __name__ == "__main__":
    main()
