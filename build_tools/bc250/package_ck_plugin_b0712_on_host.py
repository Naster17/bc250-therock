#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Package the qualified FP16 facade and resource-repaired MIOpen, then extract twice."""
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

NAME = "bc250-miopen-dl-gfx1013-b0712"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", choices=("r1", "r2", "r3"), required=True)
    args = p.parse_args()
    preflight("ck-plugin-b0712-package-" + args.attempt)
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 6 * 1024**3
    base = WS / "tmp/ck-plugin-b0712"
    out = base / ("package-" + args.attempt)
    out.mkdir()
    old = json.loads(
        (WS / "logs/ck-plugin-b0711-qualification-r4/final.json").read_text()
    )
    assert old["status"] == "passed"
    protected = dict(old["protected_sha256"])
    for p in (WS / "helpers").glob("*b0712*"):
        if p.is_file():
            protected[str(p)] = sha(p)
    for p in (base / "resources").iterdir():
        protected[str(p)] = sha(p)
    before = subprocess.check_output(["dmesg"])
    r = dict(
        status="running",
        protected_sha256=protected,
        links=old["links"],
        aot_caches=old["aot_caches"],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        runtime_changed=False,
        host105_contacted=False,
    )
    root = out / NAME
    try:
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            assert not verify_hashes(protected)
            for folder in (
                "lib",
                "runtime",
                "bin",
                "probes",
                "sources",
                "licenses",
                "examples",
                "validation",
            ):
                (root / folder).mkdir(parents=True)

            def copy(source, target):
                source = Path(source)
                protected[str(source)] = sha(source)
                shutil.copy2(source, root / target)

            copy(
                WS / "tmp/ck-plugin-b0710/build-r2/libMIOpenCKGroupedConv_gfx1013.so",
                "lib/libMIOpenCKGroupedConv_gfx1013.so",
            )
            copy(
                WS
                / "tmp/ck-plugin-b0711/native-fix-r2/miopen-build/lib/libMIOpen.so.1.0",
                "runtime/libMIOpen.so.1",
            )
            for kind, path in [
                ("b072", "b073/build-r2"),
                ("b073", "b0710/build-r2"),
                ("b0710_public", "b0710/build-r2"),
                ("b0710_half", "b0710/build-r2"),
                ("b076", "b076/build-r1"),
                ("b0711", "b0711/build-r3"),
            ]:
                copy(WS / f"tmp/ck-plugin-{path}/probe-{kind}", f"probes/probe-{kind}")
            copy(WS / "helpers/ck_plugin_b0712_entry.py", "bin/bc250-ck.py")
            copy(
                WS / "helpers/ck_plugin_b0712_references.py", "validation/references.py"
            )
            copy(WS / "helpers/ck_plugin_b0712_README.md", "README.md")
            for p in (base / "resources").iterdir():
                copy(p, ("licenses/" if "LICENSE" in p.name else "sources/") + p.name)
            # Preserve exact qualified probe/facade inputs and three resource-link recipes.
            shutil.copytree(
                WS / "tmp/ck-plugin-b0710/inputs-r2", root / "sources/facade-inputs"
            )
            for name in (
                "build_ck_plugin_b0710.sh",
                "build_ck_plugin_probe_b0710.sh",
                "build_miopen_b0711_fix.sh",
                "miopen_ck_b0711_probe.cpp",
                "miopen_ck_b076_probe.cpp",
                "miopen_ck_b072_probe.cpp",
            ):
                copy(WS / "helpers" / name, "sources/" + name)
            for name in ("target-recipes.json", "final.json"):
                copy(
                    WS / "tmp/ck-plugin-b0711/native-fix-r2" / name,
                    "sources/native-" + name,
                )
            copy(
                WS / "tmp/ck-plugin-b0711/native-inputs-r1/MIOpenConvDirUniHip.cpp",
                "sources/MIOpenConvDirUniHip.cpp",
            )
            copy(
                WS / "tmp/ck-plugin-b0710/build-r2/overlay/include/miopen/config.h",
                "sources/private-config.h",
            )
            copy(
                WS / "build/ml-libs/MIOpen/build/include/miopen/config.h",
                "sources/native-config.h",
            )
            for cohort in ("daily", "rccl"):
                dest = root / "examples" / cohort
                dest.mkdir()
                fp32 = WS / f"logs/ck-plugin-b076-r1-private-{cohort}/selected-db"
                fp16 = WS / f"logs/ck-plugin-b0711-r4-private-{cohort}/selected-db"
                for first in sorted(fp32.glob("*.txt")):
                    second = fp16 / first.name
                    protected[str(first)] = sha(first)
                    protected[str(second)] = sha(second)
                    half_lines = [
                        x for x in second.read_text().splitlines() if x.strip()
                    ]
                    if first.name.endswith(".ufdb.txt"):
                        native_search = (
                            WS
                            / f"logs/ck-plugin-b0711-r4-private-{cohort}/native-search/stdout.log"
                        )
                        protected[str(native_search)] = sha(native_search)
                        rows = json.loads(
                            native_search.read_text().strip().splitlines()[-1]
                        )["rows"]
                        keys = set()
                        for row in rows:
                            g, f = row["groups"], row["filter"]
                            key = f"{g*row['c_per_group']}-{row['h']}-{row['w']}-{f}x{f}-{g*row['k_per_group']}-{row['h']}-{row['w']}-{row['n']}-{f//2}x{f//2}-1x1-1x1-0-NCHW-FP16-F"
                            keys.add(key + (f"_g{g}" if g != 1 else ""))
                        half_lines = [
                            x for x in half_lines if x.split("=", 1)[0] in keys
                        ]
                        assert (
                            len(keys) == 6
                            and {x.split("=", 1)[0] for x in half_lines} == keys
                        )
                    lines = [
                        x for x in first.read_text().splitlines() if x.strip()
                    ] + half_lines
                    assert (
                        len(lines) == 8
                        and len({x.split("=", 1)[0] for x in lines}) == 8
                    )
                    assert (
                        sum("FP16" in x for x in lines) == 6
                        and sum("FP32" in x for x in lines) == 2
                    )
                    (dest / first.name).write_text("\n".join(sorted(lines)) + "\n")
            runtime = json.loads(
                Path("/opt/bc250-stack/current/runtime.json").read_text()
            )
            tool = runtime["image"]
            code = r"""
import hashlib,json,pathlib,subprocess
root=pathlib.Path('/package');rows=[]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for p in [root/'runtime/libMIOpen.so.1',root/'lib/libMIOpenCKGroupedConv_gfx1013.so',*sorted((root/'probes').iterdir())]:
 before=sha(p);needed=subprocess.check_output(['patchelf','--print-needed',str(p)],text=True)
 old=subprocess.check_output(['patchelf','--print-rpath',str(p)],text=True).strip()
 runpath='$ORIGIN' if p.parent.name=='runtime' else '$ORIGIN/../runtime'
 subprocess.run(['patchelf','--set-rpath',runpath,str(p)],check=True)
 assert subprocess.check_output(['patchelf','--print-needed',str(p)],text=True)==needed
 rows.append(dict(path=str(p.relative_to(root)),original_sha256=before,sha256=sha(p),original_runpath=old,runpath=runpath,needed=needed.splitlines()))
pathlib.Path('/record/elf.json').write_text(json.dumps(dict(tool_version=subprocess.check_output(['patchelf','--version'],text=True).strip(),tool_sha256=sha(pathlib.Path('/usr/local/bin/patchelf')),files=rows),indent=2)+'\n')
"""
            with (out / "patchelf.log").open("w") as log:
                subprocess.run(
                    [
                        "podman",
                        "run",
                        "--rm",
                        "--name",
                        "bc250-ck-b0712-elf-" + args.attempt,
                        "--network=none",
                        "--read-only",
                        "--cpus=2",
                        "--memory=2g",
                        "-v",
                        str(root) + ":/package:rw",
                        "-v",
                        str(out) + ":/record:rw",
                        tool,
                        "python",
                        "-c",
                        code,
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                )
            copy(out / "elf.json", "sources/elf.json")
            code = "import importlib.util,json;from pathlib import Path;s=importlib.util.spec_from_file_location('entry','/package/bin/bc250-ck.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);Path('/record/runtime-identity.json').write_text(json.dumps(m.runtime_identity(),indent=2)+'\\n')"
            subprocess.run(
                [
                    "podman",
                    "run",
                    "--rm",
                    "--name",
                    "bc250-ck-b0712-identity-" + args.attempt,
                    "--network=none",
                    "--read-only",
                    "--cpus=2",
                    "--memory=2g",
                    "-v",
                    str(root) + ":/package:ro",
                    "-v",
                    str(out) + ":/record:rw",
                    "-v",
                    "/opt/bc250-stack/current/rocm:/workspace/build/dist/rocm:ro",
                    "-e",
                    "LD_LIBRARY_PATH=/package/runtime:/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/rocm_sysdeps/lib",
                    runtime["image"],
                    "python",
                    "-c",
                    code,
                ],
                check=True,
                timeout=120,
            )
            native = json.loads((out / "runtime-identity.json").read_text())
            assert native["libMIOpen.so.1"]["sha256"] == sha(
                root / "runtime/libMIOpen.so.1"
            )
            files = {
                str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()
            }
            audit = json.loads(
                (WS / "logs/ck-plugin-b071-source-audit.json").read_text()
            )
            atomic_json(
                root / "manifest.json",
                dict(
                    format=2,
                    name=NAME,
                    architecture="gfx1013:xnack-",
                    api=5,
                    files=files,
                    runtime_libraries=native,
                    required_symbols=audit["required_symbols"],
                    source_revision="96e5a7dfc43356b1c9789a84ae9d79d41bbb0ad7",
                    plugin_source_revision="742ee1e2c0e008d73b7dccc819e6bb8defa9e7ad",
                    installed_libraries_revision="8389197c134a7337932021ab7137987a585664ae",
                    scope="FP32/FP16 packed2D forward alpha1/beta0; gfx1013 HIP-direct forward FP32 accumulation; native backward retained",
                    sdk_requirements=dict(
                        ck_instance_archive_sha256=sha(
                            WS
                            / "build/math-libs/composable_kernel/stage/lib/libdevice_conv_operations.a"
                        )
                    ),
                    example_records_per_profile=dict(FP32=2, FP16=6),
                ),
            )
            for p in root.rglob("*"):
                if p.is_file():
                    p.chmod(0o555 if p.parent.name in ("bin", "probes") else 0o444)
            archive = out / (NAME + ".tar.gz")
            with tarfile.open(archive, "w:gz") as t:
                t.add(root, arcname=NAME)
            roots = []
            for label in ("extract-a", "extract with spaces"):
                dest = base / (args.attempt + "-" + label)
                dest.mkdir()
                with tarfile.open(archive) as t:
                    t.extractall(dest, filter="data")
                extracted = dest / NAME
                assert all(sha(extracted / n) == v for n, v in files.items())
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
                runtime_identity=native,
            )
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        r.update(
            changed_hashes=verify_hashes(protected),
            links_unchanged=all(os.readlink(p) == v for p, v in old["links"].items()),
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
    print("CK_PLUGIN_B0712_PACKAGE_PASS", flush=True)


if __name__ == "__main__":
    main()
