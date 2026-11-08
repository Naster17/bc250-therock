#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage an inactive identity-pinned native repair cohort; preserve all live files."""
import argparse
import hashlib
import re
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

sys.path.insert(0, "/home/nik/bc250work/tools/plan01")
from board_preflight import reservation

BASE = Path("/opt/bc250-stack")
ROOT = BASE / "candidates/miopen-plan02-r1"
WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(p, x):
    p.write_text(json.dumps(x, indent=2) + "\n")


def elf(p):
    text = subprocess.check_output(["readelf", "-d", str(p)], text=True)
    lines = [
        l.split("[", 1)[1].split("]", 1)[0]
        for l in text.splitlines()
        if "(NEEDED)" in l or "(SONAME)" in l
    ]
    exports = subprocess.check_output(
        ["readelf", "--dyn-syms", "--wide", str(p)], text=True
    )
    names = sorted(
        {
            l.split()[-1]
            for l in exports.splitlines()
            if len(l.split()) >= 8
            and l.split()[4] in ["GLOBAL", "WEAK"]
            and l.split()[6] != "UND"
        }
    )
    return dict(
        dependencies=lines,
        exports_sha256=hashlib.sha256("\n".join(names).encode()).hexdigest(),
        exports=names,
    )


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-record", type=Path, required=True)
    args = parser.parse_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.revision)
    ROOT = BASE / ("candidates/miopen-plan02-" + args.revision)
    with reservation("plan02-stage-cohort"):
        assert not ROOT.exists()
        assert shutil.disk_usage(BASE).free > 3 * 1024**3
        ROOT.mkdir(parents=True)
        old = BASE / "current/rocm/lib/libMIOpen.so.1.0"
        original_sha = sha(old)
        source_record = json.loads(args.source_record.read_text())
        assert source_record["status"] == "built-isolated-cache-version-not-qualified"
        assert source_record["cache_key_version"] == 1
        source = Path(source_record["library"])
        assert sha(source) == source_record["library_sha256"]
        lib = ROOT / "native/libMIOpen.so.1.0"
        lib.parent.mkdir()
        shutil.copy2(source, lib)
        before = elf(source)
        runtime = json.loads((BASE / "current/runtime.json").read_text())
        code = """from pathlib import Path
import subprocess
p=Path('/candidate/libMIOpen.so.1.0')
subprocess.run(['patchelf','--set-rpath','$ORIGIN/rocm_sysdeps/lib:$ORIGIN/llvm/lib:$ORIGIN/llvm/lib/x86_64-unknown-linux-gnu:$ORIGIN:$ORIGIN/host-math/lib',str(p)],check=True)
print(subprocess.check_output(['patchelf','--version'],text=True))
"""
        subprocess.run(
            [
                "podman",
                "run",
                "--rm",
                "--network=none",
                "--cpus=2",
                "--memory=2g",
                "--read-only",
                "-v",
                str(lib.parent) + ":/candidate:rw",
                runtime["image"],
                "python3",
                "-c",
                code,
            ],
            check=True,
            timeout=120,
        )
        after = elf(lib)
        original = elf(old)
        assert before == after
        assert original["dependencies"] == after["dependencies"]
        # Recompiling a HIP-header TU renames its compiler-generated content ID.
        # No public C/C++ export may disappear or change as a result.
        public = lambda names: {
            n for n in names if not re.fullmatch(r"__hip_cuid_[0-9a-f]+", n)
        }
        assert public(original["exports"]) == public(after["exports"])

        write(
            ROOT / "elf.json",
            dict(
                before=before,
                after=after,
                installed=original,
                source_sha256=sha(source),
                candidate_sha256=sha(lib),
                installed_sha256=sha(old),
            ),
        )
        native_rocm = ROOT / "rocm"
        subprocess.run(
            ["cp", "-al", str((BASE / "current/rocm").resolve()), str(native_rocm)],
            check=True,
            timeout=120,
        )
        target = native_rocm / "lib/libMIOpen.so.1.0"
        assert not target.is_symlink()
        temporary = target.with_name(target.name + ".new")
        shutil.copy2(lib, temporary)
        os.replace(temporary, target)  # detach only the changed inode
        assert sha(old) == original_sha
        profiles = {}
        for profile, live in [
            ("daily", BASE / "current"),
            ("rccl-single", BASE / "profiles/rccl-single"),
        ]:
            dest = ROOT / "profiles" / profile
            dest.mkdir(parents=True)
            for p in live.resolve().iterdir():
                if p.name in ["runtime.json", ".runtime.lock"]:
                    continue
                (dest / p.name).symlink_to(
                    native_rocm if p.name == "rocm" else p.resolve()
                )
            os.link((BASE / "current/.runtime.lock").resolve(), dest / ".runtime.lock")
            value = json.loads((live / "runtime.json").read_text())
            value["miopen_precision_repair"] = dict(
                source_revision=source_record["source_revision"],
                cache_key_version=source_record["cache_key_version"],
                library_sha256=sha(lib),
                scope="gfx1013 HIP-direct forward FP32 accumulation",
                qualification="PLAN-02",
            )
            write(dest / "runtime.json", value)
            profiles[profile] = sha(dest / "runtime.json")
        registries = ROOT / "registries"
        registries.mkdir()
        mx = json.loads((BASE / "sdks/migraphx.json").read_text())
        mx["runtime_sha256"] = profiles
        write(registries / "migraphx.json", mx)
        ck_old = json.loads((BASE / "sdks/miopen-dl.json").read_text())
        ck = Path(ck_old["package"])
        ck_new = BASE / ("sdks/bc250-miopen-dl-gfx1013-plan02-" + args.revision)
        shutil.copytree(ck, ck_new)
        cm = json.loads((ck_new / "manifest.json").read_text())
        cm["name"] = ck_new.name
        worker = ck_new / "bin/bc250-ck.py"
        worker.write_text(worker.read_text().replace(ck.name, ck_new.name))
        cm["files"]["bin/bc250-ck.py"] = sha(worker)
        cm["runtime_libraries"]["libMIOpen.so.1"]["sha256"] = sha(lib)
        cm["native_precision_repair"] = "PLAN-02; gfx1013 HIP-direct forward only"
        write(ck_new / "manifest.json", cm)
        archive = ROOT / "ck-plan02.tar.gz"
        with tarfile.open(archive, "w:gz") as t:
            t.add(ck_new, arcname=ck_new.name)
        cr = dict(
            ck_old,
            package=str(ck_new),
            archive_sha256=sha(archive),
            manifest_sha256=sha(ck_new / "manifest.json"),
            runtime_sha256=profiles,
        )
        write(registries / "miopen-dl.json", cr)
        ort_old = json.loads((BASE / "sdks/ort-migraphx.json").read_text())
        ort = Path(ort_old["package"])
        ort_new = BASE / (
            "sdks/bc250-ort-migraphx-1.31.0-gfx1013-cp312-plan02-" + args.revision
        )
        shutil.copytree(ort, ort_new, symlinks=True)  # independent regular-file copies
        om = json.loads((ort_new / "share/bc250-ort/manifest.json").read_text())
        om["sdk_registration"] = mx
        write(ort_new / "share/bc250-ort/manifest.json", om)
        oi = json.loads((ort_new / "share/bc250-ort/sha256.json").read_text())
        oi["files"]["share/bc250-ort/manifest.json"] = sha(
            ort_new / "share/bc250-ort/manifest.json"
        )
        write(ort_new / "share/bc250-ort/sha256.json", oi)
        ore = dict(
            ort_old,
            package=str(ort_new),
            sdk_registration=mx,
            index_sha256=sha(ort_new / "share/bc250-ort/sha256.json"),
            manifest_sha256=sha(ort_new / "share/bc250-ort/manifest.json"),
            archive_sha256="pending-inactive-staging",
        )
        write(registries / "ort-migraphx.json", ore)
        tools = ROOT / "tools"
        tools.mkdir()
        names = {
            "bc250_torch.py": BASE / "tools/ort-b0611/bc250_torch.py",
            "bc250_migraphx.py": Path("/usr/local/bin/bc250-migraphx").resolve(),
            "bc250_ck.py": Path("/usr/local/bin/bc250-ck").resolve(),
            "bc250_ort.py": Path("/usr/local/bin/bc250-ort").resolve(),
        }
        for name, source_tool in names.items():
            if name == "bc250_torch.py":
                source_tool = Path("/usr/local/bin/bc250-torch")
            text = source_tool.read_text()
            text = text.replace(
                "/opt/bc250-stack/current", str(ROOT / "profiles/daily")
            ).replace(
                "/opt/bc250-stack/profiles/rccl-single",
                str(ROOT / "profiles/rccl-single"),
            )
            for command, module in [
                ("bc250-torch", "bc250_torch.py"),
                ("bc250-migraphx", "bc250_migraphx.py"),
            ]:
                text = text.replace("/usr/local/bin/" + command, str(tools / module))
            if name == "bc250_migraphx.py":
                text = text.replace(
                    'REGISTRY = BASE / "migraphx.json"',
                    f'REGISTRY = Path({str(registries/"migraphx.json")!r})',
                )
            if name == "bc250_ck.py":
                text = text.replace(
                    'REGISTRY = BASE / "miopen-dl.json"',
                    f'REGISTRY = Path({str(registries/"miopen-dl.json")!r})',
                )
                text = (
                    text.replace(ck.name, ck_new.name)
                    .replace(ck_old["manifest_sha256"], cr["manifest_sha256"])
                    .replace(ck_old["archive_sha256"], cr["archive_sha256"])
                )
            if name == "bc250_ort.py":
                text = text.replace(
                    'REGISTRY = BASE / "ort-migraphx.json"',
                    f'REGISTRY = Path({str(registries/"ort-migraphx.json")!r})',
                ).replace(ort.name, ort_new.name)
            (tools / name).write_text(text)
            (tools / name).chmod(0o755)
        from rebase_miopen_plan02_cohort import finalize

        finalize(ROOT)
        test_tools = ROOT / "test-tools"
        test_tools.mkdir()
        podman = shutil.which("podman")
        assert podman
        (test_tools / "podman").write_text(
            "#!/usr/bin/env python3\nimport os,sys\na=sys.argv[1:]\nif a[:1]==['run']:a[1:1]=['--cpus=4','--cpuset-cpus=0-3','--memory=10g','--memory-swap=12g','--pids-limit=512']\nos.execv("
            + repr(podman)
            + ",["
            + repr(podman)
            + ",*a])\n"
        )
        (test_tools / "podman").chmod(0o755)
        write(
            ROOT / "stage.json",
            dict(
                status="staged-inactive-not-qualified",
                library=str(lib),
                cache_key_version=source_record["cache_key_version"],
                source_record=source_record,
                library_sha256=sha(lib),
                profiles=profiles,
                registries={str(p): sha(p) for p in registries.iterdir()},
                tools={str(p): sha(p) for p in tools.iterdir()},
                packages=[str(ck_new), str(ort_new)],
                native_runtime_changed=False,
            ),
        )
        print(ROOT / "stage.json", flush=True)


if __name__ == "__main__":
    main()
