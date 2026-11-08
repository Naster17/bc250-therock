#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rebuild only the direct solver object and shared library in a preserved build copy."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, "/home/nik/bc250work/tools/plan01")
from board_preflight import reservation

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    out = WS / "tmp/miopen-plan02-cache-r1"
    out.mkdir()
    source = out / "inputs/conv_hip_dir2Dfwd.cpp"
    source.parent.mkdir()
    shutil.copy2("/tmp/bc250-plan02-conv_hip_dir2Dfwd.cpp", source)
    old = WS / "tmp/ck-plugin-b0711/native-fix-r2/miopen-build"
    before = sha(old / "lib/libMIOpen.so.1.0")
    record = dict(
        status="running",
        pid=os.getpid(),
        source_sha256=sha(source),
        source_revision="e1d8d1ff920",
        cache_key_version=1,
        resource=dict(cpus=4, memory="10g", memory_swap="12g", pids=512),
        old_library_sha256=before,
    )
    try:
        with reservation("plan02-cache-build"):
            assert shutil.disk_usage(WS).free > 8 * 1024**3
            target = out / "miopen-build"
            subprocess.run(
                ["cp", "-a", "--reflink=auto", str(old), str(target)],
                check=True,
                timeout=180,
            )
            assert (target / "lib/libMIOpen.so.1.0").stat().st_ino != (
                old / "lib/libMIOpen.so.1.0"
            ).stat().st_ino
            code = r"""from pathlib import Path
import json,subprocess,os
root=Path('/workspace/build/ml-libs/MIOpen/build');os.chdir(root)
env=dict(os.environ,CCACHE_MAXSIZE='20G',OMP_NUM_THREADS='4')
setup=subprocess.check_output(['python3','/workspace/source/build_tools/setup_ccache.py','--dir','/workspace/cache','--local-path','/workspace/cache/data','--no-reset-stats'],text=True)
rows=[]
for target in ['src/CMakeFiles/MIOpen.dir/solver/conv/conv_hip_dir2Dfwd.cpp.o','lib/libMIOpen.so.1.0']:
 lines=subprocess.check_output(['ninja','-t','commands','-s',target],text=True).strip().splitlines();assert len(lines)==1
 rows.append(dict(target=target,command=lines[0]));Path('/workspace/output/recipes.json').write_text(json.dumps(rows,indent=2)+'\n')
 subprocess.run(['bash','-c',setup+'\n'+lines[0]],check=True,env=env,timeout=300)
subprocess.run(['ccache','--show-stats'],env=env,check=True)
"""
            image = json.loads(
                Path("/opt/bc250-stack/current/runtime.json").read_text()
            )["image"]
            cmd = [
                "podman",
                "run",
                "--rm",
                "--network=none",
                "--cpus=4",
                "--cpuset-cpus=0-3",
                "--memory=10g",
                "--memory-swap=12g",
                "--pids-limit=512",
            ]
            for src, dst, mode in [
                (WS / "source", "/workspace/source", "ro"),
                (WS / "build", "/workspace/build", "ro"),
                (WS / "cache", "/workspace/cache", "rw"),
                (out, "/workspace/output", "rw"),
                (target, "/workspace/build/ml-libs/MIOpen/build", "rw"),
                (
                    source,
                    "/workspace/source/rocm-libraries/projects/miopen/src/solver/conv/conv_hip_dir2Dfwd.cpp",
                    "ro",
                ),
            ]:
                cmd += ["-v", f"{src}:{dst}:{mode}"]
            # Use the retained ccache build image; it contains the matching compiler/toolchain.
            cmd += ["localhost/bc250-rocm-build:10.2-ccache", "python3", "-c", code]
            record.update(command=cmd, runtime_image=image)
            (out / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
            with (out / "build.log").open("w") as log:
                subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=660
                )
            lib = target / "lib/libMIOpen.so.1.0"
            assert sha(old / "lib/libMIOpen.so.1.0") == before
            record.update(
                status="built-isolated-cache-version-not-qualified",
                library=str(lib),
                library_sha256=sha(lib),
            )
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        (out / "final.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
