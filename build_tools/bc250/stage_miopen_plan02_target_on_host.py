#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage the qualified native-only precision cohort on the deployment host."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, "/home/nik/bc250work/tools/plan01")
from board_preflight import reservation
from stage_miopen_plan02_cohort_on_host import BASE, elf, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payload", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    root = a.root
    assert root.parent == BASE / "candidates" and not root.exists()
    source = json.loads((a.payload / "stage.json").read_text())
    assert source["cache_key_version"] == 1
    lib = a.payload / "libMIOpen.so.1.0"
    assert sha(lib) == source["library_sha256"]
    with reservation("plan02-stage-target"):
        Path("/home/nik/bc250work/rocm-10.2-gfx1013/logs").mkdir(
            parents=True, exist_ok=True
        )
        assert shutil.disk_usage(BASE).free > 3 * 1024**3
        root.mkdir(parents=True)
        old = BASE / "current/rocm/lib/libMIOpen.so.1.0"
        oldsha = sha(old)
        before, after = elf(old), elf(lib)
        import re

        public = lambda names: {
            n for n in names if not re.fullmatch(r"__hip_cuid_[0-9a-f]+", n)
        }
        assert before["dependencies"] == after["dependencies"]
        assert public(before["exports"]) == public(after["exports"])
        write(root / "elf.json", dict(installed=before, candidate=after))
        native = root / "rocm"
        subprocess.run(
            ["cp", "-al", str((BASE / "current/rocm").resolve()), str(native)],
            check=True,
            timeout=180,
        )
        dest = native / "lib/libMIOpen.so.1.0"
        assert not dest.is_symlink()
        temporary = dest.with_name(dest.name + ".new")
        shutil.copy2(lib, temporary)
        os.replace(temporary, dest)
        assert sha(old) == oldsha
        d01 = json.loads((BASE / "distributed/d01.json").read_text())
        profiles = {}
        for profile, live in [
            ("daily", BASE / "current"),
            ("rccl-single", Path(d01["release"])),
        ]:
            target = root / "profiles" / profile
            target.mkdir(parents=True)
            for entry in live.resolve().iterdir():
                if entry.name in {"runtime.json", ".runtime.lock"}:
                    continue
                (target / entry.name).symlink_to(
                    native if entry.name == "rocm" else entry.resolve()
                )
            os.link(BASE / "migration-control/.runtime.lock", target / ".runtime.lock")
            value = json.loads((live / "runtime.json").read_text())
            # Both the existing daily launcher and D09 core then hold the
            # recorded migration lock through workload exit. Payload stays frozen.
            value["mutable"] = True
            value["miopen_precision_repair"] = dict(
                source_revision=source["source_record"]["source_revision"],
                cache_key_version=1,
                library_sha256=sha(lib),
                scope="gfx1013 HIP-direct forward FP32 accumulation",
                qualification="PLAN-02 target; payload frozen, cooperative lock enabled",
            )
            write(target / "runtime.json", value)
            profiles[profile] = sha(target / "runtime.json")
        tools = root / "tools"
        tools.mkdir()
        text = (BASE / "tools/distributed-d09/bc250_torch.py").read_text()
        for oldpath, newpath in [
            (BASE / "current", root / "profiles/daily"),
            (BASE / "profiles/rccl-single", root / "profiles/rccl-single"),
        ]:
            text = text.replace(str(oldpath), str(newpath))
        core = tools / "bc250_torch.py"
        core.write_text(text)
        core.chmod(0o755)
        test_tools = root / "test-tools"
        test_tools.mkdir()
        podman = shutil.which("podman")
        assert podman
        wrapper = test_tools / "podman"
        wrapper.write_text(
            "#!/usr/bin/env python3\nimport os,sys\na=sys.argv[1:]\n"
            "if a[:1]==['run']:a[1:1]=['--cpus=4','--cpuset-cpus=0-3',"
            "'--memory=10g','--memory-swap=12g','--pids-limit=512']\n"
            f"os.execv({podman!r},[{podman!r},*a])\n"
        )
        wrapper.chmod(0o755)
        write(
            root / "stage.json",
            dict(
                status="staged-inactive-not-qualified",
                source_record=source["source_record"],
                source_cohort=source,
                library_sha256=sha(lib),
                original_library_sha256=oldsha,
                cache_key_version=1,
                profiles=profiles,
                tools={str(core): sha(core)},
                registries={},
                packages=[],
                native_only=True,
                native_runtime_changed=False,
            ),
        )


if __name__ == "__main__":
    main()
