#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage a versioned SDK-only cohort without changing any active payload."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from board_preflight import reservation

WORK = Path("/home/nik/bc250work")
BASE = Path("/opt/bc250-stack")
OLD = BASE / "candidates/miopen-plan02-r4"
ROOT = BASE / "candidates/sdk-d49-r1"
OLD_SHA = "a6dc1c884c4f446e23a2afc8185b26ef9fd55b57cdd1d15324d46054b902b9c2"
NEW_SHA = "2a975370ba4eb1c2fe39e406ff36b175f554a49d0aa19febbd326f641fcb590b"
SDK = "lib/librocprofiler-sdk.so.1.4.2"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def identity_tree(root):
    records = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            value = ["symlink", os.readlink(path)]
        elif path.is_file():
            st = path.stat()
            value = [
                "file",
                st.st_dev,
                st.st_ino,
                st.st_size,
                st.st_mtime_ns,
                st.st_mode,
            ]
        else:
            value = ["directory"]
        records[str(path.relative_to(root))] = value
    return records


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    with reservation("d49-stage"):
        assert BASE.joinpath("current").resolve() == OLD / "profiles/daily"
        assert shutil.disk_usage(BASE).free > 6 * 1024**3
        qualified = json.loads(
            (WORK / "evidence/sdk-d48-tests-r3/result.json").read_text()
        )
        assert (
            qualified["status"] == "passed"
            and qualified["guard"]["status"] == "acknowledged"
        )
        source = WORK / "evidence/sdk-d48-build-r2/librocprofiler-sdk.so.1.4.2"
        assert sha(source) == NEW_SHA and sha(OLD / "rocm" / SDK) == OLD_SHA
        before = identity_tree(OLD / "rocm")
        ROOT.mkdir(exist_ok=False)
        subprocess.run(
            ["cp", "-al", str(OLD / "rocm"), str(ROOT / "rocm")],
            check=True,
            timeout=120,
        )
        target = ROOT / "rocm" / SDK
        assert not target.is_symlink()
        pending = target.with_suffix(".d49-new")
        shutil.copy2(source, pending)
        pending.replace(target)  # detach this inode; never write a shared hardlink
        after = identity_tree(ROOT / "rocm")
        assert set(before) == set(after)
        assert [name for name in before if before[name] != after[name]] == [SDK]
        assert identity_tree(OLD / "rocm") == before
        assert sha(OLD / "rocm" / SDK) == OLD_SHA
        profiles = {}
        override = dict(
            schema=1,
            component="rocprofiler-sdk",
            source_revision="143f1d878296176c738274202092bb247b7b771c",
            library_sha256=NEW_SHA,
            replaced_sha256=OLD_SHA,
            qualification="D48",
            scope="Exact -1 visibility terminator; CI checks retained; no other native component rebuild",
        )
        for name in ("daily", "rccl-single"):
            live = OLD / "profiles" / name
            dest = ROOT / "profiles" / name
            dest.mkdir(parents=True)
            for item in live.iterdir():
                if item.name == "runtime.json":
                    shutil.copy2(item, dest / item.name)
                elif item.name == ".runtime.lock":
                    os.link(item.resolve(), dest / item.name)
                else:
                    (dest / item.name).symlink_to(
                        ROOT / "rocm" if item.name == "rocm" else item.resolve()
                    )
            write(dest / "native-overrides.json", override)
            profiles[name] = sha(dest / "runtime.json")
        shutil.copytree(OLD / "registries", ROOT / "registries")
        tools = ROOT / "tools"
        tools.mkdir()
        for item in (OLD / "tools").glob("*.py"):
            source_text = (
                (WORK / "tools/sdk-d49/bc250_torch.py").read_text()
                if item.name == "bc250_torch.py"
                else item.read_text()
            )
            text = source_text.replace(str(OLD), str(ROOT))
            if item.name == "bc250_torch.py":
                text = text.replace(
                    "/opt/bc250-stack/current", str(ROOT / "profiles/daily")
                ).replace(
                    "/opt/bc250-stack/profiles/rccl-single",
                    str(ROOT / "profiles/rccl-single"),
                )
            (tools / item.name).write_text(text)
            (tools / item.name).chmod(0o755)
        write(
            ROOT / "preservation.json",
            dict(
                old=before,
                new=after,
                changed_paths=[SDK],
                method="Unchanged files share original inodes; only SDK inode atomically detached; no byte-level rehash of other files",
            ),
        )
        write(
            ROOT / "stage.json",
            dict(
                status="staged-inactive",
                base=str(OLD),
                profiles=profiles,
                native_override=override,
                tools={str(p): sha(p) for p in tools.glob("*.py")},
                registries={
                    str(p): sha(p) for p in (ROOT / "registries").glob("*.json")
                },
                runtime_metadata="runtime.json retains exact baseline identity for existing consumer packages; native-overrides.json records SDK-only delta explicitly",
            ),
        )
        print(ROOT / "stage.json")


if __name__ == "__main__":
    main()
