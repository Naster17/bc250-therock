#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reversible SDK cohort selection under stable EX runtime and board locks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

from board_preflight import preflight
from stage_sdk_d49_on_host import (
    BASE,
    WORK,
    ROOT,
    OLD,
    SDK,
    NEW_SHA,
    OLD_SHA,
    sha,
    write,
)

HELPERS = [
    Path("/usr/local/bin/bc250-torch"),
    BASE / "tools/distributed-d09/bc250_torch.py",
]
REGISTRY = BASE / "distributed/d01.json"
POINTERS = {
    BASE / "current": ROOT / "profiles/daily",
    BASE / "profiles/rccl-single": ROOT / "profiles/rccl-single",
    BASE / "native-cohort": ROOT,
    **{
        Path("/usr/local/bin/bc250-" + name): ROOT / ("tools/bc250_" + name + ".py")
        for name in ("ck", "migraphx", "ort")
    },
}


def atomic(path, data, mode=0o644):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
        stream.write(data)
    pending.chmod(mode)
    pending.replace(path)


def link(path, target):
    pending = path.with_name(path.name + ".d49-new")
    assert not pending.exists() and not pending.is_symlink()
    pending.symlink_to(target)
    pending.replace(path)


def restore(before):
    for name, target in before["pointers"].items():
        link(Path(name), target)
    atomic(REGISTRY, Path(before["registry_backup"]).read_bytes())
    for item in before["helpers"]:
        atomic(Path(item["path"]), Path(item["backup"]).read_bytes(), item["mode"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["activate", "rollback"], required=True)
    parser.add_argument(
        "--label",
        required=True,
        choices=["activate-r1", "rollback-r1", "activate-r2", "failure-rollback"],
    )
    parser.add_argument("--from-transaction", type=Path)
    args = parser.parse_args()
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    folder = WORK / "evidence/sdk-d49-install" / args.label
    folder.mkdir(parents=True, exist_ok=False)
    stage = json.loads((ROOT / "stage.json").read_text())
    assert sha(ROOT / "rocm" / SDK) == NEW_SHA and sha(OLD / "rocm" / SDK) == OLD_SHA
    for name, digest in stage["profiles"].items():
        assert sha(ROOT / "profiles" / name / "runtime.json") == digest
    for group in ("tools", "registries"):
        for name, digest in stage[group].items():
            assert sha(Path(name)) == digest
    journal = dict(status="preparing", action=args.action, pid=os.getpid())
    with (WORK / "evidence/.board-reservation.lock").open("a") as board, (
        BASE / "current/.runtime.lock"
    ).open("r") as runtime:
        fcntl.flock(board, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(runtime, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (
            os.fstat(runtime.fileno()).st_ino
            == (ROOT / "profiles/daily/.runtime.lock").stat().st_ino
        )
        journal["preflight"] = preflight("d49-" + args.label)
        before = dict(
            pointers={str(p): os.readlink(p) for p in POINTERS},
            registry_backup=str(folder / "registry.before.json"),
            helpers=[],
        )
        shutil.copy2(REGISTRY, before["registry_backup"])
        for index, target in enumerate(HELPERS):
            assert not target.is_symlink()
            backup = folder / f"{index}-helper.before.py"
            shutil.copy2(target, backup)
            before["helpers"].append(
                dict(
                    path=str(target),
                    backup=str(backup),
                    mode=target.stat().st_mode & 0o777,
                    sha256=sha(target),
                )
            )
        write(folder / "before.json", before)
        try:
            if args.action == "rollback":
                restore(json.loads(args.from_transaction.read_text()))
            else:
                assert (BASE / "current").resolve() == OLD / "profiles/daily"
                record = json.loads(REGISTRY.read_text())
                assert record["runtime_sha256"] == stage["profiles"]["rccl-single"]
                record.update(
                    release=str(ROOT / "profiles/rccl-single"),
                    native_sdk_override=stage["native_override"],
                )
                atomic(REGISTRY, (json.dumps(record, indent=2) + "\n").encode())
                for path, target in POINTERS.items():
                    link(path, target)
                data = (WORK / "tools/sdk-d49/bc250_torch.py").read_bytes()
                for target in HELPERS:
                    atomic(target, data, 0o755)
            expected = NEW_SHA if args.action == "activate" else OLD_SHA
            assert sha(BASE / "current/rocm" / SDK) == expected
            for profile in ("daily", "rccl-single"):
                assert (
                    sha(
                        BASE
                        / ("current" if profile == "daily" else "profiles/rccl-single")
                        / "runtime.json"
                    )
                    == stage["profiles"][profile]
                )
            journal.update(
                status="passed",
                sdk_sha256=expected,
                active_profile=str((BASE / "current").resolve()),
                helper_sha256=sha(HELPERS[0]),
                distributed=json.loads(REGISTRY.read_text()),
                after=preflight("d49-" + args.label + "-after"),
            )
        except BaseException as error:
            restore(before)
            journal.update(status="restored-after-failure", error=repr(error))
            raise
        finally:
            write(folder / "result.json", journal)
    print(folder / "result.json")


if __name__ == "__main__":
    main()
