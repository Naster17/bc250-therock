#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export/replay tracked Torch sources without inherited Git objects or history."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

BASE = "c779020750d7dc99a39ea8c104ef8df1fc9d0755"
NATIVE = "a5249802372882dedccb91d3fd4d73be9fcdf2bd"
HEAD = "8d8b18df558cb478727ba2179f2e01dfce20ea40"
TREES = {
    "baseline": "abdd861bee54ca83fd7ed9a742dd9db1b7536831",
    "native": "3e3edf7f1de1fd1fe4730f9ec4ed30dd038b2112",
    "source_head": "33a862732a52c8825a3d1f9c0b609d06fba0df65",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_environment():
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_NO_REPLACE_OBJECTS="1",
    )
    return environment


def git(repo, *args, input=None):
    return (
        subprocess.check_output(
            ["git", "-C", str(repo), *args], input=input, env=git_environment()
        )
        .decode()
        .strip()
    )


def entries(repo, revision):
    values = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-rz", revision], env=git_environment()
    )
    result = []
    for item in values.split(b"\0"):
        if not item:
            continue
        metadata, path = item.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        result.append(dict(path=path.decode(), mode=mode, kind=kind, oid=oid))
    return result


def export(repo, destination, patches):
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "baseline.tar.gz"
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=tar.gz",
            "--output=" + str(archive.resolve()),
            BASE,
        ],
        check=True,
        env=git_environment(),
    )
    files = entries(repo, BASE)
    gitlinks = [f for f in files if f["mode"] == "160000"]
    dependencies = []
    for item in gitlinks:
        dependency = repo / item["path"]
        record = dict(item, local_pin_available=False, license_files=[])
        try:
            top = Path(git(dependency, "rev-parse", "--show-toplevel")).resolve()
            if top == dependency.resolve():
                git(dependency, "cat-file", "-e", item["oid"] + "^{commit}")
                record["local_pin_available"] = True
                record["local_head"] = git(dependency, "rev-parse", "HEAD")
                record["license_files"] = [
                    dict(path=f["path"], oid=f["oid"])
                    for f in entries(dependency, item["oid"])
                    if Path(f["path"])
                    .name.upper()
                    .startswith(("LICENSE", "COPYING", "NOTICE"))
                ]
        except subprocess.CalledProcessError:
            pass
        dependencies.append(record)
    names = ["0001-qualified-native-torch.patch", "0002-agent-exit-lifetime.patch"]
    for name in names:
        shutil.copy2(patches / name, destination / name)
    record = dict(
        format=1,
        baseline=BASE,
        native=NATIVE,
        source_head=HEAD,
        baseline_tree=git(repo, "rev-parse", BASE + "^{tree}"),
        native_tree=git(repo, "rev-parse", NATIVE + "^{tree}"),
        source_head_tree=git(repo, "rev-parse", HEAD + "^{tree}"),
        baseline_entries=len(files),
        baseline_blob_files=sum(f["kind"] == "blob" for f in files),
        gitlinks=gitlinks,
        dependencies=dependencies,
        main_license_oid=git(repo, "rev-parse", BASE + ":LICENSE"),
        payloads={
            name: sha(destination / name) for name in ["baseline.tar.gz", *names]
        },
        scope="Complete tracked parent source and gitlink identities; dependency contents/toolchain/wheels not bundled",
        source_git_shallow=git(repo, "rev-parse", "--is-shallow-repository"),
    )
    (destination / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def replay(package, destination):
    record = json.loads((package / "manifest.json").read_text())
    assert (
        record["format"],
        record["baseline"],
        record["native"],
        record["source_head"],
    ) == (1, BASE, NATIVE, HEAD)
    assert all(record[name + "_tree"] == tree for name, tree in TREES.items())
    for name, expected in record["payloads"].items():
        assert name in (
            "baseline.tar.gz",
            "0001-qualified-native-torch.patch",
            "0002-agent-exit-lifetime.patch",
        )
        assert sha(package / name) == expected, name
    assert set(record["payloads"]) == {
        "baseline.tar.gz",
        "0001-qualified-native-torch.patch",
        "0002-agent-exit-lifetime.patch",
    }
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(package / "baseline.tar.gz") as archive:
        assert not any(".git" in Path(m.name).parts for m in archive)
        archive.extractall(destination, filter="data")
    git(destination, "init", "--quiet", "--object-format=sha1")
    git(destination, "add", "--force", "--all")
    for item in record["gitlinks"]:
        assert item["mode"] == "160000"
        git(
            destination,
            "update-index",
            "--add",
            "--cacheinfo",
            "160000," + item["oid"] + "," + item["path"],
        )
    baseline_tree = git(destination, "write-tree")
    assert baseline_tree == record["baseline_tree"], (
        baseline_tree,
        record["baseline_tree"],
    )
    actual_entries = entries(destination, baseline_tree)
    assert len(actual_entries) == record["baseline_entries"]
    assert (
        git(destination, "rev-parse", baseline_tree + ":LICENSE")
        == record["main_license_oid"]
    )
    trees = dict(baseline=baseline_tree)
    for patch, name in [
        ("0001-qualified-native-torch.patch", "native"),
        ("0002-agent-exit-lifetime.patch", "source_head"),
    ]:
        git(
            destination,
            "apply",
            "--check",
            "--index",
            "--whitespace=error",
            str((package / patch).resolve()),
        )
        git(
            destination,
            "apply",
            "--index",
            "--whitespace=error",
            str((package / patch).resolve()),
        )
        trees[name] = git(destination, "write-tree")
        assert trees[name] == record[name + "_tree"]
    assert not (destination / ".git/objects/info/alternates").exists()
    assert not (destination / ".git/shallow").exists()
    return dict(
        status="passed",
        trees=trees,
        baseline_entries=len(actual_entries),
        baseline_blob_files=record["baseline_blob_files"],
        gitlink_pins=len(record["gitlinks"]),
        license_blob=record["main_license_oid"],
        object_alternates=False,
        source_history_required=False,
        dependency_contents_bundled=False,
        native_build=False,
        payloads=record["payloads"],
    )


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("export")
    create.add_argument("--repo", type=Path, default=root / "sources/bc250-pytorch")
    create.add_argument(
        "--patches", type=Path, default=root / "patches/bc250-pytorch-complete"
    )
    create.add_argument("--destination", type=Path, required=True)
    check = sub.add_parser("replay")
    check.add_argument("--package", type=Path, required=True)
    check.add_argument("--destination", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "export":
        result = export(args.repo.resolve(), args.destination, args.patches)
        print(
            json.dumps(
                dict(
                    status="exported",
                    blob_files=result["baseline_blob_files"],
                    gitlinks=len(result["gitlinks"]),
                )
            )
        )
    else:
        result = replay(args.package, args.destination)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
