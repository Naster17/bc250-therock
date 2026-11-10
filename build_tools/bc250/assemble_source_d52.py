#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Assemble and verify fresh selected Torch sources from portable D47/D50/D51 kits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from portable_torch_d47 import replay as replay_parent, git, entries, HEAD, TREES
from portable_dependencies_d50 import replay as replay_dependencies
from verify_build_inputs_d51 import verify as verify_inputs


def blob_id(path, mode):
    value = hashlib.sha1()
    if mode == "120000":
        data = os.fsencode(os.readlink(path))
        value.update(b"blob " + str(len(data)).encode() + b"\0" + data)
    else:
        assert path.is_file() and not path.is_symlink(), path
        assert bool(path.stat().st_mode & 0o100) == (mode == "100755"), path
        value.update(b"blob " + str(path.stat().st_size).encode() + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
    return value.hexdigest()


def verify_materialized(root, expected, reference=None, raw_expected=None):
    links = {item["path"] for item in expected if item["mode"] == "160000"}
    want = {item["path"] for item in expected}
    actual = set()
    for current, dirs, files in os.walk(root, followlinks=False):
        relative = Path(current).relative_to(root)
        for name in list(dirs):
            path = relative / name
            key = path.as_posix()
            if name == ".git":
                assert relative == Path("."), path
                dirs.remove(name)
            elif key in links or (root / path).is_symlink():
                actual.add(key)
                dirs.remove(name)
        actual.update((relative / name).as_posix() for name in files)
    assert actual == want, (root, sorted(actual - want)[:5], sorted(want - actual)[:5])
    for item in expected:
        path = root / item["path"]
        if item["mode"] == "160000":
            assert path.is_dir() and not path.is_symlink(), path
        else:
            wanted = (
                blob_id(reference / item["path"], item["mode"])
                if reference is not None
                else (
                    raw_expected[item["path"]]
                    if raw_expected is not None
                    else item["oid"]
                )
            )
            assert blob_id(path, item["mode"]) == wanted, path
    return sum(item["kind"] == "blob" for item in expected)


def fresh_recipe(text, rccl=False):
    assert "python -m pip install -e . -v --no-build-isolation" in text
    lines = [
        line
        for line in text.splitlines()
        if not line.startswith("cp build/CMakeCache.txt ")
    ]
    assert len(lines) == len(text.splitlines()) - 1
    content = "\n".join(lines) + "\n"
    guard = "test ! -e /workspace/pytorch-work/build/CMakeCache.txt\n"
    if rccl:
        content = content.replace(
            "/home/nik/bc250work/rocm-10.2-gfx1013/rccl-b034/install-r1",
            "/workspace/rccl",
        )
        guard += "test -d /workspace/rccl/lib/cmake/rccl\n"
    return content.replace("set -euo pipefail\n", "set -euo pipefail\n" + guard, 1)


def assemble(parent, dependencies, inputs, destination):
    assert not destination.exists()
    destination.mkdir(parents=True)
    parent_result = replay_parent(parent, destination / "torch")
    parent_entries = entries(destination / "torch", TREES["source_head"])
    parent_raw = {
        item["path"]: blob_id(destination / "torch" / item["path"], item["mode"])
        for item in parent_entries
        if item["kind"] == "blob"
    }
    dependency_result = replay_dependencies(
        dependencies, destination / "dependency-proof"
    )
    inputs_result = verify_inputs(inputs)
    manifest = json.loads((dependencies / "manifest.json").read_text())
    nodes = sorted(manifest["nodes"], key=lambda item: (item["depth"], item["path"]))
    copied = []
    for node in nodes:
        target = destination / "torch" / node["path"]
        if target.exists():
            assert target.is_dir() and not any(target.iterdir()), target
        if node["available"]:
            proof = destination / "dependency-proof" / node["pin"]
            shutil.copytree(
                proof,
                target,
                symlinks=True,
                ignore=shutil.ignore_patterns(".git"),
                dirs_exist_ok=True,
            )
            copied.append(node["path"])
        else:
            target.mkdir(parents=True, exist_ok=True)
    checks = []
    for node in nodes:
        if node["available"]:
            proof = destination / "dependency-proof" / node["pin"]
            tree = git(proof, "write-tree")
            count = verify_materialized(
                destination / "torch" / node["path"],
                entries(proof, tree),
                reference=proof,
            )
            checks.append(
                dict(
                    path=node["path"],
                    pin=node["pin"],
                    tree=tree,
                    blob_files=count,
                    overlay=node["path"] == "third_party/gloo",
                )
            )
    tree = git(destination / "torch", "write-tree")
    assert tree == TREES["source_head"]
    parent_blobs = verify_materialized(
        destination / "torch", parent_entries, raw_expected=parent_raw
    )
    assert len(copied) == 69 and len(checks) == 69
    recipes = destination / "recipes"
    recipes.mkdir()
    for profile in ("daily", "rccl-single"):
        source = inputs / "evidence" / profile
        shutil.copytree(source, recipes / profile)
        old_recipe = source / (
            "build_torch_b052.sh" if profile == "daily" else "build_torch_b053.sh"
        )
        fresh = recipes / ("build-fresh-" + profile + ".sh")
        fresh.write_text(
            fresh_recipe(old_recipe.read_text(), rccl=profile == "rccl-single")
        )
        fresh.chmod(0o755)
        shutil.copy2(
            source / "torch_toolchain.cmake", recipes / "torch_toolchain.cmake"
        )
    result = dict(
        status="passed",
        source_head=HEAD,
        parent_tree=tree,
        parent_blob_files=parent_blobs,
        materialized_dependency_locations=len(checks),
        dependency_blob_files_with_duplicate_locations=sum(
            x["blob_files"] for x in checks
        ),
        checks=checks,
        parent_replay=parent_result,
        dependency_replay=dependency_result,
        build_input_replay=inputs_result,
        excluded_empty_paths=[n["path"] for n in nodes if not n["available"]],
        scope="Fresh materialized source and recorded recipes; verification Git indices remain unborn; no configure/codegen/native build or complete external SDK/toolchain closure",
    )
    (destination / "assembly.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(args.parent, args.dependencies, args.inputs, args.destination)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "parent_blob_files",
                    "materialized_dependency_locations",
                    "dependency_blob_files_with_duplicate_locations",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
