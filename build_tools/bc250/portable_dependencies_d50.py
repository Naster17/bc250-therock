#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export and independently replay available pinned recursive Torch dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from portable_torch_d47 import git, git_environment, entries, sha

PARENT_SHA = "8adfc0a3c057ee49a9c8fc695bb7d34bf6b717a606d7aa3233adbc3a3178d130"


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def object_id(kind, data):
    return hashlib.sha1(
        kind.encode() + b" " + str(len(data)).encode() + b"\0" + data
    ).hexdigest()


def safe_path(name):
    path = Path(name)
    assert (
        not path.is_absolute()
        and name
        and all(p not in ("..", ".git") for p in path.parts)
    ), name
    return path


def export(repo, parent, overlay, destination):
    assert sha(parent) == PARENT_SHA
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "objects").mkdir()
    shutil.copy2(parent, destination / "parent-manifest.json")
    shutil.copy2(overlay, destination / "gloo-legacy-platform.patch")
    nodes = []
    exported = {}

    def walk(relative, pin, depth):
        safe_path(relative)
        node = dict(path=relative, pin=pin, depth=depth, available=False)
        nodes.append(node)
        source = repo / relative
        if not source.exists():
            node["reason"] = "local checkout absent"
            return
        try:
            if (
                Path(git(source, "rev-parse", "--show-toplevel")).resolve()
                != source.resolve()
            ):
                node["reason"] = "local path is not an initialized independent checkout"
                return
            git(source, "cat-file", "-e", pin + "^{commit}")
        except subprocess.CalledProcessError:
            node["reason"] = "pinned commit object unavailable locally"
            return
        tree = git(source, "rev-parse", pin + "^{tree}")
        content = entries(source, pin)
        links = [v for v in content if v["kind"] == "commit"]
        node.update(available=True, tree=tree, gitlinks=links)
        if pin not in exported:
            archive = destination / "objects" / (pin + ".tar.gz")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "archive",
                    "--format=tar.gz",
                    "--output=" + str(archive.resolve()),
                    pin,
                ],
                check=True,
                env=git_environment(),
            )
            commit = subprocess.check_output(
                ["git", "-C", str(source), "cat-file", "commit", pin],
                env=git_environment(),
            )
            (destination / "objects" / (pin + ".commit")).write_bytes(commit)
            assert object_id("commit", commit) == pin
            licenses = [
                v
                for v in content
                if v["kind"] == "blob"
                and Path(v["path"])
                .name.upper()
                .startswith(("LICENSE", "COPYING", "NOTICE"))
            ]
            exported[pin] = dict(
                tree=tree,
                archive_sha256=sha(archive),
                commit_sha256=sha(destination / "objects" / (pin + ".commit")),
                blob_files=sum(v["kind"] == "blob" for v in content),
                license_files=licenses,
            )
        for child in links:
            walk(relative + "/" + child["path"], child["oid"], depth + 1)

    parent_record = json.loads(parent.read_text())
    for item in parent_record["gitlinks"]:
        walk(item["path"], item["oid"], 1)
    record = dict(
        format=1,
        parent_manifest_sha256=PARENT_SHA,
        nodes=nodes,
        objects=exported,
        gloo_overlay_sha256=sha(destination / "gloo-legacy-platform.patch"),
        complete_recursive_source_closure=all(n["available"] for n in nodes),
        scope="Available pinned dependency contents and license texts; explicit missing branches; no native build/toolchain/wheel or full offline-build claim",
    )
    write(destination / "manifest.json", record)
    return record


def exact_data_filter(member, destination):
    checked = tarfile.data_filter(member, destination)
    if checked is not None and member.issym():
        # Python's data filter normalizes trailing slash bytes in symlink
        # targets. Git records those bytes, so preserve this harmless spelling
        # only after data_filter has verified the same resolved target boundary.
        assert member.linkname.rstrip("/") == checked.linkname.rstrip("/")
        checked = checked.replace(linkname=member.linkname)
    return checked


def verify_tree(package, pin, record, destination):
    safe_path(pin)
    assert len(pin) == 40 and all(c in "0123456789abcdef" for c in pin)
    archive = package / "objects" / (pin + ".tar.gz")
    commit = (package / "objects" / (pin + ".commit")).read_bytes()
    assert object_id("commit", commit) == pin
    assert sha(archive) == record["archive_sha256"]
    assert hashlib.sha256(commit).hexdigest() == record["commit_sha256"]
    assert commit.splitlines()[0] == ("tree " + record["tree"]).encode()
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as content:
        for item in content:
            safe_path(item.name)
            assert item.isfile() or item.isdir() or item.issym(), item.name
        content.extractall(destination, filter=exact_data_filter)
    git(destination, "init", "--quiet", "--object-format=sha1")
    git(destination, "add", "--force", "--all")
    # Archives omit gitlink object contents. Recover their exact entries from
    # the export metadata, then require the commit-bound entire tree identity.
    for link in record["gitlinks"]:
        safe_path(link["path"])
        assert link["mode"] == "160000" and link["kind"] == "commit"
        git(
            destination,
            "update-index",
            "--add",
            "--cacheinfo",
            "160000," + link["oid"] + "," + link["path"],
        )
    assert git(destination, "write-tree") == record["tree"]
    actual = entries(destination, record["tree"])
    assert sum(v["kind"] == "blob" for v in actual) == record["blob_files"]
    actual_licenses = [
        v
        for v in actual
        if v["kind"] == "blob"
        and Path(v["path"]).name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
    ]
    assert actual_licenses == record["license_files"]
    return actual


def replay(package, destination):
    assert sha(package / "parent-manifest.json") == PARENT_SHA
    parent = json.loads((package / "parent-manifest.json").read_text())
    manifest = json.loads((package / "manifest.json").read_text())
    assert manifest["format"] == 1 and manifest["parent_manifest_sha256"] == PARENT_SHA
    destination.mkdir(parents=True, exist_ok=False)
    # Reconstruct the graph from the anchored parent's pins, not arbitrary
    # manifest path assertions. Missing branches are explicit leaf dispositions.
    by_path = {v["path"]: v for v in manifest["nodes"]}
    assert len(by_path) == len(manifest["nodes"])
    visited = set()
    verified = {}

    def walk(path, pin, depth):
        node = by_path[path]
        assert node["pin"] == pin and node["depth"] == depth and path not in visited
        safe_path(path)
        visited.add(path)
        if not node["available"]:
            assert node.get("reason")
            return
        obj = manifest["objects"][pin]
        assert node["tree"] == obj["tree"]
        if pin not in verified:
            obj = dict(obj, gitlinks=node["gitlinks"])
            actual = verify_tree(package, pin, obj, destination / pin)
            verified[pin] = actual
        links = [v for v in verified[pin] if v["kind"] == "commit"]
        assert node["gitlinks"] == links
        for child in links:
            walk(path + "/" + child["path"], child["oid"], depth + 1)

    for item in parent["gitlinks"]:
        walk(item["path"], item["oid"], 1)
    assert visited == set(by_path)
    assert set(verified) == set(manifest["objects"])
    complete = all(v["available"] for v in manifest["nodes"])
    assert manifest["complete_recursive_source_closure"] == complete
    patch = package / "gloo-legacy-platform.patch"
    assert sha(patch) == manifest["gloo_overlay_sha256"]
    gloo_pin = by_path["third_party/gloo"]["pin"]
    gloo = destination / gloo_pin
    git(gloo, "apply", "--check", "--index", str(patch.resolve()))
    git(gloo, "apply", "--index", str(patch.resolve()))
    overlay_tree = git(gloo, "write-tree")
    assert overlay_tree != by_path["third_party/gloo"]["tree"]
    (destination / "gloo-cuda-overlaid.cu").write_bytes(
        (gloo / "gloo/cuda.cu").read_bytes()
    )
    return dict(
        status="passed",
        nodes=len(by_path),
        available_nodes=sum(v["available"] for v in by_path.values()),
        unique_payloads=len(verified),
        blob_files=sum(v["blob_files"] for v in manifest["objects"].values()),
        license_files=sum(
            len(v["license_files"]) for v in manifest["objects"].values()
        ),
        missing=[
            dict(path=v["path"], pin=v["pin"], reason=v["reason"])
            for v in by_path.values()
            if not v["available"]
        ],
        complete_recursive_source_closure=complete,
        gloo_overlay_tree=overlay_tree,
        gloo_cuda_sha256=sha(gloo / "gloo/cuda.cu"),
        parent_manifest_sha256=PARENT_SHA,
        manifest_sha256=sha(package / "manifest.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--repo", type=Path, required=True)
    exp.add_argument("--parent", type=Path, required=True)
    exp.add_argument("--overlay", type=Path, required=True)
    exp.add_argument("--destination", type=Path, required=True)
    rep = sub.add_parser("replay")
    rep.add_argument("--package", type=Path, required=True)
    rep.add_argument("--destination", type=Path, required=True)
    rep.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "export":
        result = export(args.repo, args.parent, args.overlay, args.destination)
        print(
            json.dumps(
                dict(nodes=len(result["nodes"]), unique_payloads=len(result["objects"]))
            )
        )
    else:
        result = replay(args.package, args.destination)
        write(args.output, result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
