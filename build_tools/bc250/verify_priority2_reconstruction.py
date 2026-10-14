#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Verify exact source trees reconstructed from ordered BC-250 patch series.

Uses isolated shared bare clones and private indexes; original repositories,
worktrees, indexes and commits are never modified. A cached patch application
needs no checkout of the large source trees.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def git(arguments: list[str], *, repository: Path | None = None) -> str:
    command = ["git"]
    if repository is not None:
        command.extend(["--git-dir", str(repository)])
    result = subprocess.run(command + arguments, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command + arguments)}: {result.stderr.strip()}")
    return result.stdout.strip()


def verify(repository: Path, baseline: str, expected: str, patch_dir: Path) -> dict:
    patches = sorted(patch_dir.glob("*.patch"))
    if not patches:
        raise ValueError(f"No patches found in {patch_dir}")
    expected_revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", expected], text=True
    ).strip()
    expected_tree = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", f"{expected_revision}^{{tree}}"],
        text=True,
    ).strip()
    with tempfile.TemporaryDirectory(prefix="bc250-patch-reconstruction-") as temporary:
        isolated = Path(temporary) / "source.git"
        git(["clone", "--shared", "--bare", "--quiet", str(repository), str(isolated)])
        git(["read-tree", baseline], repository=isolated)
        for patch in patches:
            git(
                ["apply", "--cached", "--whitespace=error", str(patch.resolve())],
                repository=isolated,
            )
        actual_tree = git(["write-tree"], repository=isolated)
        result = {
            "status": "passed" if actual_tree == expected_tree else "failed",
            "baseline": baseline,
            "expected_revision": expected_revision,
            "expected_tree": expected_tree,
            "reconstructed_tree": actual_tree,
            "patches": [
                {
                    "name": patch.name,
                    "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                }
                for patch in patches
            ],
        }
        if actual_tree != expected_tree:
            result["difference"] = git(
                ["diff", "--stat", expected_tree, actual_tree], repository=isolated
            )
        return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-repo", type=Path, default=root / "rocm-libraries")
    parser.add_argument(
        "--torch-repo", type=Path, default=root / "sources/bc250-pytorch"
    )
    parser.add_argument(
        "--library-patches",
        type=Path,
        default=root / "patches/amd-mainline/rocm-libraries",
    )
    parser.add_argument(
        "--torch-patches", type=Path, default=root / "patches/bc250-pytorch"
    )
    parser.add_argument(
        "--library-base", default="3d8f334b9a72ad99a0ecd87552613970433e0500"
    )
    parser.add_argument(
        "--torch-base", default="c779020750d7dc99a39ea8c104ef8df1fc9d0755"
    )
    parser.add_argument("--library-expected", default="HEAD")
    parser.add_argument("--torch-expected", default="HEAD")
    parser.add_argument("--systems-repo", type=Path, default=root / "rocm-systems")
    parser.add_argument(
        "--systems-patches",
        type=Path,
        default=root / "patches/amd-mainline/rocm-systems",
    )
    parser.add_argument(
        "--systems-base", default="d5c528019d0fcfbbc32d96b53ac2c61df6d0e67d"
    )
    parser.add_argument("--systems-expected", default="HEAD")
    args = parser.parse_args()
    report = {"status": "running", "components": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, repository, baseline, expected, patches in (
        (
            "rocm-libraries",
            args.library_repo,
            args.library_base,
            args.library_expected,
            args.library_patches,
        ),
        (
            "rocm-systems",
            args.systems_repo,
            args.systems_base,
            args.systems_expected,
            args.systems_patches,
        ),
        (
            "torch",
            args.torch_repo,
            args.torch_base,
            args.torch_expected,
            args.torch_patches,
        ),
    ):
        try:
            report["components"][name] = verify(repository, baseline, expected, patches)
        except Exception as error:
            report["components"][name] = {"status": "failed", "error": str(error)}
    report["status"] = (
        "passed"
        if all(result["status"] == "passed" for result in report["components"].values())
        else "failed"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "components": {
                    name: result["status"]
                    for name, result in report["components"].items()
                },
            }
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
