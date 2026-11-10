#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay complete Torch deltas in private bare clones, preserving all source trees."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from verify_priority2_reconstruction import verify


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=root / "sources/bc250-pytorch")
    parser.add_argument(
        "--patches", type=Path, default=root / "patches/bc250-pytorch-complete"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revisions = json.loads((args.patches / "revisions.json").read_text())
    with tempfile.TemporaryDirectory(prefix="bc250-d46-native-") as directory:
        shutil.copy2(args.patches / "0001-qualified-native-torch.patch", directory)
        native = verify(
            args.repo,
            revisions["baseline"],
            revisions["native_revision"],
            Path(directory),
        )
    head = verify(
        args.repo, revisions["baseline"], revisions["source_head"], args.patches
    )
    result = dict(
        status="passed" if native["status"] == head["status"] == "passed" else "failed",
        native=native,
        source_head=head,
        scope="Exact complete tracked Git trees with existing local baseline objects; not a portable object bundle or native rebuild",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status=result["status"],
                native_tree=native["reconstructed_tree"],
                source_head_tree=head["reconstructed_tree"],
            )
        )
    )
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
