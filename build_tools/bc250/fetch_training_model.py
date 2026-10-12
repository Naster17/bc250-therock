# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fetch and hash the pinned small pretrained model used by the training gate."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=directory,
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"],
    )
    hashes = {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if hashes.get("model.safetensors") != MODEL_SHA256:
        raise RuntimeError(
            "Pretrained weights do not match the pinned upstream checksum"
        )
    (directory / "bc250-model-provenance.json").write_text(
        json.dumps({"model": MODEL, "revision": REVISION, "sha256": hashes}, indent=2)
        + "\n"
    )
    print(f"TRAINING_MODEL_READY {directory}", flush=True)


if __name__ == "__main__":
    main()
