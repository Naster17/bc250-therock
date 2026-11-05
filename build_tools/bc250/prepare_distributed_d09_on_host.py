#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create only owned D09 external mount fixtures, with no application/model access."""
import json
import os
from pathlib import Path


def main() -> None:
    assert os.geteuid() == 0
    root = Path("/var/tmp/bc250-d09-fixtures")
    root.mkdir()
    for name in ["read only data", "writable data", "cwd with spaces"]:
        path = root / name
        path.mkdir()
        os.chown(path, 1000, 1000)
    data = dict(
        fixture="owned-distributed-d09",
        input_ids=[[(i * 7 + j * 3) % 96 + 1 for j in range(12)] for i in range(8)],
    )
    (root / "read only data/data.json").write_text(json.dumps(data) + "\n")
    (root / "consumer.env").write_text(
        "BC250_D09_ORDER=from-file\nHF_HUB_OFFLINE=1\nHF_HUB_DISABLE_XET=1\n"
    )
    os.chown(root, 1000, 1000)
    print(root)


if __name__ == "__main__":
    main()
