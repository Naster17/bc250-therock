#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Model-independent compiler gates on the preserved native gfx1013 runtime."""

import argparse
import re
import shutil

from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attempt", default="r1")
    args, _ = parser.parse_known_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    preflight("compiler-n310-" + args.attempt)
    shutil.copy2(
        WS / "logs" / ("training-n39-resource-compiler-n310-" + args.attempt + ".json"),
        WS / "logs" / ("training-n310-resource-" + args.attempt + ".json"),
    )
    main(
        batch_id="training-n310",
        baseline_label="training-n39-guidance-r4",
        default_phase="extra",
        extra_gates=tuple(
            (name, "validate_compiler_n310.py", ["--family", name], 14)
            for name in ("dense", "gated", "cnn", "batchnorm")
        ),
        additional_sources=("validate_compiler_n310.py",),
        minimum_disk_free_bytes=4 * 1024**3,
    )
