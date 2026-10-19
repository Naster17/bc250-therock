#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reuse exact training replay on the identity-pinned installed GEMM preset."""
import hashlib
import json
import os
from pathlib import Path
import sys

from profile_training_n31 import main as profile_main


def main():
    preset = Path("/workspace/rocblas-gemm-overrides.csv")
    assert os.environ.get("ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH") == str(preset)
    assert "LD_PRELOAD" not in os.environ
    digest = hashlib.sha256(preset.read_bytes()).hexdigest()
    profile_main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    result = json.loads(output.read_text())
    result["checks"].append("installed-identity-pinned-preset")
    result["gemm_preset_sha256"] = digest
    result["n33_wrapper_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
