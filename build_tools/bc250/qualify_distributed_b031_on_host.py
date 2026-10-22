#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify CPU distributed functionality and retained native GPU consumers."""

import json
from pathlib import Path

from deploy_torch_a04_on_host import sha
from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

GATES = (
    ("distributed", "validate_distributed_b031.py", [], 31),
    ("sparse", "validate_sparse_b021.py", [], 253),
    ("triangular", "validate_sparse_n213.py", ["--group", "edge"], 80),
    ("sparse-legacy", "validate_priority2_math.py", ["--group", "sparse"], 41),
    ("fft-promotion", "validate_fft_b011.py", [], 245),
    ("fft-real", "validate_fft_n214.py", ["--group", "real"], 72),
    ("fft-contracts", "validate_fft_n214.py", ["--group", "contracts"], 3),
    ("vision-ops", "validate_vision_b041.py", ["--group", "ops"], 51),
    ("vision-integration", "validate_vision_b041.py", ["--group", "integration"], 9),
    ("audio-dsp", "validate_audio_b042.py", ["--group", "dsp"], 34),
    ("audio-native", "validate_audio_b042.py", ["--group", "native"], 40),
    ("image", "validate_image_b045.py", [], 86),
    ("video", "validate_video_b044.py", [], 55),
    ("files", "validate_codec_b043.py", [], 46),
    ("compiled-dense", "validate_compiler_n310.py", ["--family", "dense"], 14),
    ("compiled-cnn", "validate_compiler_n310.py", ["--family", "cnn"], 14),
    ("profiler", "validate_distributed_profiler_b031.py", [], 1),
    ("core", "validate_cleanup_runtime.py", [], 7),
)
SOURCES = tuple(
    sorted(
        {g[1] for g in GATES}
        | {
            "test_bc250_sparse_triangular.py",
            "validate_profiler_order.py",
            "test_bc250_profiler_order.py",
        }
    )
)

if __name__ == "__main__":
    preflight("distributed-b031")
    stage = json.loads((WS / "logs/distributed-b031-stage-r1/final.json").read_text())
    candidate = Path(stage["candidate_release"])
    assert sha(candidate / "runtime.json") == stage["candidate_runtime_sha256"]
    for name, expected in stage["native_sha256"].items():
        assert sha(candidate / "torch/torch/lib" / name) == expected, name
    main(
        batch_id="distributed-b031",
        baseline_label="sparse-b021-qualification-r5",
        include_training_app=False,
        default_phase="extra",
        release_override=candidate,
        extra_gates=GATES,
        additional_sources=SOURCES,
        minimum_disk_free_bytes=4 * 1024**3,
    )
