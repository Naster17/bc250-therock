#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline preset identity, opt-out and command construction contracts."""
import contextlib
import copy
import hashlib
import io
from pathlib import Path
import os
import tempfile
import unittest

from bc250_torch import (
    LaunchOptions,
    build_command,
    parse_launcher_arguments,
    qualified_gemm_override,
)


class TuningLauncher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.options = LaunchOptions(arguments=["-c", "pass"])
        identities = {}
        for name in ("lib/librocblas.so.5.8", "lib/rocblas/library/TensileLibrary.dat"):
            p = self.root / "rocm" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(name)
            identities[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        self.csv = self.root / "rocblas-gemm-n32.csv"
        self.csv.write_text("qualified-test-data")
        self.manifest = dict(
            uid=os.getuid(),
            gid=os.getgid(),
            image="test-image",
            rocblas_gemm_tuning=dict(
                enabled=True,
                identity_sha256=identities,
                sha256=hashlib.sha256(self.csv.read_bytes()).hexdigest(),
            ),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_qualified_and_bound_command(self):
        self.assertEqual(
            qualified_gemm_override(self.root, self.manifest, self.options), self.csv
        )
        cmd = build_command(
            self.root,
            self.manifest,
            self.root,
            self.root,
            self.options,
            [],
            self.root / "env",
        )
        self.assertIn(f"{self.csv}:/workspace/rocblas-gemm-overrides.csv:ro", cmd)
        self.assertIn(
            "ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=/workspace/rocblas-gemm-overrides.csv",
            cmd,
        )

    def test_disabled_and_explicit_override(self):
        for env in (
            {"BC250_GEMM_TUNING": "0"},
            {"ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH": "/custom.csv"},
        ):
            self.options.environment = env
            self.assertIsNone(
                qualified_gemm_override(self.root, self.manifest, self.options)
            )

    def test_previous_release_without_metadata(self):
        self.assertIsNone(qualified_gemm_override(self.root, {}, self.options))

    def test_changed_preset_or_library_falls_back(self):
        for file in (self.csv, self.root / "rocm/lib/librocblas.so.5.8"):
            old = file.read_bytes()
            file.write_text("different")
            with contextlib.redirect_stderr(io.StringIO()) as message:
                self.assertIsNone(
                    qualified_gemm_override(self.root, self.manifest, self.options)
                )
            self.assertIn("identities differ", message.getvalue())
            file.write_bytes(old)

    def test_missing_or_traversing_identity_falls_back(self):
        for name in ("../outside", "lib/missing.so"):
            manifest = copy.deepcopy(self.manifest)
            manifest["rocblas_gemm_tuning"]["identity_sha256"][name] = "invalid"
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertIsNone(
                    qualified_gemm_override(self.root, manifest, self.options)
                )

    def test_environment_optout_reaches_launcher(self):
        parsed = parse_launcher_arguments(["-c", "pass"], {"BC250_GEMM_TUNING": "0"})
        self.assertEqual(parsed.environment["BC250_GEMM_TUNING"], "0")


if __name__ == "__main__":
    unittest.main()
