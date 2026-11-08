#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the default Docker admission controls against the stack module too."""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock
import bc250_lora_checkpoint_test as prior

spec = importlib.util.spec_from_file_location(
    "stack_docker_preflight",
    Path(__file__).resolve().parents[1] / "bc250/docker_preflight.py",
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class StackDockerPreflight(prior.DockerPreflight):
    def setUp(self):
        patch = mock.patch.object(prior, "guard", guard)
        patch.start()
        self.addCleanup(patch.stop)


if __name__ == "__main__":
    unittest.main()
