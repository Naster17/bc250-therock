# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "bc250/deploy_on_host.py"
spec = importlib.util.spec_from_file_location("bc250_deployment", SCRIPT)
deployment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployment)


class ExpandedDeploymentGateTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)
        self.logs = self.workspace / "logs"
        (self.logs / "candidate-runtime").mkdir(parents=True)
        for name in (
            "expanded-build.exit",
            "expanded-validation.exit",
            "candidate-torch-validation.exit",
            "candidate-runtime/eager.exit",
            "candidate-runtime/compiler.exit",
            "candidate-runtime/stability.exit",
        ):
            (self.logs / name).write_text("0\n")
        self.evidence = {
            "status": "passed",
            "seconds": 600.1,
            "cycles": 100,
            "arch": "gfx1013:xnack-",
        }
        self.write_evidence()

    def write_evidence(self):
        (self.logs / "candidate-runtime/stability.json").write_text(
            json.dumps(self.evidence)
        )

    def test_completed_expanded_candidate_is_accepted(self):
        deployment.require_expanded_evidence(self.workspace)

    def test_failed_running_and_missing_gates_are_rejected(self):
        gate = self.logs / "expanded-validation.exit"
        for status in ("1", "running"):
            with self.subTest(status=status):
                gate.write_text(status)
                with self.assertRaises(RuntimeError):
                    deployment.require_expanded_evidence(self.workspace)
        gate.unlink()
        with self.assertRaises(FileNotFoundError):
            deployment.require_expanded_evidence(self.workspace)

    def test_short_empty_wrong_arch_and_running_soaks_are_rejected(self):
        for field, value in (
            ("seconds", 120),
            ("cycles", 0),
            ("arch", "gfx1010"),
            ("status", "running"),
        ):
            with self.subTest(field=field):
                original = self.evidence[field]
                self.evidence[field] = value
                self.write_evidence()
                with self.assertRaises(RuntimeError):
                    deployment.require_expanded_evidence(self.workspace)
                self.evidence[field] = original


if __name__ == "__main__":
    unittest.main()
