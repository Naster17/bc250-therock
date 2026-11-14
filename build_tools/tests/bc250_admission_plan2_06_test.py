# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure (no torch/GPU) tests for the PLAN2-06 broader-admission harness.

The GPU battery runs on native gfx1013 under the guardian; these tests prove the
orchestrator configs, the owned-only source contracts (no diffusers / no weight
fetching), D62-helper reuse, and the registry/battery coverage.
"""

from pathlib import Path
import sys
import unittest

BC250 = Path(__file__).resolve().parents[1] / "bc250"
if str(BC250) not in sys.path:
    sys.path.insert(0, str(BC250))

from admission_plan2_06 import run_admission  # noqa: E402

PKG = run_admission.LOCAL_TOOLS


class OrchestratorConfigTest(unittest.TestCase):
    def test_required_keys(self):
        config = run_admission.guardian_config("ad-x", "unet2d-tiny", "float32")
        self.assertTrue(
            {"mode", "label", "argv", "rank", "evidence", "cwd", "audit"} <= set(config)
        )
        self.assertEqual(config["mode"], "single")
        self.assertEqual(config["rank"], 0)

    def test_argv_targets_model_and_dtype(self):
        config = run_admission.guardian_config("ad-x", "lstm-tiny", "bfloat16")
        argv = config["argv"]
        self.assertIn("--model", argv)
        self.assertEqual(argv[argv.index("--model") + 1], "lstm-tiny")
        self.assertEqual(argv[argv.index("--dtype") + 1], "bfloat16")
        self.assertIn("admit.py", " ".join(argv))

    def test_report_paths_distinct(self):
        paths = {
            run_admission.report_path(m, d)
            for m in ("unet2d-tiny", "rnn-tiny", "lstm-tiny", "gru-tiny")
            for d in ("float32", "bfloat16")
        }
        self.assertEqual(len(paths), 8)

    def test_package_files_exist(self):
        for name in run_admission.PACKAGE_FILES:
            self.assertTrue((PKG / name).exists(), name)

    def test_reuses_qualified_guardian_and_d62(self):
        source = (PKG / "run_admission.py").read_text()
        for token in (
            "guard_nf4.py",
            "bounded_launch.py",
            "kernel_noise.py",
            "stock_clock_check",
            "idle_check",
            "admit_model_d62.py",
        ):
            self.assertIn(token, source)


class OwnedOnlyContractTest(unittest.TestCase):
    def test_no_weight_fetching_or_diffusers(self):
        for name in ("models.py", "admit.py"):
            source = (PKG / name).read_text()
            self.assertNotIn("from_pretrained", source)
            self.assertNotIn("snapshot_download", source)
            self.assertNotIn("import diffusers", source)

    def test_models_defined(self):
        source = (PKG / "models.py").read_text()
        for token in (
            "class UNet2D",
            "class RecurrentLM",
            "class SpatialAttention",
            "nn.Conv2d",
            "nn.GroupNorm",
            "scaled_dot_product_attention",
            "getattr(nn, cell)",
        ):
            self.assertIn(token, source)

    def test_reuses_d62_helpers(self):
        source = (PKG / "admit.py").read_text()
        self.assertIn("import admit_model_d62 as d62", source)
        for token in (
            "d62.state_bytes",
            "d62.route_status",
            "d62.summarize",
            "d62.new_report",
            "d62.owned_import_guard",
        ):
            self.assertIn(token, source)

    def test_registry_coverage(self):
        source = (PKG / "admit.py").read_text()
        for model in ("unet2d-tiny", "rnn-tiny", "lstm-tiny", "gru-tiny"):
            self.assertIn(f'"{model}"', source)
        for route in ("conv", "attention", "recurrent"):
            self.assertIn(f'"{route}"', source)

    def test_battery_checks_present(self):
        source = (PKG / "admit.py").read_text()
        for check in (
            "forward-vs-cpu-fp64",
            "backward-grads",
            "optimizer-step",
            "resume-bitwise-state",
            "resume-exact-continuation",
            "gfx1013-native",
            "owned-import-guard",
        ):
            self.assertIn(check, source)

    def test_d62_fixture_untouched_contract(self):
        # The D62 module is imported, not edited; it must still exist alongside.
        self.assertTrue((BC250 / "admit_model_d62.py").exists())


if __name__ == "__main__":
    unittest.main()
