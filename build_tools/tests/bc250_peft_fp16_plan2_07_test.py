# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure (no torch/GPU) tests for the PLAN2-07 FP16 PEFT harness.

The GPU battery runs on native gfx1013 under the guardian; these tests prove the
policy/bounds transcription, the D28 fixture identity, orchestrator configs and
the frozen-helper-mirroring source contracts (FP32 master + FP16 autocast +
GradScaler; existing FSDP helpers untouched).
"""

from pathlib import Path
import sys
import unittest

BC250 = Path(__file__).resolve().parents[1] / "bc250"
if str(BC250) not in sys.path:
    sys.path.insert(0, str(BC250))

from peft_fp16_plan2_07 import policy  # noqa: E402
from peft_fp16_plan2_07 import run_peft_fp16  # noqa: E402

PKG = run_peft_fp16.LOCAL_TOOLS


class PolicyTranscriptionTest(unittest.TestCase):
    def test_fixture_identity(self):
        self.assertEqual(policy.BASE_PARAM_COUNT, 21280)
        self.assertEqual(policy.TRAINABLE_PARAM_COUNT, 3072)
        self.assertEqual(policy.LORA_RANK, 4)
        self.assertEqual(policy.LORA_ALPHA, 8)
        self.assertEqual(policy.GPT2_CONFIG["n_layer"], 2)
        self.assertEqual(policy.GPT2_CONFIG["n_embd"], 32)

    def test_precision_contract(self):
        self.assertEqual(policy.PRECISION["working_parameters"], "float16")
        self.assertEqual(policy.PRECISION["master_parameters"], "float32")
        self.assertEqual(policy.PRECISION["reductions"], "float32")
        self.assertEqual(policy.PRECISION["optimizer_state"], "float32")
        self.assertTrue(policy.PRECISION["grad_scaler"])

    def test_bounds(self):
        self.assertEqual(policy.LOSS_REL_BOUND, 0.02)
        self.assertEqual(policy.GRAD_GLOBAL_REL_BOUND, 0.02)
        self.assertEqual(policy.GRAD_PER_PARAM_REL_BOUND, 0.03)
        self.assertEqual(policy.CLIP_NORM_REL_BOUND, 1e-5)

    def test_relative_norm(self):
        self.assertAlmostEqual(policy.relative_norm([1.0, 2.0], [1.0, 2.0]), 0.0)
        self.assertGreater(policy.relative_norm([2.0, 2.0], [1.0, 1.0]), 0.0)


class OrchestratorConfigTest(unittest.TestCase):
    def test_required_keys(self):
        config = run_peft_fp16.guardian_config("fp-x")
        self.assertTrue(
            {"mode", "label", "argv", "rank", "evidence", "cwd", "audit"} <= set(config)
        )
        self.assertEqual(config["mode"], "single")
        self.assertEqual(config["rank"], 0)
        self.assertIn("qualify_peft_fp16.py", " ".join(config["argv"]))

    def test_package_files_exist(self):
        for name in run_peft_fp16.PACKAGE_FILES:
            self.assertTrue((PKG / name).exists(), name)

    def test_reuses_qualified_guardian(self):
        source = (PKG / "run_peft_fp16.py").read_text()
        for token in (
            "guard_nf4.py",
            "bounded_launch.py",
            "kernel_noise.py",
            "stock_clock_check",
            "idle_check",
        ):
            self.assertIn(token, source)


class SourceContractTest(unittest.TestCase):
    def test_fp16_autocast_and_scaler(self):
        source = (PKG / "qualify_peft_fp16.py").read_text()
        self.assertIn('torch.autocast("cuda", dtype=torch.float16)', source)
        self.assertIn("torch.amp.GradScaler", source)
        self.assertIn("scaler.unscale_", source)
        self.assertIn("scaler.step(", source)

    def test_real_peft_lora(self):
        source = (PKG / "qualify_peft_fp16.py").read_text()
        self.assertIn("from peft import LoraConfig, get_peft_model", source)
        self.assertIn("get_peft_model(", source)
        self.assertIn("init_lora_weights=True", source)

    def test_battery_checks_present(self):
        source = (PKG / "qualify_peft_fp16.py").read_text()
        for check in (
            "forward-vs-cpu-fp64",
            "backward-grads-vs-cpu-fp64",
            "frozen-base-no-grads",
            "grad-scaler-inf-skip",
            "root-clip-vs-cpu",
            "multi-update-training",
            "resume-bitwise-state",
            "resume-exact-continuation",
        ):
            self.assertIn(check, source)

    def test_frozen_fsdp_helpers_untouched(self):
        # The proof must not import or modify the frozen FSDP PEFT helpers.
        source = (PKG / "qualify_peft_fp16.py").read_text()
        self.assertNotIn("bc250_fsdp_peft", source)
        self.assertNotIn("BC250BF16PeftFSDPTrainer", source)
        for helper in (
            "bc250_fsdp_peft.py",
            "bc250_fsdp_peft_bf16.py",
            "bc250_fsdp_peft_bf16_checkpoint.py",
        ):
            self.assertTrue((BC250 / helper).exists())


if __name__ == "__main__":
    unittest.main()
