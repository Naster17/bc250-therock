# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path
import unittest

BC250 = Path(__file__).resolve().parents[1] / "bc250"
SOAK_DIR = BC250 / "soak_plan2_01"
D71_DIR = BC250 / "gemm_efficiency_d71"


def load(name, directory=BC250):
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summarize = load("summarize_soak", SOAK_DIR)
runner = load("run_soak", SOAK_DIR)
kernel_noise = load("kernel_noise", SOAK_DIR)


def seg_audit(start, n, seconds=40.0, peak=6.0e9, loss=2.4, skipped=False, finite=True):
    rows = [
        {
            "step": start + i,
            "seconds": seconds,
            "finite_gradients": finite,
            "optimizer_skipped": skipped,
            "peak_allocated_bytes": peak,
            "metrics": {"loss": loss},
        }
        for i in range(n)
    ]
    return {
        "initial_adapter_sha256": f"init-{start}",
        "final_adapter_sha256": f"final-{start}",
        "rows": rows,
    }


class AggregateTest(unittest.TestCase):
    def test_aggregate_sorts_by_step(self):
        audits = [seg_audit(6, 3), seg_audit(1, 3)]
        rows = summarize.aggregate_rows(audits)
        self.assertEqual([r["step"] for r in rows], [1, 2, 3, 6, 7, 8])

    def test_steps_contiguous(self):
        self.assertTrue(
            summarize.steps_contiguous(summarize.aggregate_rows([seg_audit(1, 5)]))
        )
        self.assertFalse(
            summarize.steps_contiguous(
                summarize.aggregate_rows([seg_audit(1, 3), seg_audit(5, 2)])
            )
        )

    def test_finite_nonskipped(self):
        self.assertTrue(
            summarize.all_finite_nonskipped(summarize.aggregate_rows([seg_audit(1, 3)]))
        )
        self.assertFalse(
            summarize.all_finite_nonskipped(
                summarize.aggregate_rows([seg_audit(1, 3, skipped=True)])
            )
        )
        self.assertFalse(
            summarize.all_finite_nonskipped(
                summarize.aggregate_rows([seg_audit(1, 3, finite=False)])
            )
        )

    def test_loss_finite(self):
        self.assertTrue(
            summarize.loss_finite(summarize.aggregate_rows([seg_audit(1, 3)]))
        )
        self.assertFalse(
            summarize.loss_finite(
                summarize.aggregate_rows([seg_audit(1, 3, loss=float("nan"))])
            )
        )

    def test_adapters_changed(self):
        self.assertTrue(summarize.adapters_changed([seg_audit(1, 3)]))
        bad = seg_audit(1, 3)
        bad["final_adapter_sha256"] = bad["initial_adapter_sha256"]
        self.assertFalse(summarize.adapters_changed([bad]))


class BoundsTest(unittest.TestCase):
    def test_drift_stable(self):
        rows = summarize.aggregate_rows([seg_audit(1, 12, seconds=40.0)])
        self.assertAlmostEqual(summarize.drift_ratio(rows), 1.0)

    def test_drift_slowdown(self):
        rows = [{"step": i, "seconds": 10.0 if i < 6 else 25.0} for i in range(1, 13)]
        self.assertGreater(summarize.drift_ratio(rows), summarize.DRIFT_MAX)

    def test_memory_stable(self):
        self.assertAlmostEqual(
            summarize.memory_ratio([seg_audit(1, 3), seg_audit(4, 3)]), 1.0
        )

    def test_memory_growth(self):
        ratio = summarize.memory_ratio(
            [seg_audit(1, 3, peak=6e9), seg_audit(4, 3, peak=9e9)]
        )
        self.assertGreater(ratio, summarize.MEMORY_RATIO_MAX)


class SummaryTest(unittest.TestCase):
    def _passed_inputs(self):
        audits = [seg_audit(1, 5), seg_audit(6, 5)]
        telemetry = [{"segment": 0, "temp_c": 70.0}, {"segment": 1, "temp_c": 72.0}]
        return audits, telemetry

    def test_passed(self):
        audits, telemetry = self._passed_inputs()
        s = summarize.build_summary(
            audits, telemetry, 300.0, {"seg0": True, "seg1": True}
        )
        self.assertEqual(s["status"], "passed")
        self.assertEqual(s["total_updates"], 10)
        self.assertEqual(s["segments"], 2)
        self.assertAlmostEqual(s["accumulated_training_seconds"], 400.0)
        self.assertTrue(all(s["checks"].values()))

    def test_target_not_met(self):
        audits, telemetry = self._passed_inputs()
        s = summarize.build_summary(audits, telemetry, 1000.0)
        self.assertEqual(s["status"], "failed")
        self.assertFalse(s["checks"]["accumulated_target_met"])

    def test_skipped_fails(self):
        audits = [seg_audit(1, 5), seg_audit(6, 5, skipped=True)]
        s = summarize.build_summary(audits, [], 300.0)
        self.assertEqual(s["status"], "failed")
        self.assertFalse(s["checks"]["all_finite_nonskipped"])

    def test_noncontiguous_fails(self):
        audits = [seg_audit(1, 5), seg_audit(7, 5)]  # gap at step 6
        s = summarize.build_summary(audits, [], 300.0)
        self.assertFalse(s["checks"]["steps_contiguous"])
        self.assertEqual(s["status"], "failed")

    def test_temp_over_fails(self):
        audits, _ = self._passed_inputs()
        s = summarize.build_summary(audits, [{"temp_c": 95.0}], 300.0)
        self.assertFalse(s["checks"]["temp_within_bounds"])

    def test_no_rows_fails(self):
        self.assertEqual(summarize.build_summary([], [], 300.0)["status"], "failed")


class OrchestratorTest(unittest.TestCase):
    def test_app_config_cumulative(self):
        c0 = runner.app_config(0, 10)
        self.assertIsNone(c0["resume"])
        self.assertEqual(c0["max_steps"], 10)
        self.assertEqual(c0["save_steps"], 10)
        c1 = runner.app_config(1, 10)
        self.assertEqual(c1["max_steps"], 20)
        self.assertTrue(c1["resume"].endswith("seg0/checkpoint-10"))
        c3 = runner.app_config(3, 10)
        self.assertEqual(c3["max_steps"], 40)
        self.assertTrue(c3["resume"].endswith("seg2/checkpoint-30"))

    def test_app_config_epoch_suppression_avoided(self):
        cfg = runner.app_config(0, 10)
        self.assertGreaterEqual(cfg["epochs"], 1000)
        self.assertEqual(cfg["save_total_limit"], 2)
        self.assertIsNone(cfg["eval_data"])

    def test_base_env_no_benchmark(self):
        joined = " ".join(runner.base_env())
        self.assertNotIn("BC250_LORA_BENCHMARK", joined)
        self.assertIn("PYTHONPATH=", joined)
        self.assertIn("BC250_LORA_INITIAL_SCALE=128", joined)

    def test_guardian_config(self):
        cfg = runner.guardian_config(2)
        self.assertEqual(cfg["mode"], "single")
        self.assertEqual(cfg["label"], "soak-seg2")
        self.assertEqual(cfg["rank"], 0)
        self.assertTrue(cfg["audit"].endswith("seg2/audit-rank0.json"))
        self.assertIn("train_two_host.py", " ".join(cfg["argv"]))
        self.assertIn("--config", " ".join(cfg["argv"]))

    def test_defaults(self):
        self.assertEqual(runner.LENGTH, 1024)
        self.assertGreaterEqual(runner.TARGET_SECONDS, 7200.0)
        self.assertEqual(runner.SESSION_TIMEOUT, 900)


class KernelNoiseTest(unittest.TestCase):
    def test_filter_removes_only_benign(self):
        data = b"ok line\namdgpu 0000:01:00.0: Unsupported clock type\nother\n"
        self.assertEqual(kernel_noise.filter_benign(data), [b"ok line", b"other"])

    def test_equivalent_when_only_benign_added(self):
        before = b"[1.0] ok line\n"
        after = b"[1.0] ok line\n[2.0] amdgpu: Unsupported clock type\n"
        self.assertTrue(kernel_noise.kernel_log_equivalent(before, after))

    def test_identical_logs_equivalent(self):
        self.assertTrue(kernel_noise.kernel_log_equivalent(b"a\nb\n", b"a\nb\n"))

    def test_real_gpu_fault_not_filtered(self):
        before = b"[1.0] ok\n"
        for fault in (
            b"[2.0] amdgpu: ring timeout\n",
            b"[2.0] amdgpu: GPU reset\n",
            b"[2.0] amdgpu: VM fault\n",
            b"[2.0] Kernel panic\n",
        ):
            self.assertFalse(
                kernel_noise.kernel_log_equivalent(before, before + fault), fault
            )

    def test_benign_count(self):
        data = b"Unsupported clock type\nx\nUnsupported clock type\n"
        self.assertEqual(kernel_noise.benign_count(data), 2)
        self.assertEqual(kernel_noise.benign_count(b"clean\n"), 0)


class GuardianReuseTest(unittest.TestCase):
    def test_guardian_kernel_filter_refinement(self):
        # The soak guardian intentionally diverges from D71 ONLY in the
        # kernel-log check: it uses the pre-declared benign-noise filter.
        soak = (SOAK_DIR / "guard_soak.py").read_text()
        d71 = (D71_DIR / "guard_d71.py").read_text()
        self.assertIn("kernel_log_equivalent(before, after)", soak)
        self.assertIn("from kernel_noise import", soak)
        self.assertIn("before == after", d71)
        self.assertNotIn("kernel_log_equivalent", d71)
        # All other shared invariants remain present in the soak guardian.
        for invariant in (
            "fcntl.LOCK_SH | fcntl.LOCK_NB",
            '"--timeout",',
            "code = child.wait(timeout=460)",
            "Owned container remains; recovery stays armed",
            'result["registry_unchanged"]',
            'result["foreign_containers"]',
        ):
            self.assertIn(invariant, soak)

    def test_bounded_launch_hardened_backstop(self):
        text = (SOAK_DIR / "bounded_launch.py").read_text()
        self.assertIn('command[:0] = ["timeout", "430"]', text)
        for envelope in (
            '"--cpus=4"',
            '"--cpuset-cpus=0-3"',
            '"--memory=10g"',
            '"--memory-swap=12g"',
            '"--pids-limit=512"',
        ):
            self.assertIn(envelope, text)


if __name__ == "__main__":
    unittest.main()
