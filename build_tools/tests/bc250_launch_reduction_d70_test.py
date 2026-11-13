# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import ast
import importlib.util
from pathlib import Path
import unittest

BC250 = Path(__file__).resolve().parents[1] / "bc250"
D70_DIR = BC250 / "launch_reduction_d70"
D69_DIR = BC250 / "route_port_d69"


def load(name, directory=BC250):
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dec = load("decompose_lora_d70")
probe = load("probe_launch_d70")
summarize = load("summarize_d70", D70_DIR)
runner = load("run_d70", D70_DIR)


def span(
    cat, seconds, phase="forward", micro=0, layer=None, update=0, wall=None, w0=0.0
):
    w = seconds if wall is None else wall
    return {
        "cat": cat,
        "phase": phase,
        "micro": micro,
        "layer": layer,
        "update": update,
        "seconds": seconds,
        "wall": w,
        "w0": w0,
        "w1": w0 + w,
    }


def synthetic_update():
    """Two microbatches, two layers (linear + full), budget closes exactly."""
    spans = []
    for micro, w0 in ((0, 0.0), (1, 10.5)):
        spans += [
            span("micro_total", 10.0, micro=micro, w0=w0),
            span("backward_total", 6.0, phase="backward", micro=micro, w0=w0 + 4),
            span("layer/linear_attention", 2.0, micro=micro, layer=0),
            span("layer/full_attention", 1.5, micro=micro, layer=1),
            span("mod/mlp", 0.5, micro=micro, layer=0),
            span("mod/mlp", 0.4, micro=micro, layer=1),
            span("layer/linear_attention", 3.0, phase="backward", micro=micro, layer=0),
            span("layer/full_attention", 2.0, phase="backward", micro=micro, layer=1),
            span("mod/mlp", 0.5, phase="backward", micro=micro, layer=0),
            span("mod/mlp", 0.4, phase="backward", micro=micro, layer=1),
            span("fn/conv", 0.8, micro=micro, layer=0),
            span("fn/conv", 0.8, phase="backward", micro=micro, layer=0),
            span("fn/chunk", 1.0, micro=micro, layer=0),
            span("fn/chunk", 1.0, phase="backward", micro=micro, layer=0),
            span("fn/eager", 0.7, micro=micro, layer=1),
            span("fn/eager", 0.7, phase="backward", micro=micro, layer=1),
        ]
    spans += [
        span("clip", 0.2, micro=-1, w0=20.6),
        span("optimizer", 1.0, micro=-1, w0=21.0),
    ]
    return spans


LAYER_TYPES = ["linear_attention", "full_attention"]


class PureLogicTest(unittest.TestCase):
    def test_layer_category(self):
        self.assertEqual(
            dec.layer_category("linear_attention"), "layer/linear_attention"
        )
        self.assertEqual(dec.layer_category("full_attention"), "layer/full_attention")
        with self.assertRaises(ValueError):
            dec.layer_category("recurrent")

    def test_extract_config(self):
        self.assertEqual(dec.extract_config(["--config", "a.json"]), Path("a.json"))
        self.assertEqual(dec.extract_config(["--config=b.json"]), Path("b.json"))
        with self.assertRaises(ValueError):
            dec.extract_config(["--other"])

    def test_aggregate_budget_closure(self):
        agg = dec.aggregate_update(synthetic_update(), 0, 2, LAYER_TYPES)
        self.assertAlmostEqual(agg["wall_seconds"], 22.0)
        self.assertAlmostEqual(agg["micro_wall_seconds"], 20.0)
        top = agg["top"]
        self.assertAlmostEqual(top["mixer/full_attention"], 2.7 * 2)
        self.assertAlmostEqual(top["mixer/linear_attention"], 4.0 * 2)
        self.assertAlmostEqual(top["mlp"], 1.8 * 2)
        self.assertAlmostEqual(top["fwd_other"], 0.5 * 2)
        self.assertAlmostEqual(top["bwd_other"], 1.0 * 2)
        self.assertAlmostEqual(top["clip"], 0.2)
        self.assertAlmostEqual(top["optimizer"], 1.0)
        self.assertNotIn("host_gap", top)
        # GPU timeline spans tile 21.2s of the 22s update (sync fixture).
        self.assertAlmostEqual(agg["gpu_timeline_coverage"], 21.2 / 22.0)
        self.assertAlmostEqual(sum(agg["shares"].values()), 21.2 / 22.0)
        cpu = agg["cpu"]
        self.assertAlmostEqual(cpu["micro_wall"], 20.0)
        self.assertAlmostEqual(cpu["inter_micro_gap"], 0.5)
        self.assertAlmostEqual(cpu["pre_optimizer_gap"], 0.1)
        self.assertAlmostEqual(cpu["post_clip_gap"], 0.2)
        self.assertAlmostEqual(cpu["clip_wall"], 0.2)
        self.assertAlmostEqual(cpu["optimizer_wall"], 1.0)
        self.assertAlmostEqual(cpu["host_gap"], 0.8)
        self.assertAlmostEqual(cpu["closure_residual"], 0.0)
        self.assertAlmostEqual(agg["sub"]["fn/conv"]["forward"], 1.6)
        self.assertAlmostEqual(agg["sub"]["fn/conv"]["recompute"], 1.6)
        self.assertAlmostEqual(agg["sub"]["fn/eager"]["forward"], 1.4)
        self.assertAlmostEqual(agg["recompute_total"], (0.8 + 1.0 + 0.7) * 2)

    def test_aggregate_requires_full_update(self):
        spans = synthetic_update()
        spans = [
            s for s in spans if not (s["cat"] == "micro_total" and s["micro"] == 1)
        ]
        with self.assertRaises(ValueError):
            dec.aggregate_update(spans, 0, 2, LAYER_TYPES)

    def test_steady_median_excludes_first(self):
        rows = [{"v": 100.0}, {"v": 1.0}, {"v": 3.0}, {"v": 2.0}, {"v": 5.0}]
        self.assertAlmostEqual(dec.steady_median(rows, "v"), 2.5)
        rows = [{"v": 100.0}, {"v": 1.0}, {"v": 3.0}, {"v": 2.0}]
        self.assertAlmostEqual(dec.steady_median(rows, "v"), 2.0)
        with self.assertRaises(ValueError):
            dec.steady_median([{"v": 1.0}], "v")

    def test_progress_label_single_update(self):
        self.assertEqual(
            dec.progress_label([{"wall_seconds": 325.884}]), "cold=325.884s"
        )
        self.assertEqual(
            dec.progress_label(
                [{"wall_seconds": 40.0}, {"wall_seconds": 36.0}, {"wall_seconds": 36.2}]
            ),
            "steady=36.100s",
        )

    def _decomp(self, walls, loss=2.4717):
        rows = []
        for i, w in enumerate(walls):
            top = {
                "mixer/full_attention": w * 0.3,
                "mixer/linear_attention": w * 0.2,
                "mlp": w * 0.2,
                "fwd_other": w * 0.05,
                "bwd_other": w * 0.05,
                "clip": w * 0.02,
                "optimizer": w * 0.1,
            }
            rows.append(
                {
                    "update": i,
                    "wall_seconds": w,
                    "top": top,
                    "shares": {k: v / w for k, v in top.items()},
                    "gpu_timeline_coverage": 0.97,
                    "cpu": {
                        "micro_wall": w * 0.6,
                        "inter_micro_gap": w * 0.05,
                        "pre_optimizer_gap": w * 0.3,
                        "post_clip_gap": 0.0,
                        "clip_wall": w * 0.02,
                        "optimizer_wall": w * 0.03,
                        "host_gap": w * 0.35,
                        "closure_residual": 0.0,
                    },
                    "sub": {
                        c: {"forward": 0.1, "recompute": 0.1}
                        for c in ("fn/conv", "fn/chunk", "fn/eager")
                    },
                }
            )
        return {"updates": rows, "final_loss": loss, "d69_ref_loss": 2.4717}

    def test_build_summary_reproduction_pass(self):
        summary = dec.build_summary(self._decomp([40.0, 36.0, 36.2, 36.1]), 36.166)
        self.assertTrue(summary["reproduction_within_bound"])
        self.assertTrue(summary["loss_within_bound"])
        self.assertAlmostEqual(summary["steady_wall_seconds"], 36.1)
        self.assertAlmostEqual(summary["steady_shares"]["optimizer"], 0.1)
        self.assertAlmostEqual(summary["gpu_timeline_coverage"], 0.97)
        self.assertAlmostEqual(summary["steady_cpu"]["pre_optimizer_gap"], 36.1 * 0.3)

    def test_build_summary_reproduction_fail(self):
        summary = dec.build_summary(self._decomp([50.0, 50.0, 50.0]), 36.166)
        self.assertFalse(summary["reproduction_within_bound"])
        summary = dec.build_summary(self._decomp([36.0, 36.0], loss=2.5), 36.166)
        self.assertFalse(summary["loss_within_bound"])


class SummarizerTest(unittest.TestCase):
    def _inputs(self, selftest="works", failed=()):
        decomp = PureLogicTest()._decomp([40.0, 36.0, 36.2, 36.1])
        probe_report = {
            "verdicts": {
                "profiler": "works",
                "graph_mlp": "works",
                "graph_qwen35_aten": "blocked",
                "graph_qwen35_fused": "blocked",
                "graph_checkpoint": "blocked",
                "graphed_callables": "works",
                "instrumentation_selftest": selftest,
            },
            "failed_sections": list(failed),
        }
        return probe_report, decomp

    def test_passed_summary(self):
        probe_report, decomp = self._inputs()
        summary = summarize.build_summary(probe_report, decomp, None)
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["profiler_available"])
        self.assertEqual(summary["graph_feasibility"]["qwen35_aten_capture"], "blocked")
        ranked = summary["ranked_step_shares"]
        self.assertEqual(ranked[0]["category"], "mixer/full_attention")
        self.assertEqual(
            [r["share"] for r in ranked],
            sorted((r["share"] for r in ranked), reverse=True),
        )

    def test_selftest_failure_blocks(self):
        probe_report, decomp = self._inputs(
            selftest="failed", failed=["instrumentation_selftest"]
        )
        summary = summarize.build_summary(probe_report, decomp, None)
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(any("selftest" in r for r in summary["reasons"]))

    def test_reproduction_failure_blocks(self):
        probe_report, _ = self._inputs()
        decomp = PureLogicTest()._decomp([60.0, 60.0, 60.0])
        summary = summarize.build_summary(probe_report, decomp, None)
        self.assertEqual(summary["status"], "failed")

    def test_cold_4096_summary(self):
        probe_report, decomp = self._inputs()
        cold = PureLogicTest()._decomp([330.0])
        summary = summarize.build_summary(probe_report, decomp, None, cold)
        self.assertEqual(summary["status"], "passed")
        self.assertAlmostEqual(summary["cold_4096"]["wall_seconds"], 330.0)
        self.assertTrue(summary["cold_4096"]["within_bound"])
        self.assertAlmostEqual(
            summary["cold_4096"]["ratio"], 330.0 / summarize.D69_4096_REF_SECONDS
        )
        cold = PureLogicTest()._decomp([400.0])
        summary = summarize.build_summary(probe_report, decomp, None, cold)
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["cold_4096"]["within_bound"])


class ProbeFixtureTest(unittest.TestCase):
    def test_tiny_config(self):
        cfg = probe.tiny_config_overrides()
        self.assertEqual(cfg["num_hidden_layers"], 4)
        self.assertEqual(cfg["full_attention_interval"], 4)
        self.assertEqual(cfg["attention_dropout"], 0.0)
        self.assertTrue(cfg["tie_word_embeddings"])

    def test_parity_error_is_value_error(self):
        self.assertTrue(issubclass(probe.ParityError, ValueError))

    def test_verdict_helper(self):
        section = {"name": "x"}
        probe.verdict(section, "blocked", error="boom", detail={"a": 1})
        self.assertEqual(section["status"], "blocked")
        self.assertEqual(section["error"], "boom")


class GuardianReuseTest(unittest.TestCase):
    def test_guardian_logic_identical_to_d69(self):
        def body(path):
            tree = ast.parse(path.read_text())
            return ast.dump(ast.Module(body=tree.body[1:], type_ignores=[]))

        self.assertEqual(body(D70_DIR / "guard_d70.py"), body(D69_DIR / "guard_d69.py"))

    def test_bounded_launch_identical(self):
        self.assertEqual(
            (D70_DIR / "bounded_launch.py").read_bytes(),
            (D69_DIR / "bounded_launch.py").read_bytes(),
        )


class OrchestratorTest(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(runner.CASES["decomp-1024"], (1024, 6, False))
        self.assertEqual(runner.CASES["profile-1024"], (1024, 2, True))
        self.assertEqual(runner.CASES["decomp-4096"], (4096, 1, False))

    def test_app_config_matches_d67_settings(self):
        cfg = runner.app_config("decomp-1024", 1024, 6)
        self.assertEqual(cfg["grad_accum"], 4)
        self.assertEqual(cfg["batch_size"], 1)
        self.assertEqual(cfg["precision"], "fp16")
        self.assertEqual(cfg["attention"], "eager")
        self.assertTrue(cfg["gradient_checkpointing"])
        self.assertEqual(cfg["rank"], 128)
        self.assertEqual(cfg["alpha"], 256)
        self.assertEqual(cfg["seed"], 42)
        self.assertEqual(cfg["scheduler"], "constant")
        self.assertIn("owned-1024.jsonl", cfg["train_data"][0])

    def test_guardian_configs(self):
        probe_cfg = runner.probe_guardian_config()
        self.assertEqual(probe_cfg["label"], "d70-probe")
        self.assertNotIn("audit", probe_cfg)
        case_cfg = runner.case_guardian_config("decomp-1024")
        self.assertEqual(case_cfg["label"], "d70-decomp-1024")
        self.assertIn("audit-rank0.json", case_cfg["audit"])
        self.assertIn("decompose_lora_d70.py", " ".join(case_cfg["argv"]))
        profile_cfg = runner.case_guardian_config("profile-1024")
        self.assertIn("BC250_D70_PROFILE=1", profile_cfg["argv"])


if __name__ == "__main__":
    unittest.main()
