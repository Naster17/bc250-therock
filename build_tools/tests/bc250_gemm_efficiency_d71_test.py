# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import ast
import importlib.util
from pathlib import Path
import unittest

BC250 = Path(__file__).resolve().parents[1] / "bc250"
D71_DIR = BC250 / "gemm_efficiency_d71"
D70_DIR = BC250 / "launch_reduction_d70"


def load(name, directory=BC250):
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


census = load("census_gemm_d71")
bench = load("bench_gemm_d71")
preset = load("preset_d71", D71_DIR)
summarize = load("summarize_d71", D71_DIR)
runner = load("run_d71", D71_DIR)


def desc(shape, stride=None, dtype="float16"):
    if stride is None:
        stride = [shape[1], 1] if len(shape) == 2 else None
    return {"shape": shape, "stride": stride, "dtype": dtype}


A = desc([1024, 2048])
B = desc([2048, 128])
BIAS = desc([1024, 128])


class CensusTest(unittest.TestCase):
    def test_extract_config(self):
        self.assertEqual(census.extract_config(["--config", "a.json"]), Path("a.json"))
        with self.assertRaises(ValueError):
            census.extract_config([])

    def test_aggregate_calls(self):
        calls = [
            {"op": "mm", "args": [A, B]},
            {"op": "mm", "args": [A, B]},
            {"op": "mm", "args": [A, desc([2048, 64])]},
        ]
        rows = census.aggregate_calls(calls)
        self.assertEqual(rows[0]["count"], 2)
        self.assertAlmostEqual(rows[0]["share"], 2 / 3)
        self.assertEqual(rows[1]["count"], 1)

    def test_rocblas_layout_mm(self):
        rb = census.rocblas_layout("mm", [A, B])
        self.assertEqual(
            rb,
            {
                "transA": "N",
                "transB": "N",
                "M": 128,
                "N": 1024,
                "K": 2048,
                "batch_count": 1,
                "lda": 128,
                "ldb": 2048,
                "ldc": 128,
                "beta": 0,
                "input_type": "float16",
                "output_type": "float16",
                "compute_type": "float32",
            },
        )

    def test_rocblas_layout_transposed(self):
        a_t = desc([1024, 2048], [1, 1024])
        rb = census.rocblas_layout("mm", [a_t, B])
        self.assertEqual(rb["transB"], "T")
        self.assertEqual(rb["ldb"], 1024)

    def test_rocblas_layout_addmm(self):
        rb = census.rocblas_layout("addmm", [BIAS, A, B])
        self.assertEqual(rb["beta"], 1)
        self.assertEqual(rb["M"], 128)
        self.assertEqual(rb["N"], 1024)

    def test_rocblas_layout_bmm(self):
        a = desc([4, 1024, 2048], [2048 * 1024, 2048, 1])
        b = desc([4, 2048, 128], [2048 * 128, 128, 1])
        rb = census.rocblas_layout("bmm", [a, b])
        self.assertEqual(rb["batch_count"], 4)

    def test_rocblas_layout_unsupported(self):
        a = desc([1024, 2048], [2, 2])
        self.assertIsNone(census.rocblas_layout("mm", [a, B]))


class BenchTest(unittest.TestCase):
    def test_dtype_constants(self):
        self.assertEqual(bench.DTYPES["float16"], 150)
        self.assertEqual(bench.DTYPES["float32"], 151)
        self.assertEqual(bench.DTYPES["bfloat16"], 168)

    def test_select_layouts(self):
        census_doc = {
            "layouts": [
                {
                    "op": "mm",
                    "args": [A, B],
                    "count": 10,
                    "rocblas": census.rocblas_layout("mm", [A, B]),
                },
                {
                    "op": "bmm",
                    "args": [A, B],
                    "count": 9,
                    "rocblas": {"batch_count": 4},
                },
                {"op": "mm", "args": [A, B], "count": 8, "rocblas": None},
                {
                    "op": "addmm",
                    "args": [BIAS, A, B],
                    "count": 7,
                    "rocblas": census.rocblas_layout("addmm", [BIAS, A, B]),
                },
            ]
        }
        rows = bench.select_layouts(census_doc)
        self.assertEqual([r["op"] for r in rows], ["mm", "addmm"])


def bench_result(ratio=2.0, op="mm", count=100, default=10.0, candidate=5.0):
    args = [BIAS, A, B] if op == "addmm" else [A, B]
    return {
        "op": op,
        "args": args,
        "count": count,
        "rocblas_layout": {
            "transa": "N",
            "transb": "N",
            "m": 128,
            "n": 1024,
            "k": 2048,
            "lda": 128,
            "ldb": 2048,
            "ldc": 128,
            "beta": 1 if op == "addmm" else 0,
        },
        "selected_candidate": -490,
        "medians": {"default": default, "candidate": candidate},
        "paired_gpu_ratio": ratio,
    }


EXISTING_CSV = (
    "transA,transB,M,N,batch_count,K,alpha,beta,lda,ldb,ldc,"
    "input_type,output_type,compute_type,solution_index\n"
    "N,T,2048,1024,1,6144,1,0,2048,1024,2048,bf16_r,bf16_r,f32_r,-490\n"
)


class PresetTest(unittest.TestCase):
    def test_winner_filter(self):
        doc = {"results": [bench_result(2.0), bench_result(1.01)]}
        winners = preset.winner_rows(doc)
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0]["solution_index"], -490)

    def test_build_preset_preserves_existing(self):
        doc = {"results": [bench_result(2.0)]}
        csv_text = preset.build_preset(doc, EXISTING_CSV)
        lines = csv_text.strip().splitlines()
        self.assertEqual(lines[0], preset.CSV_HEADER)
        self.assertIn("bf16_r,bf16_r,f32_r,-490", lines[1])
        self.assertEqual(len(lines), 3)
        self.assertIn("f16_r,f16_r,f32_r,-490", lines[2])
        self.assertTrue(lines[2].startswith("N,N,128,1024,1,2048,1,0,128,2048,128,"))

    def test_build_preset_addmm_beta(self):
        doc = {"results": [bench_result(2.0, op="addmm")]}
        csv_text = preset.build_preset(doc, EXISTING_CSV)
        self.assertIn(",1,128,2048,128,f16_r", csv_text.splitlines()[2])

    def test_no_winners_keeps_existing(self):
        csv_text = preset.build_preset({"results": []}, EXISTING_CSV)
        self.assertEqual(len(csv_text.strip().splitlines()), 2)


def audit_rows(seconds, loss=2.4717):
    return {
        "rows": [
            {
                "seconds": s,
                "metrics": {"loss": loss},
                "finite_gradients": True,
                "optimizer_skipped": False,
                "peak_allocated_bytes": 6 * 2**30,
            }
            for s in seconds
        ]
    }


class SummarizerTest(unittest.TestCase):
    def _inputs(self, cand_loss=2.4717, ratio=2.0):
        census_doc = {"total_gemm_calls": 1000, "distinct_layouts": 12}
        bench_doc = {"status": "passed", "results": [bench_result(ratio)]}
        ref = audit_rows([40.0, 36.0, 36.2, 36.1, 36.0, 36.2])
        cand = audit_rows([40.0, 20.0, 20.1, 20.0, 20.2, 20.0], loss=cand_loss)
        return census_doc, bench_doc, ref, cand

    def test_passed_summary(self):
        census_doc, bench_doc, ref, cand = self._inputs()
        summary = summarize.build_summary(census_doc, bench_doc, ref, cand, 1)
        self.assertEqual(summary["status"], "passed")
        self.assertAlmostEqual(summary["candidate"]["step_speedup"], 36.1 / 20.0)
        self.assertAlmostEqual(summary["predicted_savings_seconds"], 5.0 * 100 / 1000)
        self.assertEqual(len(summary["winners"]), 1)

    def test_loss_drift_fails(self):
        census_doc, bench_doc, ref, cand = self._inputs(cand_loss=2.6)
        summary = summarize.build_summary(census_doc, bench_doc, ref, cand, 1)
        self.assertEqual(summary["status"], "failed")

    def test_reproduction_fails(self):
        census_doc, bench_doc, _, cand = self._inputs()
        ref = audit_rows([60.0, 60.0, 60.0])
        summary = summarize.build_summary(census_doc, bench_doc, ref, cand, 1)
        self.assertEqual(summary["status"], "failed")

    def test_no_candidate(self):
        census_doc, bench_doc, ref, _ = self._inputs()
        summary = summarize.build_summary(census_doc, bench_doc, ref, None, 0)
        self.assertEqual(summary["status"], "passed")
        self.assertIsNone(summary["candidate"])


class GuardianReuseTest(unittest.TestCase):
    def test_guardian_logic_identical_to_d70(self):
        def body(path):
            tree = ast.parse(path.read_text())
            return ast.dump(ast.Module(body=tree.body[1:], type_ignores=[]))

        self.assertEqual(body(D71_DIR / "guard_d71.py"), body(D70_DIR / "guard_d70.py"))

    def test_bounded_launch_hardened_backstop(self):
        # D71 intentionally diverges from D70: a `timeout 430` backstop wraps
        # podman so a container cannot outlive a killed guardian (r6 incident).
        # The resource envelope itself is unchanged.
        d71 = (D71_DIR / "bounded_launch.py").read_text()
        d70 = (D70_DIR / "bounded_launch.py").read_text()
        self.assertIn('command[:0] = ["timeout", "430"]', d71)
        for envelope in (
            '"--cpus=4"',
            '"--cpuset-cpus=0-3"',
            '"--memory=10g"',
            '"--memory-swap=12g"',
            '"--pids-limit=512"',
        ):
            self.assertIn(envelope, d71)
            self.assertIn(envelope, d70)


class OrchestratorTest(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(runner.CASES["census-1024"], (1024, 1))
        self.assertEqual(runner.CASES["step-ref-1024"], (1024, 6))
        self.assertEqual(runner.CASES["step-cand-1024"], (1024, 6))
        self.assertEqual(runner.CASES["profile-shapes-1024"], (1024, 2))

    def test_guardian_configs(self):
        census_cfg = runner.guardian_config("census-1024")
        self.assertIn("census_gemm_d71.py", " ".join(census_cfg["argv"]))
        self.assertIn("audit", census_cfg)
        bench_cfg = runner.guardian_config("bench")
        self.assertNotIn("audit", bench_cfg)
        ref_cfg = runner.guardian_config("step-ref-1024")
        self.assertIn("train_two_host.py", " ".join(ref_cfg["argv"]))
        self.assertNotIn(
            "ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH", " ".join(ref_cfg["argv"])
        )
        cand_cfg = runner.guardian_config("step-cand-1024")
        joined = " ".join(cand_cfg["argv"])
        self.assertIn("ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH", joined)
        self.assertIn("--mount-ro", joined)

    def test_app_config_matches_d67_settings(self):
        cfg = runner.app_config("step-ref-1024", 1024, 6)
        self.assertEqual(cfg["grad_accum"], 4)
        self.assertEqual(cfg["precision"], "fp16")
        self.assertEqual(cfg["attention"], "eager")
        self.assertTrue(cfg["gradient_checkpointing"])
        self.assertEqual(cfg["rank"], 128)
        self.assertEqual(cfg["seed"], 42)


if __name__ == "__main__":
    unittest.main()
