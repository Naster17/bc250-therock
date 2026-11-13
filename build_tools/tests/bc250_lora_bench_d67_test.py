# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

BC250 = Path(__file__).resolve().parents[1] / "bc250/lora_bench_d67"


def load(name):
    spec = importlib.util.spec_from_file_location(name, BC250 / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary = load("summarize_d67")
data = load("make_owned_data")
run = load("run_d67")


def row(step, seconds, tokens=16384, finite=True, skipped=False):
    return {
        "step": step,
        "seconds": seconds,
        "finite_gradients": finite,
        "optimizer_skipped": skipped,
        "input_tokens": tokens,
        "peak_allocated_bytes": 1000 + step,
        "peak_reserved_bytes": 2000 + step,
    }


class SteadyStateTest(unittest.TestCase):
    def test_excludes_first_and_summarizes(self):
        rows = [row(1, 999.0)] + [row(i, float(i)) for i in range(2, 8)]
        result = summary.steady_state(rows)
        self.assertEqual(result["measured_updates"], 6)
        self.assertEqual(result["step_median_s"], 4.5)
        self.assertEqual(result["global_tokens_per_update"], 16384)
        self.assertAlmostEqual(result["tokens_per_second"], 16384 / 4.5)
        self.assertGreater(result["step_iqr_percent"], 0.0)
        self.assertEqual(result["peak_allocated_bytes"], 1007)

    def test_small_sample_iqr_zero(self):
        rows = [row(1, 999.0), row(2, 2.0), row(3, 4.0)]
        result = summary.steady_state(rows)
        self.assertEqual(result["step_iqr_percent"], 0.0)

    def test_refuses_skipped(self):
        with self.assertRaises(ValueError):
            summary.steady_state([row(1, 1.0), row(2, 2.0, skipped=True)])

    def test_refuses_nonfinite(self):
        with self.assertRaises(ValueError):
            summary.steady_state([row(1, 1.0), row(2, 2.0, finite=False)])

    def test_refuses_unequal_tokens(self):
        with self.assertRaises(ValueError):
            summary.steady_state([row(1, 1.0), row(2, 2.0), row(3, 3.0, tokens=1)])

    def test_refuses_empty(self):
        with self.assertRaises(ValueError):
            summary.steady_state([row(1, 1.0)])


class MergeRanksTest(unittest.TestCase):
    def audit(self, world, rows):
        return {"world_size": world, "rows": rows, "trainable_parameters": 134553600}

    def test_single_rank(self):
        merged = summary.merge_ranks([self.audit(1, [row(1, 5.0)])])
        self.assertEqual(merged["world_size"], 1)
        self.assertEqual(merged["rows"][0]["seconds"], 5.0)

    def test_two_ranks_max_time(self):
        merged = summary.merge_ranks(
            [
                self.audit(2, [row(1, 5.0), row(2, 7.0)]),
                self.audit(2, [row(1, 6.0), row(2, 6.5)]),
            ]
        )
        self.assertEqual([r["seconds"] for r in merged["rows"]], [6.0, 7.0])
        self.assertEqual(merged["rows"][0]["peak_allocated_bytes"], 1001)

    def test_refuses_mixed_worlds(self):
        with self.assertRaises(ValueError):
            summary.merge_ranks([self.audit(1, []), self.audit(2, [])])

    def test_refuses_token_disagreement(self):
        with self.assertRaises(ValueError):
            summary.merge_ranks(
                [
                    self.audit(2, [row(1, 5.0)]),
                    self.audit(2, [row(1, 6.0, tokens=1)]),
                ]
            )


class RatioAndSummaryTest(unittest.TestCase):
    def test_ratio(self):
        single = summary.steady_state([row(1, 99.0), row(2, 20.0), row(3, 20.0)])
        ddp = summary.steady_state([row(1, 99.0), row(2, 16.0), row(3, 16.0)])
        self.assertAlmostEqual(summary.throughput_ratio(single, ddp), 1.25)

    def test_build_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp)
            for mode, world, seconds in (("single", 1, 20.0), ("ddp", 2, 16.0)):
                case = evidence / f"1024-{mode}"
                case.mkdir()
                for rank in range(world):
                    (case / f"audit-rank{rank}.json").write_text(
                        json.dumps(
                            {
                                "world_size": world,
                                "trainable_parameters": 134553600,
                                "rows": [
                                    row(1, 99.0),
                                    row(2, seconds),
                                    row(3, seconds),
                                ],
                            }
                        )
                    )
            result = summary.build_summary(evidence)
            self.assertEqual(len(result["cases"]), 1)
            entry = result["cases"][0]
            self.assertEqual(entry["sequence_length"], 1024)
            self.assertAlmostEqual(entry["throughput_ratio_ddp_over_single"], 1.25)

    def test_build_summary_cold_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp)
            for mode, world, seconds in (("single", 1, 326.8), ("ddp", 2, 170.0)):
                case = evidence / f"4096-{mode}"
                case.mkdir()
                for rank in range(world):
                    (case / f"audit-rank{rank}.json").write_text(
                        json.dumps(
                            {
                                "world_size": world,
                                "trainable_parameters": 134553600,
                                "rows": [row(1, seconds)],
                            }
                        )
                    )
            result = summary.build_summary(evidence)
            entry = result["cases"][0]
            self.assertNotIn("throughput_ratio_ddp_over_single", entry)
            self.assertAlmostEqual(
                entry["cold_first_update_ratio_ddp_over_single"], 326.8 / 170.0
            )
            self.assertEqual(entry["modes"]["single"]["updates_run"], 1)
            self.assertNotIn("step_median_s", entry["modes"]["single"])


class OwnedDataPureLogicTest(unittest.TestCase):
    def test_words_deterministic(self):
        a = data.words(random.Random(7), 50)
        b = data.words(random.Random(7), 50)
        self.assertEqual(a, b)
        self.assertEqual(len(a.split()), 50)
        self.assertNotEqual(a, data.words(random.Random(8), 50))

    def test_example_structure(self):
        row = data.example(random.Random(1), 10, 20)
        self.assertEqual(row["prompt"][0]["role"], "user")
        self.assertEqual(row["completion"][0]["role"], "assistant")
        self.assertTrue(row["prompt"][0]["content"].startswith(data.PROMPT_PREFIX))


class FakeTokenizer:
    """Mixed 2/3-token words: some totals are skipped by naive increments, and
    a fixed word-fraction prompt can overshoot the whole target budget."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = " ".join(m["content"] for m in messages)
        count = 7  # fixed template overhead
        for word in text.split():
            count += 3 if word[0] in "bdfhk" else 2
        return list(range(count))


class EncodingLike:
    """BatchEncoding stand-in: UserDict-like, len == key count, not a dict."""

    def __init__(self, ids):
        self.input_ids = ids

    def __len__(self):
        return 2


class EncodingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        return EncodingLike(
            super().apply_chat_template(messages, tokenize, add_generation_prompt)
        )


class TokenLengthNormalizationTest(unittest.TestCase):
    def test_batchencoding_like_normalized(self):
        row = data.example(random.Random(1), 10, 10)
        expected = FakeTokenizer().apply_chat_template(
            row["prompt"] + row["completion"]
        )
        self.assertEqual(data.token_length(EncodingTokenizer(), row), len(expected))

    def test_nested_list_normalized(self):
        class Nested(FakeTokenizer):
            def apply_chat_template(
                self, messages, tokenize=True, add_generation_prompt=False
            ):
                return [
                    super().apply_chat_template(
                        messages, tokenize, add_generation_prompt
                    )
                ]

        row = data.example(random.Random(1), 10, 10)
        expected = FakeTokenizer().apply_chat_template(
            row["prompt"] + row["completion"]
        )
        self.assertEqual(data.token_length(Nested(), row), len(expected))


class ExactLengthSearchTest(unittest.TestCase):
    def test_generate_hits_exact_targets_with_mixed_costs(self):
        # With mixed 1/2-token words some totals are unreachable for a fixed
        # prompt; generate() must vary the prompt word count to hit them all.
        tok = FakeTokenizer()
        for target in range(100, 130):
            rows = data.generate(tok, target, 2, 5)
            for row in rows:
                self.assertEqual(data.token_length(tok, row), target)

    def test_fit_deterministic_for_reachable_target(self):
        tok = FakeTokenizer()
        rows = data.generate(tok, 100, 1, 5)
        self.assertEqual(data.token_length(tok, rows[0]), 100)

    def test_generate_exact_and_diverse(self):
        rows = data.generate(FakeTokenizer(), 64, 8, 5)
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertEqual(data.token_length(FakeTokenizer(), row), 64)
        contents = {r["completion"][0]["content"] for r in rows}
        self.assertGreater(len(contents), 1)

    def test_generate_deterministic(self):
        a = data.generate(FakeTokenizer(), 64, 4, 5)
        b = data.generate(FakeTokenizer(), 64, 4, 5)
        self.assertEqual(a, b)


class ConfigTest(unittest.TestCase):
    def test_app_config_accumulation_preserves_global_batch(self):
        single = run.app_config("1024-single", 1024, "single")
        ddp = run.app_config("1024-ddp", 1024, "ddp")
        self.assertEqual(single["grad_accum"], 4)
        self.assertEqual(ddp["grad_accum"], 2)
        self.assertEqual(single["batch_size"] * single["grad_accum"], 4)
        self.assertEqual(ddp["batch_size"] * ddp["grad_accum"] * 2, 4)
        self.assertEqual(single["max_steps"], run.MAX_STEPS[1024])
        self.assertEqual(single["max_length"], 1024)
        self.assertEqual(single["dropout"], 0.05)
        self.assertEqual(single["warmup_ratio"], 0.0)

    def test_launcher_argv(self):
        ddp = run.launcher_argv("1024-ddp", "ddp", 1, 30167)
        self.assertEqual(ddp[:4], ["--node-rank", "1", "--master-port", "30167"])
        single = run.launcher_argv("1024-single", "single", 0, 30167)
        self.assertNotIn("--node-rank", single)
        for argv in (ddp, single):
            self.assertIn("BC250_LORA_BENCHMARK=1", argv)
            self.assertTrue(argv[-3].endswith("train_two_host.py"))
            self.assertEqual(argv[-2], "--config")
            self.assertTrue(argv[-1].endswith(".json"))

    def test_guardian_config(self):
        cfg = run.guardian_config("4096-ddp", "ddp", 1, 30170)
        self.assertEqual(cfg["rank"], 1)
        self.assertEqual(cfg["mode"], "ddp")
        self.assertTrue(cfg["audit"].endswith("audit-rank1.json"))
        self.assertTrue(cfg["evidence"].endswith("4096-ddp"))
        self.assertEqual(cfg["cwd"], run.PROJECT)


if __name__ == "__main__":
    unittest.main()
