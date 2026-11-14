# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure (no torch) tests for the PLAN2-02 NF4 core contract and reference.

The torch ops and GPU parity qualification run on native gfx1013; these tests
prove the dependency-free quantization contract, the independent reference, and
the nearest-code/nibble/metadata/bounds logic on the workstation.
"""

import bisect
import importlib.util
from pathlib import Path
import random
import sys
import unittest

BC250 = Path(__file__).resolve().parents[1] / "bc250"
if str(BC250) not in sys.path:
    sys.path.insert(0, str(BC250))

from nf4_plan2_02 import core, reference  # noqa: E402

NF4_DIR = BC250 / "nf4_plan2_02"


def load(name, directory):
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("run_nf4", NF4_DIR)


class CodebookTest(unittest.TestCase):
    def test_sixteen_sorted_levels_in_range(self):
        self.assertEqual(len(core.NF4_CODEBOOK), 16)
        self.assertEqual(list(core.NF4_CODEBOOK), sorted(core.NF4_CODEBOOK))
        self.assertTrue(all(-1.0 <= v <= 1.0 for v in core.NF4_CODEBOOK))
        self.assertEqual(core.NF4_CODEBOOK[0], -1.0)
        self.assertEqual(core.NF4_CODEBOOK[-1], 1.0)
        self.assertEqual(core.NF4_CODEBOOK[7], 0.0)

    def test_canonical_values(self):
        self.assertAlmostEqual(core.NF4_CODEBOOK[1], -0.6961111111114001)
        self.assertAlmostEqual(core.NF4_CODEBOOK[8], 0.07951111111154178)
        self.assertAlmostEqual(core.NF4_CODEBOOK[14], 0.7221111111126917)

    def test_midpoints_between_levels(self):
        self.assertEqual(len(core.NF4_MIDPOINTS), 15)
        for i, mid in enumerate(core.NF4_MIDPOINTS):
            self.assertGreater(mid, core.NF4_CODEBOOK[i])
            self.assertLess(mid, core.NF4_CODEBOOK[i + 1])
        self.assertEqual(list(core.NF4_MIDPOINTS), sorted(core.NF4_MIDPOINTS))


class NearestCodeTest(unittest.TestCase):
    def test_exact_levels_map_to_themselves(self):
        for j, level in enumerate(core.NF4_CODEBOOK):
            self.assertEqual(core.nearest_code(level), j)

    def test_matches_distance_argmin_away_from_boundaries(self):
        rng = random.Random(20261114)
        for _ in range(2000):
            v = rng.uniform(-1.3, 1.3)
            if any(abs(v - m) < 1e-9 for m in core.NF4_MIDPOINTS):
                continue
            distances = [abs(v - level) for level in core.NF4_CODEBOOK]
            self.assertEqual(core.nearest_code(v), distances.index(min(distances)))

    def test_matches_searchsorted_equivalent(self):
        rng = random.Random(20261114)
        values = [rng.uniform(-1.3, 1.3) for _ in range(2000)]
        values += list(core.NF4_CODEBOOK) + list(core.NF4_MIDPOINTS)
        values += [-2.0, 2.0, 0.0, -1.0, 1.0]
        for v in values:
            self.assertEqual(
                core.nearest_code(v),
                bisect.bisect_right(core.NF4_MIDPOINTS, v),
                msg=f"nearest/searchsorted disagree at {v}",
            )

    def test_ties_round_to_higher_code(self):
        for j, mid in enumerate(core.NF4_MIDPOINTS):
            self.assertEqual(core.nearest_code(mid), j + 1)

    def test_out_of_range_clamps(self):
        self.assertEqual(core.nearest_code(-5.0), 0)
        self.assertEqual(core.nearest_code(5.0), 15)


class NibbleTest(unittest.TestCase):
    def test_known_encoding(self):
        self.assertEqual(core.pack_nibbles([0, 15, 1, 14]), [0xF0, 0xE1])

    def test_round_trip_even_and_odd(self):
        rng = random.Random(7)
        for count in (2, 4, 16, 64, 3, 17, 65):
            indices = [rng.randrange(16) for _ in range(count)]
            packed = core.pack_nibbles(indices)
            self.assertEqual(len(packed), (count + 1) // 2)
            self.assertEqual(core.unpack_nibbles(packed, count), indices)

    def test_odd_count_zero_pads_high_nibble(self):
        self.assertEqual(core.pack_nibbles([5]), [0x05])
        self.assertEqual(core.unpack_nibbles([0x05], 1), [5])


class BlockScaleTest(unittest.TestCase):
    def test_zero_block_uses_unit_scale(self):
        self.assertEqual(core.block_absmax_scales([0.0] * 8, 8), [1.0])

    def test_absmax_per_block(self):
        rows = [1.0, -3.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0]
        self.assertEqual(core.block_absmax_scales(rows, 4), [3.0, 1.0])


class ReferenceTest(unittest.TestCase):
    def test_matmul_known_product(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        self.assertEqual(reference.matmul(a, b), [[19.0, 22.0], [43.0, 50.0]])

    def test_zero_weight_packs_to_zero_level(self):
        weight = [[0.0] * 16 for _ in range(4)]
        packed, scales = reference.pack_reference(weight, 16)
        self.assertEqual(scales, [1.0] * 4)
        indices = core.unpack_nibbles(packed, 4 * 16)
        self.assertEqual(indices, [7] * (4 * 16))

    def test_dequant_values_are_levels_times_scale(self):
        rng = random.Random(11)
        n, k, bs = 4, 32, 16
        weight = [[rng.uniform(-2.0, 2.0) for _ in range(k)] for _ in range(n)]
        packed, scales = reference.pack_reference(weight, bs)
        restored = reference.dequantize_reference(packed, scales, n, k, bs)
        bytes_per_row = k // 2
        for i in range(n):
            indices = core.unpack_nibbles(
                packed[i * bytes_per_row : (i + 1) * bytes_per_row], k
            )
            for b in range(k // bs):
                scale = scales[i * (k // bs) + b]
                for j in range(b * bs, (b + 1) * bs):
                    self.assertEqual(
                        restored[i][j], core.NF4_CODEBOOK[indices[j]] * scale
                    )
                    self.assertLessEqual(abs(restored[i][j]), scale + 1e-12)

    def test_pack_dequant_pack_is_idempotent(self):
        rng = random.Random(13)
        weight = [[rng.uniform(-3.0, 3.0) for _ in range(64)] for _ in range(8)]
        packed, scales = reference.pack_reference(weight, 16)
        restored = reference.dequantize_reference(packed, scales, 8, 64, 16)
        packed2, scales2 = reference.pack_reference(restored, 16)
        self.assertEqual(packed, packed2)
        self.assertEqual(scales, scales2)

    def test_quantization_error_bounded(self):
        # NF4 relative Frobenius error on Gaussian weights is a few percent;
        # pre-declared loose bound for the recipe (not a model-quality claim).
        rng = random.Random(17)
        weight = [[rng.gauss(0.0, 1.0) for _ in range(256)] for _ in range(16)]
        packed, scales = reference.pack_reference(weight, 64)
        restored = reference.dequantize_reference(packed, scales, 16, 256, 64)
        num = sum(
            (weight[i][j] - restored[i][j]) ** 2 for i in range(16) for j in range(256)
        )
        den = sum(weight[i][j] ** 2 for i in range(16) for j in range(256))
        self.assertLess((num / den) ** 0.5, 0.10)

    def test_forward_and_input_grad_match_manual(self):
        rng = random.Random(19)
        n, k, m = 4, 16, 3
        weight = [[rng.uniform(-1.0, 1.0) for _ in range(k)] for _ in range(n)]
        packed, scales = reference.pack_reference(weight, 16)
        x = [[rng.uniform(-1.0, 1.0) for _ in range(k)] for _ in range(m)]
        bias = [rng.uniform(-0.5, 0.5) for _ in range(n)]
        y = reference.nf4_forward(x, packed, scales, n, k, 16, bias)
        w = reference.dequantize_reference(packed, scales, n, k, 16)
        manual = reference.matmul(x, reference.transpose(w))
        for i in range(m):
            for j in range(n):
                self.assertAlmostEqual(y[i][j], manual[i][j] + bias[j], places=10)
        dy = [[rng.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(m)]
        dx = reference.nf4_input_grad(dy, packed, scales, n, k, 16)
        manual_dx = reference.matmul(dy, w)
        for i in range(m):
            for j in range(k):
                self.assertAlmostEqual(dx[i][j], manual_dx[i][j], places=10)

    def test_reference_rejects_bad_block_divisibility(self):
        with self.assertRaises(ValueError):
            reference.pack_reference([[0.0] * 15], 16)


class MetadataTest(unittest.TestCase):
    def valid(self):
        return dict(
            format_version=1,
            format="nf4_block_absmax",
            in_features=64,
            out_features=32,
            block_size=16,
            compute_dtype="torch.float16",
            bias=True,
            bias_trainable=False,
        )

    def test_accepts_valid(self):
        self.assertEqual(core.validate_metadata(self.valid()), self.valid())

    def test_refuses_bad_version_and_format(self):
        for key, value in (("format_version", 2), ("format", "int8")):
            state = self.valid()
            state[key] = value
            with self.assertRaises(ValueError):
                core.validate_metadata(state)

    def test_refuses_bad_block_and_divisibility(self):
        state = self.valid()
        state["block_size"] = 7
        with self.assertRaises(ValueError):
            core.validate_metadata(state)
        state = self.valid()
        state["in_features"] = 65
        with self.assertRaises(ValueError):
            core.validate_metadata(state)

    def test_refuses_bad_dtype_and_bias(self):
        state = self.valid()
        state["compute_dtype"] = "torch.int8"
        with self.assertRaises(ValueError):
            core.validate_metadata(state)
        state = self.valid()
        state["bias"] = False
        state["bias_trainable"] = True
        with self.assertRaises(ValueError):
            core.validate_metadata(state)

    def test_refuses_missing_field(self):
        state = self.valid()
        del state["block_size"]
        with self.assertRaises(ValueError):
            core.validate_metadata(state)


class BoundsTest(unittest.TestCase):
    def test_weight_shape_bounds(self):
        core.validate_weight_shape((4096, 1024))
        for shape in ((0, 4), (4097, 4), (4, 4097), (4,), (4096, 4096)):
            with self.assertRaises(ValueError):
                core.validate_weight_shape(shape)

    def test_block_size_bounds(self):
        for bs in core.BLOCK_SIZES:
            core.validate_block_size(bs)
        for bs in (0, 7, 8, 512, 64.0, "64"):
            with self.assertRaises(ValueError):
                core.validate_block_size(bs)


class OrchestratorTest(unittest.TestCase):
    def test_guardian_config(self):
        cfg = runner.guardian_config()
        self.assertEqual(cfg["mode"], "single")
        self.assertEqual(cfg["rank"], 0)
        self.assertEqual(cfg["label"], "nf4-r1")
        self.assertEqual(cfg["audit"], runner.REPORT)
        self.assertIn(runner.QUALIFY_SCRIPT, cfg["argv"])
        self.assertIn("--output", cfg["argv"])
        self.assertIn(runner.REPORT, cfg["argv"])
        self.assertTrue(cfg["evidence"].endswith("nf4-r1"))

    def test_package_files_complete(self):
        for name in runner.PACKAGE_FILES:
            self.assertTrue((NF4_DIR / name).exists(), name)

    def test_parse_sclk_levels(self):
        self.assertEqual(
            runner.parse_sclk_levels("0: 1000Mhz *\n1: 1500Mhz\n2: 2000Mhz"),
            [1000, 1500, 2000],
        )
        self.assertEqual(runner.parse_sclk_levels("0: 3500Mhz *"), [3500])
        self.assertEqual(runner.parse_sclk_levels(""), [])

    def test_stock_clock_ceiling(self):
        self.assertEqual(runner.STOCK_CLOCK_MAX_MHZ, 2000)
        self.assertLessEqual(
            max(runner.parse_sclk_levels("0: 1000Mhz\n2: 2000Mhz")), 2000
        )
        self.assertGreater(max(runner.parse_sclk_levels("0: 3500Mhz *")), 2000)


class GuardianReuseTest(unittest.TestCase):
    def setUp(self):
        self.text = (NF4_DIR / "guard_nf4.py").read_text()
        self.launch = (BC250 / "soak_plan2_01" / "bounded_launch.py").read_text()

    def test_envelope_invariants(self):
        for token in (
            "boot-guard.py",
            "fcntl.LOCK_SH",
            "child.wait(timeout=460)",
            "kernel_log_equivalent",
            "registry_unchanged",
            "foreign_containers",
            "ack.json",
            '"nonce", "boot_id", "phase"',
            "preflight()",
        ):
            self.assertIn(token, self.text, token)
        self.assertIn('"600"', self.text)

    def test_bounded_launch_backstop(self):
        self.assertIn('"timeout", "430"', self.launch)
        self.assertIn("--pids-limit=512", self.launch)

    def test_audit_asserts_all_passed(self):
        self.assertIn('result["worker"]["summary"]["all_passed"]', self.text)
        self.assertIn('result["worker"]["checks"]', self.text)


if __name__ == "__main__":
    unittest.main()
