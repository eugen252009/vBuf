#!/usr/bin/env python3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_simd_traceable_mutation import (
    A, B, C, FAMILIES, ROUNDS, action_metrics, affine_jump, descriptor_accounting,
    generated_raw, integer_states, modular_inverse, projected_seeds,
    reverse_integer_state, undo_xor_right, vector_split,
)


class SimdTraceableMutationTests(unittest.TestCase):
    def test_scalar_avx2_equivalence(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="simd-mutation-test-") as temporary:
            binary = Path(temporary) / "bench"
            subprocess.run(["g++", "-O2", "-march=znver3", "-mavx2", "-mfma", "-std=c++17",
                            str(root / "research/simd_traceable_mutation_bench.cpp"), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary), "self-test"], check=True, text=True, capture_output=True)
            self.assertIn("scalar_avx2_equivalence=PASS", result.stdout)

    def test_deterministic_forward_mutation(self):
        seeds = np.array([[1, 2], [3, 4]], dtype=np.uint32)
        ordinals = np.array([7, 8], dtype=np.uint32)
        for family in FAMILIES:
            for rounds in ROUNDS:
                np.testing.assert_array_equal(generated_raw(family, rounds, seeds, ordinals, 16), generated_raw(family, rounds, seeds, ordinals, 16))

    def test_modular_inverses(self):
        for value in A + C + (0x9E3779B1,):
            self.assertEqual((value * modular_inverse(value)) & 0xFFFFFFFF, 1)
        with self.assertRaises(ValueError):
            modular_inverse(2)

    def test_exact_integer_trace_back(self):
        seeds = np.array([0, 1, 0xDEADBEEF, 0xFFFFFFFF], dtype=np.uint32)
        ordinals = np.array([3, 17, 91, 400], dtype=np.uint32)
        lanes = np.array([0, 7, 13, 31], dtype=np.uint32)
        for family in FAMILIES[:3]:
            for rounds in ROUNDS:
                states = integer_states(family, rounds, seeds[:, None], ordinals, 32)[np.arange(4), 0, lanes]
                traced = reverse_integer_state(family, rounds, states, ordinals, lanes)
                np.testing.assert_array_equal(traced, seeds)

    def test_xorshift_inverse(self):
        values = np.array([0, 1, 0x12345678, 0xFFFFFFFF], dtype=np.uint32)
        for shift in (7, 9, 11, 13):
            mutated = values ^ (values >> np.uint32(shift))
            np.testing.assert_array_equal(undo_xor_right(mutated, shift), values)

    def test_jump_ahead_matches_steps(self):
        seed = 0x12345678
        for steps in range(20):
            value = seed
            for _ in range(steps): value = (value * 0x9E3779B1 + 0x7F4A7C15) & 0xFFFFFFFF
            jump_a, jump_b = affine_jump(0x9E3779B1, 0x7F4A7C15, steps)
            self.assertEqual(value, (jump_a * seed + jump_b) & 0xFFFFFFFF)

    def test_projection_determinism(self):
        target = np.random.default_rng(9).normal(size=(8, 32)).astype(np.float32)
        ordinals = np.arange(8, dtype=np.uint32)
        for family in FAMILIES:
            np.testing.assert_array_equal(projected_seeds(target, family, 2, ordinals), projected_seeds(target, family, 2, ordinals))

    def test_descriptor_accounting(self):
        fixed = descriptor_accounting("M1_LANE_AFFINE", 32, 32768, False); mixed = descriptor_accounting("M1_LANE_AFFINE", 32, 32768, True)
        self.assertEqual(fixed["descriptor_bits_per_segment"], 48)
        self.assertEqual(mixed["descriptor_bits_per_segment"], 51)
        self.assertEqual(fixed["position_bits_per_segment"], 0)
        self.assertGreater(mixed["total_true_bytes"], fixed["total_true_bytes"])

    def test_position_derivation(self):
        seeds = np.array([[123]], dtype=np.uint32)
        left = generated_raw("M2_AFFINE_XOR", 2, seeds, np.array([5], np.uint32), 8, True)
        right = generated_raw("M2_AFFINE_XOR", 2, seeds, np.array([6], np.uint32), 8, True)
        self.assertFalse(np.array_equal(left, right))

    def test_mutation_round_handling(self):
        seeds = np.array([[123]], dtype=np.uint32); ordinals = np.array([5], np.uint32)
        outputs = [generated_raw("M1_LANE_AFFINE", rounds, seeds, ordinals, 8) for rounds in ROUNDS]
        self.assertEqual(len(outputs), 5)
        self.assertTrue(all(not np.array_equal(outputs[0], value) for value in outputs[1:]))

    def test_tensor_orientation_and_metrics(self):
        weights = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
        vectors = np.ones((2, 8), dtype=np.float32)
        reference = vectors @ weights.T
        self.assertEqual(reference.shape, (2, 3))
        metrics = action_metrics(reference, reference * .5)
        self.assertAlmostEqual(metrics["mean_relative_l2"], .5)

    def test_untouched_split_isolation(self):
        validation, test = vector_split(69)
        self.assertFalse(set(validation) & set(test))
        self.assertEqual((len(validation), len(test)), (34, 35))


if __name__ == "__main__":
    unittest.main()
