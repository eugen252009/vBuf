#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_implicit_weight_feasibility import (
    Encoded, action_metrics, descriptor_layout, encode_fixed, exact_segmentation,
    generator_codebook, reconstruct, select_best_fixed, vector_split, validate_orientation,
)


class ImplicitWeightFeasibilityTests(unittest.TestCase):
    def test_generators_are_deterministic(self):
        for family in ("G0_HASH_SIGN", "G1_HASH_AFFINE", "G2_CENTER_DENSE", "G3_RECURRENCE", "G4_TINY_BASIS"):
            np.testing.assert_array_equal(generator_codebook(family, 32), generator_codebook(family, 32))

    def test_same_descriptor_same_values(self):
        values = np.random.default_rng(4).normal(size=256).astype(np.float32)
        encoded = encode_fixed(values, "G0_HASH_SIGN", 16, batch_segments=4)
        np.testing.assert_array_equal(reconstruct(encoded), reconstruct(encoded))

    def test_position_changes_position_aware_generator(self):
        seeds = np.arange(8, dtype=np.uint32)[:, None]
        local = np.arange(16, dtype=np.uint32)[None, :]
        from qualify_implicit_weight_feasibility import mix32
        left = mix32(seeds * np.uint32(0x9E3779B9) + local * np.uint32(0x85EBCA6B))
        right = mix32(seeds * np.uint32(0x9E3779B9) + (local + 16) * np.uint32(0x85EBCA6B))
        self.assertFalse(np.array_equal(left, right))

    def test_descriptor_bit_accounting(self):
        g0 = descriptor_layout("G0_HASH_SIGN", 16, 1024)
        g1 = descriptor_layout("G1_HASH_AFFINE", 16, 1024)
        self.assertEqual(g0["descriptor_bits_per_segment"], 24)
        self.assertEqual(g0["descriptor_bytes"], 192)
        self.assertEqual(g1["descriptor_bits_per_segment"], 40)
        self.assertEqual(g1["descriptor_bytes"], 320)
        self.assertEqual(g0["total_true_bytes"] * 8 / 1024, g0["true_bpw"])

    def test_segment_boundary_handling(self):
        values = np.arange(64, dtype=np.float32)
        encoded = encode_fixed(values, "G1_HASH_AFFINE", 8, batch_segments=3)
        self.assertEqual(reconstruct(encoded).shape, values.shape)
        with self.assertRaises(ValueError):
            encode_fixed(values[:-1], "G0_HASH_SIGN", 8)

    def test_adaptive_segmentation_exact_coverage(self):
        costs = {(0, 4): 1.0, (0, 8): 3.0, (4, 4): 1.0, (8, 4): 1.0, (8, 8): 1.5, (12, 4): 1.0}
        segments = exact_segmentation(16, (4, 8), costs)
        self.assertEqual(segments, [(0, 4), (4, 4), (8, 8)])
        self.assertEqual(sum(length for _, length in segments), 16)
        for left, right in zip(segments, segments[1:]):
            self.assertEqual(left[0] + left[1], right[0])

    def test_validation_test_isolation(self):
        validation, test = vector_split(69)
        self.assertFalse(set(validation) & set(test))
        self.assertEqual(len(validation), 34)
        self.assertEqual(len(test), 35)
        rows = [{"candidate": "a", "true_bpw": 1.5, "validation_mean_relative_l2": .3},
                {"candidate": "b", "true_bpw": 1.0, "validation_mean_relative_l2": .4}]
        self.assertEqual(select_best_fixed(rows), "a")

    def test_tensor_orientation(self):
        weights = np.arange(1024 * 5120, dtype=np.float32).reshape(1024, 5120) / 1e6
        vectors = np.ones((2, 5120), dtype=np.float32)
        self.assertTrue(validate_orientation(weights, vectors)["verified"])

    def test_functional_metric_aggregation(self):
        reference = np.array([[3., 4.], [0., 2.]], dtype=np.float32)
        candidate = np.array([[0., 4.], [0., 1.]], dtype=np.float32)
        result = action_metrics(reference, candidate)
        expected = np.mean((3 / 5, 1 / 2))
        self.assertAlmostEqual(result["mean_relative_l2"], expected)
        self.assertGreaterEqual(result["mean_cosine"], 0.0)


if __name__ == "__main__":
    unittest.main()
