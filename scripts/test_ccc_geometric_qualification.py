#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_ccc_geometric import geometric_levels, split_positions, validation_winner


class CCCQualificationTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_complete(self):
        first = split_positions(100_000)
        second = split_positions(100_000)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.size, sum(int((first == part).sum()) for part in (0, 1, 2)))
        self.assertTrue(all(np.any(first == part) for part in (0, 1, 2)))
        self.assertTrue(0.68 < np.mean(first == 0) < 0.72)
        self.assertTrue(0.13 < np.mean(first == 1) < 0.17)
        self.assertTrue(0.13 < np.mean(first == 2) < 0.17)

    def test_c3_zero_layouts_have_expected_unique_states(self):
        no_zero, _ = geometric_levels(3, "no_zero", 1.35, 1.0)
        duplicate, _ = geometric_levels(3, "duplicate_zero", 1.35, 1.0)
        tail, tail_code = geometric_levels(3, "tail", 1.35, 1.0, 1, 2.0)
        self.assertEqual(len(np.unique(no_zero)), 8)
        self.assertEqual(len(np.unique(duplicate)), 7)
        self.assertEqual(len(np.unique(tail)), 8)
        self.assertEqual(tail_code, 4)
        self.assertEqual(tail[0], 0)

    def test_promotion_uses_validation_not_test_metrics(self):
        candidates = [
            {"name": "validation_winner", "validation": {"rmse": 0.1}, "test": {"rmse": 100.0}},
            {"name": "test_winner", "validation": {"rmse": 0.2}, "test": {"rmse": 0.0}},
        ]
        self.assertEqual(validation_winner(candidates)["name"], "validation_winner")


if __name__ == "__main__":
    unittest.main()
