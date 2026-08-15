#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_ccc_c4_hard_gate import CANONICAL_BLOCKS, lloyd, pareto, vector_split
from qualify_ccc_geometric import byte_account, geometric_levels


class C4HardGateTests(unittest.TestCase):
    def test_free_codebook_is_deterministic_and_convergent(self):
        values = np.random.default_rng(1).normal(size=8192).astype(np.float32)
        left = lloyd(values, 16, "quantile")
        right = lloyd(values, 16, "quantile")
        np.testing.assert_array_equal(left[0], right[0])
        self.assertLessEqual(left[1], 512)
        self.assertGreaterEqual(left[3], 0.0)

    def test_c4_no_zero_geometry_is_unique_and_monotonic(self):
        levels, _ = geometric_levels(4, "no_zero", 1.35, 1.0)
        self.assertEqual(len(np.unique(levels)), 16)
        self.assertFalse(np.any(levels == 0))
        self.assertTrue(np.all(np.diff(np.sort(levels)) > 0))

    def test_true_bpw_accounting(self):
        accounting = byte_account(5_242_880, 4, 4096, 8)
        self.assertEqual(accounting["payload_bytes"], 2_621_440)
        self.assertGreater(accounting["true_bpw"], 4.0)

    def test_canonical_block_geometries(self):
        self.assertEqual(CANONICAL_BLOCKS["Q4_0"], (32, 18))
        self.assertEqual(CANONICAL_BLOCKS["Q4_K"], (256, 144))
        self.assertEqual(CANONICAL_BLOCKS["IQ4_XS"], (256, 136))

    def test_functional_split_isolated(self):
        validation, test = vector_split(256)
        self.assertEqual(len(set(validation) & set(test)), 0)
        self.assertEqual(len(validation) + len(test), 256)
        self.assertEqual(len(validation), 128)
        self.assertEqual(len(test), 128)

    def test_pareto_classification(self):
        rows = [{"name":"a","true_bpw":4.,"error":.1},{"name":"b","true_bpw":5.,"error":.2}]
        pareto(rows, "error")
        self.assertEqual(rows[0]["pareto"], "PARETO")
        self.assertEqual(rows[1]["pareto"], "DOMINATED")


if __name__ == "__main__":
    unittest.main()
