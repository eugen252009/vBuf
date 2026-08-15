#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_lut_predictive_correction import (
    TargetSpec, balanced_classes, class_assignments, make_prediction,
    pack_match_exceptions, restore_payload, unpack_match_exceptions,
)


class LutPredictiveCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(19)
        self.codes = self.rng.integers(0, 16, size=(8, 4, 8), dtype=np.uint8)
        self.metadata = self.rng.integers(0, 256, size=(8, 4, 4), dtype=np.uint8)

    def test_balanced_class_encoding_is_deterministic(self):
        features = np.arange(32, dtype=np.int32).reshape(8, 4)
        np.testing.assert_array_equal(balanced_classes(features, 4), balanced_classes(features, 4))

    def test_row_and_column_assignments(self):
        rows, columns = class_assignments(self.codes, 4, 2)
        self.assertEqual(rows.shape, (8,)); self.assertEqual(columns.shape, (4,))
        self.assertLessEqual(int(rows.max()), 3); self.assertLessEqual(int(columns.max()), 1)

    def test_prediction_lut_determinism(self):
        rows, columns = class_assignments(self.codes, 4, 2)
        left, left_bytes = make_prediction(self.codes, rows, columns, "L3")
        right, right_bytes = make_prediction(self.codes, rows, columns, "L3")
        np.testing.assert_array_equal(left, right); self.assertEqual(left_bytes, right_bytes)

    def test_match_exception_roundtrip(self):
        prediction = np.zeros_like(self.codes)
        _, _, payload, _ = pack_match_exceptions(self.codes, prediction, 8, 32)
        restored = unpack_match_exceptions(prediction, payload, 8, 32)
        np.testing.assert_array_equal(restored, self.codes)

    def test_xor_correction_is_exactly_reversible(self):
        prediction = np.ones_like(self.codes)
        correction = np.bitwise_xor(self.codes, prediction)
        np.testing.assert_array_equal(np.bitwise_xor(correction, prediction), self.codes)

    def test_payload_metadata_roundtrip(self):
        spec = TargetSpec("test", 8, 2, 4, 1, 4)
        restored = restore_payload(self.codes[:, :, :4], self.metadata, spec)
        self.assertEqual(len(restored), 8 * 4 * 8)
        unpacked = np.frombuffer(restored, dtype=np.uint8).reshape(8, 4, 8)
        np.testing.assert_array_equal(unpacked[:, :, 2:6], self.codes[:, :, :4])
        np.testing.assert_array_equal(unpacked[:, :, :2], self.metadata[:, :, :2])
        np.testing.assert_array_equal(unpacked[:, :, 6:], self.metadata[:, :, 2:])

    def test_orientation_and_no_activation_dependency(self):
        self.assertEqual(self.codes.shape[:2], (8, 4))
        self.assertNotIn("activation", __import__("qualify_lut_predictive_correction").__dict__)


if __name__ == "__main__":
    unittest.main()
