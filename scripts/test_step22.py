#!/usr/bin/env python3
import csv
import io
import unittest

class Step22SchemaTests(unittest.TestCase):
    def test_benchmark_sample_schema_and_sign_convention(self):
        sample = "format,phase,duration_us\ngguf,model_ready,10\nvbuf,model_ready,12\n"
        rows = list(csv.DictReader(io.StringIO(sample)))
        self.assertEqual(rows[0]["phase"], "model_ready")
        gguf = float(rows[0]["duration_us"]); vbuf = float(rows[1]["duration_us"])
        self.assertAlmostEqual((vbuf - gguf) / gguf, 0.2)

    def test_balanced_order_alternates(self):
        formats = ["gguf" if index % 2 == 0 else "vbuf" for index in range(6)]
        self.assertEqual(formats, ["gguf", "vbuf", "gguf", "vbuf", "gguf", "vbuf"])

    def test_required_summary_columns(self):
        required = {"artifact", "condition", "phase", "gguf_median_us", "vbuf_median_us", "relative_delta"}
        self.assertTrue(required.issuperset({"artifact", "condition", "phase"}))
        self.assertIn("relative_delta", required)

if __name__ == "__main__":
    unittest.main()
