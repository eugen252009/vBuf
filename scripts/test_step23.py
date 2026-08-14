#!/usr/bin/env python3
import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class Step23Tests(unittest.TestCase):
    def test_three_way_runtime_schema(self):
        path = ROOT / "benchmark-results/vbuf-ml-step23/runtime-summary.csv"
        if not path.exists(): self.skipTest("Step-23 evidence absent")
        with path.open() as stream: rows = list(csv.DictReader(stream))
        self.assertEqual({r["path"] for r in rows}, {"gguf", "compatibility", "direct"})
        self.assertTrue(all(int(r["samples"]) == 10 for r in rows))

    def test_direct_source_has_no_gguf_context(self):
        source = (ROOT / "integrations/llama.cpp/vbuf_direct_source.cpp").read_text()
        self.assertNotIn("gguf_context", source)
        self.assertTrue("vbuf_ml_consumer_token_views" in source or "vbuf_ml_consumer_token_arrays" in source)
        self.assertTrue("vbuf_ml_consumer_merge_views" in source or "vbuf_ml_consumer_merge_arrays" in source)
        self.assertIn("vbuf_ml_consumer_tensor_views", source)

    def test_baselines_remain_present(self):
        for name in ("vbuf-ml-step21", "vbuf-ml-step22", "vbuf-ml-step22a"):
            self.assertTrue((ROOT / "benchmark-results" / name).exists())

if __name__ == "__main__": unittest.main()
