#!/usr/bin/env python3
import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark-results/vbuf-ml-nano-runtime-audit"

class NanoAuditTests(unittest.TestCase):
    def test_geometry_is_physical_only(self):
        data = json.loads((OUT / "artifact-geometry.json").read_text())
        self.assertEqual(set(data), {"BF16", "Q8_0"})
        for geometry in data.values():
            self.assertEqual(geometry["set_bits"], geometry["blocks"])
            self.assertEqual(geometry["continuations"], 0)
            self.assertGreater(geometry["nano_bytes"], geometry["set_bits"])

    def test_dense_tokenizer_classification(self):
        with (OUT / "classification.csv").open() as stream: rows = list(csv.DictReader(stream))
        by_name = {row["structure"]: row for row in rows}
        self.assertEqual(by_name["Token text/offset/type/score"]["classification"], "DIRECT_VIEW_READY")
        self.assertEqual(by_name["Merge IDs"]["classification"], "DIRECT_VIEW_READY_WITH_RUNTIME_INDEX")
        self.assertEqual(by_name["Tensor payloads"]["classification"], "DIRECT_VIEW_READY")

    def test_no_wire_or_runtime_implementation(self):
        config = json.loads((OUT / "qualification-config.json").read_text())
        self.assertFalse(config["native_runtime_implemented"])
        self.assertFalse(config["wire_change"])

if __name__ == "__main__": unittest.main()
