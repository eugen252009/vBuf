#!/usr/bin/env python3
import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class Step22ATests(unittest.TestCase):
    def test_phase_schema_and_sample_count(self):
        path = ROOT / "benchmark-results/vbuf-ml-step22a/phase-summary.csv"
        if not path.exists(): self.skipTest("diagnostic evidence absent")
        with path.open() as stream: rows = list(csv.DictReader(stream))
        self.assertTrue(any(row["phase"] == "canonical_v06" for row in rows))
        self.assertTrue(all(int(row["samples"]) == 10 for row in rows))

    def test_ffi_call_count_is_machine_recorded(self):
        path = ROOT / "benchmark-results/vbuf-ml-step22a/ffi-summary.csv"
        if not path.exists(): self.skipTest("diagnostic evidence absent")
        with path.open() as stream: rows = list(csv.DictReader(stream))
        self.assertEqual({row["artifact"] for row in rows}, {"BF16", "Q8_0"})
        self.assertGreater(int(rows[0]["ffi_calls"]), 900_000)

    def test_baseline_directory_is_separate(self):
        self.assertTrue((ROOT / "benchmark-results/vbuf-ml-step22/phase-summary.csv").exists())
        self.assertTrue((ROOT / "benchmark-results/vbuf-ml-step22a/phase-summary.csv").exists())

if __name__ == "__main__": unittest.main()
