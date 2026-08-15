#!/usr/bin/env python3
import csv
import json
import unittest
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "benchmark-results/vbuf-ml-loader-residency"


class Step33Tests(unittest.TestCase):
    def test_required_artifacts(self):
        names = [
            "qualification-report.md", "qualification.json", "environment.json",
            "source-manifest.json", "storage-device.json", "baseline-decomposition.csv",
            "device-sequential-baseline.csv", "strategy-summary.csv", "cold-runs.csv",
            "warm-runs.csv", "timeline.csv", "residency.csv", "compute-stalls.csv",
            "chunk-size-sweep.csv", "queue-depth-sweep.csv", "range-coalescing.csv",
            "first-token.csv", "loader-efficiency.csv",
        ]
        for name in names:
            self.assertTrue((OUT / name).exists(), name)

    def test_qualification_and_output_equivalence(self):
        qualification = json.loads((OUT / "qualification.json").read_text())
        self.assertEqual(qualification["status"], "PASS")
        self.assertTrue(qualification["output_equivalence"])
        with (OUT / "strategy-summary.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(rows)
        self.assertTrue(all(row["output_match"] == "True" for row in rows))

    def test_loader_has_no_errors(self):
        with (OUT / "loader-efficiency.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(rows)
        self.assertTrue(all(row["errors"] == "0" for row in rows))

    def test_timeline_and_residency(self):
        with (OUT / "timeline.csv").open() as f:
            timeline = list(csv.DictReader(f))
        self.assertTrue(timeline)
        for row in timeline:
            self.assertLessEqual(float(row["COMPUTE_STARTED"]), float(row["GENERATION_COMPLETE"]))
        with (OUT / "residency.csv").open() as f:
            residency = list(csv.DictReader(f))
        self.assertTrue(residency)

    def test_conditional_sweeps_are_explicit(self):
        for name in ("chunk-size-sweep.csv", "queue-depth-sweep.csv", "range-coalescing.csv"):
            with (OUT / name).open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["status"], "NOT_REACHED")


if __name__ == "__main__":
    unittest.main()
