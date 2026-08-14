#!/usr/bin/env python3
import csv
import json
import unittest
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "benchmark-results/vbuf-ml-step27-loader-scaling"

class Step27Tests(unittest.TestCase):
    def test_matrix_and_sample_counts(self):
        with (OUT / "phase-summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({(r["model"], r["format"], r["cache"]) for r in rows}, {(m, f, c) for m in ("0.6B", "32B") for f in ("GGUF", "vBuf") for c in ("warm", "uncached-approx")})
        self.assertTrue(all(int(r["samples"]) == (10 if r["cache"] == "warm" else 3) for r in rows))
        self.assertEqual(len((OUT / "run-order.csv").read_text().splitlines()) - 1, 52)

    def test_attribution_and_exact_generation(self):
        correctness = json.loads((OUT / "correctness-summary.json").read_text())
        self.assertTrue(correctness["artifact_hashes_verified"])
        self.assertTrue(correctness["warm_generation_sequences_equal_by_model"])
        attribution = json.loads((OUT / "attribution-summary.json").read_text())
        self.assertTrue(attribution["model_ready_delta_repeatable"])
        self.assertFalse(attribution["ttfuc_advantage_survives"])
        self.assertTrue(attribution["work_deferred_past_model_ready"])

    def test_required_evidence(self):
        for name in ("qualification-config.json", "environment.json", "artifact-provenance.json", "warm-raw.csv", "uncached-approx-raw.csv", "phase-summary.csv", "fault-summary.csv", "io-summary.csv", "memory-summary.csv", "cpu-summary.csv", "small-vs-large-summary.csv", "scaling-ratios.json", "correctness-summary.json", "attribution-summary.json"):
            self.assertTrue((OUT / name).exists(), name)

if __name__ == "__main__": unittest.main()
