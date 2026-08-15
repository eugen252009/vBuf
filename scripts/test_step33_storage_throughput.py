#!/usr/bin/env python3
import csv
import json
import unittest
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "benchmark-results/vbuf-ml-storage-throughput"


class StorageThroughputTests(unittest.TestCase):
    def test_payload_and_required_artifacts(self):
        source = json.loads((OUT / "source-manifest.json").read_text())
        self.assertEqual(source["payload_end"] - source["payload_start"], source["payload_bytes"])
        for name in ("qualification-report.md", "qualification.json", "environment.json", "dd-baseline.csv", "direct-io-baseline.csv", "sequential-reader.csv", "worker-count-sweep.csv", "chunk-size-sweep.csv", "uncapped-loader.csv", "capped-loader.csv", "syscall-counts.csv", "access-order.csv", "cap-blocking.csv", "storage-utilization.csv", "residency-methods.csv", "range-coalescing.csv", "end-to-end-strategies.csv", "expected-full-load-times.csv"):
            self.assertTrue((OUT / name).exists(), name)

    def test_reader_completeness_and_accounting(self):
        with (OUT / "sequential-reader.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 7)
        payload = json.loads((OUT / "source-manifest.json").read_text())["payload_bytes"]
        self.assertTrue(all(int(r["successful_bytes"]) == payload and int(r["errors"]) == 0 and int(r["short_reads"]) == 0 for r in rows))
        self.assertTrue(all(int(r["physical_read_bytes"]) >= payload for r in rows))

    def test_chunk_and_worker_controls(self):
        with (OUT / "worker-count-sweep.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([int(r["workers"]) for r in rows], [1, 2, 4, 8])
        with (OUT / "syscall-counts.csv").open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            self.assertGreater(float(row["average_bytes_per_syscall"]), 0)

    def test_classification_is_conservative(self):
        q = json.loads((OUT / "qualification.json").read_text())
        self.assertIn(q["storage_ceiling"], ("STORAGE_BASELINE_CONFIRMED", "STORAGE_BASELINE_NOT_REPRODUCED"))
        if q["storage_ceiling"] == "STORAGE_BASELINE_NOT_REPRODUCED":
            self.assertEqual(q["root_cause"], "ROOT_CAUSE_NOT_ISOLATED")


if __name__ == "__main__":
    unittest.main()
