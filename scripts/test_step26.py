#!/usr/bin/env python3
import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark-results/vbuf-ml-step26-qwen32b-placement"

class Step26Tests(unittest.TestCase):
    def test_all_base_shifts_and_selection(self):
        with (OUT / "candidate-layouts.csv").open() as stream: rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["base_shift"]) for row in rows], list(range(3, 9)))
        selection = json.loads((OUT / "selection.json").read_text())
        self.assertEqual(selection["selected_base_shift"], 3)
        self.assertEqual(selection["selected_base_step"], 8)

    def test_plan_is_exact_and_payloads_are_not_duplicated(self):
        manifest = json.loads((OUT / "qwen3-32b-manifest.json").read_text())
        selection = json.loads((OUT / "selection.json").read_text())
        with (OUT / "candidate-layouts.csv").open() as stream: rows = list(csv.DictReader(stream))
        selected = next(row for row in rows if int(row["base_shift"]) == selection["selected_base_shift"])
        plan = manifest["placement_plan"]
        self.assertEqual(plan["base_shift"], selection["selected_base_shift"])
        self.assertEqual(plan["final_size"], int(selected["final_size"]))
        self.assertEqual(len(plan["entries"]), int(selected["block_count"]))
        self.assertEqual(json.loads((OUT / "qualification-config.json").read_text())["payload_duplication"], 0)

    def test_dense_qwen3_provenance(self):
        provenance = json.loads((OUT / "artifact-provenance.json").read_text())
        self.assertEqual(provenance["architecture"], "qwen3")
        self.assertTrue(provenance["dense"])
        self.assertEqual(provenance["layer_count"], 64)
        self.assertEqual(provenance["tensor_count"], 707)
        self.assertEqual(provenance["vocabulary_size"], 151936)
        self.assertEqual(provenance["merge_count"], 151387)

if __name__ == "__main__": unittest.main()
