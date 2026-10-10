#!/usr/bin/env python3
"""Offline integrity and comparison audit for the Qwen3-14B native-AV evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROP = ROOT / "research/results/vbuf-ml-integration/qwen3-numerical-propagation"
NATIVE = ROOT / "research/results/vbuf-ml-integration/qwen3-native-layout-av"
ALLOWED = {"PASS", "FAIL", "NOT_APPLICABLE", "NOT_TESTED", "INVALID_EVIDENCE"}

INVENTORY_FIELDS = [
    "record_id", "fixture_id", "comparison", "capacity", "phase_or_position",
    "history_identity", "paired_inputs_identical", "paired_topology_identical",
    "status", "historical_result", "repeats", "evidence", "notes",
]
CAPTURE_FIELDS = [
    "capture_id", "capacity", "phase", "location", "metadata_sha256",
    "binary_sha256", "binary_bytes", "expected_bytes", "geometry",
    "same_binary_as_primary", "identity_provenance_note",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ints(value: str, separator: str = ";") -> list[int]:
    return [int(part) for part in value.split(separator) if part]


def status_from_gate(value: str) -> str:
    if value == "PASS":
        return "PASS"
    if value == "FAIL":
        return "FAIL"
    if value == "NOT_APPLICABLE":
        return "NOT_APPLICABLE"
    if value in {"NOT_RUN_AFTER_STOP", "UNQUALIFIED", "NOT_RUN"}:
        return "NOT_TESTED"
    raise ValueError(f"unrecognized evidence status {value!r}")


def passes_existing_gate(row: dict[str, str]) -> bool:
    return (
        float(row["final_hidden_relative_rms"]) <= 0.02
        and float(row["final_hidden_cosine"]) >= 0.9998
        and float(row["final_logits_relative_rms"]) <= 0.02
        and float(row["final_logits_cosine"]) >= 0.9998
    )


def inventory_record(**values: Any) -> dict[str, str]:
    row = {field: "" for field in INVENTORY_FIELDS}
    for key, value in values.items():
        if key not in row:
            raise ValueError(f"unknown inventory field {key}")
        row[key] = str(value)
    if row["status"] not in ALLOWED:
        raise ValueError(f"invalid status {row['status']!r} in {row['record_id']}")
    return row


def read_matrix_manifest(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="|"))


def parse_repeat_lines(path: Path) -> list[dict[str, str]]:
    pattern = re.compile(r"fixture_repeat id=(\S+) repeat=(\d+) run=(\S+) status=(\S+)(.*)")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        fields = dict(re.findall(r"([A-Za-z_]+)=([^\s]+)", match.group(5)))
        rows.append({
            "fixture_id": match.group(1), "repeat": match.group(2),
            "run": match.group(3), "raw_status": match.group(4), **fields,
        })
    return rows


def add_matrix_inventory(records: list[dict[str, str]]) -> dict[str, Any]:
    manifest = read_matrix_manifest(PROP / "qwen3-native-av-fixture-matrix-v1.tsv")
    raw_path = PROP / "raw/autoregressive-fixture-matrix-20261009/matrix/fixture_summary.csv"
    raw = read_csv(raw_path)
    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        by_id[row["fixture_id"]].append(row)
    repeatability_path = PROP / "raw/autoregressive-fixture-matrix-20261009/matrix/repeatability.csv"
    repeatability = read_csv(repeatability_path)
    if len(repeatability) != 2 or any(
        row["attempts"] != "2" or row["status"] != "PASS"
        or row["canonical_tokens_repeatable"] != "yes"
        or row["candidate_tokens_repeatable"] != "yes"
        or row["canonical_capture_repeatable"] != "yes"
        or row["candidate_capture_repeatable"] != "yes"
        for row in repeatability
    ):
        raise ValueError("original matrix baseline token/capture repeatability failed")
    planned_ids = {row["fixture_id"] for row in manifest}
    if len(manifest) != 20 or set(by_id) != planned_ids:
        raise ValueError("original 20-row manifest and matrix summary do not reconcile")

    baseline = "baseline-existing-natural-repeat-p8-g25-cap512"
    baseline_rows = by_id[baseline]
    if len(baseline_rows) != 4:
        raise ValueError("baseline should contain two modes repeated twice")
    modes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in baseline_rows:
        modes[row["run_kind"]].append(row)
    if set(modes) != {"FREE_RUNNING", "COMMON_TOKEN_REPLAY"}:
        raise ValueError("baseline free-running/common-replay modes are incomplete")
    manifest_row = next(row for row in manifest if row["fixture_id"] == baseline)
    free = modes["FREE_RUNNING"]
    replay = modes["COMMON_TOKEN_REPLAY"]
    if any(row["numeric_gate"] != "PASS" for row in free) or any(row["numeric_gate"] != "FAIL" for row in replay):
        raise ValueError("baseline PASS/FAIL statuses changed; review the qualification evidence")
    full_free_history = ints(manifest_row["prompt_ids"], ",") + ints(free[0]["canonical_generated_tokens"])
    replay_history = ints(replay[0]["input_token_ids"])
    if full_free_history != replay_history:
        raise ValueError("common-token replay is not the exact free-running 33-token history")
    if any(row["same_history"] != "yes" for row in baseline_rows):
        raise ValueError("baseline candidate/reference history is not marked identical")
    for run_kind, rows in modes.items():
        if len(rows) != 2 or any(
            row[field] != rows[0][field] for row in rows[1:]
            for field in ("canonical_token_hash", "candidate_token_hash", "canonical_capture_hash", "candidate_capture_hash")
        ):
            raise ValueError(f"baseline {run_kind} repeat hashes differ")
        if run_kind == "FREE_RUNNING" and any(
            row[field] != rows[0][field] for row in rows[1:]
            for field in ("canonical_generated_tokens", "candidate_generated_tokens")
        ):
            raise ValueError("free-running baseline generated token histories differ across repeats")

    for run_kind, rows in modes.items():
        status = status_from_gate(rows[0]["numeric_gate"])
        if any(status_from_gate(row["numeric_gate"]) != status for row in rows):
            raise ValueError(f"baseline {run_kind} repeat statuses disagree")
        records.append(inventory_record(
            record_id=f"matrix-{run_kind.lower()}", fixture_id=baseline,
            comparison="candidate versus canonical final hidden/logits",
            capacity=512, phase_or_position="incremental decode through position 32" if run_kind == "FREE_RUNNING" else "32-row prefill + decode position 32",
            history_identity="exact 33-token history; prompt + generated IDs verified against common replay",
            paired_inputs_identical="YES", paired_topology_identical="YES", status=status,
            historical_result=status, repeats=2,
            evidence=str(raw_path.relative_to(ROOT)),
            notes="The two modes use the same token history but different segmentation; do not compare their statuses as an identical-topology A/B.",
        ))

    untested = 0
    for planned in manifest:
        fixture = planned["fixture_id"]
        if fixture == baseline:
            continue
        rows = by_id[fixture]
        if not rows or any(row["status"] != "NOT_RUN_AFTER_STOP" for row in rows):
            raise ValueError(f"planned fixture {fixture} was not preserved as NOT_RUN_AFTER_STOP")
        untested += 1
        records.append(inventory_record(
            record_id=f"matrix-unrun-{fixture}", fixture_id=fixture,
            comparison=planned["mode"], capacity=planned["capacity"],
            phase_or_position=planned["input_mode"], history_identity="manifest token IDs; execution not performed",
            paired_inputs_identical="NOT_TESTED", paired_topology_identical="NOT_TESTED",
            status="NOT_TESTED", historical_result="NOT_TESTED", repeats=0,
            evidence=str(raw_path.relative_to(ROOT)),
            notes="Original stop-on-first-failure matrix row; later standalone follow-ups do not backfill this row.",
        ))
    if untested != 19:
        raise ValueError(f"expected 19 original matrix fixtures not run, found {untested}")
    return {"planned": len(manifest), "original_unrun": untested, "baseline_free_history": len(full_free_history)}


def add_phase_isolation(records: list[dict[str, str]]) -> None:
    base = PROP / "raw/common-token-phase-isolation-20261009"
    summary_path = base / "fixture_summary.csv"
    rows = read_csv(summary_path)
    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_id[row["fixture_id"]].append(row)
    if len(by_id) != 2 or any(len(group) != 2 for group in by_id.values()):
        raise ValueError("phase-isolation fixture repeats are incomplete")
    histories = {row["canonical_token_hash"] for row in rows}
    token_lists = {row["input_token_ids"] for row in rows}
    if len(histories) != 1 or len(token_lists) != 1 or any(row["same_history"] != "yes" for row in rows):
        raise ValueError("phase isolation does not use one exactly matched token history")
    repeat_path = base / "repeatability.csv"
    repeat_rows = read_csv(repeat_path)
    if len(repeat_rows) != 2 or any(
        row["canonical_capture_repeatable"] != "yes" or row["candidate_capture_repeatable"] != "yes"
        for row in repeat_rows
    ):
        raise ValueError("phase-isolation capture repeatability failed")
    for fixture, group in by_id.items():
        raw_statuses = {row["status"] for row in group}
        stable_rows = [{key: value for key, value in row.items() if key != "repeat"} for row in group]
        if len(raw_statuses) != 1 or stable_rows[0] != stable_rows[1]:
            raise ValueError(f"phase-isolation repeats disagree for {fixture}")
        status = "FAIL" if raw_statuses == {"KNOWN_NEGATIVE_REPRODUCED"} else status_from_gate(group[0]["status"])
        phase = "native prefill only; canonical decode" if "prefill" in fixture else "canonical prefill; native decode only"
        records.append(inventory_record(
            record_id=f"phase-isolation-{fixture}", fixture_id=fixture,
            comparison="candidate versus canonical final hidden/logits",
            capacity=512, phase_or_position=phase, history_identity=f"token-hash:{next(iter(histories))}",
            paired_inputs_identical="YES", paired_topology_identical="YES", status=status,
            historical_result=status, repeats=2, evidence=str(summary_path.relative_to(ROOT)),
            notes="Both phase trials replay the exact same 33 input IDs, capacity, and position-32 endpoint; only the candidate AV phase differs.",
        ))


def add_existing_propagation(records: list[dict[str, str]]) -> None:
    base = PROP / "raw/propagation-20261008-final"
    path = base / "native_candidate_summary.csv"
    for row in read_csv(path):
        status = status_from_gate(row["numeric_gate"])
        records.append(inventory_record(
            record_id=f"propagation-{row['case']}", fixture_id="natural-repeat-p32-cap512-propagation",
            comparison="candidate versus canonical final hidden/logits", capacity=512,
            phase_or_position=f"{row['case']} at position {row['position']}",
            history_identity="natural-repeat 32-token prompt; fixed recorded runner fixture",
            paired_inputs_identical="YES", paired_topology_identical="YES", status=status,
            historical_result=status, repeats=1, evidence=str(path.relative_to(ROOT)),
            notes="Qualification diagnostic only; pass does not validate candidate or offset separate failures.",
        ))

    intervention_path = base / "single_layer_summary.csv"
    intervention_rows = read_csv(intervention_path)
    if len(intervention_rows) != 20:
        raise ValueError("expected the 5-layer x 4-position intervention grid")
    for row in intervention_rows:
        status = status_from_gate(row["existing_numeric_gate"])
        records.append(inventory_record(
            record_id=f"intervention-layer{row['intervention_layer']}-position{row['position']}",
            fixture_id="natural-repeat-p8-g25-single-layer-intervention",
            comparison="single native-AV intervention versus canonical output",
            capacity=512, phase_or_position=f"decode position {row['position']}; layer {row['intervention_layer']}",
            history_identity="fixed natural-repeat token path; intervention AV inputs verified bitwise equal",
            paired_inputs_identical="YES", paired_topology_identical="YES", status=status,
            historical_result=status, repeats=1, evidence=str(intervention_path.relative_to(ROOT)),
            notes="One of 20 isolated layer/position interventions; does not represent full-candidate selection.",
        ))


def add_topology_and_prefix(records: list[dict[str, str]]) -> None:
    base = PROP / "raw/native-av-divergence-topology-final-20261009-070840"
    topology_path = base / "topology_comparison.csv"
    topology = read_csv(topology_path)
    if len(topology) != 8:
        raise ValueError("expected four topology comparisons repeated twice")
    topology_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in topology:
        topology_groups[row["comparison"]].append(row)
    for comparison, group in topology_groups.items():
        if len(group) != 2 or {row["repeat"] for row in group} != {"1", "2"}:
            raise ValueError(f"topology repeat records incomplete for {comparison}")
        stable_rows = [{key: value for key, value in row.items() if key != "repeat"} for row in group]
        if stable_rows[0] != stable_rows[1]:
            raise ValueError(f"topology results differ across repeats for {comparison}")
    for row in topology:
        status = "PASS" if passes_existing_gate({
            "final_hidden_relative_rms": row["hidden_relative_rms"],
            "final_hidden_cosine": row["hidden_cosine"],
            "final_logits_relative_rms": row["logits_relative_rms"],
            "final_logits_cosine": row["logits_cosine"],
        }) else "FAIL"
        arms = row["comparison"].split("_vs_")
        same_topology = len(arms) == 2 and arms[0].split("_", 1)[-1] == arms[1].split("_", 1)[-1]
        records.append(inventory_record(
            record_id=f"topology-{row['repeat']}-{row['comparison']}",
            fixture_id=row["fixture_id"], comparison=row["comparison"], capacity=512,
            phase_or_position=f"33-token endpoint; position {row['position']}",
            history_identity=f"token-hash:{row['token_hash']}", paired_inputs_identical="YES",
            paired_topology_identical="YES" if same_topology else "NO", status=status,
            historical_result=status, repeats=1,
            evidence=str(topology_path.relative_to(ROOT)),
            notes="Same token history. Canonical incremental-vs-batched is a topology control; native incremental-vs-batched isolates topology sensitivity, not an identical-graph A/B.",
        ))

    prefix_path = base / "prefix_equivalence.csv"
    prefix_rows = read_csv(prefix_path)
    if len(prefix_rows) != 6:
        raise ValueError("expected canonical/native long-vs-short prefix controls and short A/B, twice")
    prefix_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prefix_rows:
        prefix_groups[row["comparison"]].append(row)
    for comparison, group in prefix_groups.items():
        if len(group) != 2 or {row["repeat"] for row in group} != {"1", "2"}:
            raise ValueError(f"prefix repeat records incomplete for {comparison}")
        stable_rows = [{key: value for key, value in row.items() if key != "repeat"} for row in group]
        if stable_rows[0] != stable_rows[1]:
            raise ValueError(f"prefix results differ across repeats for {comparison}")
    for row in prefix_rows:
        comparison = row["comparison"]
        if comparison in {"canonical_incremental_long_vs_short", "native_incremental_long_vs_short"}:
            if row["hidden_bitwise_equal"] != "yes" or row["logits_bitwise_equal"] != "yes":
                raise ValueError("short/long prefix bitwise control failed")
            status = "PASS"
            note = "Harness/session reproducibility control: short and long runs have bitwise-identical arrays; not a candidate-versus-canonical pass."
        else:
            status = "PASS" if (
                float(row["logits_relative_rms"]) <= 0.02 and float(row["logits_cosine"]) >= 0.9998
                and float(row["hidden_relative_rms"]) <= 0.02 and float(row["hidden_cosine"]) >= 0.9998
            ) else "FAIL"
            note = "Same 9-token prefix and one-row position-8 topology; unchanged final-output gate applies."
        records.append(inventory_record(
            record_id=f"prefix-{row['repeat']}-{comparison}", fixture_id=row["fixture_id"],
            comparison=comparison, capacity=row["capacity"],
            phase_or_position=f"position {row['position']} after {row['prefix_tokens']} input tokens",
            history_identity=f"prefix-token-hash:{row['prefix_token_hash']}",
            paired_inputs_identical="YES", paired_topology_identical=row["step_geometry_equal"].upper(),
            status=status, historical_result=status, repeats=1,
            evidence=str(prefix_path.relative_to(ROOT)), notes=note,
        ))


def add_sequence_and_followups(records: list[dict[str, str]]) -> None:
    sequence_path = PROP / "raw/sequence-context32-20261009/run.log"
    line = next((line for line in sequence_path.read_text(encoding="utf-8").splitlines() if line.startswith("sequence_result ")), None)
    if line is None:
        raise ValueError("bounded autoregressive sequence result is missing")
    fields = dict(re.findall(r"([A-Za-z_]+)=([^\s]+)", line))
    repeat_path = PROP / "raw/sequence-context32-repeat-20261009/run.log"
    repeat_line = next((line for line in repeat_path.read_text(encoding="utf-8").splitlines() if line.startswith("sequence_result ")), None)
    if repeat_line is None:
        raise ValueError("repeated bounded autoregressive sequence result is missing")
    repeat_fields = dict(re.findall(r"([A-Za-z_]+)=([^\s]+)", repeat_line))
    stable_fields = {key: value for key, value in fields.items() if key != "progress_csv"}
    stable_repeat_fields = {key: value for key, value in repeat_fields.items() if key != "progress_csv"}
    if fields.get("completed") != "yes" or fields.get("tokens_equal") != "yes" or stable_fields != stable_repeat_fields:
        raise ValueError("autoregressive sequence control is incomplete or non-repeatable")
    sequence_status = status_from_gate(fields["numeric_gate"])
    records.append(inventory_record(
        record_id="natural-repeat-p8-g25-incremental-final", fixture_id="natural-repeat-p8-g25-incremental",
        comparison="candidate versus canonical final hidden/logits", capacity=512,
        phase_or_position="33 one-row decode steps; final position 32",
        history_identity="same 33 generated/input token IDs; token equality confirmed",
        paired_inputs_identical="YES", paired_topology_identical="YES", status=sequence_status,
        historical_result=sequence_status, repeats=2,
        evidence=f"{sequence_path.relative_to(ROOT)}; {repeat_path.relative_to(ROOT)}",
        notes="Two runs repeat the same endpoint; intermediate output drift is non-monotonic and is not a per-position gate.",
    ))

    followup_root = PROP / "raw/post-stop-followup-20261010"
    for fixture, directory, expected in (
        ("natural-repeat-p32-g1-cap512", "natural-repeat-p32-g1-cap512", "PASS"),
        ("repeated-token-id-p8-g25-cap512", "repeated-token-id-p8-g25-cap512", "PASS"),
    ):
        log = followup_root / directory / "run.log"
        matched = [row for row in parse_repeat_lines(log) if row["fixture_id"] == fixture]
        stable_matched = [
            {key: value for key, value in row.items() if key not in {"repeat", "run"}}
            for row in matched
        ]
        if len(matched) != 2 or any(row["raw_status"] != expected for row in matched) or stable_matched[0] != stable_matched[1]:
            raise ValueError(f"post-stop follow-up repeat/status mismatch for {fixture}")
        records.append(inventory_record(
            record_id=f"post-stop-{fixture}", fixture_id=fixture,
            comparison="candidate versus canonical final hidden/logits", capacity=512,
            phase_or_position="guarded free-running endpoint", history_identity="same token trajectory in A/B; see run manifest",
            paired_inputs_identical="YES", paired_topology_identical="YES", status=expected,
            historical_result=expected, repeats=2, evidence=str(log.relative_to(ROOT)),
            notes="Separate post-stop follow-up; does not backfill the original stopped matrix or validate the candidate.",
        ))

    bad_log = followup_root / "short-natural-reviewer-p1-g2-cap64/run.log"
    bad_rows = [row for row in parse_repeat_lines(bad_log) if row["fixture_id"] == "short-natural-reviewer-p1-g2-cap64"]
    bad_stable = [{key: value for key, value in row.items() if key not in {"repeat", "run"}} for row in bad_rows]
    if len(bad_rows) != 2 or any(row["raw_status"] != "NUMERIC_GATE_FAIL" for row in bad_rows) or bad_stable[0] != bad_stable[1]:
        raise ValueError("expected preserved historical capacity-64 runner misclassification")
    records.append(inventory_record(
        record_id="capacity64-obsolete-runner-classification", fixture_id="short-natural-reviewer-p1-g2-cap64",
        comparison="runner classification of an out-of-scope metric", capacity=64,
        phase_or_position="short decode", history_identity="same captured history; repeated",
        paired_inputs_identical="YES", paired_topology_identical="YES", status="INVALID_EVIDENCE",
        historical_result="NOT_APPLICABLE", repeats=2, evidence=str(bad_log.relative_to(ROOT)),
        notes="Preserved old runner labeled contract NOT_APPLICABLE as NUMERIC_GATE_FAIL; do not interpret as a numerical failure.",
    ))

    corrected_log = followup_root / "contract-scope-control/run.log"
    corrected = [row for row in parse_repeat_lines(corrected_log) if row["fixture_id"] == "short-natural-reviewer-p1-g2-cap64"]
    corrected_stable = [{key: value for key, value in row.items() if key not in {"repeat", "run"}} for row in corrected]
    if len(corrected) != 2 or any(row["raw_status"] != "NUMERIC_NOT_APPLICABLE" for row in corrected) or corrected_stable[0] != corrected_stable[1]:
        raise ValueError("corrected capacity-64 contract-scope result is missing")
    records.append(inventory_record(
        record_id="capacity64-contract-v2", fixture_id="short-natural-reviewer-p1-g2-cap64",
        comparison="candidate versus canonical; final-output contract scope", capacity=64,
        phase_or_position="short decode", history_identity="token hash recorded in live contract output",
        paired_inputs_identical="YES", paired_topology_identical="YES", status="NOT_APPLICABLE",
        historical_result="diagnostic metrics only", repeats=2,
        evidence=str(corrected_log.relative_to(ROOT)),
        notes="Policy v2 minimum capacity is 512; preserved metrics do not constitute a pass or failure.",
    ))


def add_capacity_history(records: list[dict[str, str]]) -> None:
    logs = {
        64: "cap64-natural32.log", 256: "cap256-natural32-diagnostic.log",
        512: "cap512-natural32-final-guard.log", 1024: "cap1024-natural32-diagnostic.log",
        1032: "cap1032-natural32-diagnostic.log",
    }
    for capacity, name in logs.items():
        path = NATIVE / "raw" / name
        text = path.read_text(encoding="utf-8")
        matches = re.findall(r"(?:native_trial_numeric_probe|real_qwen_compare) [^\n]*numeric_gate=(PASS|FAIL)", text)
        if not matches:
            raise ValueError(f"historical capacity-{capacity} numeric result missing from {path}")
        historical = matches[-1]
        current = historical if capacity == 512 else "NOT_APPLICABLE"
        reason = (
            "Capacity 512 is inside the current candidate guard and policy-v2 scope; this single historical endpoint still does not validate the candidate."
            if capacity == 512 else
            "Historical endpoint is retained, but the current final-output contract/guard does not admit this capacity."
        )
        records.append(inventory_record(
            record_id=f"capacity-history-{capacity}", fixture_id=f"natural-repeat-p32-cap{capacity}",
            comparison="historical candidate versus canonical endpoint", capacity=capacity,
            phase_or_position="32-row prefill + position-32 decode",
            history_identity="same recorded natural-repeat prompt across capacity sweep",
            paired_inputs_identical="YES", paired_topology_identical="YES", status=current,
            historical_result=historical, repeats=1 if capacity != 64 else 3,
            evidence=str(path.relative_to(ROOT)), notes=reason,
        ))


def capture_expected_bytes(meta: dict[str, str]) -> int:
    rows = int(meta["query_rows"])
    visible = int(meta["visible_context"])
    dims, kv_heads, q_heads = 128, 8, 40
    values = dims * visible * kv_heads * 2
    probabilities = visible * rows * q_heads * 4
    positions = rows * 4
    outputs = dims * rows * q_heads * 4
    keys = values
    queries = dims * q_heads * rows * 4
    scores = visible * rows * q_heads * 4
    return values + probabilities + positions + 3 * outputs + keys + queries + 4 * scores


def read_meta(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def add_capture_integrity() -> list[dict[str, str]]:
    locations: list[tuple[str, int, str, Path]] = []
    qk_root = NATIVE / "raw/av-boundary-qk"
    repeat_root = NATIVE / "raw/av-boundary-qk-repeat"
    for capacity in (256, 512, 1024, 1032):
        for phase in ("prefill", "decode"):
            stem = f"boundary-capacity-{capacity}-{phase}-layer-00"
            locations.append(("primary-qk", capacity, phase, qk_root / stem))
            locations.append(("repeat-qk", capacity, phase, repeat_root / stem))
    locality = PROP / "raw/native-av-divergence-prefill-locality-final-20261009-070358/raw"
    locations.extend([
        ("candidate-locality-prefill", 512, "prefill", locality / "local/prefill32/boundary-capacity-512-prefill-layer-00"),
        ("candidate-locality-decode", 512, "decode", locality / "capacity/cap-512-decode32/boundary-capacity-512-decode-layer-00"),
    ])

    primary_hash: dict[tuple[int, str], str] = {}
    output = []
    for location, capacity, phase, stem in locations:
        meta_path, binary_path = stem.with_suffix(".meta"), stem.with_suffix(".bin")
        if not meta_path.is_file() or not binary_path.is_file():
            raise ValueError(f"missing boundary capture pair: {stem}")
        meta = read_meta(meta_path)
        expected_geometry = {
            "format": "qwen3-native-av-boundary-v1", "capacity": str(capacity),
            "layer": "0", "device_id": "0", "phase": phase,
            "query_heads": "40", "kv_heads": "8", "head_dim": "128",
            "value_type": "F16", "probability_type": "F32", "qk_capture": "yes",
            "oracle": "ascending-visible-position-FP64-accumulator-output-F32",
        }
        for key, expected in expected_geometry.items():
            if meta.get(key) != expected:
                raise ValueError(f"{meta_path}: {key}={meta.get(key)!r}; expected {expected!r}")
        rows = int(meta["query_rows"])
        visible = int(meta["visible_context"])
        expected_shape = (32, 32) if phase == "prefill" else (1, 33)
        if (rows, visible) != expected_shape:
            raise ValueError(f"unexpected {phase} capture extents {(rows, visible)}")
        data = binary_path.read_bytes()
        expected_size = capture_expected_bytes(meta)
        if len(data) != expected_size:
            raise ValueError(f"{binary_path}: {len(data)} bytes, expected {expected_size}")
        key = (capacity, phase)
        digest = sha256(data)
        if location in {"primary-qk", "candidate-locality-prefill", "candidate-locality-decode"}:
            primary_hash[key] = primary_hash.get(key, digest)
            same = digest == primary_hash[key]
        else:
            same = digest == primary_hash.get(key, "")
        if not same:
            raise ValueError(f"capture repeat hash mismatch for capacity={capacity} phase={phase} location={location}")
        output.append({
            "capture_id": f"cap{capacity}-{phase}-layer0", "capacity": str(capacity), "phase": phase,
            "location": location, "metadata_sha256": sha256(meta_path.read_bytes()),
            "binary_sha256": digest, "binary_bytes": str(len(data)),
            "expected_bytes": str(expected_size),
            "geometry": f"rows={rows};visible={visible};QH=40;KVH=8;D=128;V=F16;P=F32",
            "same_binary_as_primary": "YES" if same else "NO",
            "identity_provenance_note": "v1 metadata omits model/token identity; run manifest/log supplies association.",
        })
    return output


def validate_records(records: list[dict[str, str]], captures: list[dict[str, str]]) -> dict[str, int]:
    ids = [row["record_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate inventory record ID")
    counts = defaultdict(int)
    for row in records:
        counts[row["status"]] += 1
    if not captures or any(row["same_binary_as_primary"] != "YES" for row in captures):
        raise ValueError("one or more repeated boundary captures are not byte-identical")
    if not any(row["status"] == "INVALID_EVIDENCE" for row in records):
        raise ValueError("historical capacity-64 status-misclassification evidence was omitted")
    return dict(counts)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true", help="validate evidence without writing files")
    args = parser.parse_args()
    if not args.check_only and args.output_dir is None:
        parser.error("--output-dir is required unless --check-only is used")

    records: list[dict[str, str]] = []
    matrix = add_matrix_inventory(records)
    add_phase_isolation(records)
    add_existing_propagation(records)
    add_topology_and_prefix(records)
    add_sequence_and_followups(records)
    add_capacity_history(records)
    captures = add_capture_integrity()
    counts = validate_records(records, captures)

    summary = {
        "inventory_records": len(records),
        "status_counts": counts,
        "original_matrix_planned": matrix["planned"],
        "original_matrix_unrun": matrix["original_unrun"],
        "baseline_exact_history_tokens": matrix["baseline_free_history"],
        "boundary_capture_records": len(captures),
        "boundary_capture_hash_groups": len({(row["capacity"], row["phase"]) for row in captures}),
        "non_authorizing": True,
    }
    if not args.check_only:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / "fixture-inventory.csv", INVENTORY_FIELDS, records)
        write_csv(output / "capture-integrity.csv", CAPTURE_FIELDS, captures)
        (output / "audit-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("qwen3_native_av_evidence_audit=PASS " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"qwen3_native_av_evidence_audit=FAIL: {error}", file=sys.stderr)
        sys.exit(1)
