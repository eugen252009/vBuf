#!/usr/bin/env python3
"""Validate the replay tool's self-describing numerical evaluation JSONL."""

import json
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: vbuf_numerical_contract_json_contract.py REPLAY_TOOL TOPOLOGY_CSV AV_CSV")
    result = subprocess.run(
        [sys.argv[1], sys.argv[2], sys.argv[3]],
        check=True,
        capture_output=True,
        text=True,
    )
    records = []
    for line in result.stdout.splitlines():
        prefix = "numerical_contract_historical_replay "
        if line.startswith(prefix):
            records.append(json.loads(line[len(prefix):]))
    if len(records) != 18:
        raise AssertionError(f"expected 18 evaluation JSON records, found {len(records)}")
    statuses = {record["evaluation_status"] for record in records}
    if statuses != {"PASS", "FAIL", "NOT_TESTED"}:
        raise AssertionError(f"unexpected replay evaluation statuses: {sorted(statuses)}")
    for record in records:
        if record["policy_id"] != "vbuf.numerical-contracts" or not record["policy_sha256"]:
            raise AssertionError("evaluation JSON omitted its policy identity/digest")
        if not record["replayed_metrics_only"]:
            raise AssertionError("historical JSON record was not marked replay-only")
    print("vbuf_numerical_contract_json_contract=PASS records=18 parse=jsonl replay_only=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
