#!/usr/bin/env python3
"""Compare vBuf's recorded first-turn IDs with llama.cpp's matched-template probe."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path


def read_jsonl_gzip(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence

    vbuf_log = evidence / "raw/runs/vbuf-04/server.jsonl.gz"
    vbuf_ids = None
    for event in read_jsonl_gzip(vbuf_log):
        line = event.get("line", "") if isinstance(event, dict) else ""
        if "vbuf_request id=chatcmpl-vbuf-1" in line and "prompt_token_ids=" in line:
            match = re.search(r"prompt_tokens=(\d+).*?prompt_token_ids=([0-9,]+)", line)
            if not match:
                raise SystemExit("could not parse the vBuf first-turn prompt token log")
            vbuf_ids = [int(token) for token in match.group(2).split(",")]
            if int(match.group(1)) != len(vbuf_ids):
                raise SystemExit("vBuf prompt token count does not match its recorded ID list")
            break
    if vbuf_ids is None:
        raise SystemExit("vBuf pilot has no recorded first-turn prompt token IDs")

    probe_log = evidence / "raw/template-probe/llama-matched/server.jsonl.gz"
    probe = next((event for event in read_jsonl_gzip(probe_log)
                  if isinstance(event, dict) and event.get("kind") == "chat_template_probe"), None)
    if probe is None:
        raise SystemExit("matched llama template probe is missing")
    llama_ids = [int(token) for token in probe["prompt_token_ids"]]
    same = llama_ids == vbuf_ids
    ids_text = ",".join(map(str, llama_ids)).encode("ascii")

    request_path = evidence / "raw/harness/template-probe-request.json"
    template_path = evidence / "raw/harness/qwen3-vbuf-matched.jinja"
    template_meta = json.loads((evidence / "raw/harness/qwen3-vbuf-matched-template.json").read_text())
    report = {
        "schema": "qwen3-pi-ab-prompt-equivalence-v1",
        "status": "PASS" if same else "FAIL",
        "vbuf_reference": "excluded exploratory vbuf-04 first Pi request; generation outcome is not scored",
        "llama_probe": "matched-template-probe; apply-template plus tokenize only; no Pi/model generation",
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "source_template_sha256": template_meta["original_template_sha256"],
        "rendered_prompt_utf8_bytes": len(probe["prompt"].encode("utf-8")),
        "rendered_prompt_sha256": hashlib.sha256(probe["prompt"].encode("utf-8")).hexdigest(),
        "vbuf_prompt_token_count": len(vbuf_ids),
        "llama_prompt_token_count": len(llama_ids),
        "llama_prompt_token_ids_sha256": hashlib.sha256(ids_text).hexdigest(),
        "token_ids_identical": same,
        "first_mismatch_index": next((i for i, (left, right) in enumerate(zip(vbuf_ids, llama_ids))
                                       if left != right), None),
        "task_user_message_sha256": hashlib.sha256((evidence / "benchmark-task-v1/pi-user-message.txt").read_bytes()).hexdigest(),
        "task_system_prompt_sha256": hashlib.sha256((evidence / "benchmark-task-v1/system-prompt.txt").read_bytes()).hexdigest(),
        "qualification_boundary": "This establishes identical first-turn token IDs for the captured Pi request, not agent success or backend numerical parity.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
