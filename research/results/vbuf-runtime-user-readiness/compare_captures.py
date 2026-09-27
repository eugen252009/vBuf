#!/usr/bin/env python3
"""Compare independent callback captures with native production boundaries."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("oracle", type=Path)
parser.add_argument("native", type=Path)
parser.add_argument("--logits-max-abs", type=float)
parser.add_argument("--expected-logits", type=int, default=6)
parser.add_argument("--logits-only", action="store_true")
args = parser.parse_args()
logits_checked = 0
failed = False
for path in sorted(args.native.glob("*-result_output.f32" if args.logits_only else "*.f32")):
    other = args.oracle / path.name
    if not other.exists():
        print(json.dumps({"tensor": path.stem, "oracle": "not captured"}))
        if path.stem.endswith("result_output"):
            failed = True
        continue
    actual, expected = np.fromfile(path, dtype="<f4"), np.fromfile(other, dtype="<f4")
    if actual.shape != expected.shape:
        print(json.dumps({"tensor": path.stem, "actual_shape": actual.shape, "expected_shape": expected.shape}))
        failed = True
        continue
    error = np.abs(actual - expected)
    if path.stem.endswith("result_output"):
        logits_checked += 1
        if args.logits_max_abs is not None:
            failed |= (not np.isfinite(error).all() or float(error.max()) > args.logits_max_abs
                       or int(actual.argmax()) != int(expected.argmax()))
    print(json.dumps({"tensor": path.stem, "elements": actual.size,
        "max_abs": float(error.max()), "rms": float(np.sqrt(np.mean(error.astype(np.float64)**2))),
        "expected_abs_max": float(np.abs(expected).max()), "actual_first": actual[:4].tolist(),
        "expected_first": expected[:4].tolist()}))
if args.logits_max_abs is not None:
    failed |= logits_checked != args.expected_logits
    print(json.dumps({"logits_gate": "FAIL" if failed else "PASS", "positions": logits_checked,
                      "max_abs_allowed": args.logits_max_abs}))
    raise SystemExit(1 if failed else 0)
