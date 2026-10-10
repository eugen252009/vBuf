#!/usr/bin/env python3
"""Repair only the escaped header separator in the temporary raw sweep CSV."""
import pathlib
import sys

if len(sys.argv) != 3:
    raise SystemExit(f"usage: {sys.argv[0]} RAW.csv FIXED.csv")
raw = pathlib.Path(sys.argv[1]).read_text()
needle = "top1_margin\\n0,"
if raw.count(needle) != 1:
    raise SystemExit(f"expected exactly one escaped header separator, found {raw.count(needle)}")
fixed = raw.replace(needle, "top1_margin\n0,", 1)
pathlib.Path(sys.argv[2]).write_text(fixed)
print(f"wrote corrected header; data rows were not changed: {sys.argv[2]}")
