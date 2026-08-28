#!/usr/bin/env python3
"""Fail if the generic portable graph boundary imports model identity."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ROOT / "rust/vbuf-runtime/src/ffi.rs",
    ROOT / "rust/vbuf-runtime/src/graph.rs",
    ROOT / "rust/vbuf-runtime/src/lowering.rs",
    ROOT / "rust/vbuf-runtime/src/device.rs",
    ROOT / "integrations/ggml/include/vbuf_portable_graph_ffi.h",
    ROOT / "integrations/ggml/include/vbuf_portable_graph_adapter.h",
    ROOT / "integrations/ggml/tools/portable_graph_poc22_adapter.cpp",
)
FORBIDDEN = re.compile(
    r"(?<![A-Za-z0-9_])(deepseek|qwen|phi|gemma|gguf|model_family|parse_layer)(?![A-Za-z0-9_])|blk\."
)
CUDA_FORBIDDEN = re.compile(r"cuda|cublas|cudart|nvidia|cudevice|cudastream")


def main() -> int:
    violations = []
    cuda_violations = []
    for path in FILES:
        text = path.read_text(encoding="utf-8").lower()
        if path.name.endswith(".rs") and "#[cfg(test)]" in text:
            text = text.split("#[cfg(test)]", 1)[0]
        for match in FORBIDDEN.finditer(text):
            violations.append(f"{path.relative_to(ROOT)}: {match.group(0)}")
        if path.name in {"device.rs", "graph.rs", "lowering.rs", "generic.rs", "ffi.rs"}:
            for match in CUDA_FORBIDDEN.finditer(text):
                cuda_violations.append(f"{path.relative_to(ROOT)}: {match.group(0)}")
    if violations:
        print("FORBIDDEN_LEAKAGE_COUNT=" + str(len(violations)))
        print("\n".join(violations))
        return 1
    print("FORBIDDEN_LEAKAGE_COUNT=0")
    print("CUDA_TYPE_LEAKAGE_COUNT=" + str(len(cuda_violations)))
    if cuda_violations:
        print("\n".join(cuda_violations))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
