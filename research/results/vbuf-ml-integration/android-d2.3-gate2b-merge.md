# vBuf-ML D2.3 and Gate 2B Merge Qualification

Date: 2026-08-19

## Merge

- Destination: `vbuf-ml`
- Source: `vbuf-ml-riscv`
- Merge commit: `ee57377`
- Merge mode: normal non-squash merge, no conflicts

The merge preserves the D2.3 direct runtime/materialization seam and adds the
portable lowering, versioned Rust graph ABI, generic GGML adapter, and their
contract tests. The persistent vBuf wire format and runtime ownership policy
were not changed.

## Boundary Checks

- The Rust ABI exports graph structure and operation attributes only.
- The native adapter consumes a validated ready `PersistentTensorRef` and a
  retained lease; it performs no source lookup, model detection, or tensor-name
  interpretation.
- `scripts/verify_portable_graph_neutrality.py` reports
  `FORBIDDEN_LEAKAGE_COUNT=0`.
- Physical payload ranges remain distinct from semantic payload lengths in the
  existing D2.3 path.

## Qualification Results

### Rust workspace

Command: `cargo test --workspace`

Result: pass. All workspace unit, integration, format, range, vBuf v0.6,
vBuf-ML, and portable-runtime tests passed.

Command: `cargo fmt --all -- --check`

Result: pass after formatting the merged `vbuf-runtime` sources.

### Native substrate

The pinned ggml revision `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d` was used
from the existing local checkout.

Commands:

- `cmake -S integrations/ggml -B /tmp/opencode/vbuf-ml-merged-build ...`
- `cmake --build /tmp/opencode/vbuf-ml-merged-build -j2`
- `ctest --test-dir /tmp/opencode/vbuf-ml-merged-build --output-on-failure`

Result: 19/19 tests passed, including the portable graph adapter contract,
all existing GGML dependency/materializer/residency contracts, and the
portable graph neutrality test.

## Qualification Boundary

The Android ARM64 D2.3 result remains the previously recorded qualification
in `research/results/vbuf-android-arm64-chat-poc/phase-d2.3-materialization-readiness.md`.
This merge was qualified on the host Rust and native GGML surfaces; Android
hardware generation/parity was not rerun during this merge.
