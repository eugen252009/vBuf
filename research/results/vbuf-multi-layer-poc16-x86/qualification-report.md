# POC16 Multi-Layer Transformer Execution

## Core Result

The x86 core composition executes real `blk.1 -> blk.2 -> blk.3` activation
forwarding for token positions 0 and 1 with independent reference parity.

- Per-block attention and final block parity: PASS for all six block/token
  combinations, max absolute error `0`.
- Final multi-layer token-0 parity: PASS, max absolute error `0`.
- Final multi-layer token-1 parity: PASS, max absolute error `0`.
- Block output to next block input: PASS, boundary copy bytes `0`.
- Block handoff timings: `25,930 ns`, `31,670 ns`, `39,370 ns`, and `27,770 ns`
  in the measured cold sequence.
- Per-layer runtime state: PASS; each layer owns separate K/V slots.

## Routing

Token 0:

```text
blk.1: 56,11,50,34,3,47
blk.2: 18,37,53,44,23,61
blk.3: 60,2,19,35,9,53
```

Token 1:

```text
blk.1: 56,29,11,3,50,26
blk.2: 37,18,44,53,23,49
blk.3: 60,2,53,35,15,59
```

All selected experts executed. Aggregate unselected expert graphs,
acquisitions, source reads, and materializations were reported as zero.

## Working Set

- Logical persistent bytes per block: `24,128,512`.
- Total logical persistent bytes across three blocks: `72,385,536`.
- Peak active persistent bytes: `1,892,352`.
- Peak active fraction: `0.0261417`.
- Peak resident bytes: `8,312,832`.
- Cold source reads/bytes: `172 / 139,249,664`.
- Warm source reads/bytes: `172 / 139,249,664`.
- Cache thrash: observed; the 8 MiB budget does not retain the complete
  multi-block working set.
- Execution-preparation copies/repack/transcode: `0/0/0`.

The current harness records per-block residency traces and payload timing
events. The output demonstrates bounded residency and reload behavior, but the
aggregate eviction/reload counters still need a dedicated summary pass.

## State

Each block has independent K/V runtime state. State bytes per block are:

- After token 0: `20,480` bytes.
- After token 1: `40,960` bytes.

The generic `multi_layer_state_contract` passes cross-slot isolation.

## Qualification Gaps

This package is not yet a complete POC16 qualification because:

- The middle-block failure selector currently does not fail at the intended
  block boundary and requires correction before failure semantics can be PASS.
- RV2 exact-range replay for the three-block span has not yet been executed.
- A dedicated aggregate eviction/reload/thrash report is still required.

These are reported as gaps, not fabricated PASS results.
