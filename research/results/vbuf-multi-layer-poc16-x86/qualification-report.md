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
- Cold source reads/bytes: `174 / 141,707,264`.
- Warm source reads/bytes: `174 / 141,707,264`.
- Cache thrash: observed; the 8 MiB budget does not retain the complete
  multi-block working set.
- Execution-preparation copies/repack/transcode: `0/0/0`.

The aggregate residency trace reports `102` unique identities, `348` load
events, `696` hits, `684` misses, `340` evictions, and `246` reload events.
Reloaded bytes are `198,818,816`, a `0.701512` reload-byte fraction. Block
reload bytes are `66,888,704`, `67,339,264`, and `64,590,848` for `blk.1`,
`blk.2`, and `blk.3`; the top ten reload offenders are recorded in the raw
execution log.

## State

Each block has independent K/V runtime state. State bytes per block are:

- After token 0: `20,480` bytes.
- After token 1: `40,960` bytes.

The generic `multi_layer_state_contract` passes cross-slot isolation.

## Failure Semantics

- Middle-block attention failure: PASS; `12` injections, one completed earlier
  block, no later block execution, invalid final output, cleanup PASS.
- Middle-block selected-expert failure: PASS; `6` injections, one completed
  earlier block, no later block execution, invalid final output, cleanup PASS.

## Remaining Scope

- RV2 exact-range replay for the three-block span remains transport-only;
  RV2 compute is not executed because of the pinned RVV FP16 blocker.
