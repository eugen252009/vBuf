# Reconstructed POC7 Provenance

This directory is a **reconstructed qualification package**. It was created
after the original POC sequence because no dedicated POC7 report or
`research/results/vbuf-source-selection-poc7-x86/` directory survives in the
repository.

## Historical Surviving Evidence

- `research/results/vbuf-remote-range-poc6-x86/qualification-report.md`
  qualifies exact local/HTTP range sources, source-independent execution,
  asynchronous materialization, fallback-related source behavior, payload
  identity, and parity. Its historical gate still says
  `READY_FOR_SOURCE_SELECTION_POLICY_POC: NO`, documenting the state at POC6,
  not a later failure.
- `research/results/vbuf-storage-hierarchy-poc8-x86/readiness-reconciliation.md`
  explicitly states that POC1-POC7 had qualified exact range sources,
  asynchronous materialization, source selection, fallback, leases, parity,
  and the real FFN fixture.
- `research/results/vbuf-moe-tensor-wave-poc10-x86/readiness-reconciliation.md`
  lists source selection among the already-satisfied prerequisites for later
  tensor-wave/MoE work.
- `git show --stat 0f6bdef` shows the current
  `vbuf_source_selection.h/.cpp` implementation and
  `source_selection_contract.cpp` first appearing in the recovered history.

## Reconstructed Provenance

The smallest capability consistent with these surviving references is generic
source selection plus fallback integration before/inside materialization. The
original standalone POC7 report, original measurements, original command log,
and original milestone commit are not recoverable.

## Current Evidence

The current implementation and focused contracts were rerun for this package.
Those results are current implementation evidence, not historical POC7 output.
The act of connecting the POC6 and POC8 references to this dedicated report is
reconstructed provenance.

```text
ORIGINAL_DEDICATED_POC7_EVIDENCE: MISSING
FIRST_RECOVERED_IMPLEMENTATION_COMMIT: 0f6bdef
ORIGINAL_POC7_MILESTONE_COMMIT: NOT_RECOVERED
HISTORICAL_MEASUREMENTS_REUSED_AS_POC7: NO
HISTORICAL_MEASUREMENTS_FABRICATED: NO
```
