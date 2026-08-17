# Reconstructed POC7 Source Selection Qualification

> This is a reconstructed qualification report. It is not the original POC7
> report and does not claim that the original dedicated evidence survived.

## Scope

POC7 is reconstructed as the generic policy answering only:

```text
Which eligible source should provide a requested exact byte range?
```

It does not select tensors, schedule prefetch, manage residency, execute
graphs, or change the vBuf format.

## Historical Evidence

POC6 proves the source/materialization substrate needed by policy. POC8 later
records source selection and fallback as already qualified by POC1-POC7. No
historical POC7-specific benchmark or command output is available and none is
asserted here.

## Current Implementation

Implementation files:

- `integrations/ggml/include/vbuf_source_selection.h`
- `integrations/ggml/src/vbuf_source_selection.cpp`
- `integrations/ggml/include/vbuf_materializer.h`
- `integrations/ggml/src/vbuf_materializer.cpp`

Current consumers include:

- `integrations/ggml/tools/tensor_wave_poc3.cpp`
- `integrations/ggml/tools/moe_tensor_wave_poc10.cpp`
- `integrations/ggml/tools/multi_expert_moe_poc12.cpp`

Direct tests:

- `integrations/ggml/tests/source_selection_contract.cpp`
- `integrations/ggml/tests/source_selection_fallback_contract.cpp`
- `integrations/ggml/tests/range_source_contract.cpp`
- `integrations/ggml/tests/materializer_contract.cpp`

## Selection Semantics

`SourceDescriptor` records source identity/kind, availability, exact-range
capability, measured throughput, fixed latency, and the `RangeSource` object.

`SourceSelectionPolicy::select`:

1. Marks a source eligible only when it is available, range-capable, and has a
   non-null source object.
2. Estimates completion as fixed latency plus requested bytes divided by
   measured throughput.
3. Treats zero/unknown throughput as `UINT64_MAX`, ranking it after known
   metrics.
4. Orders equal estimates deterministically by source ID, then source index.
5. Returns the first eligible candidate as primary and the second as fallback.

Current verification covers:

- Deterministic primary selection: PASS.
- Deterministic tie handling: PASS.
- Unknown metric handling: PASS.
- Fallback candidate selection: PASS.

## Fallback Semantics

`LocalVbufRangeMaterializer` attempts its primary `RangeSource` first. If that
read fails and a fallback source exists, it retries the same requested offset
and length through the fallback source. The focused fallback contract verifies:

- one primary attempt;
- one fallback attempt;
- exact requested range preservation;
- exact payload identity with the artifact bytes;
- successful materialization and clean release.

Source selection itself does not perform the read, prefetch, residency insert,
or execution. It supplies the source ordering; materialization owns retry and
payload lifecycle.

## Architecture Boundaries

- `SourceSelectionPolicy`: decides **where** bytes should come from.
- `PrefetchPlanner`: decides dependency-driven **when** tensors may be
  requested.
- `TensorResidencyStore`: owns completed resident buffers and leases.
- `TensorDependencyExecutor`: executes graph dependencies and consumers.
- vBuf persistent format: remains unchanged.

No regression or boundary absorption was found. No source-policy branch on
DeepSeek, MoE, MLA, block number, tensor name, or architecture enum exists in
the generic implementation.

## Current Verification

```text
source_selection_contract: PASS
source_selection_fallback_contract: PASS
range_source_contract: PASS
materializer_contract: PASS
striped_materializer_contract: PASS
full x86 CTest: 14/14 PASS
git diff --check: PASS
ccc index: PASS
```

Later POC behavior remains preserved by the current full CTest and the existing
POC8-POC15 evidence packages. No POC7 benchmark was created to fill the
historical gap.

## Limitations

- The original dedicated POC7 evidence is missing.
- Original POC7 dates, command lines, benchmark values, exact source metrics,
  and historical output logs are unrecoverable.
- `0f6bdef` is the first recovered commit containing the current implementation;
  it is not asserted to be the original POC7 milestone commit.

## Provenance Classification

- POC6 and POC8 statements: `HISTORICAL_SURVIVING_EVIDENCE`.
- Current implementation and test results: `CURRENT_IMPLEMENTATION_EVIDENCE`.
- This report and the POC6→POC7→POC8 linkage: `RECONSTRUCTED_PROVENANCE`.

## Final Status

```text
ORIGINAL_DEDICATED_POC7_EVIDENCE: MISSING
POC7_HISTORICAL_QUALIFICATION_REFERENCED_BY_LATER_EVIDENCE: YES
SOURCE_SELECTION_IMPLEMENTATION_PRESENT: YES
DETERMINISTIC_PRIMARY_SELECTION: PASS
DETERMINISTIC_TIE_HANDLING: PASS
UNKNOWN_METRIC_HANDLING: PASS
FALLBACK_CANDIDATE_SELECTION: PASS
PRIMARY_FAILURE_TO_FALLBACK: PASS
FALLBACK_PAYLOAD_IDENTITY: PASS
SOURCE_POLICY_ONLY_DECIDES_SOURCE: PASS
PREFETCH_SEMANTICS_CHANGED: NO
RESIDENCY_SEMANTICS_CHANGED: NO
VBUF_FORMAT_CHANGE_REQUIRED: NO
RECONSTRUCTED_POC7_EVIDENCE: PASS
FULL_X86_CTEST: PASS
GIT_DIFF_CHECK: PASS
ORIGINAL_IMPLEMENTATION_COMMIT: 0f6bdef (first recovered implementation; original milestone not recovered)
HISTORICAL_MEASUREMENTS_FABRICATED: NO
```
