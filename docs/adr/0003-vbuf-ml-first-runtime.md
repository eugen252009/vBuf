# ADR 0003: Narrow first runtime and model qualification

- **Status:** Proposed; concrete selection required before ML metadata implementation
- **Scope:** vBuf-ML only
- **Related plan decision:** E

## Context

Metadata requirements cannot be derived from “all ML models.” They must be traced to operations performed by a concrete inference runtime and model architecture. Selecting fields because GGUF contains them would violate the first-principles requirement.

## Proposed decision

Start with:

- one pinned llama.cpp revision;
- one small, openly redistributable decoder-only model architecture;
- one unquantized F16 representation;
- one common GGML quantized representation supported by identical runtime kernels.

The exact runtime revision and fixtures are not selected by this ADR. Their hashes, licenses, provenance, and redistribution rules must be recorded before Step 10 or runtime integration work.

## Invariants

- The first model is a qualification scope, not a universal-format claim.
- Every required metadata field maps to a concrete runtime consumer or validation operation.
- Unsupported architectures fail explicitly.
- GGUF and vBuf-ML comparisons use equivalent tensor bytes, kernels, runtime settings, prompts, and timing boundaries.
- Conversion/repacking is excluded from the equivalent inference timing path.
- Negative and null performance results remain valid outcomes.

## Non-goals

- selecting BaseStep or generic vBuf layout from model requirements;
- serializing execution graphs without runtime evidence;
- designing new quantization or kernels;
- broad architecture coverage in the first qualification.

## Decision required

Before ML metadata implementation, select and pin the runtime revision, model fixture, F16 source, quantized source, and expected correctness outputs.
