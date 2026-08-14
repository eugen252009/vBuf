# ADR 0002: Descendant profiles compose generic vBuf primitives

- **Status:** Accepted boundary; navigation/finalization candidates remain unselected
- **Scope:** BASE/profile dependency boundary
- **Related plan decisions:** B, F, H, I

## Context

A descendant needs a deterministic way to identify its own contract and may need efficient navigation. That does not justify moving descendant semantics into vBuf. Earlier vBuf drafts proposed a generic Nano-Index, while the current implementations linearly scan generic Key-IDs. The usefulness and wire shape of Nano, checkpoints, a region directory, and a finalization envelope have not been qualified.

## Decision

1. vBuf exposes only generic physical representation and independently promoted generic primitives.
2. A descendant assigns meaning to generic payloads and numeric roles inside its own profile contract.
3. Dependency direction is always `descendant -> vBuf`; vBuf never depends on or recognizes descendant semantics.
4. No Nano, rank/select checkpoint, generic region directory, finalization envelope, or generic integrity mechanism is selected by this ADR.
5. Step 5A must select or reject navigation candidates using generic workloads before any corresponding wire contract is frozen.
6. If every finalized artifact fails qualification, no finalization envelope is added.
7. Canonical blocks remain authoritative. Derived acceleration structures never authorize unsafe access or replace canonical validation.

## Generic promotion test

A BASE feature must be domain-neutral, independently useful outside vBuf-ML, safely skippable when optional, large-file safe, and smaller than an equivalent composition only when evidence supports that claim. Files not using it must not pay mandatory hot-path cost.

## Profile-owned semantics

The following remain outside generic vBuf:

- tensor names, ranks, shapes, and representations;
- model and tokenizer metadata;
- quantization semantics;
- runtime execution or scheduling behavior;
- profile-specific integrity policy.

A generic directory, if selected, may map a generic numeric ID/kind to a validated range. It may not contain tensor metadata. Nano, if selected, maps candidate physical starts and never resolves Key-ID or profile meaning.

## Open gates

- Decision B: Nano/directory/coexistence/no-artifact result from Step 5A.
- Decision F: profile-local versus independently justified generic integrity.
- Decision H: finalized-artifact discovery only after at least one artifact survives.
- Decision I: explicit fast-path versus full-conformance policy for any retained derived artifact.

## Consequences

vBuf-ML scaffolding may be created only after its parent and dependency boundary are stable. ML wire semantics remain blocked until the required BASE qualification and specification steps complete.
