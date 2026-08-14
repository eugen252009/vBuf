# vBuf-ML tensor representations (Step 12)

## Qualification result

No packed/quantized representation is selected in profile 0.1.

The repository's Decision D remains unresolved between a profile-local
namespaced registry and reuse of external numeric IDs. Decision E also does not
name a first real model fixture or pinned upstream revision. Selecting Q4_K,
Q8_0, or another packed layout without those authorities would invent unstable
semantics, so quantized implementation is intentionally blocked.

The exact evidence gap is:

```text
first target architecture/model fixture
+ pinned upstream GGML/llama.cpp revision
+ authoritative source layout and size functions
```

## Current representation

Representation ID `0` is `CanonicalPrimitive`.

Its storage meaning is derived entirely from the canonical vBuf descriptor:

```text
semantic
physical form
bit width
count
payload length
alignment
```

The profile-local contract validates shape product, primitive compatibility,
checked byte size, and canonical payload range. It stores no duplicate dtype,
block-size, offset, length, or alignment fields.

The contract descriptor is:

```text
id = CanonicalPrimitive
logical elements per block = determined by canonical descriptor
physical bytes per block = determined by canonical descriptor
required payload alignment = 1 (canonical alignment remains authoritative)
```

## Quantized representations

Future packed representations will use stable profile-local IDs and exact
external layout contracts. A selected contract must specify:

```text
upstream project and pinned revision
source layout name
logical elements per block
physical bytes per block
field ordering and byte order
row/divisibility rules
payload-size formula
alignment requirements
```

Canonical vBuf may store the selected layout as opaque byte payloads. The ML
representation identity will then validate exact block counts and payload size
without decoding or repacking the tensor. Generic vBuf will not learn GGML or
quantization semantics.

No quantized bytes, fake layouts, aliases, or external enum copies are
currently normative. Unknown representation IDs fail closed.
