# vBuf-ML layout and alignment policy (Step 13)

Step 13 defines downstream writer policy only. Canonical v0.6 remains
responsible for block geometry, padding, BaseStep, and checked ranges.

## Normative rules

- `BaseStep = 1 << BaseShift` remains generic v0.6 behavior.
- Requested payload alignment must be a power-of-two multiple of BaseStep.
- Invalid alignments fail before writing.
- Payload alignment is encoded as the existing v0.6 payload-shift refinement.
- Semantic indexes contain no offsets, lengths, or physical alignment fields.
- Each Key-ID occurrence is assigned deterministically by planned output order.
- Identical semantic input and writer configuration produce identical bytes.

`CanonicalPrimitive` requests use BaseStep alignment by default. No SIMD,
page, direct-I/O, GPU, or quantized alignment is assumed.

## Writer ordering

The downstream `LayoutPlan` uses this deterministic control-first order:

```text
Bootstrap
ModelMetadata
TensorDirectory
TokenizerControl
TokenizerPayload
TensorPayload
Auxiliary
```

Within a class, the caller supplies a unique deterministic source-order key.
This policy is a writer recommendation for profile-owned output and is not a
new decoding requirement. Semantic references remain Key-ID plus occurrence,
so changing legal payload alignment does not change semantic identity.

## Hot/cold behavior

Control regions are placed before large payload regions so model-open can stop
after canonical validation, bootstrap, model metadata, and TensorDirectory.
Tokenizer and tensor payloads remain direct canonical ranges and are not read
merely because they are physically nearby.

The policy does not add `HOT` flags, Nano, finalization, transfer plans, or
backend fields. It also does not claim that page alignment is faster.

## Accounting

Canonical v0.6 reports block/header/payload geometry. A layout qualification
must separately account for:

```text
global header
block headers
payload bytes
inter-block BaseStep padding
profile control bytes
```

The planner does not call all non-payload bytes padding.

## Streaming

The same deterministic plan can be emitted through known-size or indefinite
canonical v0.6 writers. References use identities and occurrences rather than
relocations or raw offsets.

Quantized and backend-specific alignment remains unresolved with the Step 12
first-target evidence gap.
