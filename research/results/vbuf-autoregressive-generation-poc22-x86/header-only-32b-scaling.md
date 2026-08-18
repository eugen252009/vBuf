# Header-Only vBuf 32B Scaling Check

## Scope

This is a structural/bootstrap-size scaling check only. It reuses the same
experimental valid-vBuf metadata-record derivation as the Qwen3-0.6B audit.
It does not continue SourceDescriptor/TensorRef design, remote semantic
bootstrap, SIMD, Nano, runtime, or inference work.

## Artifacts

| Metric | Qwen3-0.6B | Qwen3-32B | Growth |
|---|---:|---:|---:|
| Full artifact bytes | 637,925,504 | 34,816,197,376 | 54.5772x |
| Valid header-only vBuf bytes | 24,288 | 52,872 | 2.1769x |
| Physical block/header count | 337 | 734 | 2.1780x |
| TensorDirectory entries | 310 | 707 | 2.2806x |
| Model metadata entries | 11 | 11 | 1.0000x |
| Bootstrap block bytes | 64 | 96 | 1.5000x |
| TensorDirectory payload bytes | 13,933 | 31,900 | 2.2890x |
| Tokenizer metadata bytes | 164 | 164 | 1.0000x |
| Header-only/full ratio | 0.0000380734 | 0.00000151860 | 0.0399x |
| Header-only percentage | 0.00380734% | 0.000151860% | 0.0399x |
| Full/header-only reduction factor | 26,265.0x | 658,499.7x | 25.07x |

The exact 32B vBuf SHA-256 from the established Step 27 provenance is
`84597064d5b3530572959345286368b17e891e640b5958bba7f0b67980cd119d`.
The 0.6B vBuf SHA-256 is
`2982cedd0ddc12d762d12ff3426bcc105b2cee1c675ca13bbb5ca4c3cce9a998`.

## Scaling Attribution

The measured derivative uses one ordinary metadata record per physical vBuf
block. At BaseStep 8 its size is:

```text
24-byte global header + physical_block_count * (8-byte anchor + 64-byte metadata record)
```

Thus the observed derivative growth is driven primarily by physical block
count, which is approximately `2.18x`, not by payload bytes, which grow
`54.58x`. TensorDirectory entries grow `2.28x`; model metadata and tokenizer
control structures remain effectively fixed. The benchmark confirms that many
logical metadata items remain densely encoded in directory/array payloads;
one logical tensor is not automatically one original physical header.

The derivative contains zero model payload bytes. It contains only the
metadata-record bytes required by the experimental profile. Optional source
hash headers were not present in this measured artifact; the source identity
hash design remains optional and binary.

## Validity Gate

```text
HEADER_ONLY_32B_VALID_VBUF: YES
GENERIC_VBUF_READER_OPENS_32B_DERIVATIVE: YES
GENERIC_VBUF_PARSE: PASS
STRUCTURAL_VALIDATION: PASS
MODEL_PAYLOAD_BYTES_COPIED: 0
HEADER_ONLY_IS_VALID_VBUF: YES
SECOND_CONTAINER_FORMAT_INTRODUCED: NO
SPECIAL_NON_VBUF_PARSER_REQUIRED: NO
ORIGINAL_VBUF_REMAINS_AUTHORITATIVE: YES
COMPLETE_VBUF_ML_DISCOVERY_PARITY: NOT_QUALIFIED
SIMD_REBENCHMARKED: NO
NANO_REBENCHMARKED: NO
```

The generated 32B artifact was validated by the normal generic v0.6 parser.
This does not establish that the current vbuf-ml `TensorDirectory` can use
external ranges; that known semantic seam remains intentionally deferred.

## Classification

```text
REMOTE_BOOTSTRAP_SCALING: STRONGLY_FAVORABLE
HEADER_ONLY_GROWS_SUBLINEAR_TO_MODEL_BYTES: YES
ARRAY_LIKE_METADATA_SCALING_CONFIRMED: YES
HEADER_ONLY_SIZE_PRIMARY_SCALING_DRIVER: PHYSICAL_HEADERS
OVERALL_RECOMMENDATION: INVESTIGATE_FURTHER
NEXT_STEP: EXTERNAL_SOURCE_DESCRIPTOR_TENSORREF_SEAM
```

This strengthens the remote-bootstrap case substantially: for the measured
32B artifact, a valid metadata-only vBuf is approximately 658,500 times
smaller than the authoritative payload artifact. It does not by itself
authorize a production profile because complete semantic bootstrap parity is
still unqualified.
