# Header-Only Derived vBuf Benchmark

## Provenance

```text
host: x86_64 Debian Linux 6.12.101
cpu: AMD Ryzen 7 5800X 8-Core Processor
logical_cpus: 16
compiler: rustc 1.95.0 (LLVM 22.1.2)
optimization: cargo build --release
repository_revision: 5c276b0a97082d4432306f5c12510dbf20163a8e
source_artifact: research-models/Qwen3-0.6B-Q8_0.vbuf
source_artifact_bytes: 637925504
warmups: 3
samples: 10
cache_condition: warm process/file-backed cache; true cold cache not controlled
hash: FNV-1a over raw F32 bit-pattern fields in element order
```

Command:

```text
cargo build --release --manifest-path rust/Cargo.toml --bin header_only_format_bench
target/release/header_only_format_bench /tmp/header-only-bench research-models/Qwen3-0.6B-Q8_0.vbuf
```

The benchmark covered header counts `32`, `128`, `377`, `1000`, and `10000`,
and BaseStep values `8`, `16`, `32`, `64`, `128`, and `256`. It measured:

```text
CURRENT_CANONICAL
HEADER_ONLY_GENERIC_VBUF
HEADER_ONLY_SCALAR
HEADER_ONLY_SIMD
HEADER_ONLY_DIRECT
HEADER_ONLY_NANO
```

Every variant validated the vBuf structure and matched the deterministic
metadata digest. The generic variant used `parse_v06`; scalar/direct used the
same validated dense layout with fixed arithmetic; SIMD loaded the fixed
metadata fields in AVX2 lanes; Nano reconstructed and enumerated structural
start bits.

## Actual 337-Record Artifact

Times are warm median milliseconds for the structural metadata digest path:

| Variant | Median | Mean | Min | P95 |
|---|---:|---:|---:|---:|
| Current canonical | 0.016849 | 0.0169 | 0.0168 | 0.0172 |
| Header-only generic vBuf | 0.019990 | 0.0200 | 0.0199 | 0.0201 |
| Header-only scalar | 0.016500 | 0.0165 | 0.0164 | 0.0166 |
| Header-only SIMD | 0.019355 | 0.0194 | 0.0193 | 0.0195 |
| Header-only direct | 0.016500 | 0.0165 | 0.0164 | 0.0167 |
| Header-only Nano | 0.020385 | 0.0205 | 0.0203 | 0.0212 |

The derivative was `24288` bytes versus `637925504` bytes for the source
artifact. Despite the byte reduction, generic header-only traversal was
approximately `0.84x` the canonical speed, scalar/direct approximately
`1.02x`, SIMD approximately `0.87x`, and Nano approximately `0.83x`.

## Size and BaseStep Results

At 377 records, canonical/generic/scalar/SIMD/direct/Nano medians in ms were:

| BaseStep | Canonical | Generic | Scalar | SIMD | Direct | Nano |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.018410 | 0.024289 | 0.018510 | 0.021760 | 0.018524 | 0.022665 |
| 16 | 0.019150 | 0.023195 | 0.019310 | 0.022490 | 0.020450 | 0.023205 |
| 32 | 0.021345 | 0.024850 | 0.020930 | 0.024140 | 0.020949 | 0.024709 |
| 64 | 0.024594 | 0.028155 | 0.024175 | 0.027454 | 0.024190 | 0.027974 |
| 128 | 0.045024 | 0.062445 | 0.037780 | 0.038580 | 0.035275 | 0.038965 |
| 256 | 0.044235 | 0.066454 | 0.062540 | 0.065789 | 0.062665 | 0.066300 |

At 10000 records and BaseStep 8, medians were `0.483876 ms` canonical,
`0.590556 ms` generic, `0.484762 ms` scalar, `0.578771 ms` SIMD,
`0.483882 ms` direct, and `0.604496 ms` Nano. No SIMD or Nano crossover was
observed through 10000 records or any tested BaseStep.

## Correctness

```text
CANONICAL_METADATA_DIGEST: b80a77bbe074dc9e
HEADER_SCALAR_METADATA_DIGEST: b80a77bbe074dc9e
HEADER_SIMD_METADATA_DIGEST: b80a77bbe074dc9e
ORIGINAL_VBUF_METADATA_DIGEST: b80a77bbe074dc9e
HEADER_ONLY_VBUF_METADATA_DIGEST: b80a77bbe074dc9e
SEMANTIC_METADATA_PARITY: PASS
METADATA_SEMANTIC_PARITY: PASS
HEADER_ONLY_VBUF_VALIDATION: PASS
```

The digest covers count, source/block offsets, payload lengths, key and
representation fields, continuation, width, and logical count. It does not
claim tensor-payload equivalence because no payload is copied.

## Remote Bootstrap Extension

The same Qwen3-0.6B source was exposed through the repository Range server.
The valid 24288-byte bootstrap was fetched with `Range: bytes=0-24287` and
returned `206`. Its first tensor record resolved the source range
`171965160..173079271` (offset `171965160`, length `1114112`). A second HTTP
Range request returned exactly `1114112` bytes, matching the local file-source
read. The source artifact was never downloaded by this test.

```text
BOOTSTRAP_ARTIFACT_BYTES: 24288
FULL_MODEL_ARTIFACT_BYTES: 637925504
BOOTSTRAP_TO_MODEL_SIZE_RATIO: 0.0000380734
BYTES_TRANSFERRED_TO_METADATA_READY: 24288
BYTES_TRANSFERRED_TO_FIRST_PAYLOAD_REQUEST: 1138400
NUMBER_OF_RANGE_REQUESTS_BEFORE_FIRST_COMPUTE: 2
BYTES_TRANSFERRED_TO_FIRST_USEFUL_COMPUTE: NOT_MEASURED
LOCAL_STORAGE_REQUIRED_BEFORE_METADATA_READY: 24288
LOCAL_STORAGE_REQUIRED_BEFORE_FIRST_PAYLOAD_REQUEST: 1138400
FULL_MODEL_LOCAL_STORAGE_REQUIRED_FOR_HEADER_ONLY_BOOTSTRAP: NO
HTTP_RANGE_USED: YES
FULL_SOURCE_DOWNLOADED: NO
```

The full-artifact deployment contract still requires the full local file;
full-path network bytes were not measured and are not fabricated here. The
header-only structural POC resolves ranges before parsing/opening the source,
but complete current vbuf-ML semantic discovery is not yet parity-qualified.

## Interpretation

The layout benefit in bytes is confirmed, but a time benefit is not. SIMD adds
no benefit over scalar and is consistently slower in this fixed-record POC.
Nano adds construction and enumeration work without a measured crossover.
The remote bootstrap transfer reduction is established for structural records,
but semantic source-indirected `TensorDirectory`/`TensorRef` support remains an
open architecture seam. This is why the overall recommendation is
`INVESTIGATE_FURTHER`, not a production build recommendation.
