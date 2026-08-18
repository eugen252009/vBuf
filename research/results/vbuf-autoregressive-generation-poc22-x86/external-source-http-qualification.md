# External Source HTTP Qualification

HTTP is represented only as a `SourceLocator::Http` value and a normal
`RangeSource` implementation. TensorDirectory and semantic discovery do not
know or branch on HTTP.

The contract seam is therefore transport-neutral and accepts the established
controlled HTTP Range implementation at the resolver boundary. Full HTTP
bootstrap discovery and returned-byte parity were not reached in this slice:
the existing 24,288-byte header-only artifact fails normal vbuf-ML discovery
with `BootstrapNotFound` because it contains only the earlier structural
metadata-record derivative.

```text
HTTP_RANGE_USED: NOT_REACHED_FOR_SEMANTIC_BOOTSTRAP
FULL_SOURCE_DOWNLOADED: NO
SOURCE_ARTIFACT_PARSED_BEFORE_RANGE: NO
HTTP_TENSOR_BYTES_MATCH_LOCAL: NOT_REACHED
SOURCE_HASH_DOES_NOT_FORCE_FULL_DOWNLOAD: YES
```
