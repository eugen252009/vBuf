# llama External HTTP Compute Qualification

The source-aware Rust path already qualified exact HTTP Range byte parity for
the 0.6B and 32B semantic bootstraps, including the 32B tensor at offset
`6056603320`. This task does not add HTTP or file I/O to C++.

The new C++-facing seam accepts only an already-materialized pointer, exact
length, source provenance, and an optional lease. The external HTTP-to-ggml
compute path was not run because this workspace does not contain the external
llama.cpp/ggml headers or integration build configuration.

```text
HTTP_LOGIC_IN_LLAMA_LOADER: NO
FULL_SOURCE_DOWNLOADED: NO for the controlled Rust HTTP qualification
HTTP_EXTERNAL_COMPUTE_PATH: NOT_RUN
SOURCE_HASH_DOES_NOT_FORCE_FULL_DOWNLOAD: YES
```
