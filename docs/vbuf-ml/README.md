# vBuf-ML descendant boundary

`vbuf-ml` is a downstream profile crate. It consumes canonical, validated
vBuf state and generic checked ranges; it does not define or override generic
physical validity.

Dependency direction is one-way:

```text
vbuf-ml -> vbuf-core / vbuf-layout
vbuf-core -X-> vbuf-ml
vbuf-layout -X-> vbuf-ml
```

The crate currently defines a small profile bootstrap, tensor-directory semantic
index, model-metadata index, and vocabulary-only tokenizer view. These add only
ML names, relationships, shapes, and domain validation over canonical generic
values; they do not duplicate physical descriptors. Quantization, tokenizer
algorithms, merges, backends, placement, and conversion remain deferred.
A request from this descendant is not by itself
a reason to promote a feature into generic vBuf; promotion requires independent
downstream-neutral utility and generic qualification.

The boundary follows:

```text
raw bytes -> canonical v0.6 validation -> checked ranges -> bootstrap roles -> tensor semantics
```

The bootstrap and tensor directory are located through canonical generic
blocks. Their semantic references resolve by generic Key-ID and physical
occurrence, never by trusted profile offsets.

Step 11 keeps `VocabularyOnly` normative and qualifies lazy versus eager
access with file-backed mmap fixtures. Step 12 keeps `CanonicalPrimitive`
normative while quantized layouts await a pinned first target and upstream
revision. Step 13 adds only deterministic downstream writer ordering and
payload-alignment planning; BaseStep remains generic. Runtime-local indexes and future algorithm-specific structures remain
derived or deferred. Optional SHA-256 payload integrity is a separate semantic
role and is verified only when a runtime requests it. Partial loading derives runtime-local
read plans from canonical checked ranges; mmap and positioned-read paths remain
separate from semantic selection.

Step 16 real-model placement qualification is documented in
[`step16-real-model-placement.md`](step16-real-model-placement.md). It analyzes
the ignored local Qwen3-0.6B Q8_0 and BF16 GGUF artifacts without conversion or
runtime integration. Step 17 pins llama.cpp commit
`4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c` and qualifies exact F32, BF16, and
Q8_0 contracts in [`step17-upstream-representation-qualification.md`](step17-upstream-representation-qualification.md).
Step 18 builds deterministic, read-only conversion manifests without writing
model bytes; see [`step18-conversion-manifest.md`](step18-conversion-manifest.md).
Step 19A resolves the pinned Qwen3 KV-head and key/value head-dimension
metadata gaps. Step 19B qualifies the pinned GPT2/Qwen2 tokenizer path and
adds the `Gpt2BpeQwen2` storage profile; see
[`step19b-qwen3-tokenizer.md`](step19b-qwen3-tokenizer.md). Step 20 implements the deterministic manifest-driven writer; see
[`step20-gguf-conversion.md`](step20-gguf-conversion.md). Step 21 adds the
validated Rust/C vBuf consumer adapter and proves BF16/Q8_0 CPU parity through
the pinned llama.cpp/GGML runtime. See
[`step21-llama-consumer.md`](step21-llama-consumer.md). Step 22 records the
neutral unoptimized CPU baseline without changing the loader or format; see
[`step22-baseline-benchmark.md`](step22-baseline-benchmark.md). Step 22A
attributes the warm adapter/model-construction overhead without optimization;
see [`step22a-loader-overhead-attribution.md`](step22a-loader-overhead-attribution.md).
Step 23 adds a source-neutral llama construction seam and direct vBuf source;
see [`step23-native-llama-source.md`](step23-native-llama-source.md).
The Nano/direct-view audit is documented in
[`nano-runtime-audit.md`](nano-runtime-audit.md). Step 24 borrowed runtime
views are documented in
[`step24-borrowed-runtime-views.md`](step24-borrowed-runtime-views.md). Step 25
runtime-local tokenizer indexes are documented in
[`step25-runtime-tokenizer-index.md`](step25-runtime-tokenizer-index.md). Step 26
Qwen3-32B placement qualification is documented in
[`step26-qwen32b-placement.md`](step26-qwen32b-placement.md). Step 27 controlled
loader scaling and first-use attribution is documented in
[`step27-loader-scaling.md`](step27-loader-scaling.md). Step 28 layer-span
prefetch and wave-readiness qualification is documented in
[`step28-layer-prefetch.md`](step28-layer-prefetch.md). Step 29 host-relative
layer compute and bulk-I/O qualification is documented in
[`step29-layer-io.md`](step29-layer-io.md). Reopened, **PARTIAL** Step 30
reconstructable-weight research is documented in
[`step30-reparameterization.md`](step30-reparameterization.md). The corrected
full tensor/group/layer algorithm matrix is documented in
[`step30-structured-matrix.md`](step30-structured-matrix.md), with the 22-family
completion and residual-quantization continuation in
[`step30-continuation.md`](step30-continuation.md). The downstream-only
alignment-slack audit is documented in
[`step30-slack-audit.md`](step30-slack-audit.md). The bounded Qwen3-32B
Contextual Correction Code assessment is recorded in
[`../../benchmark-results/vbuf-ml-step30-ccc-assessment/representation-assessment.md`](../../benchmark-results/vbuf-ml-step30-ccc-assessment/representation-assessment.md). CCC Stage-2 canonical-quantizer and real-hidden-state evidence is in
[`../../benchmark-results/vbuf-ml-step31-ccc-canonical/stage2-report.md`](../../benchmark-results/vbuf-ml-step31-ccc-canonical/stage2-report.md). Independent-branch reconciliations are recorded in
[`../../benchmark-results/vbuf-ml-step32-ccc-reconciliation/reconciliation-report.md`](../../benchmark-results/vbuf-ml-step32-ccc-reconciliation/reconciliation-report.md) and
[`../../benchmark-results/vbuf-ml-step33-ccc-structured-reconciliation/reconciliation-report.md`](../../benchmark-results/vbuf-ml-step33-ccc-structured-reconciliation/reconciliation-report.md), and
[`../../benchmark-results/vbuf-ml-step34-ccc-c4-reconciliation/reconciliation-report.md`](../../benchmark-results/vbuf-ml-step34-ccc-c4-reconciliation/reconciliation-report.md).
Tokenizer execution and chat-template rendering remain consumer/runtime responsibilities.

The base remains a small compositional vocabulary: efficient composition is
preferred over maximal base functionality.
