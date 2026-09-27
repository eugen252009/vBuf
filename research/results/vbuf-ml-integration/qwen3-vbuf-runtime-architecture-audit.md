# QWEN3 VBUF RUNTIME ARCHITECTURE AUDIT

Date: 2026-09-26

## Status

The pre-implementation audit identified a viable incremental architecture path. Since then, a small production metadata/dispatch seam and strict Qwen3 dense tensor catalog have been added and contract-tested. **No Qwen3 inference support is implemented or qualified**. The full 9 GB model has not been downloaded. Exact GGUF header/metadata and all 443 tensor descriptors were available from the previously fetched 64 MiB prefix, without reading the remaining weight file.

The audit corrects an overly broad interpretation in the earlier larger-model note: this repository already contains an architecture-neutral Rust graph seed/executor, generic CPU GQA attention, and a narrow C++ portable-graph adapter. Those pieces are useful reusable infrastructure, but they are **not the canonical C++ `VbufGenerationSession` execution path**, are not connected to the compatibility server, and do not support this Q4_K_M tensor inventory end to end.

## Current Direct Inference Path

`vbuf_compat_server.cpp` includes `autoregressive_poc22.cpp`; PoC22 includes the PoC16 layer runner and earlier POC operations. The server owns `VbufTokenizer` and one `VbufGenerationSession`.

The direct generation path is:

1. `load_metadata` opens the vBuf-ML artifact, gets borrowed tensor views/physical ranges, and retains the artifact buffer/handle.
2. `VbufGenerationSession` creates one `LayerPlan` per requested block. `make_plan` aliases each physical block's names to `blk.1.*` because the lower-level routines expect those exact names.
3. `run()` resolves `token_embd.weight`, `output_norm.weight`, and `output.weight` by name. It embeds one token, loops positions, runs one transformer layer at a time, computes final RMSNorm/output projection, and feeds the greedy token back.
4. For attention it calls the `AttentionTensors`/`compute_token` implementation in `attention_poc14.cpp`; for FFN it selects the special dense layer-0 implementation or the routed/shared MoE implementation in PoC13/16.
5. Payload lookup/materialization uses the shared vBuf `RangeSource`, `LocalVbufRangeMaterializer`, `ResidentTensorMaterializer`, and `TensorResidencyStore`; TensorWave wraps GGML CPU tensor operations.
6. KV is request-local, represented by dynamically sized `RuntimeStateSlot` vectors, but the direct path chooses fixed DeepSeek K/V widths and rebuilds KV on each new HTTP request.

The path now reads a validated architecture/config descriptor and rejects unsupported architectures before execution. The only executable graph remains the DeepSeek-specific one; Qwen3 catalog metadata is not yet bound to a graph. Low-level calls still encode DeepSeek assumptions.

## Architecture Boundary Found In Repository

`rust/vbuf-runtime` has an architecture-neutral `ExecutionGraph`, semantic lowering, request-local generic KV state, F32 CPU graph execution, and a CUDA backend. Its generic CPU implementation already contains RMSNorm, matrix multiplication, SiLU, elementwise multiply, head reshape, split-half RoPE, causal MHA/GQA, residual, and state behavior. The Step 32E GLM runner builds a model-specific graph over that API.

However:

- `rust/vbuf-runtime/README.md` explicitly says the portable graph is not bound to the C++ PoC22/GGML executor.
- The generic Rust materializer in the Step 32E runner handles BF16, F8_E4M3, and F32, not GGML Q4_K/Q6_K.
- Its ordinary F32 matmul is a scalar reference implementation, not the production path for 14.8B quantized weights.
- The C++ portable-graph GGML adapter is a separate, narrow adapter. It currently dispatches RMSNorm, MatMul/IndexedMatMul, SiLU, residual, TopK, and ABI-v2 attention; it does not implement RoPE/head reshape/elementwise multiply and is not called by `VbufGenerationSession` or the server.
- The portable graph C ABI has explicit attributes for attention but does not expose the Rust graph's full RoPE/head-reshape/elementwise operation attributes to C++.

This suggests a **small shared runtime plus architecture-specific block builders** is possible; it does not justify copying the PoC22 runtime or claiming model support from the graph seed alone.

## Component Classification

Legend: `GENERIC`, `DEEPSEEK-SPECIFIC`, `ARCHITECTURE-SPECIFIC BUT EXTENSIBLE`, `MISSING FOR QWEN3`.

| Component | Current direct path | Qwen3 requirement | Classification / evidence |
|---|---|---|---|
| Architecture dispatch | PoC22 now reads validated architecture/config metadata; it retains DeepSeek execution and rejects Qwen3 before entering that graph. | Select a Qwen3 builder, reject unsupported architectures explicitly. | Metadata dispatch seam **IMPLEMENTED**; Qwen3 execution builder **MISSING**. |
| GGUF/vBuf metadata | Direct generator reads architecture and numeric metadata through the consumer FFI into a neutral descriptor. | Use validated dimensions/norm/RoPE in the Qwen3 builder. | Metadata seam **IMPLEMENTED**; graph use of parameters **MISSING FOR QWEN3**. |
| Tensor discovery | Generic tensor views/range lookup plus a strict Qwen3 dense catalog for 443 tensors, shapes, representation IDs, and payload lengths. The selected artifact has no optional model tensors; every catalog entry is required. | Bind catalog entries into a Qwen3 layer plan. | Catalog contract **IMPLEMENTED AND UNIT-TESTED**; Qwen graph binding **MISSING**. It has not yet been run against a vBuf sidecar generated from the full artifact. |
| Tensor acquisition | `HttpRangeSource`, local vBuf range materializer, leases, bounded materialization and residency are shared. | Fetch source ranges using the vBuf-ML external-source profile and same materializer. | **GENERIC** and reusable. Do not introduce GGUF acquisition policy into a backend loader. |
| Tensor representation | Rust vBuf-ML representation enum and GGUF conversion plan include Q4_K (representation 6) and Q6_K (14). The pinned C++ adapter mapping has been expanded and contract-tested for both. | Bind Q4_K/Q6_K matrices and F32 norms to pinned GGML CPU. | Q4_K/Q6_K borrowed tensor mapping and zero-block CPU matmul **QUALIFIED**; nonzero real tensor row dequantization **QUALIFIED** against pinned GGML row-dequantization routines. Full model matmul/graph remains **MISSING FOR QWEN3**. |
| Embedding | `token_embd.weight` name is shared; row lookup/dequant graph hardcodes width 2048 and validates that geometry. | `[5120,151936]` Q4_K; output width 5120. | Row acquisition **GENERIC**; dimensions/quantized lookup **ARCHITECTURE-SPECIFIC BUT EXTENSIBLE**; current implementation **DEEPSEEK-SPECIFIC**. |
| Q/K/V projections | DeepSeek q projection plus compressed `attn_kv_a`, `attn_kv_a_norm`, `attn_kv_b`. | Separate `attn_q`, `attn_k`, `attn_v`; q/k per-head RMSNorm. | **DEEPSEEK-SPECIFIC** current; Qwen graph **MISSING FOR QWEN3**. GGML matmul and RMSNorm primitives are reusable. |
| Attention | `attention_poc14.cpp`: 16 heads, MLA `q_head=192`, 128 non-RoPE + 64 RoPE key dimensions, 512 latent KV, value width 128; DeepSeek attention scale. | Standard Q/K/V, 40 query heads, 8 KV heads, 128 dim, GQA ratio 5, scale `1/sqrt(128)`. | Current **DEEPSEEK-SPECIFIC**. Generic GQA exists in Rust reference executor and the separate C++ portable attention test; **not integrated** in PoC22. |
| RoPE | `deepseek_rotary` applies DeepSeek partial rotary/YaRN semantics and DeepSeek scale. | Full 128-dim Qwen RoPE, theta 1,000,000, no GGUF scaling. | Current **DEEPSEEK-SPECIFIC**. Split-half generic CPU/CUDA RoPE exists elsewhere; **missing from direct GGML path**. |
| Normalization | GGML RMSNorm is a reusable primitive, but width/epsilon/order are embedded in POC helpers. | Pre-attention 5120; q/k head norm 128; post-attention 5120; final 5120; epsilon 1e-6. | Primitive **GENERIC**; layer ordering/shapes **ARCHITECTURE-SPECIFIC BUT EXTENSIBLE**. |
| MLP | First block has a dense gate/up/down SwiGLU helper; later blocks require fixed DeepSeek MoE (64 experts, TopK 6, shared experts), width 2048. | All 40 blocks are dense SwiGLU; intermediate 17408; no MoE. | SwiGLU primitive **GENERIC**; current plan **DEEPSEEK-SPECIFIC**; parameterized dense plan is **ARCHITECTURE-SPECIFIC BUT EXTENSIBLE**. |
| MoE / routing | Fixed DeepSeek router, 64 expert identities, TopK=6, shared-expert tensors and accumulation rules. | No MoE metadata/tensors; do not route. | **DEEPSEEK-SPECIFIC**, optional in an architecture plan. Generic scheduler/storage must remain shared. |
| KV state | `RuntimeStateSlot` is generic storage, but direct allocation hardcodes K width `16*192` and V width `16*128`; F32; per-request. | Separate K/V, each `[positions,8,128]`, GQA mapping; retain across tokens; current 4096 cap. | Storage **GENERIC**; current geometry **DEEPSEEK-SPECIFIC**; Qwen layout **MISSING FOR QWEN3**. Rust/C++ portable test states support generic GQA but are not on direct path. |
| Output projection | `output_norm.weight`, `output.weight`; norm epsilon and hidden width are fixed; Vocab/rows handled by existing direct helper. | Qwen `output_norm` F32 and `output.weight` Q6_K `[5120,151936]` (GGUF axes). | **ARCHITECTURE-SPECIFIC BUT EXTENSIBLE**; Q6_K representation mapping exists, but complete Qwen head path is **MISSING FOR QWEN3**. |
| Tokenizer | Rust persists BPE vocabulary/merge/token types; server rebuilds a byte-BPE index. `encode_parts` does not scan token-type/control/user-defined strings as atomic added tokens. | Qwen GPT-2 BPE / `qwen2`, 151936 IDs; control and user-defined marker IDs must round-trip. | Base tokenizer data is **GENERIC**; Qwen special-token behavior **MISSING FOR QWEN3** until qualified against reference. |
| Chat template | Server implements a small Jinja-like subset of message loop/role/content, not arbitrary Jinja; message DTO only role/content. | Qwen template has tool-conditioned branch, JSON tool serialization, tool calls/results, thinking and history logic. | Current evaluator **MISSING FOR QWEN3**. Use a model conversation adapter; do not put tool semantics in graph/runtime. |
| Generation / output | Greedy direct loop is single-token decode and has DeepSeek block path; output stream emits text token callback. | Same greedy runtime seam, Qwen stop IDs, native call parser, structured `AssistantOutput`. | Loop/scheduler **ARCHITECTURE-SPECIFIC BUT EXTENSIBLE**; Qwen integration **MISSING FOR QWEN3**. |
| Materialization / residency | Shared vBuf-ML range materializer, leases, bounded cache and tracing. | Same tensor identity/source offsets, Q4_K/Q6_K ranges, limited active layer working set. | **GENERIC**, preserve. Note C++ `load_metadata` calls `read_file` on the semantic artifact; use a small external-source semantic bootstrap, not a self-contained 9 GB vBuf copy. |
| Layer scheduling / prefill | Direct PoC22 iterates positions and layers serially; PoC16 has row-batched FFN experiments, but direct server does not use the generic Rust graph runner. | Correct 40-layer ordered causal path; start with bounded single-token/staged blocks, optimize prefill later. | Scheduler/residency controls **GENERIC**; current layer call sequence **DEEPSEEK-SPECIFIC**. |
| CPU backend | GGML CPU TensorWave for supported matrix/norm/SwiGLU ops; explicit CPU backend selection. | Add quantized Q4_K/Q6_K matrix types and architecture operations; portable correctness first. | Backend **GENERIC**; type/operation coverage **MISSING FOR QWEN3**. |
| CUDA backend | C++ direct path selects CPU. Separate Rust generic graph CUDA backend and GLM qualification code are not the direct Qwen path. | Optional later; no reason to block CPU correctness. | **MISSING FOR QWEN3 DIRECT PATH**; no CUDA work now. |
| OpenAI/tool/SSE | OpenAI DTO/validation/serialization is model-independent; server rejects native tools until adapter. | Keep protocol generic; Qwen adapter maps native markers into normalized DTO. | Protocol **GENERIC / previously qualified**; Qwen conversation binding **MISSING FOR QWEN3**. |

## Exact Qwen3-14B Q4_K_M GGUF Metadata

Source candidate: `bartowski/Qwen_Qwen3-14B-GGUF`, commit `bd080f768a6401c2d5a7fa53a2e50cd8218a9ce2`, file `Qwen_Qwen3-14B-Q4_K_M.gguf`.

- Expected size: `9,001,753,632` bytes.
- Repository LFS SHA-256: `915913e22399475dbe6c968ac014d9f1fbe08975e489279aede9d5c7b2c98eb6` (remote repository metadata; not verified by a full local hash yet).
- Previously fetched prefix: bytes `0..67,108,863` (64 MiB) via HTTP Range. It includes GGUF header/metadata/tensor directory and first tensor payload bytes. It is not a complete/loadable artifact.
- Exact parsed GGUF fields: architecture `qwen3`; block count 40; context 32768; embedding 5120; FFN 17408; Q heads 40; KV heads 8; key/value dimensions 128; RoPE theta 1,000,000; RMS epsilon `1e-6`; GGUF file type 15 (`Q4_K_M`); vocabulary 151936; tokenizer model `gpt2`, pre-tokenizer `qwen2`; BOS 151643; EOS 151645; add-BOS false.
- Exact tensor inventory from tensor-directory metadata: 443 tensors: 40 × 11 layer tensors plus `token_embd.weight`, `output_norm.weight`, `output.weight`. Every layer has `attn_k.weight (Q4_K, 5120×1024)`, `attn_k_norm.weight (F32,128)`, `attn_norm.weight (F32,5120)`, `attn_output.weight (Q4_K,5120×5120)`, `attn_q.weight (Q4_K,5120×5120)`, `attn_q_norm.weight (F32,128)`, `attn_v.weight (Q6_K,5120×1024)`, `ffn_down.weight (Q6_K,17408×5120)`, `ffn_gate.weight (Q4_K,5120×17408)`, `ffn_norm.weight (F32,5120)`, `ffn_up.weight (Q4_K,5120×17408)`. Outer tensors are `token_embd.weight (Q4_K,5120×151936)`, `output_norm.weight (F32,5120)`, and `output.weight (Q6_K,5120×151936)`. Shapes above preserve GGUF dimension ordering.
- MLP activation is SiLU-gated (SwiGLU) from the exact upstream Qwen3 config (`hidden_act=silu`); the GGUF has no separate activation metadata field.
- No MoE/expert tensor names or MoE metadata are present; this is dense Qwen3.
- Template: exact GGUF `tokenizer.chat_template` length 4614 bytes, SHA-256 `5595e8741bb5500f9d395e05a0aefe6ca1e8d54c5dd8b98bcf7760d895fdc6e6`. The previous audit records its full tool schema/call/result behavior. User-defined tool strings have IDs `<tool_call>` 151657, `</tool_call>` 151658, `<tool_response>` 151665, `</tool_response>` 151666; `<|im_start|>` 151644 and `<|im_end|>` 151645 are control tokens.

## Concrete Graph Gap Table

| Feature | DeepSeek-V2-Lite path | Qwen3-14B requirement | Status |
|---|---|---|---|
| Tensor normalization/residual | GGML RMSNorm/residual; 2048 width; fixed `blk.1` names | 5120 width, four norm sites per block | Reuse primitives; new parameterized builder/catalog required |
| Q/K/V | Q projection + MLA KV-A/norm/KV-B | Separate Q/K/V matrices; per-head q/k RMSNorm | Qwen graph missing |
| Attention | MLA compressed state, 16 query heads, DeepSeek-specific scale | GQA 40/8, dimension 128, 5 Q heads per KV head | C++ portable GQA contract exists separately; direct execution missing |
| RoPE | 64 rotary dimensions in 192-wide MLA head plus YaRN/DeepSeek scale | Full 128 rotary dimensions, theta 1e6, standard Qwen mode | Generic reference exists; direct backend missing |
| MLP | Block 0 dense SwiGLU; later routed MoE and shared expert | Dense SwiGLU in all 40 blocks | Parameterize/share existing dense primitive; no Qwen builder yet |
| MoE | Router, score logic, TopK 6/64, shared experts | None | Must be optional; Qwen must not enter DeepSeek router |
| Quantization | Current C++ GGML binder supports existing format types | Q4_K/Q6_K/F32 | Q4_K/Q6_K mapping/CPU descriptor tests done; whole model path not qualified |
| Embedding/head | F32/quantized path fixed to width 2048 | Q4_K embedding and Q6_K output, width 5120 | Parameterize and add full graph use |
| KV | Float state sized `16*192` and `16*128`; request-local | Float state `[positions,8,128]` K and V | Reuse storage abstraction; pass architecture geometry |
| Tokenizer | BPE merges, basic special handling | Qwen user-defined/control token recognition | Added-token/special scanning and parity still missing |
| Conversation rendering | Restricted message-role/content template evaluator | Complex canonical tool template | Separate adapter; not yet implemented |
| Runtime/backend link | PoC22 calls POC13/14/16 directly | Architecture builder over common acquisition/CPU execution | Establish direct `ModelArchitecture` dispatch; preserve DeepSeek implementation |

## Qualification Baseline

Observed during audit, before Qwen production changes:

- `cargo test --manifest-path rust/Cargo.toml -p vbuf-runtime --lib`: **PASS**, 24 tests. Includes generic F32 MHA/GQA attention, state, generation, and lowering contracts; does not qualify a Qwen graph or Q4_K/Q6_K execution.
- Pinned-GGML (`2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`) CMake target `vbuf_portable_graph_adapter_contract`: **BUILT AND TESTED**, exit 0. Its attention test is a small MHA/GQA adapter contract, not a full block and not connected to server generation.
- `python3 scripts/verify_portable_graph_neutrality.py`: **PASS**, no model-name leakage in generic portable graph boundary.
- Full pinned-GGML CMake build: **PASS**, including the compatibility server and all probe/test targets. CTest: **30/30 PASS**, including the new architecture contract and DeepSeek semantics contract.
- The initially failing DeepSeek two-block rerun served the raw GGUF where the semantic sidecar requires the converted `.vbuf` payload. The wrong source produced a non-finite embedding row, later reported as `non-finite router score`. With the sidecar-matched `.vbuf` source, the current dirty worktree passes the complete HTTP regression suite twice; a separate clean worktree at the base revision also passes the exact request. No production numerical fix was needed. See [`deepseek-two-block-regression-investigation.md`](deepseek-two-block-regression-investigation.md).
- No Qwen inference or HTTP request was run.

## Refactoring Performed

Added `vbuf_model_architecture.{h,cpp}` as a separate direct-runtime integration library (not part of the generic `vbuf_region_executor`) with architecture identity/config validation and a strict Qwen3 dense tensor catalog; added a contract test covering the inspected 443-entry inventory, required names/shapes/types/payload lengths, duplicates, missing/extra tensors, and invalid ranges. The direct generation metadata loader now reads architecture/numeric metadata and Qwen3 tensor descriptors; Qwen3 is explicitly rejected before DeepSeek execution. The FFI struct mirrors the existing Rust `repr(C)` metadata contract.

This is an initial boundary only: it does not parameterize the DeepSeek graph, implement Qwen operations, or connect a Qwen plan to inference. Shared vBuf source/materialization/residency and GGML execution remain the intended services. DeepSeek behavior is unchanged in the graph code, but current-worktree HTTP regression did not pass (see qualification results).

## Full-Artifact Gate Decision

Metadata establishes the required Qwen computation without unknown exotic layers: standard Q/K/V projections, Q/K RMSNorm, split-half RoPE, standard GQA attention, dense SwiGLU, RMSNorm, and output projection. The required Q4_K/Q6_K storage types exist in the vBuf representation enum and pinned GGML type system, but the current C++ adapter mapping is missing those two types. These are bounded adapter/operator gaps with known representations, not evidence that the model requires replacing GGML or the storage runtime. Thus the operations are **reasonably implementable**, but not yet implemented.

Per the task gate, do not use a full-model load as a substitute for this work. The metadata/architecture seam and Qwen tensor inventory contract are now implemented; Q4_K/Q6_K binder/materialization contracts and direct-CPU operations are not. Continue with type-binding tests and isolated generic operations (Q/K RMSNorm, standard RoPE, GQA, dense SwiGLU) while resolving the DeepSeek regression. Acquire the full artifact outside the worktree only after those bounded contracts pass.

## Next Required Step

1. Isolate the current DeepSeek HTTP failure against the prior recorded baseline; preserve unrelated in-progress FFN/MoE changes and report actual results.
2. Add pinned-GGML Q4_K/Q6_K binding and payload-layout contract tests, independent of a model download.
3. Implement and unit-test the needed direct CPU operations and a small Qwen3 block builder while retaining DeepSeek dispatch unchanged.
4. Once these contracts pass, acquire and hash-verify the exact Q4_K_M artifact outside the worktree; build a sidecar and proceed to staged real-model text and tool qualification.

## Qualification Update — Verified Artifact and Source Binding

The earlier “full artifact not downloaded” and “no Qwen sidecar” statements are superseded by this dated update. The preceding DeepSeek regression is green with its correct `.vbuf` source; it was not repeated as an open investigation.

### Artifact and conversion identity

- Downloaded the exact candidate outside the worktree to `/home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.gguf`.
- Exact size `9,001,753,632` bytes; full SHA-256 **verified** as `915913e22399475dbe6c968ac014d9f1fbe08975e489279aede9d5c7b2c98eb6`, matching the pinned repository LFS identity.
- Generated and validated a Qwen conversion manifest and converted artifact with the repository’s vBuf-ML converter. Converted range payload: `Qwen_Qwen3-14B-Q4_K_M.vbuf`, size `9,000,232,144`, SHA-256 `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- Generated a 4,450,377-byte semantic bootstrap whose persisted source ID 1 declares exactly the converted `.vbuf` size/hash above. The consumer FFI identity was read and independently matched against `stat` and `sha256sum` of that payload. Therefore the serving payload for this sidecar is the converted `.vbuf`, **not** the GGUF.
- The converter was invoked with its Rust workspace symlinked from the external cache directory; all model artifacts/evidence remain outside the worktree.

### Runtime source identity enforcement

The current format already held a source SHA-256 and declared size, exposed through consumer FFI; previously the direct HTTP path did not apply them. The direct generation metadata loader now requires all bound tensors to use one source ID, checks each FFI source range against the validated tensor range, reads the persisted SHA-256 identity, and passes expected size/hash to `HttpRangeSource`. An identity-bound HTTP response must provide a matching full-source SHA-256 `ETag` (quoted or bare) and a `Content-Range` total equal to the sidecar-declared size. The range server emits an ETag from a one-time full-file SHA-256 computed at server startup; it is not recomputed per request or model open.

- Correct Qwen sidecar + converted `.vbuf`: actual HTTP range identity probe **PASS**.
- Correct DeepSeek sidecar + converted `.vbuf`: bounded two-block HTTP suite **PASS** after this check was enabled.
- DeepSeek sidecar + original `.gguf`: HTTP request returned 500 at embedding materialization before any generated tokens; it did not reach the former non-finite activation/router failure. The `.gguf` source has a different total size and is rejected by the response `Content-Range` identity check. A same-size wrong-SHA fixture and wrong-size fixture were also **REJECTED** by the identity-bound range client; a truncated-range fixture was rejected on length validation.
- Identity qualification relies on a trusted range endpoint truthfully binding its ETag to its complete served object. The local qualification server computes that value itself. Endpoints that do not advertise this SHA-256 ETag fail closed for identity-bound model sources.

### Real sidecar and quantized row checks

- Server startup against the real Qwen semantic bootstrap **PASS**: consumer open, source identity, metadata, and all 443 catalog descriptors validated without downloading model payloads over HTTP.
- Real Q4_K_M inventory exposed a correction to the original fixture assumption: `attn_v.weight` and `ffn_down.weight` are Q4_K in 20 layers and Q6_K in 20 layers each. The catalog now accepts Q4_K or Q6_K for those two mixed-quantized roles while retaining strict tensor names, shapes, payload lengths, and allowed representation checks. The other matrix roles retain their exact Q4_K/Q6_K requirements. Both fixture contract and real sidecar catalog pass.
- Added representation mapping for GGML Q4_K/Q6_K and CPU binding/zero-block matmul checks. `vbuf_tensor_adapter_qualification`: **PASS**.
- Using the sidecar’s real source offsets and actual converted payload bytes, the GGML CPU `ggml_get_rows` output was compared bit-for-bit with pinned GGML `dequantize_row_q4_K` for `token_embd.weight`, row 0 (5,120 values; 2,880 source bytes): **0 unequal values, max absolute difference 0**.
- The same comparison for `blk.0.attn_v.weight`, row 0, Q6_K (5,120 values; 4,200 source bytes): **0 unequal values, max absolute difference 0**.
- These are representation/range/row-dequantization checks, not a full projection or transformer-block reference qualification.

### Regression and current blocker

- Pinned-GGML Release build: **PASS**; CTest: **30/30 PASS**; `git diff --check`: **PASS**.
- DeepSeek bounded HTTP regression with correct sidecar-matched `.vbuf`: **PASS** (chat/completions/text SSE/protocol bounds). The wrong-source setup now fails closed at source identity validation rather than contaminating tensors.
- Qwen real tensor catalog/source identity/Q4_K/Q6_K row checks: **PASS**.
- Qwen3 direct generation graph and minimal block: **NOT IMPLEMENTED / NOT QUALIFIED**. The current direct `VbufGenerationSession` explicitly rejects Qwen3 before payload inference. Remaining graph work includes a parameterized Qwen layer plan; dynamic-width embedding gather; Q/K/V projections; per-head Q/K RMSNorm; Qwen RoPE; GQA state/attention; dense SwiGLU; residual ordering; and output path. No Qwen model inference request or full-block numerical comparison was attempted.
- Native tool calling and Pi remain **NOT STARTED**, as required. No generation, memory/VRAM, latency, or full-model runtime-resource qualification is claimed.

The next bounded implementation gate is an isolated Qwen3 single-position CPU graph/block builder over the existing vBuf-ML materializer and GGML operations, with explicit block geometry and numerical/reference checks. Preserve the DeepSeek builder and all unrelated in-progress changes; do not connect protocol/tool parsing or Pi until real Qwen text and block qualification is green.
