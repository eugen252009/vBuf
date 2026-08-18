# CPU_REPACK Allocation Scope

The pinned llama request of `2711642112` bytes is a shared backend buffer
created during model loading. `ggml_backend_alloc_ctx_tensors_from_buft_impl`
walks one ggml context, sums `GGML_PAD(ggml_backend_buft_get_alloc_size(t),
alignment)` for each unallocated tensor, and passes the sum to
`alloc_tensor_range`. For this DeepSeek GGUF context, CPU_REPACK contains 27
IQ4_NL tensors: `blk.0.ffn_down.weight` followed by
`blk.1.ffn_down_exps.weight` through `blk.26.ffn_down_exps.weight`.

The request is exactly the sum of their allocated spans. `blk.0.ffn_down.weight`
starts at offset zero and occupies `12607488` bytes; the next tensor starts at
that exact offset. No padding was observed between these placements.

The CPU_REPACK allocator delegates to `ggml_aligned_malloc`, and the resulting
buffer is retained in `llama_model::pimpl->ctxs_bufs` for the model lifetime.
The audit exited after placement, before any `set_tensor` call, so reserved,
committed, and touched physical memory are not distinguished here.

The prior comparison of llama's `2711642112` bytes with vBuf's `12607488`
bytes was an invalid cross-scope comparison. The tensor-local execution storage
needed for block 0 is `12607488` bytes, but the pinned llama allocation owner
is a shared backend buffer. A future vBuf reproduction must first choose a
valid shared-buffer or tensor-local backend seam that preserves this ownership
and lifetime distinction; this task does not choose or implement that seam.
