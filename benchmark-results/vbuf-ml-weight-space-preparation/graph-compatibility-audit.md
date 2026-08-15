# Graph Compatibility Audit

This is an analytical audit only. No graph was modified.

## Actual seam

`attn_norm-0` is weighted RMS-normalized hidden state and feeds all separate Q, K, and V projections, not K alone. Q/K subsequently receive per-head weighted RMSNorm and RoPE. Attention output and FFN output rejoin the unchanged residual stream.

## Permutation

Local K-only column permutation requires an explicit activation gather. A persistent hidden-coordinate permutation can be propagated only by coordinated conversion of embeddings, normalization parameters, all Q/K/V input columns, attention/FFN output rows, residual layers, and final output. Classification: `PERMUTATION_LAYOUT_FOLDABLE` only at whole-model scope; otherwise `PERMUTATION_RUNTIME_COST_REQUIRED`.

## Signed permutation

Pure signs are diagonal and can fold locally by compensating the `attn_norm`/QKV fan-out. The permutation component has the whole-model constraint above. Sign propagation must keep residual coordinates consistent; it must not be pushed through Q/K RoPE or FFN nonlinear intermediates.

## Diagonal scaling

Local nonzero scaling is a valid fan-out gauge: scale `attn_norm` output coordinates and inversely scale input columns of Q, K, and V (and active LoRA input factors). General nonuniform scaling is not a global RMSNorm symmetry. Classification: local `FOLDABLE_INTO_ADJACENT_WEIGHTS`, not a general residual-basis transform.

## Hadamard

Block Hadamard does not commute with learned diagonal RMSNorm weights. It requires an explicit post-norm activation transform or a coordinated whole-model orthogonal residual-basis conversion. Q/K outputs must remain unchanged to avoid head, GQA, KV-cache, and RoPE constraints. Classification: `GRAPH_PROPAGATION_CONSTRAINED`.

## Source references

- `/tmp/ccc-llama-pinned/src/models/qwen3.cpp:76-141`
- `/tmp/ccc-llama-pinned/src/llama-graph.cpp:1556-1658`
- `/tmp/ccc-llama-pinned/src/llama-graph.cpp:1707-1779`
- `/tmp/ccc-llama-pinned/src/llama-graph.cpp:2800-2817`
