# Experimental PORTABLE_SEMANTIC_V1

This is an experimental lowering boundary, not a vBuf v0.6 format change or a
permanent cross-model standard. It is proven descriptively against Qwen3.6,
Phi-4-mini, and Gemma 3, and imported for the qualified DeepSeek-V2-Lite
manifest.

## Primitive/operator kinds

`Embedding`, `MatMul`, `IndexedMatMul`, `Norm`, `Activation`, `Add`, `Mul`,
`Div`, `Reshape`, `View`, `Permute`, `Concat`, `Repeat`, `RoPE`, `Attention`,
`Softmax`, `TopK`, `Gather`, `Scatter`, `Combine`, `StateRead`, `StateWrite`,
`Dependency`, `ResidualAdd`, and `LMHead`.

Composite operations such as attention, SSM, and MoE are allowed only when
their attributes, tensor dependencies, state effects, and dynamic alternatives
are explicit. No family-name operation is part of v1.

## Tensor bindings

Each semantic binding has a stable semantic key, a generic TensorRef, shape,
representation contract, and optional generic storage view. Source names and
backend representation IDs are importer provenance only. A view may select a
slice of shared immutable storage without duplicating payload bytes.

## Aliases

An alias relates two semantic roles to the same underlying TensorRef/storage.
Weight tying is persistent semantic truth. Backend views, repacking, and buffer
handles are excluded.

## State

StateRefs carry kind, shape/schema, sequence/layer scope, lifetime, access,
initialization/reset, and explicit read/write transition semantics. KV key and
value state, recurrent state, convolution state, and position state are
optional. Runtime contents are transient and never serialized as model truth.

## Position and masks

Position transforms carry semantic parameters such as theta/base, rotary
dimensions, scaling, local/global scope, and context switch. Attention carries
causal visibility and optional local window/global scope. Concrete masks,
frequency tables, and cache layouts are derived runtime/backend state.

## Routing

Dynamic alternatives contain a runtime selector and selector-to-TensorRef
alternative mapping. Router outputs and current selections are transient;
expert catalog roles, selection policy, and architecturally observable capacity
rules are persistent importer semantics.

## Dependencies and regions

Portable execution is a dependency graph of values, tensors, state reads/writes,
and region boundaries. Layer/expert labels are importer hints, not runtime
control flow. Acquisition derives from required dependencies.

## Intentionally unsupported

The current v1 experiment does not standardize arbitrary tokenizer pipelines,
multimodal vision graphs, sparse layouts, distributed cuts, backend fusion,
device placement, source scheduling, or a persistent wire encoding. These are
explicitly outside vBuf v0.6.
