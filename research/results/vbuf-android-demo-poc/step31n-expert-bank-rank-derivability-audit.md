# Step 31N Expert-Bank Rank Derivability Audit

Date: 2026-08-21
Branch: `vbuf-ml`
Starting commit: `964ca45`
Status: **AUDIT COMPLETE; NO RUNTIME OR FORMAT CHANGE**

## Scope and Answer

Step 31N determines whether the third dimension of the routed expert tensors
is canonical vBuf-ML semantic information or a backend view over a family of
equal-shaped expert matrices.

The answer is:

```text
CANONICAL_VBUF_ML_BANK_GEOMETRY: PRESERVED AS RANK-3 TENSOR DESCRIPTORS
RANK3_SEMANTIC_REQUIREMENT_FOR_TensorRef: NO
RANK3_BACKEND_VIEW: LOSSLESSLY DERIVABLE FROM CANONICAL BANK + SELECTION
PHYSICAL_LAYOUT: CONTIGUOUS_FIXED_STRIDE_BANK
ZERO_COPY_RANK3_VIEW: POSSIBLE WHEN THE FULL BANK SPAN IS AVAILABLE
CURRENT_RANK2_LOWERING: CORRECT FOR INDIVIDUAL EXPERT EXECUTION
CURRENT_LOSS: EXPLICIT BANK-MEMBER PROVENANCE IS NOT CARRIED BY THE
  ISOLATED TensorWaveGraphView
```

The importer does not permanently replace the canonical rank-3 expert bank
with independent rank-2 semantic tensors. The rank-2 form is created by the
qualification execution path after selection. No new canonical rank-3
requirement is justified by `ggml_mul_mat_id`.

## Evidence Classification

- **CODE-AUDITED:** the GGUF importer records source shape, source type, source
  offset, and source payload size without rewriting tensor payload bytes.
- **CODE-AUDITED:** the vBuf-ML tensor directory and C ABI expose the original
  rank, dimensions, representation, and full payload range.
- **CODE-AUDITED:** DeepSeek and Qwen MoE loaders require routed expert tensors
  with rank-3 shapes and expert count on dimension 2.
- **CODE-AUDITED:** `make_expert` derives a rank-2 selected view by slicing the
  full bank at `expert * (bank_payload / expert_count)`.
- **CODE-AUDITED:** `PersistentTensorRef` retains the selected source range,
  tensor ID, and name, but not the parent dimensions or explicit expert ordinal.
- **HOST-VERIFIED:** the offline derivation helper checked all 78 routed banks
  across 26 MoE layers in the qualified DeepSeek manifest.
- **DERIVED:** per-expert strides, offsets, spans, quantized row strides, and
  zero-copy descriptor feasibility.
- **NOT_IMPLEMENTED:** no grouped backend operation, TensorWave change,
  materializer change, TensorRef change, persistence change, residency change,
  or model-format change.

## Original Imported Geometry

The Step 31M source is the qualified real artifact:

```text
SOURCE: DeepSeek-V2-Lite.IQ2_XXS.gguf
SOURCE_SHA256: 3b7da33584bebf89afcdbdd2e7a8e3e47e11092971559371f13b510f475e3c0c
SOURCE_SIZE_BYTES: 5640619552
IMPORT_MANIFEST: /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS-manifest.json
EXPERT_COUNT: 64
MOE_LAYERS_CHECKED: 26
```

The original GGUF tensor records and the vBuf-ML conversion manifest use the
following geometry for every routed layer:

| Bank | Original name suffix | Original rank | Original dims | Expert axis | Expert count | Per-expert dims | Type | Total payload | Per-expert payload |
|---|---|---:|---|---:|---:|---|---|---:|---:|
| Gate | `ffn_gate_exps.weight` | 3 | `[2048, 1408, 64]` | 2 | 64 | `[2048, 1408]` | IQ2_XXS | 47,579,136 | 743,424 |
| Up | `ffn_up_exps.weight` | 3 | `[2048, 1408, 64]` | 2 | 64 | `[2048, 1408]` | IQ2_XXS | 47,579,136 | 743,424 |
| Down | `ffn_down_exps.weight` | 3 | `[1408, 2048, 64]` | 2 | 64 | `[1408, 2048]` | IQ4_NL | 103,809,024 | 1,622,016 |

The source manifest is built by `scripts/build_step18_manifest.py`. Its
`tensor_plans` copy `tensor.shape`, `tensor.type_name`,
`tensor.absolute_start`, and `tensor.payload_size` into `source_shape`,
`source_ggml_type`, `source_offset`, and `source_payload_bytes`. The converter
then emits the same shape and range into the vBuf-ML tensor directory with
`payload_action: COPY_BYTES`.

The third dimension is therefore a bank-member axis: it names one of 64
same-shaped expert matrices in the tensor family. It is semantically relevant
to expert selection, but rank 3 itself is not an additional primitive required
by vBuf-ML. The semantic facts are the bank identity, member count, member
shape, and mapping from expert ordinal to bytes.

## Rank-3 to Rank-2 Trace

| Boundary | Rank | Dims | Expert index known | Parent bank known | Base offset known | Stride known |
|---|---:|---|---|---|---|---|
| GGUF parser / import manifest | 3 | Full bank shape | No selected member yet | Yes, tensor name and source tensor record | Yes, source offset | Derivable as payload / dimension 2 |
| vBuf-ML `TensorDescriptor` | 3 | Full bank shape | No selected member yet | Yes, descriptor identity and `TensorRef` | Yes, `TensorRef.offset()` | Derivable from shape, type, and payload |
| `VbufMlTensorView` C ABI | 3 | Full bank shape | No selected member yet | Yes, name and full payload view | Physical range API exposes it | Derivable, not a field |
| POC `Meta` lookup | 3 | Full bank shape | Selection supplies it later | Yes, `id`, name, full payload, full offset | Yes | Derivable |
| `make_expert(full, expert)` | 2 local view | First two full dims | Yes, function argument and `slice_offset` | Yes, local full name/id/payload/offset | Yes, local full offset | Yes, `bytes = full_len / 64` |
| `ExpertTensor::ref()` | 2 | Selected matrix dims | No explicit ordinal in returned ref | Name and tensor ID still equal bank identity | Selected offset only | No field; derivable only with parent bank |
| `TensorWaveGraphView.persistent` | 2 | Selected matrix dims | No | Name and tensor ID only | Selected `source_offset` | No |
| `MaterializedTensor` | 2 | Copied selected dims | No | No parent relation in value | No separate parent base | No |
| `BorrowedGgmlTensor` in current adapter | 2 | Selected matrix dims | No | No parent relation | Bound selected payload only | No bank stride |

The exact lowering code is in
`integrations/ggml/tools/router_driven_moe_poc11.cpp:196-210`:

```text
result.dimensions = { full.dimensions[0], full.dimensions[1] }
result.bytes = full.view.payload_len / full.view.dimensions[2]
result.slice_offset = result.bytes * expert
```

The selected `PersistentTensorRef` is then made at lines 94-98 from the
selected payload pointer, selected length, rank 2, and `offset + slice_offset`.
`multi_expert_moe_poc12.cpp:136-151` creates one such graph member per selected
TopK rank. The TensorWave executor only sees those rank-2 persistent refs.

The information-loss boundary is therefore the transition from the local
`ExpertTensor`/`make_expert` state to the isolated `PersistentTensorRef` stored
in `TensorWaveGraphView`. The bytes are not lost. The canonical bank descriptor
and explicit member relation are no longer present in that graph view.

The Rust semantic path has a different boundary: it does not lower the bank to
rank 2. `DeepSeekMoELoader` and `QwenMoELoader` validate rank-3 descriptors
directly, and `VbufMlTensorView` reports the original rank and dimensions.

## Physical Layout Proof

The source contains one tensor payload per bank, not 64 independent GGUF tensor
records. The importer uses `COPY_BYTES`, so it does not reorder bytes within a
bank. Quantized payload geometry gives the exact plane stride:

```text
IQ2_XXS:
  row bytes = (2048 / 256) * 66 = 528
  gate/up plane bytes = 1408 * 528 = 743,424
  bank bytes = 64 * 743,424 = 47,579,136

IQ4_NL:
  row bytes = (1408 / 32) * 18 = 792
  down plane bytes = 2048 * 792 = 1,622,016
  bank bytes = 64 * 1,622,016 = 103,809,024
```

For each bank, member `i` is the byte interval:

```text
expert_offset(i) = bank_base + i * expert_stride
expert_end(i)    = expert_offset(i) + expert_payload_bytes
```

The offline helper `scripts/audit_step31n_expert_bank_rank.py` checked:

```text
BANKS_CHECKED: 78
LAYERS_CHECKED: 26
GATE_LAYER_1_STRIDE: 743424
UP_LAYER_1_STRIDE: 743424
DOWN_LAYER_1_STRIDE: 1622016
ALL_OFFSETS_MATCH_FORMULA: YES
ALL_BANK_SPANS_EQUAL_PAYLOAD: YES
ALL_BANKS_NON_OVERLAPPING: YES
EXPERT_ORDER: 0,1,2,...,63
```

Representative source ranges are:

| Bank | Base offset | End offset | Span | Adjacent member delta |
|---|---:|---:|---:|---:|
| Layer 1 gate | 348,536,352 | 396,115,488 | 47,579,136 | 743,424 |
| Layer 1 up | 396,115,488 | 443,694,624 | 47,579,136 | 743,424 |
| Layer 1 down | 244,727,328 | 348,536,352 | 103,809,024 | 1,622,016 |

There are no per-expert headers or interspersed unrelated bytes inside any
bank span. The manifest treats each bank as one source tensor and the exact
packed representation contract accounts for every byte. Thus the layout class
is:

```text
PHYSICAL_LAYOUT: CONTIGUOUS_FIXED_STRIDE_BANK
EXPERTS_CONTIGUOUS: YES
EXPERT_INDEX_MATCHES_PHYSICAL_ORDER: YES
EXPERT_INDEX_MAPPING_REQUIRED: NO; identity 0..63 is canonical
```

This conclusion is about the bytes inside each bank. Gate and up are separate
banks; they are not physically interleaved as per-expert gate/up pairs. A
combined gate/up bank would require a different layout or a pointer-pair
operation and is not derivable as one contiguous rank-3 tensor from the two
existing banks.

## Zero-Copy Rank-3 View

For the existing full bank payload, a backend descriptor can be constructed
without changing bytes:

```text
rank = 3
dims = [D0, D1, expert_count]
base = canonical bank payload pointer
nb[0] = quantized block byte size
nb[1] = row bytes
nb[2] = per-expert payload bytes
total span = expert_count * nb[2]
```

The GGML `ggml_view_3d` API explicitly accepts `nb1`, `nb2`, and an offset.
For this bank, the view can use the existing payload as its base with no
repacking. The quantized row width is block-aligned and every expert plane is
an integer number of rows.

```text
ZERO_COPY_RANK3_BACKEND_VIEW_POSSIBLE: YES
PAYLOAD_COPY_REQUIRED: NO when the full bank span is materialized/borrowed
WEIGHT_REPACK_REQUIRED: NO
MODEL_FORMAT_CHANGE_REQUIRED: NO
```

This does not mean selected rank-2 materialization can be stitched into a
rank-3 view. Six independently allocated selected slices do not form one
contiguous allocation. A future grouped backend must either retain/materialize
the full bank span or use a different pointer-array/grouped representation.
That is an acquisition and working-set decision, not evidence that rank 3 is a
canonical TensorRef requirement.

## GGML `ggml_mul_mat_id` Contract

The checked-in GGML source documents and asserts the following:

```text
as  -> [cols, rows, n_expert]
b   -> [cols, n_expert_used, n_tokens]
ids -> [n_expert_used, n_tokens] with GGML_TYPE_I32
c   -> [rows, n_expert_used, n_tokens]
```

The constructor requires:

- `as` is non-transposed and rank 3 with `as->ne[3] == 1`;
- `b` has `b->ne[3] == 1`;
- `ids->ne[2] == ids->ne[3] == 1`;
- `ids->ne[1] == b->ne[2]`;
- `as->ne[0] == b->ne[0]`;
- `ids->ne[0] % b->ne[1] == 0`, allowing broadcast activation columns;
- each ID is in `[0, as->ne[2])` in the CPU implementation.

The CPU implementation does not require a separate per-expert header or a
copy. It indexes the selected bank plane as `src0->data + cur_a * nb02` and
uses `nb00 == ggml_type_size(src0->type)` and `nb10 == ggml_type_size(src1->type)`
assertions to reject permuted matrix rows. The output must be contiguous in the
normal GGML sense. The implementation supports the IQ2_XXS and IQ4_NL dot
traits used by the Step 31M source.

Therefore the current bank is compatible with the raw GGML contract. The
current TensorWave adapter is not compatible because its operation enum only
contains ordinary `MulMat`, and its persistent input is already a selected
rank-2 view.

## Compatibility Matrix

| Requirement | Current vBuf payload/state | Derivable | Missing at isolated TensorWave graph boundary |
|---|---|---|---|
| Expert count | `MoeParameters` and rank-3 dim 2 | Yes | Bank relation not carried |
| Rank-3 dims | `TensorDescriptor.dimensions` | Yes | Full descriptor not carried |
| Bank base | `TensorRef.offset()` | Yes | Parent base unavailable from selected ref alone |
| Expert stride | Full payload / dim 2 | Yes | Not stored in selected ref |
| Per-expert payload | Representation contract and bank payload | Yes | Selected length only is present |
| Dtype/quantization | Tensor representation enum | Yes | Retained on selected ref |
| Quant block layout | Representation contract | Yes | Backend can derive from dtype/dims |
| Selected IDs | TopK selection in runtime | Yes | Not in `PersistentTensorRef`; selection remains outside graph |
| Physical order | `COPY_BYTES`, GGML axis-2 convention | Yes | No mapping table needed |
| Contiguous storage | One source tensor payload span | Yes | Full span may not be materialized |
| Source identity | `TensorRef.source_id`, tensor ID, name | Yes | Identity is retained but not enough alone |

The current canonical state contains all numeric information needed for exact
derivation. The missing object at the backend lowering boundary is one
backend-neutral parent-bank/member relation, not a new rank-3 persistent
format. It can be represented conceptually as:

```text
bank identity -> canonical bank descriptor
selected member -> expert ordinal / existing source offset
```

All bank shape, base, stride, payload length, and representation values can be
looked up or derived from the canonical descriptor. The selected ordinal is
already present in routing state and can also be checked from
`(selected_offset - bank_base) / stride`.

## Shape Versus Provenance

The rank-3 descriptor and an indexed tensor family describe the same underlying
payload here:

```text
rank-3 backend view:
  [D0, D1, E] + base + strides

backend-neutral family:
  bank identity + element shape [D0, D1] + count E + canonical range
  + member ordinal(s)
```

The family form is the more general semantic boundary. It supports:

- Backend A: iterate selected ordinals and issue ordinary rank-2 matmuls.
- Backend B: construct the rank-3 GGML view and I32 ID tensor.
- Backend C: construct grouped GEMM pointer arrays or a device-native bank
  descriptor.

The runtime should own the family identity, member selection, ordering,
lifetime, and source/residency decisions. Each backend may choose its own
execution view. GGML must not dictate source acquisition, persistence, tensor
lifetime, residency, expert naming, or router semantics.

## Architecture Classification

```text
OPTION_A_KEEP_RANK2_ONLY: INSUFFICIENT FOR THE CURRENT ISOLATED GRAPH VIEW;
  the lowerer needs the canonical bank/member relation somewhere.
OPTION_B_KEEP_RANK2_PLUS_BANK_PROVENANCE: RECOMMENDED;
  preserve a backend-neutral parent-bank/member relation until lowering while
  keeping selected execution tensors rank 2.
OPTION_C_INDEXED_TENSOR_FAMILY: SEMANTICALLY VALID GENERALIZATION of B if more
  backends or multi-token grouped execution need the relation.
OPTION_D_CANONICAL_RANK3_REQUIRED: REJECTED; rank 3 adds no bytes or semantics
  that cannot be represented by the bank family relation.
```

The practical recommendation is **Option B**, with the design allowed to grow
into Option C if the runtime needs a first-class indexed family for reasons
independent of GGML. This is not a request to change `TensorRef` now. It is a
provenance requirement for any future grouped lowering seam.

The historical rank-2 decision was not wrong. A selected expert is naturally a
rank-2 matrix for ordinary matmul, and the current path remains correct. Step
31M found an additional raw-backend opportunity; it did not invalidate the
semantic or runtime boundary.

## Backend Independence

```text
BACKEND_A_RANK2_LOOP: COMPATIBLE
  Use bank family members as selected rank-2 views and preserve TopK rank.

BACKEND_B_GGML_MUL_MAT_ID: COMPATIBLE
  Lower the same bank family to [D0,D1,E] plus I32 IDs after full-span
  availability is established.

BACKEND_C_GROUPED_GEMM: COMPATIBLE
  Lower the same bank family to pointer arrays, explicit strides, or a device
  bank descriptor without exposing GGML types upward.
```

The common abstraction is not GGML rank 3. It is a homogeneous indexed bank
with canonical source/provenance and selection order.

## Required Boundary Changes

No current change is required merely to establish derivability:

```text
GGML_RUNTIME_ASSUMPTIONS_LEAK_UPWARD: NO
TENSORREF_CHANGE_REQUIRED: NO
MATERIALIZER_CHANGE_REQUIRED: NO for the audit or rank derivation
PERSISTENCE_CHANGE_REQUIRED: NO
RESIDENCY_CHANGE_REQUIRED: NO
MODEL_FORMAT_CHANGE_REQUIRED: NO
```

If a future grouped implementation is authorized, it must add a transient
backend-neutral lowering input that retains the canonical bank relation and
selection IDs. It must not make `TensorRef`, `MaterializedTensor`, or the
persistent model format GGML-aware. It must also explicitly choose whether to
materialize the full bank or keep the current selected-slice acquisition path.

## Round-Trip Result

The offline helper constructed ephemeral rank-2 member records from each
canonical bank, reconstructed rank 3, and compared the result to the original
manifest record:

```text
OFFLINE_RANK3_DERIVATION_TEST_ADDED: YES
ROUNDTRIP_RANK3_RANK2_RANK3_PASS: YES
GATE_ROUNDTRIP_PASS: YES
UP_ROUNDTRIP_PASS: YES
DOWN_ROUNDTRIP_PASS: YES
ALL_78_BANKS_PASS: YES
DERIVED_GEOMETRY_MATCHES_ORIGINAL: YES
DERIVED_OFFSETS_MATCH_ORIGINAL: YES
DERIVED_EXPERT_ORDER_MATCHES_ORIGINAL: YES
```

The test is structural and manifest-driven. It does not execute inference and
does not claim that a future Android backend will be faster.

## Required Classification Fields

```text
ORIGINAL_EXPERT_BANK_RANK: 3
ORIGINAL_EXPERT_BANK_DIMS: gate/up [2048,1408,64]; down [1408,2048,64]
EXPERT_AXIS: 2
EXPERT_COUNT: 64
CURRENT_SELECTED_EXPERT_RANK: 2
CURRENT_SELECTED_EXPERT_DIMS: gate/up [2048,1408]; down [1408,2048]
RANK3_TO_RANK2_BOUNDARY: make_expert -> ExpertTensor::ref -> PersistentTensorRef
CURRENT_RUNTIME_PRESERVES_PARENT_BANK_IDENTITY: YES in canonical TensorDirectory;
  NO as a complete descriptor in isolated TensorWaveGraphView
CURRENT_RUNTIME_PRESERVES_EXPERT_INDEX: YES in MoeDirectory/TopK selection;
  NO as an explicit PersistentTensorRef field
CURRENT_RUNTIME_PRESERVES_BANK_BASE: YES in canonical TensorRef;
  NO in selected TensorRef alone
CURRENT_RUNTIME_PRESERVES_EXPERT_STRIDE: DERIVABLE, NOT STORED in selected ref
CURRENT_RUNTIME_PRESERVES_EXPERT_COUNT: YES in MoeParameters and bank dim 2
PHYSICAL_LAYOUT: CONTIGUOUS_FIXED_STRIDE_BANK
EXPERTS_CONTIGUOUS: YES
FIXED_EXPERT_STRIDE: YES
EXPERT_STRIDE_BYTES: gate/up 743424; down 1622016
TOTAL_BANK_SPAN: gate/up 47579136; down 103809024
EXPERT_INDEX_MATCHES_PHYSICAL_ORDER: YES
EXPERT_INDEX_MAPPING_REQUIRED: NO
GGML_MUL_MAT_ID_REQUIRED_RANK: lhs rank 3; rhs rank 3; ids rank 2
GGML_EXPERT_AXIS: lhs dimension 2
GGML_REQUIRES_CONTIGUOUS_BANK: contiguous per-plane rows and valid nb02;
  current full bank is contiguous
GGML_SUPPORTS_EXISTING_QUANTIZATION: YES, IQ2_XXS and IQ4_NL CPU traits
GGML_ACCEPTS_ZERO_COPY_VIEW: YES at raw descriptor level
RANK3_DERIVABLE_FROM_CURRENT_STATE: YES from canonical bank + selection;
  NO from isolated selected PersistentTensorRef alone
DERIVED_RANK3_DIMS: exact original dims
DERIVED_BANK_BASE: exact canonical source offset
DERIVED_EXPERT_STRIDE: exact payload / expert count
DERIVED_TOTAL_SPAN: exact canonical payload length
DERIVED_GEOMETRY_MATCHES_ORIGINAL: YES
DERIVED_OFFSETS_MATCH_ORIGINAL: YES
DERIVED_EXPERT_ORDER_MATCHES_ORIGINAL: YES
GATE_BANK_DERIVABLE: YES
UP_BANK_DERIVABLE: YES
DOWN_BANK_DERIVABLE: YES
RANK3_INFORMATION_LOST: NO canonical bytes/geometry; YES explicit relation at
  isolated TensorWave graph boundary
MISSING_INFORMATION: canonical parent-bank descriptor is not carried with the
  selected rank-2 TensorWave persistent refs
MINIMUM_PROVENANCE_REQUIRED: one backend-neutral bank-member relation; all
  numeric geometry is already canonical or derivable
ZERO_COPY_RANK3_BACKEND_VIEW_POSSIBLE: YES when full bank span is available
PAYLOAD_COPY_REQUIRED: NO
WEIGHT_REPACK_REQUIRED: NO
MODEL_FORMAT_CHANGE_REQUIRED: NO
RANK3_IS_CANONICAL_SEMANTIC_REQUIREMENT: NO
RANK3_IS_BACKEND_VIEW: YES
RANK2_DECISION_WAS_WRONG: NO
RECOMMENDED_ARCHITECTURE: KEEP_RANK2_PLUS_BANK_PROVENANCE
BACKEND_A_RANK2_LOOP_COMPATIBLE: YES
BACKEND_B_GGML_MUL_MAT_ID_COMPATIBLE: YES after backend lowering
BACKEND_C_GROUPED_GEMM_COMPATIBLE: YES after backend lowering
GGML_RUNTIME_ASSUMPTIONS_LEAK_UPWARD: NO
TENSORREF_CHANGE_REQUIRED: NO
MATERIALIZER_CHANGE_REQUIRED: NO for derivability
PERSISTENCE_CHANGE_REQUIRED: NO
RESIDENCY_CHANGE_REQUIRED: NO
MODEL_FORMAT_CHANGE_REQUIRED: NO
OFFLINE_RANK3_DERIVATION_TEST_ADDED: YES
ROUNDTRIP_RANK3_RANK2_RANK3_PASS: YES
GATE_ROUNDTRIP_PASS: YES
UP_ROUNDTRIP_PASS: YES
DOWN_ROUNDTRIP_PASS: YES
STEP31N_REPORT_UPDATED: YES
STEP_BY_STEP_UPDATED: YES
CHECKLIST_UPDATED: YES
ANDROID_RUN_REQUIRED: NO
RUNTIME_BEHAVIOR_CHANGED: NO
MODEL_ARTIFACTS_COMMITTED: NO
COMMIT_PERFORMED: STEP31N_COMMIT
PUSH_PERFORMED: NO
```

## Verification Boundary and Next Step

The audit helper is the only new executable logic and is not connected to
inference. The next implementation experiment, if separately authorized, is a
host-only transient backend-lowering prototype that consumes the canonical bank
descriptor plus existing TopK selection, constructs either ordinary rank-2
views or a GGML rank-3/ID view, and compares structural metadata and output
parity. It must not change TensorWave, TensorRef, materialization, persistence,
residency, or the model format until a separate qualification decision.
