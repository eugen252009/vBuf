# vBuf-Native Region Runtime Audit

## Scope And Evidence

This is a source and architecture audit only. No llama.cpp, ggml, vBuf format,
model parameter, or runtime behavior was changed. The pinned source was inspected
read-only on `ssh pi` at:

```text
/home/eugen/llama-vbuf-pinned
```

The pinned source identity documented by this repository is
`4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`. Relevant local evidence and seams
are:

- `integrations/llama.cpp/vbuf_direct_source.{h,cpp}`
- `patches/llama.cpp/0002-source-neutral-model-source.patch`
- `patches/llama.cpp/0003-copy-user-tensor-data-to-backend.patch`
- `docs/vbuf-ml/step23-native-llama-source.md`
- `docs/vbuf-ml/step28-layer-prefetch.md`
- `research/vbuf-rv2-memory-ownership-audit.md`

## Executive Answer

Yes: the smallest viable direction is a vBuf-native execution runtime retaining
ggml as the tensor, graph, scheduler, backend, and kernel substrate. The
current full-model requirement is not a ggml invariant. It is primarily imposed
by llama.cpp's model loader/model object, architecture graph builders, model-
scoped backend buffers, and context/KV lifetime.

The smallest useful cut is:

```text
Rust/vBuf runtime
    ModelView and validated tensor/range access
    model semantics and region tensor selection
    ExecutionRegion construction
    activation and architecture state ownership
    CPU/GPU residency and prefetch policy
    region sequencing and placement
             |
             | small C/C++ ggml execution shim
             v
ggml
    tensor descriptors and operations
    graph construction/allocation
    backend graph execution and scheduling
    backend transfers/events
    CPU/RVV and CUDA kernels
```

This is `VBUF_RUNTIME_ON_GGML`, not a modified llama runtime and not a new
compute stack. The significant work is extracting/reimplementing the
architecture graph recipe for a region, not replacing ggml.

The first proof must be CPU-only and local: one real DeepSeek layer, borrowed
or range-loaded weight storage, a ggml graph containing only that layer, and an
intermediate activation compared with the pinned llama reference. Full
inference, distributed execution, and network protocols are later stages.

## 1. Responsibility Boundary

### Current dependency map

```text
vBuf artifact
    |
    v
Rust ConsumerModel / BorrowedModelView
    |
    +-- validated metadata, tokenizer, tensor descriptors, payload views
    +-- physical tensor ranges and layer spans
    |
    v
VbufDirectSource : llama_model_source
    |
    v
llama_model_loader
    |
    +-- full tensor inventory and weights_map
    +-- architecture and hparams
    +-- vocabulary and tokenizer
    +-- model tensor descriptors
    |
    v
llama_model / llama_model_base
    |
    +-- architecture implementation and all layer weight references
    +-- device selection and static tensor placement
    +-- model-scoped backend buffers
    |
    v
llama_context
    |
    +-- KV/memory state and sequence state
    +-- graph builder and graph reservation
    +-- ggml backend scheduler
    +-- prompt/decode/output path
    |
    v
ggml
    |
    +-- tensor descriptors, operations, views, graphs
    +-- graph allocation and reuse
    +-- backend buffers and tensor binding
    +-- backend scheduling/splits/transfers
    |
    +-- CPU backend / RVV kernels
    +-- CUDA backend / CUDA kernels
```

### Responsibility table

| Subsystem | Current responsibility | Full model required? | Region reuse |
|---|---|---:|---:|
| vBuf Rust consumer | Validate artifact, expose semantic views and payload ranges | No, although current `ConsumerModel` opens the full mmap | Directly reusable |
| `llama_model_loader` | Full metadata, tensor inventory, architecture lookup, weights map, tensor registration, load callbacks | Yes in current API | Not region-local |
| `llama_model` | Own architecture-wide hparams, all model tensors, model buffers, device placement | Yes | Not reusable as the region owner |
| Architecture classes | Convert all model tensors and global hparams into a full graph recipe | Yes in current class contract | Recipe can be extracted per region |
| `llama_context` | KV/cache, sequence state, graph reserve, scheduler, decode lifecycle | Context-wide state, not all weights by principle | State concepts reusable; object is too coupled |
| ggml tensor descriptors | Type, shape, strides, operation graph relationships, data/buffer association | No | Directly reusable |
| ggml graph | Nodes and dependencies present in that graph | No | Directly reusable |
| ggml graph allocator | Transient graph tensor allocation/reuse | No | Directly reusable |
| ggml backend buffers | Storage for bound tensors; backend-specific ownership | Only for tensors in the graph | Directly reusable with region buffers |
| ggml backend scheduler | Assign graph nodes, split execution, copy intermediates, synchronize | No | Directly reusable |
| CPU/RVV backend | Execute supported ops over CPU tensor data | No | Directly reusable |
| CUDA backend/kernels | Execute supported ops over device tensors and transfers | No | Directly reusable, but device weights need upload |

The source-neutral llama seam confirms the distinction. The current patch adds a
semantic source interface with typed metadata, full `tensor_count()`, indexed
tensor descriptors, and borrowed payload pointers
(`patches/llama.cpp/0002-source-neutral-model-source.patch:146-209,
488-533`). It enters the ordinary model builder at the loader/model boundary;
it does not create a region executor.

## 2. Where Full-Model Assumptions Enter

### Loader and model

The current semantic-source constructor explicitly enumerates every tensor and
populates the model-wide `weights_map` before model creation. The current
adapter does the same in `llama_vbuf_loader.cpp:144-165` and
`vbuf_direct_source.cpp:120-127`.

The source-neutral patch then proceeds through the common path:

```text
llama_model_loader
 -> load_hparams
 -> load_vocab
 -> llama_model_base / architecture object
 -> load_tensors
 -> backend buffers
 -> llama_context
 -> graph/decode
```

This is a useful compatibility seam, but its contract remains model-wide.
`llama_model_source` has no region, layer range, tensor lease, acquire, release,
or residency operation.

### Architecture graph builders

`/home/eugen/llama-vbuf-pinned/src/llama-model.cpp:2472-2489` calls
`build_arch_graph()` and then adds pooling, sampling, dense output, and final
outputs. The DeepSeek implementation is
`/home/eugen/llama-vbuf-pinned/src/models/deepseek2.cpp`.

Its main graph builder loops over all layers:

```text
deepseek2.cpp:470    for (int il = 0; il < n_layer; ++il)
deepseek2.cpp:474    attention norm
deepseek2.cpp:487-642 attention / MLA / KV operations
deepseek2.cpp:648-689 residual and dense/MoE FFN
deepseek2.cpp:692-699 final residual and next-layer input
deepseek2.cpp:702+   final output norm and output head
```

The graph recipe is not intrinsically required to mention every layer, but this
implementation is written as a complete model graph and takes a complete
`llama_model` object. A vBuf region runtime must supply an equivalent recipe
whose tensor references are limited to the selected region and whose input and
output are explicit.

### Model-scoped buffers

The pinned `llama_model_base::load_tensors` path creates backend buffers before
calling the payload loader and retains them in the model. That is why the
current vBuf path copies borrowed pointers into backend-owned storage. The
explicit patch is:

```text
destination = t->data
set_tensor_data(t, ...)
source = t->data
t->data = destination
ggml_backend_tensor_set(t, source, 0, ggml_nbytes(t))
```

Source: `patches/llama.cpp/0003-copy-user-tensor-data-to-backend.patch:4-15`.

This is a llama loader policy, not evidence that ggml requires a full model
copy. The memory audit measured the resulting source/destination coexistence.

## 3. Partial Graph Execution

### Direct ggml graph construction

ggml builds a graph by recursively expanding dependencies from a selected output
(`ggml_build_forward_expand`, pinned `ggml/src/ggml.c:7199-7201`). If the
selected output depends only on a region's operations and input activation,
the graph need not contain operations or weight tensors from other regions.

The graph API itself has no complete-model registry. It operates on the tensor
objects referenced by the graph.

### Existing graph views are not region ownership

`ggml_graph_view` (`ggml/src/ggml.c:7388-7404`) creates a shallow node-range
view over an existing graph. It reuses the original node array, use counts, and
tensor objects. This can represent a scheduler split, but it does not create a
standalone region with independent descriptor or weight lifetime.

Therefore:

- A graph containing only layer N is feasible if constructed directly.
- A graph view of a full llama graph is not sufficient as the architecture
  boundary by itself.
- A layer N..M graph must receive its incoming activation and any required
  state as explicit graph inputs.
- The graph does not require the complete model tensor set if the region graph
  does not reference it.

### Allocation and release

The ggml graph allocator API explicitly supports graph-local allocation:

- `ggml_gallocr_reserve()` reserves from a measure graph.
- `ggml_gallocr_alloc_graph()` allocates graph tensors and resets/reuses its
  allocation buffers (`ggml/src/ggml-alloc.c:1051-1097`).
- `ggml_backend_sched_alloc_graph()` and
  `ggml_backend_sched_graph_compute()` provide scheduler integration.
- `ggml_backend_sched_reset()` is documented to deallocate the previous graph
  allocation and leave its tensors with dangling pointers; callers must discard
  those tensors and create new ones (`ggml/include/ggml-backend.h:339-351`).

This is enough for a region runtime to own a short-lived ggml context, graph,
transient compute allocation, and region tensor descriptors. It must obey the
following lifetime rule:

```text
construct descriptors and graph
    -> bind weight/input/state storage
    -> allocate graph
    -> execute
    -> wait for backend completion
    -> discard graph descriptors and compute allocation
    -> release region weight buffer
```

For asynchronous GPU execution, the release point is after backend events have
completed, not immediately after graph submission.

### What must outlive execution

- Every tensor descriptor referenced by the graph.
- Every weight buffer and its underlying bytes.
- Input activation and persistent state until the final consumer operation.
- Backend buffers and event objects until asynchronous work completes.
- The graph context and operation metadata until graph submission/compute has
  completed according to backend contract.

No object outside the selected graph's dependencies is required by ggml merely
because it exists in the model file.

## 4. Activation And State Boundaries

For a standard transformer region, the conceptual operation is sufficient in
principle:

```text
execute_region(region, input_activation, execution_state)
    -> output_activation, updated_execution_state
```

For the DeepSeek2 graph in this pinned source, the boundary is not only a hidden
activation.

### Transient per-region data

- Layer weight tensors, including attention, normalization, output, and FFN
  or expert weights.
- Attention projections and intermediate Q/K/V tensors.
- MoE router scores, selected expert intermediates, and temporary FFN values.
- Residual and normalization temporaries.
- ggml graph descriptors and compute scratch for the region.

### Persistent across region boundaries

- Hidden activation `inpL` / `cur` between consecutive layers.
- Token positions, sequence IDs, output IDs, attention masks, and RoPE-related
  inputs.
- KV/cache state for every layer whose state is needed on later tokens. The
  pinned `llama_kv_cache` is explicitly layer-indexed (`llama-kv-cache.cpp`
  accessors around `get_k_storage`, `get_k`, and `get_v`).
- DeepSeek MLA state and compressed K/V representations used by the selected
  attention form. `deepseek2.cpp:465-466, 539-604` constructs the relevant
  input and cache paths.
- Optional adapter/cvector state. The DeepSeek graph calls `build_cvec(cur,
  il)` at `deepseek2.cpp:694-695`.
- Architecture-wide hparams and tensor shape rules, although these are metadata
  and planning state rather than full weight storage.

### Boundary cases

- Embedding is before layer 0.
- Final output normalization and the language-model head are after the final
  transformer layer (`deepseek2.cpp:700-715`).
- A middle region can emit a hidden activation, but it cannot emit final logits
  without the final norm/head region.
- MoE routing is local to a layer, but the region must acquire all weights
  referenced by the graph's routing implementation or provide a verified
  selected-expert execution path.
- Models with recurrent state, state-space blocks, sliding-window variants,
  MTP blocks, or other non-transformer semantics may need more than a hidden
  activation and layer-local KV. The region ABI must not assume all models are
  simple contiguous transformer blocks.

For the current DeepSeek model, contiguous layer boundaries are viable for a
region execution contract, but the state object must explicitly include
layer-indexed KV and the architecture's optional state paths.

## 5. Weight And Storage Ownership

### ggml's actual invariant

The generic ggml tensor has descriptor fields and a `data` pointer, but backend
operations also rely on `tensor->buffer` being a valid backend buffer. The
generic backend binding path (`ggml/src/ggml-backend.cpp:2052-2065`) requires:

- no existing tensor buffer/data for `ggml_backend_tensor_alloc`;
- the supplied address to lie within the backend buffer range;
- backend tensor initialization to succeed.

This is a descriptor/storage association invariant, not a requirement that
storage be allocated by ggml's ordinary allocator.

### Borrowed CPU storage is already supported

The pinned CPU backend advertises:

```text
buffer_from_host_ptr = true
```

at `ggml/src/ggml-cpu/ggml-cpu.cpp:391-401`. Its implementation calls
`ggml_backend_cpu_buffer_from_ptr(ptr, size)` at `:417-422`.

The buffer wrapper in `ggml/src/ggml-backend.cpp:2341-2353` has:

- no-op `free_buffer`, explicitly because the pointer is not owned;
- normal tensor set/get operations;
- no special tensor initialization requirement.

`ggml_backend_cpu_buffer_from_ptr` requires tensor-aligned storage at
`:2428-2430`, then wraps the pointer without taking ownership. This is direct
evidence that CPU/RVV can retain externally owned immutable weight storage
through a ggml backend buffer. The current vBuf adapter does not use this API;
the llama patch instead copies into an ordinary backend buffer.

### Descriptor, storage, graph, backend ownership

| Object | Owner in current llama path | Small runtime owner |
|---|---|---|
| Model/tensor semantic metadata | `llama_model_loader` / `llama_model` | Rust `ModelView` plus region descriptors |
| ggml tensor descriptor | llama model/context ggml contexts | Region execution context |
| Weight bytes | ggml backend buffer, plus retained vBuf source | Borrowed vBuf map or region-owned range buffer |
| Compute graph | llama context/scheduler | Region execution object |
| Compute allocation | scheduler/gallocr | Region execution object |
| CPU backend buffer wrapper | ggml | Region-owned wrapper with external vBuf lifetime |
| CUDA device storage | CUDA/ggml backend | Residency manager, region lease |

### CPU versus CUDA

The host-pointer capability is backend-specific. In the pinned source:

- CPU: `buffer_from_host_ptr = true`.
- CUDA: `buffer_from_host_ptr = false` in `ggml-cuda/ggml-cuda.cu`.

Therefore CPU/RVV can plausibly execute directly from a vBuf-backed mapped
region after constructing a `CPU_Mapped` buffer wrapper. CUDA should be treated
as requiring device-owned region storage and an upload/copy path. This is not a
reason to force CPU into CUDA-style full-model copying.

## 6. Existing Multi-Backend And Pipeline Behavior

ggml already separates two concepts that the new runtime must also keep
separate.

### Graph split and activation transfer

`ggml_backend_sched_split_graph` in the pinned
`ggml/src/ggml-backend.cpp:1054-1484` assigns graph nodes to backends and
creates split graph views. `ggml_backend_sched_compute_splits` around
`:1594-1777` copies cross-backend inputs, waits on events where supported, and
executes each split.

This is already close to:

```text
backend A graph split
    -> activation transfer
backend B graph split
```

It is operation/activation scheduling, not dynamic weight residency.

### Static model placement

llama's `llama_meta_device_get_split_state` and related model code classify
tensors by names, layer IDs, shapes, split axes, and device count. The context
then enables pipeline mode only under device and capability conditions and
creates a scheduler with multiple copies/events (`llama-context.cpp` around
`:429-458, :607-641`; `ggml_backend_sched_new`).

This provides:

- static layer placement;
- backend graph splitting;
- activation/intermediate transfer;
- asynchronous event and copy machinery where supported.

It does not provide:

- a region acquire/release API;
- dynamic weight load or eviction;
- a model larger than all local weight storage;
- a graph whose architecture boundary is an externally supplied activation.

The missing capability is primarily dynamic region ownership and architecture
region construction, not basic partial computation or backend transfer.

## 7. Dynamic Residency Feasibility

### CPU direct mapped mode

The minimum-copy CPU path is feasible in principle:

1. Select one region's tensor payload ranges from the validated vBuf directory.
2. Keep the vBuf mmap or map only the selected range.
3. Create `ggml_backend_cpu_buffer_from_ptr` over the aligned region span.
4. Create only the region's ggml weight descriptors.
5. Bind each tensor address within that external buffer.
6. Build and execute the region graph.
7. Synchronize, discard graph/descriptor state, and release the buffer wrapper.

The current vBuf layout already has layer-major physical spans. Step 28 reports
one stable physical span per layer and three global spans, with no wire-format
change (`docs/vbuf-ml/step28-layer-prefetch.md:28-55`). This is sufficient for
region selection; no new persistent full-model index is demonstrated as
necessary.

However, a full-file mmap plus `madvise` is not a strict bounded-residency
mechanism. Step 28 measured that explicit page touching moves faults, while
`MADV_WILLNEED` did not reliably complete prefetch
(`docs/vbuf-ml/step28-layer-prefetch.md:81-115,154-165`). Strict local bounds
therefore need one of:

- range mappings that are unmapped after the region;
- exact range reads into a region-owned buffer;
- a carefully measured OS residency policy with an explicit bounded contract.

### Existing range-loading support

`rust/vbuf-ml/src/range_loading.rs` already provides:

- `ReadPlan` and physical-range coalescing (`:56-154`);
- `execute_plan()` that reads only planned ranges into owned byte vectors
  (`:187-197`);
- `MmapSource` and `PositionedFileSource` (`:200-256`).

This is enough to prove selective local residency without changing the vBuf
format. It is a source/loading facility, not yet a tensor/backend lease.

### CUDA and other accelerators

For CUDA, region weight bytes can be read into host staging storage, uploaded to
device buffers, and released after `ggml_backend_sched_synchronize()` or the
corresponding backend event. The existing ggml API has asynchronous tensor copy
and event interfaces (`ggml/include/ggml-backend.h:86-128`). The region runtime
must own the device buffer and ensure no queued graph operation references it
before eviction.

### Async loading and waves

A wave scheduler needs at least:

```text
region N compute lease
region N+1 load/prefetch lease
activation/state boundary
completion event or explicit synchronization
eviction after last use
```

The CPU backend in this pinned source advertises `async=false` and `events=false`.
Therefore ggml's CPU backend does not provide a backend event for overlapping
CPU compute with CPU loading. A separate loader thread and explicit ownership
barrier can still overlap file reads with CPU compute, but this is outside the
current scheduler contract and must be measured in a later POC.

For CUDA, backend async/events may support real copy/compute overlap, but the
runtime must use a bounded prefetch window and device memory accounting.

## 8. Strategy Comparison

### Strategy A: modify llama.cpp into a dynamic-residency runtime

Required changes:

- `llama_model_loader`: region-aware tensor acquisition, deferred materialization,
  source leases, and partial model inventory.
- `llama_model`: replace model-scoped weight ownership with region leases and
  dynamic device placement.
- architecture classes: add layer-range graph construction and explicit
  activation/state inputs/outputs.
- `llama_context`: separate persistent KV/state from transient region graph and
  weight storage.
- scheduler integration: rebuild/allocate region graphs and synchronize before
  eviction.
- public APIs: expose region execution and activation/state transfer.
- backend placement: distinguish static split from dynamic residency.

| Area | Scope |
|---|---|
| Loader/model ownership | DEEP |
| Architecture graph builders | DEEP |
| Context/KV lifecycle | DEEP |
| Public API | DEEP |
| ggml scheduler | MODERATE integration |
| CPU/RVV kernels | NONE |
| CUDA kernels | NONE |
| vBuf format | NONE |

This preserves llama's broad feature set but carries its model-wide assumptions
into the new design. It is not the smallest viable architecture.

### Strategy B: replace llama runtime, retain ggml

Rust/vBuf owns:

- validated `ModelView` and model semantics;
- tensor selection by region and physical ranges;
- region weight leases and residency policy;
- hidden activation, KV/cache, position, and sequence state;
- region ordering, placement, and future wave scheduling.

ggml owns:

- region tensor descriptors and operations;
- graph construction/allocation;
- backend buffer wrappers and tensor binding;
- scheduler graph splitting and activation transfer;
- backend synchronization/events;
- CPU/RVV and CUDA kernel dispatch.

The minimum ABI is not a general runtime framework. It is a narrow execution
shim exposing existing ggml operations:

```text
backend/device initialization and capability query
external CPU buffer or backend-owned region buffer creation
tensor descriptor creation/binding
ggml operation and graph construction for one region
graph allocation and compute
activation/state transfer and backend synchronization
buffer/graph/context release
```

There are two practical forms:

1. A small C ABI wrapper around ggml C APIs, with the DeepSeek region graph
   recipe implemented in C++ where existing llama graph helpers are useful.
2. A standalone C++ `DeepSeekRegionBuilder` linked against ggml, called from a
   Rust residency/orchestration layer.

The second is likely the smallest first step because the current architecture
recipe is C++ and uses DeepSeek-specific helpers. It still avoids
`llama_model`, `llama_context`, full model buffers, tokenizer runtime, and
llama's model loader.

| Area | Scope |
|---|---|
| vBuf ModelView/region selection | LOCAL |
| Rust residency manager | MODERATE |
| DeepSeek region graph recipe | DEEP but isolated |
| C/C++ ggml execution shim | LOCAL |
| ggml core | NONE |
| ggml CPU/RVV backend | NONE |
| ggml CUDA backend | NONE |
| llama.cpp runtime | NONE for production path |
| vBuf format | NONE |

This is the recommended strategy.

### Strategy C: replace llama.cpp and ggml

This would require replacing:

- tensor and graph representation;
- graph allocation and lifetime analysis;
- backend scheduler and split execution;
- CPU quantized kernels and RVV dispatch;
- CUDA backend and kernels;
- asynchronous copies/events;
- device memory management;
- operation correctness and numerical qualification.

No demonstrated requirement forces this. ggml already supplies the exact
compute substrate needed for a selected graph and has a borrowed CPU buffer API,
backend split scheduling, graph allocation reuse, and backend transfer APIs.

| Area | Scope |
|---|---|
| Graph/tensor substrate | FUNDAMENTAL |
| CPU/RVV kernels | FUNDAMENTAL |
| CUDA/backend support | FUNDAMENTAL |
| Numerical qualification | FUNDAMENTAL |
| vBuf format | NONE |

Strategy C is not justified by this audit.

## 9. Dependency-Cut Recommendation

The smallest supported cut is:

```text
Rust/vBuf
    ModelView
    region tensor/range selection
    ExecutionRegion state
    ResidencyManager
    WaveScheduler later
    PlacementPlanner later
        |
        v
isolated ggml region executor
    ggml contexts/tensors/ops
    ggml graph allocator
    backend buffer binding
    backend scheduler and transfers
    CPU/RVV/CUDA kernels
```

Do not use `llama_model_source` as the long-term region ABI. It is explicitly a
model-construction source interface and its contract requires a complete tensor
inventory and model-wide metadata. A region runtime should consume the vBuf
borrowed view/range APIs directly and create only the selected descriptors.

Do not fork or replace ggml's kernels. The CPU backend's `CPU_Mapped` buffer
wrapper is direct evidence that externally owned immutable storage is already a
supported backend concept. The current vBuf duplication is caused by the llama
loader patch choosing `ggml_backend_tensor_set`, not by a ggml prohibition.

## 10. Minimal Proof Of Concept

### POC 1: one real layer

Use the qualified DeepSeek model and the leading dense layer 0.

Steps:

1. Open and validate the vBuf artifact through the existing borrowed view.
2. Select only layer-0 tensors and their physical span.
3. Create ggml descriptors for layer 0 and an input activation.
4. Bind layer weights to a CPU mapped buffer or a range-owned buffer.
5. Build only the layer-0 ggml graph.
6. Execute on the CPU/RVV backend.
7. Return the `l_out` activation.

Correctness checks:

- tensor type, shape, byte range, alignment, and bound-buffer checks;
- no layer-1+ weight descriptor or destination buffer created;
- output activation shape and checksum;
- max absolute and relative difference against a pinned llama intermediate
  capture at layer 0;
- process RSS and `smaps` mapping breakdown;
- peak resident model-weight bytes less than the full model weight bytes.

### POC 2: sequential layer boundary

Execute layer 0, release its graph/weight lease after synchronization, then
execute layer 1 from the returned activation. For the DeepSeek model, layer 1
is an MoE layer, so this intentionally tests expert/router region selection.

Correctness checks:

- compare layer-0 and layer-1 intermediate outputs independently with the
  full llama reference;
- compare selected expert/router outputs if the graph exposes them;
- verify KV/cache updates for the selected token positions;
- verify no layer-0 destination buffer remains after its release point;
- verify resident weight RSS is bounded by the active region plus configured
  window.

### POC 3: selective residency

Use exact range loading or per-region mappings rather than relying only on
`MADV_WILLNEED`.

```text
layer 0 resident
layer 1 absent
    -> execute layer 0
    -> synchronize and release layer 0
    -> acquire layer 1
    -> execute layer 1
```

Correctness and memory checks:

- same intermediate and final activation results as POC 2;
- `/proc/<pid>/smaps` and per-mapping tables at acquire/execute/release;
- no full-model anonymous destination allocation;
- peak resident model-weight storage below the sum of all model weights;
- no use-after-release under ASan/UBSan where available;
- explicit backend synchronization before every release.

### POC 4: prefetch overlap

While region N executes, load region N+1 into a bounded host buffer or issue a
backend-supported asynchronous upload. Measure:

- load start/end;
- compute start/end;
- wait time at the region boundary;
- peak host and device residency;
- page faults and I/O;
- activation parity with the sequential control.

The CPU backend's lack of async/events means the first overlap experiment may
need an external loader thread and an explicit barrier. CUDA can use backend
events if its selected path supports them.

## 11. Risks And Unknowns

- **DeepSeek graph extraction:** the current graph recipe is architecture-
  specific C++ and includes MLA, KV, MoE, residual, and optional cvector paths.
  A region builder must preserve operation order and exact tensor semantics.
- **Intermediate reference capture:** current public llama APIs expose final
  outputs, not every layer activation. A temporary callback/instrumented
  reference is needed for POC correctness.
- **CPU mapped alignment:** `ggml_backend_cpu_buffer_from_ptr` requires
  `TENSOR_ALIGNMENT`; vBuf payload offsets and selected span bases must be
  checked, with a small aligned region base adjustment if needed.
- **Direct mapped residency:** a full mmap with demand paging does not provide
  a hard RSS bound. Exact range mapping or range-owned buffers are stronger.
- **MoE selection:** a region graph may reference all expert weights even when
  only top-k experts are used unless the graph recipe supports selected-expert
  execution.
- **KV placement:** KV can remain persistent in one backend or be partitioned by
  layer, but cross-device/network movement must be explicit later.
- **Backend differences:** CPU supports borrowed host pointers in this pinned
  source; CUDA does not advertise the same capability. The region runtime must
  make storage policy backend-specific.
- **Async lifetime:** `ggml_backend_sched_reset` leaves old graph tensors with
  dangling pointers. Region objects must be discarded and rebuilt, not reused
  accidentally after reset.
- **Non-transformer models:** recurrent/state-space and hybrid models may need
  a state transition contract that is not a simple hidden-activation boundary.
- **Distributed execution:** activation serialization and placement are future
  concerns; they should not be introduced into the local POC.

## 12. Final Recommendation

```text
RECOMMENDATION:
VBUF_RUNTIME_ON_GGML
```

Evidence:

- ggml can construct and execute a graph from only the tensors it references;
- ggml graph allocation can be reset/rebuilt independently of model semantics;
- ggml already has backend split, activation transfer, synchronization, and
  event mechanisms;
- the pinned CPU backend explicitly supports non-owned host-pointer buffers;
- ordinary RVV kernels already execute the original vBuf representations;
- vBuf already exposes validated borrowed views, physical ranges, and stable
  layer spans;
- current full-model residency and source/destination duplication arise in
  llama's model loader/model/context ownership path;
- replacing ggml would reimplement capabilities already present and qualified.

The next action should be the isolated POC 1, not a llama.cpp redesign and not a
full native compute stack.

## Addendum: Lower Architecture Semantics Before Runtime

### Addendum Result

The original recommendation should be refined from "a DeepSeek region builder"
to an importer/lowering step that produces a generic execution profile. The
runtime does not need to branch on `DeepSeek`, `Llama`, or `Qwen` if the profile
contains the required generic operations, tensor references, dependencies, and
state accesses.

```text
DeepSeek / Llama / Qwen importer
    |
    | architecture-specific validation and lowering
    v
generic execution profile
    |
    | tensor refs, ops, dependencies, state descriptors, locality hints
    v
vBuf-ML optional execution profile
    |
    v
architecture-neutral region planner/runtime
    |
    v
ggml backend graph and kernels
```

The importer may know the source architecture. The runtime need not retain that
knowledge as a dispatch condition. The generic profile must be expressive
enough for the real DeepSeek graph, including dynamic expert selection and
layer-indexed state updates.

```text
RUNTIME_REQUIRES_MODEL_ARCHITECTURE_KNOWLEDGE: NO
```

This is conditional on using a generic execution representation that includes
the primitives below. The current llama.cpp runtime does not have that profile;
its architecture-specific behavior is therefore an importer/lowering gap, not
proof that architecture-specific runtime code is necessary.

### 1. Architecture-Specific Behavior Found

The pinned DeepSeek implementation is in
`/home/eugen/llama-vbuf-pinned/src/models/deepseek2.cpp` and its shared graph
helpers are in `src/llama-graph.cpp`.

| Current behavior | Source evidence | Classification | Can lower before runtime? |
|---|---|---|---|
| Architecture name and hparams | `deepseek2.cpp:8-43`; `llama-model.cpp:1100+` | `IMPORT_TIME` | Yes. Store typed dimensions, counts, and attributes in the profile. |
| Tensor names, shapes, representations, layer grouping | `deepseek2.cpp:99-157`; vBuf tensor directory | `IMPORT_TIME` | Yes. Produce `TensorRef` records and dependencies. |
| RMS normalization | `deepseek2.cpp:474,563,651`; `llama-graph.cpp` norm helpers | `GENERIC_EXECUTION_SEMANTIC` | Yes, as `Norm(kind=RMS, epsilon, axes)`. |
| Linear projections and residual adds | `deepseek2.cpp:487-648,692` | `GENERIC_EXECUTION_SEMANTIC` | Yes, as `MatMul`, indexed `MatMul`, `Add`, and shape/view ops. |
| RoPE | `deepseek2.cpp:555-561`; `ggml_rope_ext` | `GENERIC_EXECUTION_SEMANTIC` | Yes, as `RoPE` with explicit position input and parameters. |
| Attention score/value computation | `deepseek2.cpp:602-641`; `llama-graph.cpp:2523-2602` | `GENERIC_EXECUTION_SEMANTIC` | Yes. Lower to primitive matmul/softmax/matmul or a generic `Attention` composite. |
| MLA absorption path | `deepseek2.cpp:566-604` | `GENERIC_EXECUTION_SEMANTIC` | Yes. It is a graph pattern and tensor-layout choice, not a required runtime architecture branch. |
| Dense leading FFN | `deepseek2.cpp:654-660` | `GENERIC_EXECUTION_SEMANTIC` | Yes, as generic gated FFN dataflow. |
| MoE router score calculation | `llama-graph.cpp:1947-2003` | `GENERIC_EXECUTION_SEMANTIC` | Yes, as matmul plus activation/normalization and selection. |
| Expert group/top-k selection | `llama-graph.cpp:2005-2034` | `GENERIC_EXECUTION_SEMANTIC` | Yes, as `TopK`/`Argsort` plus `Gather`/`SetRows`. |
| Expert indexed matmul | `llama-graph.cpp:2089-2215`; `ggml_mul_mat_id`/`build_lora_mm_id` | `GENERIC_EXECUTION_SEMANTIC` | Yes, as `IndexedMatMul` or gather + matmul. |
| Shared expert branch | `deepseek2.cpp:678-689` | `GENERIC_EXECUTION_SEMANTIC` | Yes, as an ordinary parallel branch and `Add`. |
| Expert weighted combination | `llama-graph.cpp:2222-2251` | `GENERIC_EXECUTION_SEMANTIC` | Yes, as `Gather`, `Mul`, and `Reduce/Add`. |
| Expert tensor range lookup | current `deepseek_moe.rs:20-68`; `moe.rs:62-90` | `IMPORT_TIME` | Yes. Lower to generic alternatives mapping selection IDs to tensor ranges. |
| KV/cache access | `llama-kv-cache.cpp:1210-1336`; DeepSeek attention inputs | `GENERIC_EXECUTION_SEMANTIC` | Yes, as `StateRead`/`StateWrite` with slices and indices. |
| Position/sequence state | `llama-context.cpp`, graph inputs, KV cache context | `GENERIC_EXECUTION_SEMANTIC` | Yes, as explicit state/input descriptors. |
| Cvector/adapters | `deepseek2.cpp:694-695`; `llama-context.h:284` | `GENERIC_EXECUTION_SEMANTIC` | Yes if modeled as optional state/input update; otherwise it is a profile feature. |
| Final norm and output head | `deepseek2.cpp:700-715` | `GENERIC_EXECUTION_SEMANTIC` | Yes, as ordinary graph nodes after the final region. |
| Fused attention or backend kernel choice | `ggml_flash_attn_ext`, CPU/RVV/CUDA backend implementations | `BACKEND_SPECIFIC` | Yes. Keep fusion as an implementation of a generic operation. |
| DeepSeek model-name dispatch | `MoeLoaderKind::for_architecture` in `moe.rs:49-58` | `IMPORT_TIME` | It should not be a runtime dispatch requirement. |

The only behavior that cannot be reduced to a fixed static tensor dependency
set is expert acquisition based on router output. That is data-dependent
planning, not irreducible architecture knowledge. It requires a generic dynamic
dependency/resource mechanism described below.

### 2. Minimum Generic Execution Representation

The minimum profile for the demonstrated DeepSeek graph is not a universal ML
compiler IR. It is a compact typed dataflow recipe with explicit storage and
state references.

#### Required tensor concepts

```text
TensorRef {
    id
    dtype / representation
    shape and strides
    immutable_weight | activation | state | temporary
    physical range or alternative ranges
}
```

The existing vBuf tensor directory already supplies most immutable-weight
fields. Shapes, representations, byte ranges, and alignment are
`REQUIRED_FOR_CORRECTNESS`. Layer/expert grouping is useful but not required if
the profile carries generic dependencies and ranges.

#### Required operation concepts

| Generic primitive | Current DeepSeek requirement | State interaction | Backend-independent? |
|---|---|---|---:|
| `MatMul` | Q/K/V, projections, dense FFN, router, output head | No | Yes |
| `IndexedMatMul` | `ggml_mul_mat_id` for selected experts | Reads runtime indices | Yes as a semantic op; backend implementation varies |
| `Add`, `Mul`, `Div` | Residuals, expert weights, scaling | No | Yes |
| `Norm` | RMS norms before attention/FFN/output | No | Yes |
| `Activation` | SiLU, sigmoid, softplus, model gating | No | Yes |
| `Reshape/View/Permute/Concat/Repeat` | MLA and expert layout construction | No | Yes |
| `RoPE` | Q/K rotary position encoding | Reads positions | Yes as a semantic op |
| `Attention` or decomposed attention ops | K/Q/V score, mask, softmax, value aggregation | Reads/writes KV state | Yes as a semantic op; fused implementation backend-specific |
| `Softmax` | Attention and normalized expert weights | No | Yes |
| `TopK/Argsort` | Expert and optional group selection | Produces dynamic indices | Yes |
| `Gather/GetRows` | Expert probability and selected-row extraction | Reads indices | Yes |
| `Scatter/SetRows/Combine` | Expert selection masks and output aggregation | May write temporary/state | Yes |
| `StateRead` | KV, positions, sequence metadata, optional cvector | Reads persistent state | Yes |
| `StateWrite` | KV/cache updates and optional state updates | Writes persistent state | Yes |
| `Dependency` | Dataflow and lifetime ordering | No | Yes |

`Attention` can be represented as a composite node whose lowering expands to
matmul/softmax/matmul, or retained as a generic fused semantic operation with
backend implementations. The runtime still does not need to know that the
operation came from DeepSeek MLA.

`IndexedMatMul` is the smallest useful generic MoE primitive if selective expert
residency is a goal. Expanding it into ordinary matmuls after routing is also
valid, but the planner then needs a generic dynamic graph phase.

#### Generic state descriptor

```text
StateDescriptor {
    id
    dtype
    shape
    lifetime: step | sequence | session | region-chain
    scope: global | stream | layer-indexed | region-indexed
    access: read | write | read-write
    placement: backend-neutral policy plus current placement
}
```

`StateRead` and `StateWrite` carry a descriptor ID, slice/index expression,
and dependency edge. The runtime does not need to know that state ID `kv.7`
means a DeepSeek K/V cache. The importer assigns the state descriptor and
connects it to the lowered graph.

### 3. Layer Versus ExecutionRegion

The addendum's distinction is correct:

```text
Layer
    semantic / physical grouping hint

ExecutionRegion
    actual runnable dependency subgraph
```

The lower-level runtime should accept:

```text
ExecutionRegion {
    required_nodes
    required_tensors
    inputs
    outputs
    state_reads
    state_writes
    dependencies
}
```

An importer may annotate a region with `layer_id`, `expert_id`, or physical
locality spans. Those annotations are placement and acquisition hints, not
required runtime control-flow concepts.

This matters for:

- embedding and output-head regions that are not transformer layers;
- expert-only subgraphs after router selection;
- fused or compiler-partitioned regions crossing a layer boundary;
- distributed regions cut at an activation dependency rather than a semantic
  layer boundary.

The runtime can expose convenience selectors such as `.layer(n)` and
`.expert(n)` in an importer/helper API, while the executor sees only node/tensor
IDs and dependencies.

### 4. Generic Persistent State

The four storage classes are distinct:

| Class | Lifetime | Runtime treatment |
|---|---|---|
| Immutable model weights | Region lease / model source lifetime | Acquire views or backend buffers; never mutate |
| Transient intermediates | One graph/region execution | ggml allocator-owned and recyclable |
| Persistent execution state | Across region chain/token steps | `StateDescriptor`-owned storage and explicit read/write |
| External activation input/output | Region boundary or transport boundary | Explicit input/output lease, copy, or serialization |

DeepSeek KV semantics can be lowered to generic state operations because the
current llama code already accesses cache storage by layer ID and tensor slice.
The importer records that relationship; the runtime performs state access and
placement without a DeepSeek cache class.

What remains architecture-specific is the lowering decision: which state IDs
exist, their dimensions, how the attention operation reads/writes them, and
which position/state transforms are required. That is import-time semantics,
not a runtime architecture branch.

### 5. MoE Without A DeepSeek Runtime Feature

The current generic graph helpers show the exact required dataflow:

```text
router MatMul
    -> activation / normalization
    -> optional group scoring
    -> TopK / Argsort
    -> selected expert IDs
    -> Gather expert weights
    -> IndexedMatMul expert projections
    -> activation and down projection
    -> multiply by selected probabilities
    -> combine selected expert outputs
    -> add shared-expert output
```

Evidence: `llama-graph.cpp:1947-2251`. The current ggml graph already makes
expert indices runtime data: `ggml_argsort_top_k`, `ggml_get_rows`,
`ggml_set_rows`, and `build_lora_mm_id` consume selected IDs.

There are two phases for selective expert residency:

```text
Phase 1: static graph/dataflow
    execute router and selection nodes
    produce selected expert IDs

Phase 2: data-dependent acquisition
    map selected IDs to generic TensorRefs/ranges
    acquire only those alternatives
    execute indexed expert subgraph
    release expert leases after synchronization
```

The smallest generic mechanism is an indexed/dynamic dependency edge:

```text
DynamicTensorSet {
    selector: TensorRef
    alternatives: selector_value -> TensorRef/range set
    consumer_nodes
}
```

This is a runtime resource-planning construct, not `DeepSeekExpert`. The
executor does not interpret architecture names; it evaluates the selector and
acquires the referenced alternatives. A simpler first POC may conservatively
acquire all experts in one region, then add the dynamic acquisition boundary.

The current `vbuf-ml/src/moe.rs` is already close to the desired storage-side
separation: it calls itself an architecture-neutral expert catalog and stores
layer/expert/role-to-child references. However, `MoeLoaderKind` and
`deepseek_moe.rs` still select/validate by architecture. Those are appropriate
import-time validators, not runtime primitives.

### 6. Persistent Representation Versus Planning

#### vBuf Core

Keep vBuf Core architecture-neutral and limited to generic binary structure,
validated ranges, physical blocks, and navigation. No graph or backend policy
belongs in core.

#### vBuf-ML existing profile

The existing tensor directory, model metadata, representation contracts, and
optional generic MoE catalog are useful. Classify them as:

- Tensor descriptor/type/shape/range: `REQUIRED_FOR_CORRECTNESS`.
- Alignment and representation contract: `REQUIRED_FOR_CORRECTNESS`.
- MoE catalog mapping to physical children: `OPTIONAL_OPTIMIZATION_HINT` for
  dense execution, `REQUIRED_FOR_CORRECTNESS` when using selective expert
  acquisition without rescanning names.
- Architecture string and architecture-specific validator: `IMPORT_TIME`, not
  runtime dispatch.
- Layer/expert physical grouping: `OPTIONAL_OPTIMIZATION_HINT`.

#### Optional lowered execution profile

If the goal is a runtime that does not know the source architecture, a generic
execution profile must exist somewhere after import. It can be:

- persisted as an optional vBuf-ML profile;
- kept as a separate sidecar/import artifact;
- derived at startup by an importer, in which case the importer remains a
  deployment dependency.

The profile should contain only:

- generic op descriptors and attributes;
- TensorRefs and dependencies;
- StateDescriptors and state access edges;
- region boundaries or arbitrary region node sets;
- dynamic selector-to-alternative mappings for MoE;
- optional locality/acquisition hints.

Op descriptors, dependencies, and state descriptors are
`REQUIRED_FOR_CORRECTNESS` for a standalone architecture-neutral runtime. They
are not required in vBuf Core and need not be canonical if a trusted importer
always runs first.

Backend buffer types, CUDA upload policy, CPU_REPACK choices, device IDs,
alignment workarounds, and prefetch windows are `BACKEND_SPECIFIC` or
`OPTIONAL_OPTIMIZATION_HINT`; they should not become canonical model semantics.

### 7. Dead-Work Elimination

A generic execution profile makes dependency pruning explicit:

```text
requested output/state update
    ^
backward dependency traversal
    ^
required generic nodes
    ^
required TensorRefs and StateRefs
    ^
acquisition set
```

For a dense region this is straightforward. Everything unreachable can remain
unloaded, unallocated, untransformed, untransferred, and unexecuted.

For MoE, the traversal is staged because the expert set depends on runtime
router output. The router is reachable first; the selected expert alternatives
become reachable only after `TopK`/selector evaluation. This provides a basis
for:

- selective layer execution;
- selective expert execution;
- bounded region residency;
- no repack for unselected representations;
- backend-specific materialization only for acquired tensors;
- activation-only inter-region transfer;
- later distributed execution at arbitrary dependency boundaries.

This is better described as execution tree shaking than ordinary graph
optimization. It can only eliminate work represented as a dependency or
dynamic-alternative edge. An opaque fused node must expose its required tensor
alternatives and state effects to the planner or be conservatively acquired.

### 8. Architecture-Aware Versus Lowered Runtime

| Property | Architecture-aware runtime | Lowered generic runtime |
|---|---|---|
| Initial implementation | Smaller first integration | Higher importer/profile cost |
| Runtime branching | DeepSeek/Qwen/Llama branches | Generic op/state/selector dispatch |
| Persistent format | Tensor metadata plus runtime code | Optional generic execution profile |
| Partial regions | Requires architecture graph changes | Region is a node/dependency subset |
| Expert residency | Architecture-specific router logic | Generic selector and alternatives |
| Dead-work elimination | Limited by hard-coded builders | Direct dependency traversal |
| Wave processing | Coupled to architecture runtime | Generic region leases and dependencies |
| Distributed execution | Requires per-architecture transport cuts | Activation/state edges are transportable |
| New architectures | New runtime graph/state code | New importer/lowering frontend |
| Debugging/reference | Easier against llama code | Requires profile parity checks |
| Long-term boundary | Retains llama assumptions | Keeps runtime architecture-neutral |

The lowered design has more upfront work but directly satisfies the requested
runtime contract. It also prevents vBuf-ML from becoming a collection of
architecture-specific runtime loaders. Existing `deepseek_moe.rs` and
`qwen_moe.rs` should be understood as import validators/contract checkers in
this design.

### 9. Revised Architecture-Neutral POC

The POC should not call a DeepSeek-specific execution function.

#### POC A: lower one real layer

```text
DeepSeek artifact
    -> DeepSeek importer validates metadata/tensor shapes
    -> importer lowers layer 0 into generic ops/TensorRefs/StateRefs
    -> generic profile is held in memory or serialized as a test sidecar
    -> generic runtime selects region node set
    -> CPU_Mapped ggml buffers bind vBuf payloads
    -> generic ggml graph executes
    -> activation output
```

The runtime must not inspect an architecture string. Correctness checks:

- profile validation and TensorRef byte-range checks;
- generic op graph contains no unresolved architecture operation;
- no tensors outside the lowered region are acquired;
- output activation shape, checksum, max absolute and relative error against
  a pinned llama layer-0 intermediate;
- backend buffer range/alignment checks;
- peak resident model weight bytes below full model weight size.

#### POC B: two generic regions

```text
generic region 0
    -> activation and generic state updates
generic region 1
    -> activation and generic state updates
```

Compare both intermediate activations and state writes with the pinned llama
reference. The runtime process contains no `model_arch == DEEPSEEK` branch.

#### POC C: generic dynamic expert alternatives

First lower a DeepSeek MoE layer conservatively with all expert alternatives,
then split the graph after the generic router/top-k selector. In the second
phase, acquire only the TensorRefs selected by the generic selector and compare
the combined output with the conservative graph.

This isolates the new requirement: data-dependent acquisition, not
architecture-specific execution.

#### POC D: generic wave prefetch

Use two generic `ExecutionRegion` objects and a residency manager. Prefetch the
next region while the current region executes where the backend permits it;
measure load/compute overlap, waits, peak RSS/VRAM, and output parity. No
network or distributed layer is needed.

### 10. Revised Recommendation

The addendum strengthens rather than weakens the previous recommendation:

```text
RECOMMENDATION:
VBUF_RUNTIME_ON_GGML
```

The boundary should be revised to include a generic lowered execution profile:

```text
architecture-specific importer
    -> generic execution profile
    -> vBuf-ML optional profile storage
    -> architecture-neutral runtime
    -> ggml graph/backend substrate
```

The burden of retaining architecture-specific runtime behavior is not met by
the current source. DeepSeek-specific code currently performs import-time
validation and constructs a graph recipe; its arithmetic/dataflow and state
effects can be represented generically. The concrete exception is not a
DeepSeek semantic but a generic dynamic-residency requirement: after a selector
produces IDs, the runtime must resolve and acquire data-dependent tensor
alternatives. That mechanism is necessary for selective MoE residency and
should be generic.

The runtime therefore needs to know generic operation semantics, tensor
dependencies, state effects, and dynamic acquisition edges. It does not need to
know whether the importer called the source DeepSeek, Llama, or Qwen.
