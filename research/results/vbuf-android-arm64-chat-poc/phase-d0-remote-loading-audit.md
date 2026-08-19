# Phase D0 Remote Loading Audit

This document records the evidence and reasoning behind the next remote-loading
optimization work. It is intentionally a read-only audit: no runtime behavior,
vBuf format, vbuf-ML semantics, model semantics, or backend residency policy is
changed by D0.

## Scope

The audited path is Android ARM64 model open with a local semantic bootstrap and
tensor payloads served through HTTP Range requests. The pinned llama.cpp source
tree is `/tmp/llama.cpp-step21`.

The audit covers:

- semantic bootstrap discovery;
- tensor descriptor and payload materialization;
- HTTP Range transport;
- llama.cpp tensor construction and callback binding;
- backend allocation and payload ownership;
- `EAGER_ALL` residency;
- measured and unmeasured performance costs.

The audit does not authorize lazy residency, eviction, prefetching, batching,
format changes, or model-loader redesign. Those require a separately approved
architecture phase.

## Baseline And Phase C Evidence

The selected Phase C production configuration is C1 persistent HTTP connection
reuse:

| Metric | Result |
|---|---:|
| Model-open median | 44,451 ms |
| Final-state capture | 43,848 ms |
| Requests | 310 |
| Connections | 1 |
| Transferred bytes | 633,495,552 |
| Overfetch | 0 bytes |
| Effective throughput | 14.252 MB/s |
| Improvement over Phase B | 8,752 ms / 16.450% |

Phase C also established:

- a historical 256 MiB Phase C large-range source-transfer characterization of
  29.763 MB/s using Pixel toybox netcat over direct HTTP Range; a separate ARM64
  source-transfer snapshot used ADB reverse HTTP;
- complete model-open throughput of 11.907 MB/s in the Phase B baseline;
- a historical arithmetic ratio of 0.400 that is not directly comparable to the
  current direct-Wi-Fi Android HttpRangeSource path;
- positive-gap coalescing regressed to approximately 82-85 seconds;
- concurrency 2, 4, and 8 all regressed versus C1;
- all 310 payloads matched byte-for-byte;
- remote generation and local SELF regression passed;
- the backend residency policy remained `EAGER_ALL`.

The transport-only lower bound for the complete payload is approximately:

```text
633,495,552 bytes / 29.763 MB/s = approximately 21.3 seconds
```

This is only a lower bound. It does not isolate HTTP request overhead,
materialization, hashing, allocation, or llama.cpp model construction.

Source: `phase-c/README.md` and `phase-c/final-selection.json`.

## Exact Call Graph

The current remote open path is:

```text
MainActivity.openModel()
  -> NativeInference.open(...)
  -> llama_model_load_vbuf_remote(...)
  -> make_vbuf_remote_source(...)
  -> VbufRemoteSource::VbufRemoteSource(...)
       -> VbufMlAdapter(bootstrap_path, metadata_only=true)
       -> metadata, architecture, tensor descriptors, source offsets
       -> LocalVbufRangeMaterializer(HttpRangeSource)
  -> llama_model_init_from_source(...)
  -> llama_model_load(...)
  -> llama_model_loader(source, callback, ...)
       -> source->tensor(i, ...) for every source tensor
            -> VbufRemoteSource::materialize_tensor(i)
                 -> LocalVbufRangeMaterializer::request(...)
                 -> worker thread
                      -> posix_memalign(...)
                      -> HttpRangeSource::read_range(...)
                      -> payload hash
                 -> LocalVbufRangeMaterializer::wait(i)
                 -> retain ready tensor
  -> llama_model_create(...)
  -> llama_model_base::load_tensors(...)
       -> backend weight-buffer allocation
       -> llama_model_loader::load_all_data(...)
            -> set_vbuf_remote_tensor_data(...) for every llama tensor
                 -> linear source tensor scan
                 -> retained payload pointer assigned to ggml_tensor::data
```

Relevant implementation files:

- `integrations/android-vbuf-chat/app/src/main/java/com/eugen/vbufchat/MainActivity.java`
- `integrations/android-vbuf-chat/app/src/main/cpp/vbuf_android_chat.cpp`
- `integrations/llama.cpp/llama_vbuf_loader.cpp`
- `integrations/llama.cpp/vbuf_remote_source.cpp`
- `integrations/ggml/src/vbuf_materializer.cpp`
- `integrations/ggml/src/vbuf_range_source.cpp`
- `/tmp/llama.cpp-step21/src/llama-model-loader.cpp`
- `/tmp/llama.cpp-step21/src/llama-model.cpp`
- `/tmp/llama.cpp-step21/src/llama.cpp`

## Stage Analysis

### 1. Semantic Bootstrap Discovery

`vbuf_ml_consumer_open_metadata` opens the persisted source metadata profile.
This is discovery only and does not materialize external payloads.

The adapter then obtains model metadata, architecture, tensor count, tensor
names, shapes, representations, source IDs, offsets, and lengths. The remote
source requires every tensor to use `SourceId 1`.

Status: confirmed.

### 2. Source Tensor Inventory

`VbufRemoteSource` builds an in-memory descriptor array and a corresponding
`PersistentTensorRef` array. Each reference contains the tensor view and the
physical source offset. No payload is copied during this constructor phase.

Status: confirmed.

### 3. Semantic Loader Materialization

The pinned llama.cpp semantic loader creates a metadata-only ggml context and
iterates over every source tensor. For each tensor it calls
`VbufRemoteSource::tensor(...)`.

`tensor(...)` calls `materialize_tensor(...)` when the tensor is not already in
the retained map. `materialize_tensor(...)` starts one worker and immediately
waits for that worker before returning.

Therefore the current loader path is structurally serialized one tensor at a
time, even though the materializer API is expressed as request/wait.

Status: confirmed.

### 4. Owned Payload Materialization

For each tensor, `LocalVbufRangeMaterializer`:

1. allocates an aligned destination buffer with `posix_memalign`;
2. starts a worker thread;
3. requests exactly the tensor's physical range;
4. receives the response into temporary 4 KiB buffers;
5. copies response chunks into the owned destination;
6. hashes the entire payload with FNV-1a;
7. publishes a `MaterializedTensor` view and shared ownership lease;
8. retains the allocation until model destruction.

The remote HTTP path therefore has a network receive-to-stack-buffer step and a
stack-buffer-to-owned-payload `memcpy`. It does not allocate a full response
body temporary.

Status: confirmed.

### 5. HTTP Range Transport

`HttpRangeSource` uses one persistent socket in C1. Each `read_range` call is
protected by `socket_mutex_`, sends one HTTP/1.1 request, validates status,
content length, and content range, and reads exactly the requested number of
bytes.

Keep-alive removes connection creation cost but does not remove:

- one request and response header exchange per tensor;
- header parsing and validation;
- one mutex-protected critical section per request;
- temporary receive buffers and payload copies;
- sequential request scheduling from the caller.

Status: confirmed.

### 6. llama.cpp Backend Construction

The Android caller uses default model parameters with `n_gpu_layers = 0` and
`no_alloc = false`.

The model creates ggml tensor descriptors in no-alloc contexts. Before remote
payload callbacks run, llama.cpp allocates backend weight buffers for those
contexts.

Later, `load_all_data` sees an empty file list and invokes the supplied callback
for every tensor. The remote callback does not copy payload bytes into the
backend buffer. It assigns the retained materializer pointer directly to
`ggml_tensor::data`.

This means the current path has:

- retained materializer-owned payload allocations;
- backend weight-buffer allocations made before callback binding;
- no corresponding payload memcpy into the backend allocation.

The backend allocation remains model-owned even when the tensor data pointer is
replaced. The exact resident impact is not yet measured because untouched pages
may remain non-resident, but the allocation is a credible source of memory
pressure and must not be optimized away by assumption.

Status: allocation and pointer replacement confirmed. Exact resident-page impact
is inferred and requires measurement.

## One-Tensor Lifecycle

```text
TensorRef(offset, length)
  -> aligned owned allocation
  -> HTTP range request
  -> 4 KiB receive buffer
  -> memcpy into owned allocation
  -> full-payload FNV-1a scan
  -> retained shared lease
  -> source tensor descriptor returns payload pointer
  -> llama backend buffer already allocated
  -> callback replaces ggml_tensor::data with retained pointer
  -> payload remains live for model lifetime
```

There is no release during normal model construction. The materializer release
path is used during destruction, not for per-layer or per-tensor reclamation.

## Repeated Per-Tensor Work

The following work repeats for every tensor:

- worker thread creation and join;
- HTTP request construction and response parsing;
- socket mutex acquisition;
- payload hashing;
- materializer state recording;
- `/proc/self/smaps_rollup` parsing for trace RSS values;
- source descriptor/name/dimension copying;
- linear name lookup in `set_vbuf_remote_tensor_data`.

The callback scans the source inventory from index zero for each target tensor.
With 310 tensors this is approximately 96,100 logical lookup iterations before
accounting for tied or duplicated tensor handling.

## Serialization Dependency Graph

```text
source->tensor(i)
  -> materialize_tensor(i)
       -> request(i)
       -> worker(i)
            -> HTTP mutex
            -> request/response
            -> payload copy
            -> payload hash
       -> wait(i)
  -> next source tensor
```

The caller waits before requesting the next tensor. The HTTP mutex independently
prevents concurrent use of the persistent socket. Phase C measurements show
that adding concurrency 2, 4, or 8 did not improve this complete path and
regressed model-open time.

## Physical Layout And Range Policy

Each request uses the exact tensor payload offset and length from the tensor
directory. C1 transfers all 310 logical ranges with zero overfetch.

Positive-gap coalescing was tested, but the physical gaps are 16 bytes and the
resulting one-range request regressed model-open time to approximately 82-85
seconds. This rules out broad coalescing as the next unqualified optimization.

## Residency Conclusion

The current path is `EAGER_ALL` in two senses:

1. the semantic loader materializes every tensor before model construction
   completes;
2. the source retains every materialized payload for the model lifetime.

This is consistent with the Phase C qualification boundary. The full artifact
is not downloaded to device storage, but essentially all tensor payload bytes
are transferred and retained during model open.

No lazy loading, eviction, prefetching, or residency redesign should be inferred
from the current source abstraction.

## Memory And Copy Accounting

Confirmed payload movement on the HTTP path:

1. kernel/network data to a temporary receive buffer through `recv`;
2. temporary receive buffer to aligned owned payload through `memcpy`;
3. full owned payload read for hashing.

Not present in the remote callback path:

- an explicit owned-payload-to-backend payload copy;
- a full-file temporary download;
- HTTP response-body allocation equal to the tensor size.

Potential allocation duplication:

- materializer-owned payload storage is approximately the transferred payload;
- llama.cpp backend storage is allocated before the callback rebinds data;
- duplicate or tied tensor descriptors can add padding or repeated backend
  allocation requirements.

The Phase C observation was `735,592 KiB` native heap PSS and `879,496 KiB`
total PSS. These measurements establish the memory scale but do not identify
which pages belong to materializer storage, backend buffers, allocator metadata,
or model/context state.

## Evidence Classification

### Measured

- Phase B and C model-open timings.
- Request, connection, byte, and overfetch counts.
- Transport characterization.
- Correctness and generation gates.
- Android PSS observations.
- Regression of coalescing and concurrency variants.

### Derived From Source

- One worker is joined before the next tensor request.
- HTTP requests are serialized by `socket_mutex_`.
- Every tensor is materialized during the semantic loader loop.
- Payloads are retained until model destruction.
- The callback assigns the retained pointer rather than copying payload data.
- The callback performs a linear source scan for each llama tensor.
- The materializer hashes every payload.

### Inferred And Requiring Measurement

- Exact time contribution of hashing.
- Exact time contribution of RSS trace collection.
- Exact backend allocation size and resident-page count.
- Whether backend allocation pages are touched during model construction.
- Exact split between HTTP overhead, materializer overhead, and llama.cpp
  construction overhead.

## D0 Decision

The evidence supports the following optimization direction:

- retain C1 persistent HTTP connection reuse;
- do not reintroduce positive-gap coalescing or unbounded concurrency;
- do not change `EAGER_ALL` or model semantics in this phase;
- measure phase boundaries and memory ownership before selecting the next code
  optimization.

## Recommended Next Experiment

Perform one instrumentation-only run before changing architecture or transport
policy. Record timestamps and counters around:

- semantic descriptor loading;
- worker creation and join;
- HTTP request, header, and body handling;
- payload hashing;
- backend allocation, including buffer sizes;
- callback lookup and pointer binding;
- final model construction.

Also record whether backend-allocated pages become resident and whether the
backend buffer remains distinct from the materializer payload allocation.

This experiment will identify whether the next safe optimization should target
serialized request/materialization overhead, repeated callback work, hashing and
trace overhead, or duplicate backend allocation. It changes no public format,
model semantics, or residency policy.
