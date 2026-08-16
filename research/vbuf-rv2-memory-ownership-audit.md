# RV2 vBuf Memory Ownership Audit

## A. Executive Result

The approximately 2.5 GiB vBuf RSS delta is resident file-backed source data,
not an additional CPU_REPACK allocation. In the primary REPACK-OFF comparison,
the vBuf consumer keeps a full 4,993,335,296-byte read-only mapping of the vBuf
artifact alive while the patched llama.cpp source path copies every tensor into
an approximately 4,758.96 MiB anonymous ggml CPU model buffer. At payload
materialization end, the vBuf source mapping had 2.548 GiB RSS/PSS and the
anonymous destination had 4.657 GiB RSS. The corresponding GGUF run had no
model-file mapping and approximately 4.717 GiB in its anonymous runtime model
arena. The measured peak event RSS was 4.751 GiB for GGUF and 7.262 GiB for
vBuf, a 2.511 GiB difference in this audit run. The committed median difference
was approximately 2.50 GiB; the new evidence explains it as primarily
file-backed source residency coexisting with the backend destination.

Raw evidence is under `research/results/vbuf-rv2-memory-ownership/`. The
previous benchmark directories were not modified.

## B. GGUF Ownership Graph

This audit uses `load_mode=none`, matching the qualified benchmark. The native
loader therefore does not retain a GGUF mmap in this run. In the pinned source,
`llama_model_loader::load_all_data()` reads tensor bytes from the file directly
into the already allocated backend tensor when `use_mmap` is false. The relevant
source is `/home/eugen/llama-vbuf-pinned/src/llama-model-loader.cpp`, around
lines 1745-1754 in the qualified source.

```text
GGUF file
    |
    | read_raw(tensor destination)
    v
anonymous ggml CPU / CPU_REPACK model storage
    |
    +--> CPU model buffer:       2171.14 MiB
    +--> CPU_REPACK buffer:      2587.83 MiB
    |
    +--> source file mapping:    none observed
```

The GGUF path's model buffer contains runtime tensor storage, including
alignment and backend buffer allocation overhead. Its two named model buffer
classes total 4758.97 MiB. The source file is not present as a large mapping in
any captured `smaps` snapshot.

## C. vBuf Ownership Graph

The Rust consumer opens the artifact with `memmap2::Mmap::map`; the consumer
handle owns that mapping and the exported tensor views borrow payload slices
from it. The C++ direct source assigns each GGML tensor's `data` pointer to the
borrowed vBuf payload pointer. The pinned user-source seam then sees a changed
source pointer and performs `ggml_backend_tensor_set()` into the original
backend destination.

```text
vBuf artifact, 4,993,331,814 bytes
    |
    | memmap2 Mmap::map, full file, PROT_READ, MAP_SHARED
    v
validated Rust consumer mapping, 4,993,335,296 bytes page-rounded
    |
    | tensor view payload pointers
    v
set_tensor_data / set_vbuf_direct_tensor_data
    |
    | ggml_backend_tensor_set() copy
    v
anonymous ggml CPU model buffer: 4758.96 MiB
```

The mapping is visible in `/proc/<pid>/maps` as `r--s` with the vBuf path and a
4.650 GiB virtual range. It remains present through `BEFORE_MODEL_FREE` and is
absent at `AFTER_MODEL_FREE`. The adapter registry's `shared_ptr` source keeps
the Rust consumer handle alive until `llama_model_free_vbuf_direct()` removes
the model and destroys the source.

## D. Peak RSS Decomposition

The following is from the lifetime audit's `PAYLOAD_MATERIALIZATION_END` and
`MODEL_OPEN_END` snapshots. Values are resident bytes from `smaps`; nominal
buffer sizes come from the llama logs.

| Component | GGUF | vBuf REPACK OFF | Delta / observation |
|---|---:|---:|---:|
| Anonymous CPU model storage | 4.717 GiB observed; 4758.97 MiB named buffers | 4.657 GiB observed; 4758.96 MiB named buffer | same intended destination capacity |
| Resident vBuf source mapping | 0 | 2.548 GiB at payload end; 2.507 GiB at model open end | +2.5 GiB |
| CPU_REPACK storage | included in GGUF CPU model total, 2587.83 MiB | 0 | storage is rearranged, not saved |
| CPU output buffer | 0.39 MiB | 0.39 MiB | no material delta |
| CPU KV buffer | 67.50 MiB | 67.50 MiB | no delta |
| CPU compute buffer | 30.03 MiB | 30.03 MiB | no delta |
| Heap, libraries, metadata, other | approximately 30 MiB | approximately 20 MiB plus mapping metadata | noise-scale difference |
| Observed event peak RSS | 4.751 GiB | 7.262 GiB | +2.511 GiB |

At `PAYLOAD_MATERIALIZATION_END`, the large mappings were:

| Case | Mapping | Virtual size | RSS | PSS | Classification |
|---|---|---:|---:|---:|---|
| GGUF | anonymous runtime arena | 4.774 GiB | 4.717 GiB | 4.717 GiB | backend model storage |
| vBuf OFF | anonymous runtime arena | 4.659 GiB | 4.657 GiB | 4.657 GiB | backend model storage |
| vBuf OFF | `/home/eugen/DeepSeek-V2-Lite.IQ1_S.vbuf` | 4.650 GiB | 2.548 GiB | 2.548 GiB | clean file-backed source pages |

The small residual between mapping sums and rollup RSS is heap, shared
libraries, stacks, metadata, and sampling order. It is tens of MiB, not an
unexplained GiB-scale allocation. `phase-memory.csv` records the full rollup
fields: `Rss`, `Pss`, `Pss_Anon`, `Pss_File`, `Private_Clean`,
`Private_Dirty`, `Anonymous`, `Locked`, `Swap`, and related fields.

## E. Source/Destination Coexistence

Yes. The source and destination are simultaneously resident during the
materialization interval.

| Boundary | vBuf source mapping RSS | vBuf anonymous destination RSS | vBuf rollup RSS |
|---|---:|---:|---:|
| `PAYLOAD_MATERIALIZATION_BEGIN` | approximately 34 MiB | approximately 21 MiB | approximately 78 MiB |
| `PAYLOAD_MATERIALIZATION_END` | 2.548 GiB | 4.657 GiB | 7.228 GiB |
| `MODEL_OPEN_END` | 2.507 GiB | 4.725 GiB | 7.226 GiB |
| `EXECUTION_READY` | 2.507 GiB | 4.729 GiB | 7.262 GiB |
| `BEFORE_MODEL_FREE` | 2.507 GiB | not resident in that snapshot | 2.543 GiB |
| `AFTER_MODEL_FREE` | absent | absent | 0.026 GiB |

The last two rows are lifetime controls, not claims that the backend allocation
was freed at context destruction. Linux residency can change between the
event write and the snapshot; the source mapping control is unambiguous: it is
present before model destruction and gone immediately after
`llama_model_free_vbuf_direct()`.

The source pages are clean file-backed pages. They are naturally reclaimable by
the kernel, but the mapping remains valid and the handle remains alive, so
there is no explicit unmap or source lifetime release after each tensor copy.
The current adapter therefore retains a source view for the entire model
lifetime even after the backend destination has been materialized.

## F. REPACK ON/OFF Explanation

The earlier A/B result is consistent with this audit. REPACK ON names two
backend model buffers, 2171.14 MiB CPU and 2587.83 MiB CPU_REPACK. REPACK OFF
names one ordinary CPU buffer of 4758.96 MiB. The ON total is 4758.97 MiB,
within printed-size rounding of OFF.

Thus CPU_REPACK does not add another full model representation in this
composition. It partitions the same destination capacity between two ggml
buffer types. Both variants retain the vBuf source mapping, so changing the
partition does not remove the approximately 2.5 GiB source residency.

## G. Linux Residency Behavior

The vBuf mapping is `r--s`, full-file, read-only, and its resident pages are
reported by `smaps` as `Pss_File` and `Private_Clean`. `Shared_Clean` is zero in
these single-process snapshots because no other process shares those pages;
that does not make them anonymous. They remain file-backed clean pages and are
included in RSS/PSS.

The GGUF run has no corresponding large model-file mapping. Its materialization
faults are file reads into anonymous backend destinations. The vBuf run's
major/minor fault counts are preserved in the raw logs and phase traces; a
major fault is treated here as a page-fault event, not automatically as an I/O
time measurement. The source mapping's resident set grows as payloads are
copied and is later partially reclaimable under memory pressure.

The bounded `mincore()` experiment was attempted at every captured large
mapping. The qualified RV2 kernel returned `ENOMEM` for these multi-gigabyte
ranges, including when split into 256 MiB chunks. `residency.csv` preserves
those failures. The audit therefore does not claim mincore success; the
residency conclusion is based on kernel-generated `smaps` RSS/PSS and the
per-mapping clean/dirty breakdown, which directly reconciles with rollup RSS.

## H. Correct Ownership Layer

| Bytes | Current owner | Classification |
|---|---|---|
| vBuf file mapping and borrowed views | Rust consumer handle / adapter registry | `REQUIRED_BY_CURRENT_VBUF_ADAPTER`; source lifetime is longer than copy lifetime |
| CPU model destination | ggml CPU backend buffer | `REQUIRED_BY_CURRENT_GGML_RUNTIME` for this patched path |
| CPU_REPACK destination | ggml CPU_REPACK buffer | `OPTIONAL_OPTIMIZATION`; not additional total capacity here |
| KV, output, compute buffers | ggml llama context/scheduler | `REQUIRED_BY_CURRENT_GGML_RUNTIME` |
| 2.5 GiB resident source pages after copying | current vBuf adapter's retained mapping | `AVOIDABLE_DUPLICATION` and `SOURCE_MAPPING_REMAINS_UNNECESSARILY_RESIDENT` after materialization, subject to direct-pointer lifetime requirements |

"Required by the current ggml runtime" is deliberately narrower than "required
by any native CPU runtime." This audit did not change or test a zero-copy CPU
backend contract.

## I. Native-Runtime Implication

The direct source already demonstrates that validated vBuf payload pointers can
be exposed as ordinary tensor data pointers, and the source/destination copy is
visible in the pinned seam. The audit proves that the current generic llama
composition chooses owned backend storage and retains the source mapping; it
does not prove that the CPU/RVV backend can safely execute every quantized
operation directly from read-only mapped pages.

The minimum storage required by the current path is therefore one full backend
weight representation plus context/compute storage. A native runtime that can
honor read-only mapped tensor storage could remove the second representation,
but that requires a separate backend capability and correctness qualification.
No such implementation was made here.

## J. Recommendation

Evidence-backed classifications:

- `CURRENT_VBUF_ADAPTER_DUPLICATES_WEIGHT_STORAGE`
- `GGML_CPU_RUNTIME_REQUIRES_OWNED_WEIGHT_BUFFER` for the current patched llama/ggml path
- `SOURCE_MAPPING_REMAINS_UNNECESSARILY_RESIDENT`
- `EXTRA_RSS_IS_PRIMARILY_FILE_BACKED`
- `MORE_EVIDENCE_REQUIRED` before claiming `DIRECT_MMAP_EXECUTION_IS_FEASIBLE`

No memory architecture or runtime behavior was changed by this audit.
