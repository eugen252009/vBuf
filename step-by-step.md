# vBuf-ML: commit-sized implementation plan

## 0. Purpose and non-negotiable boundary

This document plans an ML/model-container **descendant profile** built on vBuf. It does not authorize implementing vBuf-ML yet.

The dependency direction is permanent:

```text
reusable, domain-neutral implementation utilities
                    ^
                    |
generic vBuf base format and vbuf-core
                    ^
                    |
vBuf-ML profile, converters, loaders and benchmarks
```

- Generic vBuf remains useful without ML knowledge.
- `vbuf-core` must never import, recognize, enumerate, or document model architectures, tensors, tokenizers, quantization schemes, GGUF keys, or llama.cpp behavior.
- vBuf-ML may constrain generic vBuf primitives and assign profile-local meaning to payload bytes and Key-IDs.
- A feature is promoted into generic vBuf only after a separate domain-neutral justification and conformance plan. “vBuf-ML needs it” is not sufficient.
- This boundary does **not** mean that vBuf is only an opaque envelope. The base should provide a small set of efficient generic layout primitives—bounded byte regions, explicit lengths, alignment, stable numeric identification, optional direct indexing, and skippable extension framing—when they are independently useful to telemetry, analytics, storage and other descendants. Profiles compose those primitives and own their semantics.
- Existing telemetry, analytics, SoA/AoS, streaming, C and TypeScript use cases remain valid.
- mmap/zero-copy access is a first-class property, but no unsafe typed view may be created before complete bounds, alignment, representation and arithmetic validation.
- Optimizations remain hypotheses until isolated measurements support them.

### Change classification used below

1. **BASE** — required generic vBuf changes. No ML terminology or semantics.
2. **INFRA** — reusable downstream/runtime infrastructure with no ML wire semantics. It may depend on vBuf; vBuf does not need to depend on it.
3. **ML** — vBuf-ML profile, metadata, tensor representations, tools and integration.

A commit must not mix these classes unless the commit is documentation-only. In particular, an ML feature must not be hidden in a BASE commit.

### Promotion test for generic primitives

A wire feature belongs in BASE only when all of the following are true:

1. its fields can be named and validated without downstream vocabulary;
2. at least one existing generic vBuf use case and one plausible independent descendant benefit, or two existing non-ML use cases benefit;
3. generic readers can use or skip it without loading a profile implementation;
4. files that do not need it pay no mandatory payload or hot-path parsing cost;
5. streaming, finalized-file and compatibility behavior are explicit;
6. checked offset/size/alignment rules can be specified independently of payload meaning;
7. a smaller composition of existing primitives cannot provide the same property safely and efficiently.

Initial placement guidance:

| Capability | Initial owner | Reason |
|---|---|---|
| aligned bounded byte/primitive-array blocks | BASE | existing domain-neutral vBuf purpose |
| explicit portable primitive encodings | BASE | required for all interoperable vBuf users |
| checked `u64` ranges and skippable framing | BASE | safety, large files and forward compatibility are universal |
| optional finalized-file envelope | BASE candidate in Steps 5A–5B | justified only if at least one generic derived artifact survives qualification |
| optional Nano-Index | BASE candidate in Steps 5A/5C | generic physical extent topology/enumeration and hypothesized consumer-side parallel partitioning; retained only with measured benefit |
| optional rank/select checkpoints | BASE candidate in Steps 5A/5E | generic ordinal navigation; depends on a retained Nano-Index and measured benefit |
| optional direct block/region directory | BASE candidate in Steps 5A/5D | generic columns and descendants both need non-linear logical lookup |
| checked arithmetic and mmap/range source helpers | INFRA | reusable implementation behavior, not necessarily wire semantics |
| generic typed key/value or string-table codec | INFRA/profile first | potentially reusable, but base-wire promotion needs a second proven consumer |
| checksum algorithm plumbing | INFRA; BASE extension only by ADR | integrity is general, but coverage/framing/cost policy is unresolved in current specs |
| tensor names, shapes and representation registry | ML | meaning is inherently tensor-specific |
| model/tokenizer/quantization metadata | ML | downstream semantics |

Promotion is one-way compatibility work and should be conservative; efficient composition, not maximal base functionality, is the objective.

### Supplied design-lineage context

The project lineage is application/performance-driven: a barcode-oriented game led to large OpenFoodFacts ingestion in Bun/JavaScript, allocation and GC pressure motivated BumpArena, and BumpArena then exposed the need for persistent bytes that a runtime could consume without reconstructing an object graph. vBuf's recurring objective is therefore to minimize unnecessary representation transitions:

```text
stored bytes → validate → direct view / runtime adapter / native consumer
```

vBuf began in TypeScript; Rust was added later to test whether the layout itself explained the unexpectedly near-native prepared-view behavior. Cheap JavaScript consumption is architectural evidence, not a secondary binding concern. BumpArena/shadow-object framing remains downstream runtime adaptation rather than base wire semantics.

The base should provide a small compositional binary vocabulary, not a universal object type system or schema compiler. A simple mapper may lower flexible high-level values into portable numeric arrays, opaque bytes, and multiple generic blocks. AoS and SoA are ordinary choices of real array layout; variable/composite values and strings may use multiple canonical blocks. The container should largely disappear after validation establishes a compatible direct view.

This supplied lineage is recorded separately from historical written specifications and implementation behavior. Analogies such as “tar plus typed/native-friendly geometry and an optional compact physical index” are architectural intuition only, never normative grammar.

---

## 1. Evidence from the current repository

The plan must begin from the implementation that exists, not only from the desired architecture.

- `spec/spec_0.5-alpha.md` defines a 16-byte global header and a 64-bit atomic block anchor.
- `rust/src/lib.rs`, `ts/vbuf.ts`, and `c/vbuf.h` implement/read the v0.5-style anchor and aligned payloads.
- `rust/src/lib.rs::VBufInstance::open` mmaps the complete file, and `get_as<T>` exposes direct mapped slices.
- `benchmark-results/vbuf/README.md` validates that prepared scans of current real vBuf payloads are close to equivalent native Rust layouts for the measured one-million-record workload. It does not compare vBuf with GGUF or model inference.
- `benchmark-results/README.md` explicitly limits the Protobuf evidence to one Prost workload and rejects cross-format conclusions.
- `spec/spec_0.1-draft.md` through `spec/spec_0.4-alpha.md`, the current README, and v0.5 disagree about headers, checksums, indexing, slot size and navigation. Old drafts are evidence of design history, not an interoperable specification.
- Current Rust unit tests pass, but `test_round_trip_logic` checks only the magic, `test_alignment_integrity` checks only that output is nonempty, and two tests silently skip when an external file is absent. They are not a format conformance suite.

Relevant GGUF evidence:

- The upstream GGUF specification defines a typed metadata map, tensor count, tensor names, rank, dimensions, `ggml_type`, direct tensor-data offsets and configurable alignment: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>.
- The public API and metadata value types are visible in <https://github.com/ggml-org/ggml/blob/master/include/gguf.h>.
- These are comparison inputs, not requirements to copy. A vBuf-ML field is justified only by a concrete runtime, conversion, validation, distribution or reproducibility need.

### Historical Nano-Index findings

The Nano-Index is an existing generic vBuf architectural proposal, not an ML feature:

- `spec/spec_0.2-draft.md` defines a trailing bit vector over the Data Payload: one bit per 16-byte slot, `1` for a Cell start, and `0` for continuation or empty/padding. Its header has `HasIndex` and `IsStream` flags and a `u64 DataLen` described as payload length.
- `spec/spec_0.4-alpha.md` carries the concept forward for homogeneous blocks and claims POPCNT can find the Nth column/key. Its header instead has a `u32 DataLen` and no defined `HasIndex`, Nano-Index offset, encoded length, or finalization record.
- Neither `spec/spec_0.5-alpha.md` nor the Rust, TypeScript or C v0.5-style implementation writes or reads a Nano-Index.
- Current implementations advance zero padding and subsequent anchors on 8-byte boundaries (`curr += 8`, tail alignment to 8), while the historical Nano-Index literally describes a 16-byte slot grid.
- The corrected/current design intent is broader and was not stated precisely by those historical drafts: the Nano slot is exactly the file’s canonical `BaseStep`, and the bitmap describes physical extent segmentation. Historical one-bit-per-16-byte behavior is the `BaseStep = 16` instance, not a permanent Nano-specific quantum. The stronger extent-topology interpretation supplied for v0.6 qualification is recorded as current design intent, not retroactively attributed to the old text.

What the idea can soundly guarantee, if corrected and validated:

- compact topology of candidate canonical physical extents: `1` begins an extent and `0` begins no new extent;
- compact enumeration of candidate physical block/extent starts;
- candidate upper span boundaries from the next set bit or indexed-region end;
- cheap sequential bit scanning and block/extent counting;
- a structural cross-check against canonical block parsing;
- optional inspection/debug information that a reader can leave untouched;
- a possible bounded starting point for recovery attempts, never proof that candidate bytes are a valid block.

What a raw Nano-Index cannot guarantee:

- constant-time selection of the Nth set bit without an auxiliary rank/select structure;
- direct lookup by Key-ID, because it contains no Key-ID-to-bit mapping;
- the byte-level meaning of a `0` slot: it may cover continuation data, payload, or padding belonging to the active physical extent;
- canonical payload length or processing cost from the Nano-derived physical span;
- validity of any marked header without canonical header/range validation;
- tensor or other domain-semantic lookup.

For corrected v0.6 planning, let `B = BaseStep`, `R = indexed_region_size`, and use checked ceiling division:

```text
slot_size  = B
slot_count = ceil(R / B)
nano_bytes = ceil(slot_count / 8)
slot(i)    = indexed_region_start + i * B
```

`nano[i] == 1` iff a new candidate canonical block/physical extent starts at `slot(i)`. For a non-empty valid topology the first indexed slot is set. A `0` means only that no new extent starts at that slot; it does not identify payload or any domain meaning. The asymptotic storage ratio is one bit per `B` bytes, or `1/(8*B)` of the indexed region:

| BaseStep | Asymptotic Nano overhead |
|---:|---:|
| 8 B | 1.56250000% |
| 16 B | 0.78125000% |
| 32 B | 0.39062500% |
| 64 B | 0.19531250% |
| 128 B | 0.09765625% |
| 256 B | 0.048828125% |

The v0.2 statement `DataLen / 128` is only the asymptotic `BaseStep = 16` case and is incomplete for fewer than eight slots or a partial final Nano byte. v0.6 must define indexed-region start/length, require checked ceiling arithmetic, require unused final Nano-byte bits to be zero, and state whether a valid indexed region may end in a partial final BaseStep. The recommended candidate is a BaseStep-aligned region start with a permitted partial final slot; no block start is representable except `start + i*B`, and the final candidate still requires a complete canonical header/range.

For consecutive set-slot ordinals `s_i` and `s_(i+1)`, Nano proposes physical extent `[start + s_i*B, start + s_(i+1)*B)`; the final proposed extent ends at the indexed-region end. Consecutive `1` bits therefore describe adjacent BaseStep extents, while a long run of `0` slots remains part of one physical extent. These are topology bounds, not canonical payload bounds: they may include payload, internal/alignment padding, or representation-defined continuation. Canonical headers/counts/ranges remain the source of truth, and ignored-Nano parsing must produce the same block sequence. Duplicate Key-IDs do not affect the bitmap; they matter only to a separate logical lookup policy.

Variable/composite representations may use multiple canonical physical blocks rather than forcing one semantic object into one block. A string representation is one possible downstream example, but Nano records only each new physical extent and intervening continuation slots; it does not know that extents belong to a string or to one higher-level object.

---

## 2. Architectural blockers and decisions required before implementation

Planning can describe alternatives, but implementation must stop at these gates. They cannot be silently worked around.

### Decision A — authoritative generic vBuf wire format

**Execution status:** A1 accepted by the instruction to begin this plan; corrected v0.6 is the only parent path that may unblock vBuf-ML. Legacy-v0.5 remains compatibility evidence/read support only.

**Blocker:** vBuf-ML cannot be stable when its parent wire format is ambiguous.

Concrete contradictions:

1. `spec/spec_0.5-alpha.md` gives magic `0x56425546`; implementations use `0x46554256`, whose little-endian bytes are actually `VBUF`.
2. v0.5 defines `BaseStep = 16 << AShift`; Rust and TypeScript decode `1 << AShift`.
3. v0.5 describes a 16-byte slot grid; current writers can place the next anchor on an 8-byte boundary.
4. v0.5 defines a 32-bit `DataLen`; the Rust writer leaves it zero and large model files commonly exceed 4 GiB.
5. README and older drafts promise CRC/index behavior absent from v0.5 implementations.

**Alternatives:**

- **A1, recommended:** define a corrected generic vBuf v0.6, retain explicit legacy-v0.5 reading only where the old bytes are unambiguous, and require vBuf-ML to use v0.6 or later.
- **A2:** document/freeze current Rust/TypeScript behavior only as an explicit legacy-v0.5 profile and add safe legacy validation. This preserves evidence and readable bytes but does not resolve the BaseStep contradiction and therefore does not unblock Nano or vBuf-ML.
- **A3:** make vBuf-ML depend directly on ambiguous v0.5 behavior. Rejected because independent implementations could disagree about offsets.

**Recorded decision:** A1 is approved for this execution path. A2 remains a legacy documentation/reader option only and is not an alternative parent for vBuf-ML.

### Decision B — generic navigation primitives and child-profile identification

**Execution status:** qualification pending Step 5A; no Nano, checkpoint, directory, or framing candidate is selected.

**Blocker:** a generic reader must be able to navigate a vBuf-ML file without understanding ML, while an ML reader must identify its profile deterministically. The base must provide useful composition primitives without adopting profile semantics.

There is independent, non-ML evidence for a generic direct-navigation primitive: `spec/spec_0.2-draft.md` and `spec/spec_0.4-alpha.md` proposed a Nano-Index for generic cells/blocks, the README promises optional O(1) lookup, and current telemetry/SoA readers linearly scan by generic Key-ID. The old bit-vector is not necessarily the right design—it finds block positions rather than fully validating ranges—but random access is not an ML-only requirement.

**Alternatives:**

- **B1:** vBuf-ML uses ordinary generic byte/blob blocks and defines all navigation inside one opaque profile manifest. This protects the base but makes every descendant reinvent safe range directories and leaves generic readers with linear scans.
- **B2:** add only a generic profile/application identifier to the base header. This identifies semantics but does not solve direct navigation.
- **B3, candidate to qualify rather than preselect:** define an optional generic finalized-file region/block directory. A base entry contains only domain-neutral facts such as numeric Key-ID/kind, flags, direct `u64` offset, `u64` byte length, required alignment, and enough framing to skip unknown optional entries. The directory knows nothing about tensors, metadata or models. Streaming files can omit it or finalize one later. A profile-local bootstrap block still owns profile magic/version and semantic mapping.
- **B4:** restore a corrected raw Nano-Index as physical extent topology, optionally with simple checkpoints, and determine whether extent starts/spans plus canonical headers meet enough generic needs to avoid B3.

**Decision required:** approve qualification of B1–B4 in Step 5A, not B3 itself. Freeze no directory until Nano-only, directory-only and coexistence designs have been measured against generic workloads. Profile names, tensor names, typed model metadata, shapes and quantization remain excluded in every result.

### Decision C — corrected v0.6 compatibility policy

**Execution status:** accepted policy: safely readable legacy files where unambiguous; canonical new writes use corrected v0.6 only. Byte-compatible new v0.5 writing is not a vBuf-ML requirement.

**Blocker:** safe large-file support and corrected alignment may change bytes and parser behavior.

**Alternatives:** strict v0.6-only writing with optional legacy reader; dual-mode writer; or permanent v0.5 byte compatibility.

**Recorded decision:** old files remain readable where safe and unambiguous; new canonical writes use v0.6 only.

### Decision D — tensor representation identity

**Execution status:** deferred; D1 remains the recommendation and no tensor-directory wire work may begin before explicit selection.

**Blocker:** a bit width does not identify a quantized representation. Equivalent-kernel comparison requires exact packed-byte semantics.

**Alternatives:**

- **D1, recommended for qualification:** a namespaced representation registry. vBuf-ML entries use compact local IDs resolved by a manifest table to a tuple such as `(namespace=ggml, type=Q4_K, representation_version=...)`. Preserve the exact GGML packed bytes for the first converter/loader.
- **D2:** copy `ggml_type` numeric values directly into the permanent vBuf-ML wire format. Simple, but couples format evolution to an external enum.
- **D3:** invent new vBuf quantization immediately. Rejected for initial qualification because equivalent-kernel comparison would be impossible or confounded by repacking.

**Decision required:** approve D1 or D2 before freezing the tensor directory.

### Decision E — initial runtime/model scope

**Execution status:** deferred; a concrete fixture and pinned llama.cpp revision are required before ML metadata implementation.

**Blocker:** metadata requirements depend on an actual runtime and architecture. “All ML models” is not a testable first target.

**Alternatives:** select one llama.cpp-supported, openly redistributable small model architecture and one unquantized plus one quantized representation; or attempt broad architecture support immediately.

**Decision required:** name the initial architecture/model fixtures and target llama.cpp revision. Recommendation: one small decoder-only model, F16 and one common GGML quantization, then expand from observed loader requirements.

### Decision F — checksums

**Execution status:** deferred. Base integrity remains unselected; profile-local integrity work is optional and blocked until Step 14.

**Blocker:** current vBuf documentation promises checksums that current v0.5 does not define or implement consistently.

**Alternatives:**

- keep generic vBuf v0.6 integrity-neutral and define an optional cold vBuf-ML integrity section;
- define a generic optional block-checksum extension only after a domain-neutral requirement;
- make checksums mandatory, which adds startup/I/O cost and prevents exact hot-path comparison.

**Decision required:** recommendation is optional profile-local per-tensor CRC32C plus optional cryptographic digest, stored away from hot payloads. Do not call detection “self-healing.”

### Decision G — canonical `BaseStep` and derived Nano geometry

**Execution status:** resolved for v0.6. `BaseStep = 1 << BaseShift`, with legal `BaseShift` values 3 through 8 inclusive (8–256 bytes). Values outside that range are rejected before offset derivation or view construction.

**Observed contradiction:** historical Nano drafts literally use 16-byte slots; v0.5 says `BaseStep = 16 << AShift`; current Rust/TypeScript readers derive alignment as `1 << AShift`; current writers/readers can place block anchors on an 8-byte boundary. The repository therefore has no single interoperable definition of the base physical geometry.

**Evidence:** `spec/spec_0.2-draft.md:36-60` and `spec/spec_0.4-alpha.md:45-60` define 16-byte slots. `spec/spec_0.5-alpha.md:6-17,40-53` defines dynamic `16 << AShift` alignment and describes next 16-byte slots. Those drafts and the README repeatedly motivate alignment by native/SIMD access, but do not provide portable measurements that establish one universal width. `rust/src/lib.rs` writes `alignment.trailing_zeros()` and reads `1 << a_shift`, then tail-aligns anchors to 8; `ts/vbuf.ts` and `c/vbuf.h` likewise use absolute power-of-two alignment/8-byte stepping. The validated existing baseline shows prepared real-vBuf SoA/AoS scans near the corresponding native Rust layouts for its one-million-record workload, but it does not isolate BaseStep or compare scalar/aligned/SIMD paths.

**Design-intent clarification supplied now:** BaseStep was intended as vBuf's generic physical-layout performance knob for efficient direct native CPU consumption, including naturally aligned loads, SIMD-friendly payload positions, predictable starts, simple address calculation, vectorized traversal and favorable cache behavior. This intent is domain-neutral and does not select 16, 32, 64, a page size, or any contemporary ISA width.

Keep four records distinct: historical written behavior (`16`-byte slots and later `16 << AShift`, with SIMD claims but incomplete evidence); current implementation behavior (`1 << AShift` with 8-byte anchor stepping); original performance intent supplied now (hardware-neutral native/SIMD-friendly physical geometry); and corrected normative v0.6 behavior (`1 << BaseShift` over the accepted legal range). Do not rewrite history to make these agree.

**Why it matters:** BaseStep is upstream of both canonical blocks and Nano. It must preserve efficient direct native consumption without making small or multi-block representations pay unreasonable padding. Nano cannot safely describe canonical starts until vBuf itself has one unambiguous BaseStep, but Nano size, word/page coincidences, rank/select convenience and vBuf-ML needs are secondary effects and cannot select BaseStep. An independently configurable Nano quantum would create two physical geometries and permit unrepresentable block starts.

**Required v0.6 base decisions before Nano semantics:**

- minimum legal `BaseStep`;
- whether legal values are powers of two and the maximum supported value;
- whether `AShift` means an absolute exponent (`1 << AShift`), a relative exponent (`minimum << AShift`), or is replaced by an explicit field;
- checked shift/conversion limits in Rust, C and TypeScript;
- whether `BaseStep` is relative to file offset zero or an explicit indexed-region start;
- requirement that every canonical block start is `indexed_region_start + i*BaseStep` for integer `i`;
- block-tail padding needed to make every following start representable;
- payload alignment as `BaseStep` or a stricter alignment that must be an integer multiple of `BaseStep`;
- whether stricter per-payload alignment is encoded generically and how it is validated;
- canonical handling of a partial final BaseStep.

**Possible base encodings/layouts:**

- **G1:** minimum 8 bytes with absolute exponent encoding, compatible in spirit with current `1 << AShift` decoding after strict validation.
- **G2:** minimum 16 bytes with relative exponent encoding, matching the literal `16 << AShift` v0.5 text.
- **G3:** replace ambiguous `AShift` with an explicit/versioned BaseStep encoding; simplest to interpret but costs header/version work.
- **G4:** choose another generic base rule supported by layout/benchmark evidence.

These alternatives select base geometry, not Nano geometry. For every accepted result:

```text
canonical block-start granularity = Nano slot granularity = BaseStep
```

Examples: BaseStep 8/16/32/64 imply one Nano bit per 8/16/32/64 bytes respectively. Nano has no separate quantum field unless future concrete evidence proves independence necessary.

```text
native CPU / SIMD-friendly generic physical geometry
                         ↓
                  vBuf BaseStep
                         ↓
            canonical block/payload geometry
                         ↓
          optional Nano description of that geometry
                         ↓
               generic/profile consumers
```

No descendant may select or override BaseStep through Nano metadata. BaseStep remains a generic file-layout decision even when a demanding descendant exposes its costs.

**Recommended first-principles resolution:** qualify and choose the legal v0.6 BaseStep encoding/set from generic block-header size, portable alignment, direct native-access behavior and generic canonical-layout requirements before Step 3 is frozen. Evaluate the whole-system trade-off: padding/packing density and small/multi-block cost; native scalar, aligned scalar and optional SIMD/vectorized access; cache/traversal behavior; and, secondarily, Nano/checkpoint overhead. Use identical payloads and algorithms across access variants, report actual alignment and ISA assumptions, preserve a scalar reference, and make unsupported SIMD paths optional. Independent base-layout measurements may inform that prerequisite decision, but Nano results may not. Step 5A then remeasures production layout and artifact trade-offs among the already legal BaseSteps and may inform writer defaults—not redefine the wire-legal set. Do not restore 16-byte starts merely to preserve history, preserve 8-byte starts merely to preserve current implementation behavior, or make BaseStep coarse enough to penalize ordinary small/composite representations merely for SIMD/Nano convenience.

**Recorded decision:** v0.6 uses the absolute exponent above; `data_region_start` and every block start lie on that BaseStep grid, the next block uses checked `align_up(previous_block_end, BaseStep)`, and the final block needs no tail padding. Stricter payload alignment is encoded orthogonally as a power-of-two multiple of BaseStep and never changes Nano geometry. The legal range is a hardware-neutral interoperability contract, not a claim that every value is equally fast or that one is universally optimal; no normative writer default is selected without generic qualification. Legacy bytes are not reclassified as v0.6 merely because their implementation interpretation happens to match.

### Decision H — locating optional finalized-file artifacts

**Execution status:** deferred until Step 5A evidence; no finalization envelope is selected.

**Observed contradiction:** v0.2 says the Nano-Index trails `DataLen` payload and has a `HasIndex` flag, while v0.4 retains the index but removes a defined flag/location/length mechanism; v0.5 defines `DataLen == 0` as indefinite stream and omits the Nano-Index.

**Evidence:** `spec/spec_0.2-draft.md:18-32,53-60`, `spec/spec_0.4-alpha.md:16-24,56-60`, and `spec/spec_0.5-alpha.md:10-17`.

**Why it matters:** a reader cannot safely distinguish payload, index, future finalized artifacts and truncation unless discovery and lengths are canonical. Large files also cannot use v0.4/v0.5’s `u32 DataLen`.

**Possible interpretations:**

- patch a seekable v0.6 header with `u64` data length and artifact offsets;
- append a fixed-size generic finalization footer that points to a bounded artifact table, allowing payload bytes to remain untouched;
- reserve a generic finalization-manifest block discoverable from the end;
- omit generic finalization and let each optional structure invent discovery, which is rejected as unsafe duplication.

**Recommended first-principles resolution:** qualify a small fixed footer plus bounded generic artifact table against a patchable header. Both must preserve canonical payload bytes, use checked `u64` ranges, and allow unfinished streams to remain valid without artifacts. Do not add the envelope unless at least one independently qualified artifact survives Step 5A.

**Decision required:** choose the discovery mechanism after Step 5A evidence and before any Step 5B wire freeze. Dependent Nano-Index, region-directory and integrity finalization work stops until then.

### Decision I — acceleration trust and consistency validation

**Execution status:** deferred until an acceleration artifact survives Step 5A; canonical validation remains authoritative meanwhile.

**Observed tension:** canonical blocks are the source of truth, but proving that a derived Nano or region directory has neither omitted nor invented a canonical target requires comparing it with a canonical scan—the work the acceleration artifact is intended to avoid.

**Evidence:** historical specs define Nano bit meaning but no validation/trust model, block-count commitment, index checksum or behavior when bits disagree with headers. The candidate region directory creates the same general completeness problem unless its target contract and validation policy are explicit.

**Why it matters:** bounds-checking and validating each selected Nano `1` or directory range prevents unsafe access, but cannot reveal an omitted entry elsewhere. In topology terms, a missing Nano bit can falsely merge canonical extents and an extra bit can falsely split one; neither candidate span may authorize access. A malformed derived artifact could otherwise return incomplete or false partitions while appearing structurally safe.

**Possible interpretations:**

- always fully cross-check at open, preserving strongest conformance but removing cold-start/index benefits;
- provide an explicit full-validation operation and a fast indexed path that validates every accessed candidate/header/range, detects discovered conflicts, and documents that completeness of an unverified derived artifact is not established;
- require an authenticated whole-file digest from a trusted producer. This detects byte changes but does not by itself prove that the producer generated a semantically consistent index;
- make a derived index/directory authoritative, rejected because it violates the canonical-block principle.

**Recommended first-principles resolution:** separate memory safety from full conformance. Every path is bounds-safe and validates each used header/range under the artifact’s target contract. A full conformance API compares all entries/starts and is required by validators/converters. Fast readers may use an unverified optional artifact under an explicit policy and must fall back or fail on any observed conflict; no documentation may call that artifact independently authoritative.

**Decision required:** approve the explicit validation-policy split, or require full cross-check before any accelerated access. Steps 5C and 5D cannot freeze reader API/error behavior before this decision.

### Decision J — forward continuation semantics across physical blocks

**Execution status:** resolved by the supplied original design intent, but it exposes a concrete corrective blocker in completed Steps 3–4 that must be repaired before Step 5.

**Observed contradiction:** historical v0.5 says `Chain = 1` means a subsequent block belongs to the same Key-ID, and the supplied design intent makes the direction explicit: the bit on block A answers whether the next canonical block continues A. Current `spec/spec_0.6.md` and Step 4 instead interpret a set bit on block B as “B continues the preceding block,” require the first block to clear it, and encode the shared valid-chain fixture in that reverse direction.

**Recorded resolution:** the current block's continuation bit points forward. If `Continuation == 1`, a next canonical block MUST exist and MUST have the same Key-ID; if it is zero, the current logical chain terminates. Thus A=1, B=1, C=0 represents three physical blocks and one logical value. Every member remains a complete canonical physical block with its own start/header/range. The final physical block MUST clear continuation. An unchained duplicate Key-ID remains legal and begins independently.

**Required corrective supplement before Step 5:** update only the v0.6 chain paragraph/field name as needed, normative malformed/valid vectors, Rust/TypeScript/C validation, and shared fixtures/tests. Preserve every unrelated Step 3/4 safety and geometry decision. This is a directional semantic correction, not a BaseStep, block-layout, Nano, or profile redesign.

**Permanent orthogonality:** Nano marks all physical block/extent starts, including every block in a logical chain. Continuation groups adjacent canonical blocks logically and never suppresses or invents a Nano bit. Nano cannot infer logical grouping; continuation cannot redefine physical boundaries.

---

## 3. Proposed descendant layout to validate, not yet freeze

Subject to Decisions A–J, generic finalization follows this state model:

```text
write / stream canonical blocks
            ↓
valid unfinished vBuf
            ↓ optional finalize; derive artifacts from writer-known facts or canonical re-scan
valid finalized vBuf with zero or more independently qualified artifacts
```

Finalization must not rewrite canonical block/payload representation. A seekable header patch or appended footer may record discovery metadata only if Decision H selects it. A reader that does not request an artifact must not touch its pages.

The recommended compositional architecture to qualify is:

```text
Generic vBuf finalized file
├── generic header
├── generic aligned blocks/regions
│   ├── vBuf-ML bootstrap/manifest bytes
│   │   ├── profile magic and version
│   │   ├── profile feature flags
│   │   └── model/shard identity
│   ├── vBuf-ML model/runtime metadata bytes
│   ├── vBuf-ML tokenizer metadata bytes
│   ├── vBuf-ML name/shape/representation tables
│   ├── vBuf-ML fixed-width tensor directory
│   ├── optional vBuf-ML integrity bytes
│   └── contiguous tensor payload region(s)
└── optional generic finalized-file artifacts (independently selected)
    ├── Nano-Index: physical block-start bits
    ├── optional rank/select checkpoints, only if qualified
    └── optional region directory: numeric ID/kind → range/alignment
```

None of the three finalized artifacts above is assumed mandatory or assumed to coexist. Step 5A must compare them and may reject one or all. The canonical block representation remains sufficient and authoritative. If Nano survives, its slot geometry is derived solely from the canonical file BaseStep shown above.

If selected, the Nano-Index answers only where candidate physical extents start and where the next candidate topology boundary lies; it does not provide canonical payload length or identity. If selected, the generic region directory provides direct, bounds-checkable access by generic numeric identity. It does not know that one region is a tokenizer or that another contains tensors. vBuf-ML assigns profile-local numeric roles after validating its bootstrap. Tensor identity, shape, representation and direct tensor offsets remain in the vBuf-ML tensor directory because those are downstream semantics.

Generic block/index/directory fields remain generic. Key-ID values select regions or blocks; they are not tensor IDs and carry no model semantics in vBuf. Generic applications can use the same structures for telemetry columns, records, BumpArena-like descendants or other random-access payloads.

A vBuf-ML reader uses whichever generic navigation artifact is present, or canonical scanning when permitted, then reads the profile bootstrap and tensor directory before locating a tensor. It must not scan preceding tensor payloads in the qualified direct-lookup path. Hot tensor pages must not contain or require parsing metadata. The tensor directory and physical payload order are independent.

The exact generic finalized-file artifacts are outputs of BASE qualification/specification commits; exact profile records are later ML outputs. Nothing is frozen until prototypes demonstrate bounds, representability and actual consumer needs.

---

## 4. Commit-sized steps

### Step 1 — Record decisions and freeze the qualification question

**Class:** documentation only, separating BASE/INFRA/ML decisions.

**Goal:** record Decisions A–I, turn prerequisite choices into accepted ADRs, explicitly leave evidence-dependent choices open, and state the falsifiable claim: whether vBuf-ML improves any measured container/loading property while preserving identical tensor representation and kernels.

**Files/modules likely changed:**

- add `docs/adr/0001-vbuf-wire-authority.md`
- add `docs/adr/0002-descendant-profile-framing.md`
- add `docs/adr/0003-vbuf-ml-first-runtime.md`
- add `docs/adr/0004-tensor-representation-identity.md`
- update `README.md` only to point to the authoritative current spec; no ML feature list
- update `step-by-step.md` decision status

**Relevant existing material:** all files in `spec/`, current `README.md`, `rust/src/lib.rs`, `ts/vbuf.ts`, `c/vbuf.h`, benchmark scope statements.

**Existing invariants:** no wire or runtime behavior changes.

**New invariants:** each later commit cites accepted decisions; benchmark claims are limited to predeclared metrics.

**Tests/qualification:** documentation review against all three implementations; verify no ADR describes an ML concept as a generic base requirement.

**Architectural boundaries:** ADRs may reject a proposed base feature. They may not implement it.

**Expected commit outcome:** no unresolved parent-format assumption is embedded in vBuf-ML work.

**Dependencies:** none; blocks Steps 3, 6 and 8 onward.

---

### Step 2 — Build a byte-level baseline and legacy-behavior evidence corpus

**Class:** BASE.

**Goal:** preserve current valid bytes, malformed/truncated bytes, and each implementation’s actual legacy behavior before correcting or tightening it. Step 2 observes defects; it is not a strict malformed-input conformance gate.

**Files/modules likely changed:**

- add `tests/fixtures/v05/` with tiny byte-preservation fixtures produced by the current Rust and TypeScript legacy behavior
- add `tests/fixtures/v05/manifest.json` containing size and digest
- add `rust/tests/v05_compat.rs`
- add `ts/v05_compat.test.ts`
- add `scripts/verify_vbuf_fixtures.py`
- update `c/makefile` or add a small non-benchmark C conformance target

**Relevant existing code/specification:** `VBufWriter`, `VBufInstance`, `VBufWriter` in TypeScript, C inline reader, `rust/dual_test.vbuf`, and current unit tests.

**Existing invariants:** existing valid SoA/AoS use remains representable; fixture generation is not timed benchmark work; Step 2 does not repair readers.

**New invariants:** fixture bytes are immutable; every fixture records which implementation produced/derived it, which ambiguities it exercises, and the observed Rust/TypeScript/C behavior. The corpus faithfully captures valid bytes, malformed bytes, silent acceptance, clamping, partial reads, unsafe/incomplete pointer exposure, and inconsistent rejection without normalizing legacy defects. Every unsafe or non-rejecting observation carries a pending expected-v0.6 rejection annotation for Step 4.

**Tests/qualification:** cross-language read of valid scalar widths and columns; payload offset checks; malformed/truncated fixtures assert each current implementation’s observed behavior independently and need not agree. In particular, preserve the TypeScript case declaring two Float64 values but truncated by one element, where legacy `getCol(2)` returns `Float64Array(1)`; annotate that v0.6 must reject before constructing/returning a typed view. Preserve the C-facing incomplete-range pointer result without dereferencing beyond available bytes and annotate mandatory v0.6 rejection.

**Architectural boundaries:** fixtures contain generic numeric/byte columns only, never models or tensors.

**Expected commit outcome:** a reviewable byte/behavior baseline that prevents accidental format drift and gives Step 4 immutable malformed cases to promote into mandatory v0.6 rejection tests.

**Dependencies:** Step 1 compatibility decision.

---

### Step 3 — Publish one normative generic vBuf specification

**Class:** BASE.

**Goal:** create a precise corrected v0.6 specification according to Decision A; define BaseStep as the hardware-neutral canonical physical granularity supporting efficient direct native consumption; legacy-v0.5 behavior may be documented separately but cannot substitute as the vBuf-ML parent.

**Files/modules likely changed:**

- add `spec/spec_0.6.md`
- add `spec/compatibility.md`
- mark `spec/spec_0.1-draft.md` through `spec/spec_0.5-alpha.md` as historical with links to the authority
- update generic portions of `README.md`

**Relevant existing material:** all historical specs and actual Rust/TS/C offset calculations.

**Required specification details:** exact byte order and magic bytes; version encoding; finalized versus indefinite stream; file/header lengths; 64-bit-safe size policy; normative minimum/legal/maximum `BaseStep` and unambiguous encoding; checked exponent/field derivation; hardware-neutral canonical-physical-granularity semantics without an ISA-specific promise; canonical block-start geometry; block-tail padding; payload alignment and optional stricter per-payload multiples of BaseStep; count and overflow rules; payload-start and next-block formulas including overflow checks; legal semantic/physical combinations; duplicate Key-ID behavior; zero-length payloads; partial final BaseStep policy; unknown/reserved bits; extension skipping; canonical writer behavior; legacy detection. The specification defines legal values, not a universal performance winner; any writer default is non-normative unless separately qualified.

**Existing invariants:** little parsing, deterministic alignment, generic semantic categories, streaming where selected by the decision.

**New invariants:** two conforming implementations calculate identical offsets; all sizes/offsets have specified integer widths; unknown skippable content has a known length; reserved bits are rejected or ignored exactly as specified.

**Tests/qualification:** work examples by hand and by script; for every accepted BaseStep test candidate block offsets 0, 1, BaseStep−1, BaseStep, BaseStep+1, 65535 and near integer limits, accepting only offsets representable relative to the canonical region origin; test overflow counts; reject invalid stricter payload alignments; Rust/C/TypeScript derive identical BaseStep; independent review against fixture bytes. Before claiming or selecting a preferred default, use deterministic generic layouts spanning tiny/high-count multi-block, mixed and large contiguous payloads to report padding/packing density, block traversal/address calculation, cache observations, and equivalent unaligned/native, aligned-scalar and optional SIMD/vectorized paths where practical. Use identical payload bytes and work, report alignment/ISA/compiler assumptions, retain a scalar reference, and relate results to—but do not overgeneralize—the existing prepared-view near-native baseline.

**Architectural boundaries:** no tensor, model, tokenizer, quantization, GGUF or llama.cpp terminology in the normative base specification.

**Expected commit outcome:** downstream profiles have a stable parent grammar.

**Dependencies:** Steps 1–2 and Decisions A/C/G for the canonical grid; finalized-artifact details remain deferred to Steps 5A–5B.

---

### Step 4 — Make the generic reader safe before extending it

**Class:** BASE.

**Goal:** validate a file completely enough that safe APIs cannot create invalid references from attacker-controlled bytes.

**Files/modules likely changed:**

- split `rust/src/lib.rs` into `rust/src/header.rs`, `rust/src/block.rs`, `rust/src/reader.rs`, `rust/src/error.rs`, and `rust/src/lib.rs`
- add `rust/tests/malformed.rs`
- add fuzz targets under `rust/fuzz/`
- tighten `c/vbuf.h` and add checked C result/error APIs without removing legacy symbols until compatibility policy allows
- update `ts/vbuf.ts` validation and errors

**Relevant existing code:** `VBufInstance::open`, `get_as<T>`, `vbuf_get_col_ptr`, `vbuf_get_generic`, TypeScript `getCol`.

**Existing invariants:** valid files remain mmap-backed and payload access remains zero-copy.

**New invariants:** checked arithmetic precedes every addition/multiplication/alignment; payload range is inside the declared and physical file; alignment is valid; no safe arbitrary-`T` construction; semantic representation is checked; no panic crosses FFI; invalid shifts/counts/truncation return structured errors.

**Implemented API safety boundary:** canonical v0.6 uses a distinct `VBufV06`/validated-descriptor path and concrete primitive view methods. Rust's former unrestricted safe legacy `get_as<T>` is narrowed to the unsafe-to-implement `LegacyV05Pod` marker plus runtime width/range/alignment checks; local native benchmark records must opt in explicitly. The legacy generic native-memory writer is now an `unsafe` API because arbitrary `T` padding/representation cannot be proven initialized or portable. TypeScript retains the old behavior under `LegacyV05Instance` (with the old export only as a deprecated alias) and introduces `VBufV06`. C retains named legacy symbols for fixture compatibility and adds opaque `vbuf_v06_*` handles that are returned only after full-file validation. This is the smallest API change that prevents arbitrary Rust representations or incomplete v0.6 ranges from becoming safe views/pointers.

**Tests/qualification:** promote every Step 2 `expected_v06: reject` annotation into mandatory Rust/TypeScript/C rejection tests; require rejection before typed-view or pointer construction for declared ranges crossing EOF, including the preserved two-Float64 truncation fixture and C-facing incomplete-range case; fuzz open/iterate/get; Miri or equivalent tests for typed access; sanitizers for C; malicious `AShift`, count overflow, zero-sized types, wrong semantics, misalignment, truncated overflow header and data beyond EOF.

**Architectural boundaries:** known portable primitive views belong in generic vBuf; arbitrary structs remain bytes or require an explicitly unsafe caller contract. No ML types.

**Expected commit outcome:** the Step 2 transition is complete—legacy observed behavior becomes v0.6 checked range/bounds behavior with mandatory malformed-input rejection—and safe handling of untrusted downloaded containers is possible without copying valid payloads.

**Dependencies:** Step 3.

---

### Step 5 — Make the generic writer canonical and portable

**Class:** BASE.

**Goal:** produce deterministic, canonical and portable block-stream bytes rather than dumping unconstrained native object representations. Optional finalized-file artifacts are deliberately deferred to Steps 5A–5E.

**Files/modules likely changed:**

- add `rust/src/writer.rs` and `rust/src/primitive.rs`
- update `ts/vbuf.ts`
- update C writer declarations/implementation
- add `rust/tests/canonical_writer.rs` and cross-language fixture generation

**Relevant existing code:** generic `write_column<T>`, TypeScript `writeColumn`, `vbuf_write_atomic_column`, and `DataLen` handling.

**Existing invariants:** sequential writing and aligned payloads remain supported.

**New invariants:** portable primitive encodings are explicit; canonical block streams are deterministic; base payload length/finalization state follows the accepted core spec; non-power-of-two/unsupported alignment fails; native structs are not presented as portable wire values; arrays are emitted as real contiguous primitive arrays; AoS/SoA are layout compositions rather than base feature flags; simple multi-block composition is available for variable/composite representations without growing a universal object type system; forward continuation follows Decision J and terminates on the final physical block. This step does not emit Nano, directory, rank/select or integrity artifacts.

**Tests/qualification:** byte-identical Rust/TS/C primitive output; read-after-write for every primitive; empty and large logical counts; forward continuation sequences `0`, `1→0`, and `1→1→0`; reject continuation on a final block and continuation into a different Key-ID; unchained duplicate IDs remain legal; verify each chained member remains a separately enumerable physical block; interrupted/indefinite stream behavior; endian simulation where practical. Include TypeScript-first direct-consumption checks so portability work does not accidentally make JavaScript require object reconstruction.

**Architectural boundaries:** base primitives remain numbers, strings/bytes only if accepted as general-purpose, arrays/containers and opaque payloads. No tensor aliases.

**Expected commit outcome:** vBuf-ML can safely store opaque profile bytes and exact tensor bytes without relying on Rust ABI layout.

**Dependencies:** Steps 3–4 and the Decision J corrective supplement.

---

### Step 5A — Re-qualify generic finalized-file structural navigation

**Class:** BASE design/benchmark; no production wire-format implementation.

**Goal:** remeasure the already legal BaseSteps as a whole-system generic physical-layout trade-off, then determine whether Nano physical-extent topology, Nano plus simple checkpoints, a generic region directory, or a smaller combination materially improves generic vBuf consumption, including whether complete Nano-derived extents provide useful work partitions for parallel CPU consumers.

**Rationale and questions answered:**

| Structure | Question it can answer | Question it cannot answer alone |
|---|---|---|
| canonical block stream | “What does the authoritative next header say?” | direct physical or key lookup without scanning |
| raw Nano-Index | “Where do candidate canonical physical extents start, and where is the next candidate extent boundary?” | “Where is Key-ID X?”, canonical payload length, and true O(1) Nth selection |
| Nano + checkpoints | “In which bounded slot range are the Nth extent start and its next boundary?” | “Where is Key-ID X?” |
| generic region directory | “What validated range is associated with generic ID/kind X?” | what that ID means to a profile |
| vBuf-ML tensor directory | “Where is tensor X, and what are its shape/representation semantics?” | generic physical structure enumeration |

The three design possibilities must remain open:

1. Nano-Index and region directory independently survive and coexist.
2. an extended Nano-Index meets actual generic navigation needs, reducing/eliminating the directory;
3. Nano remains physical extent topology only while a directory is independently justified for logical lookup.

A fourth valid result is that one or both generic artifacts fail qualification and are omitted.

**Files/modules likely added:**

- add `docs/design/nano-index-requalification.md`
- add `docs/adr/0005-finalized-navigation-and-nano-index.md` after the evidence selects/rejects candidates and resolves Decisions B/H/I
- add `benchmarks/vbuf-navigation/README.md`
- add `benchmarks/vbuf-navigation/Cargo.toml`, `benchmarks/vbuf-navigation/src/main.rs`, and an evidence-only `benchmarks/vbuf-navigation/src/parallel_partition.rs`
- add `scripts/run_vbuf_navigation_bench.sh`
- add `scripts/validate_vbuf_navigation_reports.py`
- later store raw reports under `benchmark-results/vbuf-navigation/`; do not commit conclusions without raw data

**Historical references:** `spec/spec_0.1-draft.md:11,50-68`, `spec/spec_0.2-draft.md:5-11,18-20,24-32,36-60,68-87`, `spec/spec_0.4-alpha.md:12,45-76`, v0.5 SIMD/cache-grid and anchor text in `spec/spec_0.5-alpha.md`, 8-byte stepping in `rust/src/lib.rs`, `ts/vbuf.ts`, `c/vbuf.h`, and README SIMD/O(1) claims. Treat their performance language as design evidence, not proof of one width.

**Existing invariants:** canonical blocks alone remain sufficient; optional indexes are derived; a reader can ignore every artifact; no artifact changes payload bytes; streaming/unfinished files require none.

**New qualification invariants:** generated Nano bits are derived only from a successfully parsed canonical stream; Nano reads the already validated file BaseStep and has no independent quantum; Step 5A treats the Step 3 legal BaseStep set as an input and cannot redefine it; `slot(i) = indexed_region_start + i*BaseStep`; for a non-empty topology the first valid extent start is set; each later `1` is a candidate BaseStep-aligned canonical physical start, including every physical block in a forward-continuation chain; `0` creates no new extent and carries no payload/domain meaning; consecutive set bits propose adjacent extent bounds; long zero runs remain one proposed physical extent until the next set bit or indexed-region end; continuation is read only from canonical headers and never changes Nano bits or extent boundaries; every marked start and Nano-derived upper span remain subordinate to canonical header/range validation; indexed-region start/length and partial-final-slot policy are explicit; unused bits in the final Nano byte are zero; all slot/index arithmetic is checked; benchmark corpora and query distributions are deterministic and recorded.

**BaseStep performance qualification:** hold payload bytes, block topology and computation constant while comparing legal BaseSteps. Separate unaligned/native access, aligned scalar access, and SIMD/vectorized variants where supported; preserve the same scalar/reference result and report effective pointer alignment, compiler flags, ISA/features and whether aligned or unaligned load instructions are used. Unsupported SIMD is an optional missing variant, not a failed format. Measure traversal/address arithmetic, prepared-view scans, cache/fault behavior and end-to-end downstream-neutral work in addition to bytes. No single-host result may be called universally optimal. Keep the existing near-native prepared-view baseline as motivation/comparison evidence, while recognizing that it did not isolate BaseStep or SIMD.

**Rank/select candidates to measure, not assume:**

- Raw Nano size: `S = ceil(R/B)` slots and `ceil(S/8)` bytes for indexed size `R` and BaseStep `B`; asymptotic overhead is `1/(8*B)`, not a fixed 0.78125%. Nth selection is worst-case `O(S/64)` index words, not O(1).
- Fixed-slot checkpoint candidate: one `u64` cumulative count per `C` slots. Array size is `ceil(S/C)*8`, physical coverage is `C*B`, asymptotic payload overhead is `8/(C*B)`, and the post-checkpoint Nano scan is at most `ceil(C/64)` words. For `C=4096` and B=8/16/32/64, coverage is 32/64/128/256 KiB and checkpoint overhead is 0.0244140625/0.01220703125/0.006103515625/0.0030517578125% respectively; the scan bound remains 64 words.
- Denser fixed-slot candidate: `C=512`; for B=8/16/32/64, coverage is 4/8/16/32 KiB and overhead is 0.1953125/0.09765625/0.048828125/0.0244140625%; scan at most 8 Nano words.
- Fixed-physical-coverage candidate: approximately one checkpoint per 64 KiB, requiring `C = 65536/B` when integral. Its asymptotic checkpoint overhead is 0.01220703125% for all B, while the Nano scan bound changes with B: 128/64/32/16 words for B=8/16/32/64.
- Sparse “every Hth block” hints may be prototyped only with an explicit `8 * ceil(block_count/H)` byte formula plus framing. Their overhead is block-density-dependent and must not be advertised as a fixed percentage.
- A candidate generic directory must disclose the exact target contract (for example whole canonical block versus validated payload range), entry width, and lookup algorithm. A provisional sorted 32-byte entry may be measured with binary search (`O(log region_count)`), but neither that contract, width nor algorithm is a wire decision; a hash table must not be substituted silently.

Step 5A must compare fixed-slot, fixed-physical-span and empirically chosen checkpoint rules. It must benchmark Nth extent start, Nth start plus next extent boundary, sequential extent enumeration, and parallel extent-range construction. Checkpoint lookup still includes checkpoint search (`O(log checkpoint_count)` with binary search unless a different validated rule is selected), bounded Nano word scan, and in-word select. No prototype may be labeled constant-time without a proved bound independent of file size. Formulas above are exact for candidate arrays but exclude common finalization/artifact framing not yet chosen by Decision H; every report must add exact framing bytes.

#### Parallel work-partition hypothesis

**Hypothesis, not a format guarantee:** because Nano marks physical extent starts over BaseStep geometry, a consumer may divide Nano into independently enumerable set-bit ranges, pair each start with the next set bit/indexed-region end, and reduce serial start/end discovery or coordination for generic extent-parallel work. Nano exposes candidate topology bounds only; canonical validation still determines the actual block/payload range, and the consumer owns scheduling, worker count, affinity, NUMA policy, queues, aggregation, and cost estimation.

**Current status:** `POSSIBLE BUT NOT YET JUSTIFIED`. No repository benchmark currently demonstrates an end-to-end parallel benefit. Change this label only from validated raw evidence produced by this step.

For a Nano `u64`, physical coverage is `64*BaseStep`:

| BaseStep | Physical bytes represented by one Nano `u64` |
|---:|---:|
| 8 B | 512 B |
| 16 B | 1 KiB |
| 32 B | 2 KiB |
| 64 B | 4 KiB |
| 128 B | 8 KiB |

The BaseStep=64/page-size correspondence is only a benchmark candidate. The base format must not prefer it unless total generic layout evidence—including padding—supports that BaseStep independently. Measure page and page-multiple coverage effects on mmap locality, readahead, major/minor faults, cache footprint, parallel validation, and NUMA-observed behavior where hardware permits.

**Candidate consumer partition strategies:**

1. **Physical Nano-range partitioning:** assign disjoint contiguous slot/Nano-word ranges. Ownership is determined solely by the start bit’s slot; pair each owned start with the next set bit (which may lie in the next slice) or indexed-region end. A physical extent crossing worker slice boundaries remains owned by the start-owning worker. Measure boundary handoff/lookahead plus trivial setup/no shared iterator against density and work imbalance.
2. **Extent-count-balanced partitioning:** compute total/prefix POPCNT and assign approximate set-bit ordinal ranges. Test raw prefix scanning, selected checkpoints/sparse hints, and one-time serial prefix construction. Equal set-bit/extent count is not equal work.
3. **Byte/work-balanced partitioning:** use Nano-derived physical extent span, validated payload bytes, or a consumer-supplied deterministic cost estimate, then snap boundaries deterministically to candidate canonical extent starts. Account for canonical header parsing or other preprocessing needed to obtain costs; physical extent span is not payload length.
4. **Dynamic work queue control:** not a Nano design, but required as a consumer-side comparison so static Nano partitioning is not credited for benefits a queue achieves more simply.

**Equivalent benchmark paths:**

- A: canonical serial scan and workload;
- B: canonical parallel processing after prior serial canonical boundary discovery;
- C: Nano physical-range partitioning with start-plus-next-boundary extent construction;
- D: Nano extent-count-balanced partitioning;
- E: Nano plus each selected checkpoint/hint policy for extent-balanced boundaries;
- F: Nano-guided byte/work-balanced partitioning where the workload supplies a meaningful cost;
- G: validated dynamic-queue control using the same discovered block set and same per-block work;
- H: region-directory-assisted partitioning only when the selected directory target contract enumerates the identical generic blocks/ranges; otherwise it is not an equivalent competitor.

All timed paths must perform equivalent per-block canonical header/range validation and process the same blocks/bytes. Distinguish “validate every block referenced by an already valid/generated artifact” from “fully prove artifact-to-canonical completeness”; the latter must include the Decision I full cross-check in every compared path that claims full conformance. Boundary discovery, prefix construction, directory lookup, queue setup, and aggregation must be separately reported and included in end-to-end totals. No result may be attributed to parallelism when paths differ in validation work or page-cache state.

**Generic workloads and corpora:**

- header-only per-block validation plus deterministic counters/statistics, approximately block-count proportional; report separately whether artifact completeness was pre-established or cross-checked inside the timed path;
- payload CRC32C/hash after validation, approximately byte proportional;
- deterministic mixed synthetic cost, explicitly separated from I/O, to expose scheduling imbalance;
- optional generic conversion/transcoding, derived-index construction, inspection/statistics, and block transforms only when outputs can be compared exactly.

Use dense small blocks, high-count tiny/multi-block and variable/composite representations, sparse large blocks, mixed/uniform/highly skewed sizes, uniform/clustered starts, small files dominated by setup, and large files where scaling/vectorization is plausible. This must expose when a coarse BaseStep loses packing density even if it improves alignment or reduces Nano bytes. Cover all representative surviving BaseSteps and warm/cold mmap. vBuf-ML may later reuse a successful primitive but is not qualifying evidence for BASE promotion.

**Parallel correctness invariants/tests:**

- on a valid or fully conformance-checked Nano, each canonical block/extent start is owned exactly once: none duplicated or skipped;
- physical slice boundaries have deterministic half-open slot ranges and cannot split start ownership;
- empty slices and worker counts greater than extent count are valid;
- a physical extent spanning multiple worker slices belongs only to the start-owning worker, with its candidate upper bound supplied by the next set bit or indexed-region end;
- extent-balanced boundaries resolve to canonical starts; byte/work boundaries snap deterministically to a candidate canonical start with a specified tie rule;
- each set bit and each next-set-bit-derived span remain candidates until canonical header/range validation succeeds; a Nano span never expands the canonical valid data range;
- malformed/missing bits cannot safely merge canonical blocks, extra bits cannot authorize fake headers, and malformed canonical structure produces the same failure class as the serial validator;
- because an unverified Nano can omit entries under Decision I, “no skipped blocks” is established by full conformance comparison in qualification/validator paths, not assumed by a fast path;
- worker count does not change observable validation results or deterministic aggregation;
- parallel consumers cannot alter BaseStep, Nano geometry, artifact semantics, or canonical bytes.

**Parallel metrics:** total wall time and CPU time; speedup and scaling efficiency at 1/2/4/8/... workers where hardware permits; extents/blocks, Nano-derived physical span, and validated payload bytes per worker; extent-count, byte-count, and measured-time imbalance; synchronization/queue overhead; partition-boundary discovery time; Nano/checkpoint and canonical pages touched; major/minor faults; cache/RSS effects; finalization/index construction cost; exact artifact bytes; and complete padding-plus-artifact structural cost.

**Promotion criterion:** parallel partitioning becomes an additional generic Nano justification only if multiple generic workloads/corpora show reproducible end-to-end benefit over canonical parallel boundary discovery and appropriate dynamic-queue controls, without changing validation, bytes processed, or cache methodology. Checkpoints gain a second justification only if they materially reduce real partition setup/end-to-end time across more than one workload while still passing existing storage/complexity criteria.

**Explicit rejection criteria:** record `NOT USEFUL ENOUGH TO MATTER` if gains disappear after including Nano construction/finalization, partition discovery, canonical validation, thread/queue overhead, page-cache controls, or padding/artifact cost; if skew makes static slicing consistently inferior; or if benefit exists only for one synthetic partition query. Record `POSSIBLE BUT NOT YET JUSTIFIED` when correctness is demonstrated but evidence is narrow/noisy. Use `SUPPORTED BY EVIDENCE` only with reproducible scoped results and preserved raw reports. None of these outcomes adds scheduler semantics to vBuf or scheduling metadata to Nano/region directories.

**Tests:** property-test bit generation against canonical enumeration for every surviving BaseStep; represent the same logical block/extent topology under different legal BaseSteps; verify every bit maps to `indexed_region_start + i*BaseStep`; require the first bit for every non-empty topology; verify every later `1` is a representable canonical start; verify `0` creates no extent; consecutive `1` bits produce adjacent candidate extents; long zero runs produce one candidate extent through the next set bit/end; derive first/last/Nth start-plus-next-boundary spans; reject a stream with an unrepresentable block start; correct first/last slot and partial final slot handling; fewer than eight slots; partial final Nano byte; no blocks; all-start bits; one huge extent; padding inside an extent creates no false start; checked huge-region arithmetic; malformed/contradictory BaseStep/index metadata rejected before pointer construction; missing bits cannot authorize merged-block access; extra bits cannot authorize fake headers; Rust/C/TypeScript geometry parity; Nano-ignored parsing yields the same authoritative block sequence; candidate/header/span consistency; duplicate Key-IDs leave Nano bits unchanged; corrupted-header recovery uses later set bits only as validated candidates; run the parallel extent ownership/aggregation properties above across partition strategies and worker counts.

**Artifact deployment variants:** independently measure embedded Nano, locally reconstructed Nano, and locally reconstructed-plus-cached Nano. Construction, persistence/cache validation, cold-start page I/O, reuse count, and invalidation/version costs belong in end-to-end totals. Derivability does not imply local reconstruction is always cheaper; optional embedding does not imply every distributed file should carry it.

**Benchmark/qualification requirements:** compare only equivalent operations: canonical scan versus Nano for physical enumeration/count; raw Nano versus Nano+checkpoints for Nth physical start; canonical Key-ID/range scan versus region directory for logical lookup. A Nano-assisted Key-ID scan is a separately labeled combined path that includes header inspection/map construction; Nano alone is never reported as answering a Key-ID query. Test every legal representative BaseStep surviving base qualification—at minimum 8, 16, 32 and 64 bytes if accepted. Corpora: small/large files, high block count, tiny/multi-block variable/composite layouts, one/few large contiguous blocks, mixed small/large blocks, and equivalent topologies/payload bytes across BaseSteps. Operations: unaligned/native, aligned-scalar and supported SIMD/vectorized prepared-view workloads; canonical block traversal/address calculation; sequential extent enumeration; total extent/block count; Nth extent start; Nth start plus next boundary; random extent/block ordinal access; parallel extent-range construction; random Key-ID lookup; consistency validation; and separately labeled corruption-diagnostic candidate scanning. Environments: cold/warm mmap with validated cache methodology. Metrics: wall time, cycles where reliable, faults, bytes/pages touched, RSS/cache footprint, block density, exact Nano/checkpoint/directory bytes, construction/finalization cost, validation time, scalar/SIMD alignment fixups where observable, and padding waste caused by BaseStep. Report the whole-system result as `padding waste + Nano bytes + checkpoint bytes + region-directory bytes + other finalized-artifact bytes + measured native/scalar/SIMD/cache/traversal effects`, keeping byte costs and timing/cache metrics in separate units rather than inventing an unsupported scalar score. Omit only absent structures/unsupported optional paths. Do not optimize Nano percentage, page correspondence or one ISA in isolation. Use balanced order, repeated independent runs and raw samples.

**Non-goals/architectural boundaries:** no ML fixture is required to justify a result; no tensor names/shapes/types; no corruption “recovery” claim; no production parser changes; no sophisticated succinct structure beyond the small checkpoint candidates.

**Downstream impact:** results apply to generic telemetry, columns, records, inspection/debug tools, BumpArena-like descendants and future profiles. vBuf-ML is only a later consumer.

**Migration/version implications:** benchmarks must model each candidate v0.6 BaseStep selected by Decision G and current legacy v0.5 separately. A legacy file can receive a v0.6 Nano only if every canonical start is representable under its declared v0.6 BaseStep and the rest of the v0.6 conversion contract is satisfied; otherwise it must be rewritten.

**Expected commit outcome:** raw evidence, corrected complexity claims, a written select/reject recommendation for each artifact, and a scoped parallel-partition result labeled `SUPPORTED BY EVIDENCE`, `POSSIBLE BUT NOT YET JUSTIFIED`, or `NOT USEFUL ENOUGH TO MATTER`. No wire or scheduler commitment.

**Dependencies:** Steps 3–5 and Decisions A/G. Results feed Decisions B/H.

#### Step 5A evidence recorded — 2026-08-14

The evidence runner is `rust/src/bin/v06_navigation_bench.rs`; its outputs are
experimental, in-memory artifacts and do not define a v0.6 wire structure.
Reproduce with `scripts/run_v06_navigation_bench.sh LABEL`. The measured raw
CSV, generated interpretation, and host/compiler metadata are preserved under
`benchmark-results/vbuf-navigation/`. Validation is performed by
`scripts/validate_v06_navigation.py`.

**Measured facts:** six frozen BaseSteps (8, 16, 32, 64, 128, 256 bytes) and
eight generic layouts completed 20 samples for 18 operation/variant paths per
layout/BaseStep (17,280 raw rows across 864 groups). The layouts include many tiny blocks, few
large blocks, mixed blocks, AoS-style and SoA-style compositions,
continuation composites, opaque workloads, and zero/partial cases. Canonical
payload bytes and block topology were held constant while grid/padding
changed. The raw report records file, payload, Nano, checkpoint, and directory
byte counts separately. Median file sizes across the corpus were 73,752 /
131,094 / 262,150 / 524,294 / 1,048,582 / 2,097,158 bytes for BaseStep 8 /
16 / 32 / 64 / 128 / 256, while the corresponding median payload was 45,056
bytes. These pooled medians describe intentionally heterogeneous stress
corpora; exact canonical-header and padding columns are required for layout-
level interpretation. They show that coarse BaseSteps can make non-payload
structural overhead dominate high-block-count small/composite layouts, not that
any BaseStep is globally optimal.

**Correctness facts:** property tests compare Nano-derived set slots with every
validated canonical physical block start for every corpus/BaseStep; continuation
members remain separate; first/last/every-ordinal selection agrees with raw
set-bit enumeration for the experimental checkpoint helper; final-byte unused
bits are zero; invalid BaseStep and near-`u64::MAX` Nano arithmetic fail
closed. The current benchmark also includes raw bitmap select plus
next-boundary work, fixed-slot cumulative checkpoints, exact validated-range
directory lookup, and checksummed parallel controls; these are qualification
paths, not wire artifacts. Canonical parsing remains the
authority. No Nano artifact was added to a reader path, so it cannot authorize
bytes.

**Measured-path boundary:** the run measures warm in-process operations and
records compiler/CPU metadata. It does not claim controlled cold/warm mmap,
page-fault, SIMD load, native pointer-alignment, or NUMA results; those remain
missing evidence rather than failures. Rayon paths are workload controls, not
format scheduler semantics. Embedded serialized load, local reconstruction, and
reconstructed-cache load are measured as distinct in-memory operations, but
persistence, invalidation, reuse amortization, and page-I/O qualification remain
outstanding.

**Conservative decisions:**

- **BaseStep:** measured behavior supports retaining the frozen legal set as a
generic tuning input. No value is promoted as universally optimal and no core
contract is changed.
- **Nano:** **POSSIBLE BUT NOT YET JUSTIFIED**. Physical enumeration is correct,
but this run does not establish end-to-end benefit after construction,
validation, artifact bytes, persistence, or cache costs. Nano remains optional
and non-normative.
- **Rank/select checkpoints:** **POSSIBLE BUT NOT YET JUSTIFIED**. Raw bitmap
select plus next-boundary and fixed-slot cumulative checkpoints at 512, 4096,
and 65536 slots now have correctness and construction/lookup measurements, but
no checkpoint is selected without broader deployment evidence.
- **Generic region directory:** **POSSIBLE BUT NOT YET JUSTIFIED**. The
experiment now compares the same first validated `(block,payload_start,length)`
range contract for canonical scan and sorted lookup, while construction and
storage remain separately accounted for. The current run is insufficient to
justify a generic lookup artifact or logical-ID wire contract.
- **Parallel CPU partitioning:** **POSSIBLE BUT NOT YET JUSTIFIED**. Canonical,
Nano-guided, and dynamic Rayon controls now perform checksummed payload work,
but worker scaling, skew-balanced partitioning, queue costs, and cold-page
end-to-end qualification remain insufficient for promotion.
- **Embedded vs reconstructed:** **POSSIBLE BUT NOT YET JUSTIFIED**. The measured
paths now distinguish serialized load, reconstruction, and cache-load mechanics,
but they remain warm in-memory operations without cold-start/reuse evidence.

These are measured-fact, inference, and decision distinctions; no optional
artifact is made normative and no finalization envelope is added. The raw
report records the benchmark source hash for reproducibility.

---

### Step 5B — Specify the generic finalization envelope and selected artifact contracts

**Class:** BASE specification only.

**Goal:** if Step 5A selects at least one artifact, define how a valid stream becomes a finalized file without changing canonical payload representation.

**Rationale:** independently optional artifacts still need one unambiguous, large-file-safe discovery contract; allowing each artifact/profile to invent one would duplicate parsing and weaken forward compatibility.

**Files/modules likely changed:**

- update `spec/spec_0.6.md`
- add `spec/finalization.md`
- add selected-format test-vector descriptions under `spec/vectors/`
- update `spec/compatibility.md`

**Historical references:** v0.2 `HasIndex`/`IsStream`/trailing index; v0.4 index with no discovery contract; v0.5 `DataLen == 0` indefinite stream.

**Existing invariants:** an unfinished stream is a valid canonical block sequence; readers need not touch finalization artifacts; payload bytes and offsets are not rewritten by finalization unless the accepted base header itself is patched as specified.

**New invariants:** finalized/unfinished detection is unambiguous; all artifact types have ID, flags, checked `u64` offset/length and required/skippable behavior; artifacts derive physical geometry from the already decoded canonical BaseStep and cannot carry a contradictory private quantum; indexed-region start/length are explicit; artifacts cannot overlap canonical data or one another unless explicitly permitted; unknown optional artifacts are skipped; unknown required artifacts fail; append/truncation behavior and footer/header precedence are deterministic; artifact validity never overrides canonical parsing.

**Tests/qualification:** hand-worked empty/small/>4-GiB sparse examples across accepted BaseSteps; footer/header truncation at every field; contradictory BaseStep/artifact metadata; indexed-region alignment and partial-final-slot policy; unknown artifacts; duplicate artifact IDs; overflow/overlap; finalize without payload rewrite; unfinished file remains readable; ignored artifacts cause no page touches in an instrumented reader.

**Non-goals/architectural boundaries:** the envelope does not require Nano, directory, checksums or ML; it is not added if Step 5A rejects all independently useful artifacts.

**Downstream impact:** descendants can discover generic acceleration structures but must assign their own semantics.

**Migration/version implications:** this is v0.6+ behavior. Legacy readers must not misread appended artifact bytes as blocks; compatibility behavior follows Decision C. Legacy v0.5 files require canonical conversion before v0.6 finalization unless they are proven to satisfy the complete v0.6 contract; representable block geometry alone is insufficient.

**Expected commit outcome:** one reviewable, domain-neutral finalization contract; no implementation yet.

**Dependencies:** Step 5A and Decision H.

---

### Step 5B.1 — Implement the selected generic finalization envelope

**Class:** BASE, conditional on Step 5A retaining at least one generic finalized artifact.

**Goal:** implement the Step 5B discovery contract independently of Nano, checkpoints, directories, integrity data, and profile semantics.

**Rationale:** Step 5B freezes only the wire contract. Without a separate envelope implementation commit, the first artifact commit would silently mix common finalization parsing/writing with artifact-specific behavior.

**Files/modules likely changed:**

- add `rust/src/finalization.rs`
- update `rust/src/{reader,writer,error}.rs`
- add generic finalization discovery to `ts/vbuf.ts` and checked C APIs in `c/vbuf.h`/Rust FFI
- add `rust/tests/finalization.rs` and cross-language fixtures containing unknown optional test artifacts

**Existing invariants:** canonical and unfinished streams parse without this envelope; payload bytes and canonical BaseStep remain unchanged; no artifact pages are touched unless discovery/use is requested.

**New invariants:** implementations derive BaseStep only from the base header; footer/header/artifact-table ranges use checked `u64`; unknown optional artifacts are exposed/skipped and unknown required artifacts fail; no artifact parser is needed to parse canonical blocks; envelope presence never makes an artifact authoritative.

**Tests/qualification:** all Step 5B vectors; empty artifact table; truncation/overflow/overlap; contradictory private geometry rejected; ignored-artifact page-touch check; Rust/C/TypeScript discovery parity; legacy and unfinished inputs take their specified non-finalized paths.

**Architectural boundaries:** numeric artifact framing only—no Nano bits, Key-ID index, tensor fields, checksums, or profile roles.

**Migration/version implications:** v0.6+ only; legacy-v0.5 bytes must satisfy the complete conversion policy before receiving an envelope.

**Expected commit outcome:** selected artifacts can be added in later small commits without duplicating or redefining generic finalization.

**Dependencies:** Step 5B. Omit this step if Step 5A rejects all finalized artifacts.

---

### Step 5C — Implement the selected optional Nano-Index

**Class:** BASE, conditional on Step 5A selecting it.

**Goal:** generate, discover and consume a corrected one-bit-per-canonical-BaseStep physical extent-topology map while keeping canonical parsing authoritative.

**Rationale:** restore the historical generic structural-map benefit only if Step 5A demonstrates that bit scanning saves enough reader work to justify its bytes and consistency surface.

**Files/modules likely changed:**

- add `rust/src/nano_index.rs`
- update `rust/src/{reader,writer}.rs`
- add TypeScript support in `ts/vbuf.ts`
- add checked C APIs/declarations in `c/vbuf.h` and Rust FFI
- add `rust/tests/nano_index.rs`, TS/C conformance tests and cross-language fixtures

**Historical references:** v0.2 and v0.4 Nano-Index sections plus corrections recorded by Step 5A/5B.

**Existing invariants:** files without Nano remain valid; ignoring Nano yields identical blocks; streaming writers can emit blocks without retaining the whole index in memory.

**New invariants:** `B` comes only from the validated base header; one bit maps exactly `indexed_region_start + i*B`; Nano has no independent slot quantum; every canonical block/extent start in the indexed region is representable and set; a non-empty topology begins with a set first slot; `0` means only “no new extent starts”; consecutive set bits or set-bit-plus-region-end derive candidate physical extent spans that never override canonical payload/range validation; `slot_count=ceil(region_size/B)` and `nano_bytes=ceil(slot_count/8)` use checked arithmetic; partial-final-slot and indexed-region-boundary rules match Step 5B; tail bits are zero; generation uses writer-known starts or canonical re-scan; readers validate BaseStep/location/length before pointer construction or bit access; marked starts and derived spans are cross-checked before use; mismatch never authorizes unsafe access.

**Tests:** all Step 5A structural/extent edge cases; the same topology under each legal representative BaseStep; required first start; exact first/last/partial slot mapping; consecutive starts, long continuation runs, and Nth start-plus-next-boundary; malformed/truncated index; contradictory BaseStep metadata; missing and extra starts; impossible/unrepresentable start; padding inside an extent creates no start; derived spans cannot authorize bytes outside canonical ranges; huge checked and >4-GiB sparse regions; final-byte unused bits; Rust/TS/C geometry and byte identity; index-ignored canonical sequence equality.

**Benchmark/qualification:** rerun accepted Step 5A benchmarks against production code for every surviving representative BaseStep and report generation/finalization cost, cold/warm extent enumeration, Nth start-plus-next-boundary, parallel extent-range construction, page touches, exact artifact bytes, extent density and combined padding-plus-index structural cost.

**Non-goals/architectural boundaries:** no Key-ID lookup table, profile semantics, tensor knowledge, automatic repair or promise that every `1` is valid without parsing.

**Downstream impact:** generic consumers may enumerate candidate extent starts and topology spans cheaply. If and only if Step 5A supports the parallel hypothesis, consumers may also treat disjoint set-bit ordinal ranges as deterministic candidate extent work boundaries; vBuf still defines no scheduler. Profiles may use ordinal physical navigation but cannot derive canonical payload length or semantic lookup from Nano alone.

**Migration/version implications:** written only after canonical v0.6 BaseStep is unambiguously decoded; never silently append Nano to legacy bytes whose starts or alignment violate that declared geometry.

**Expected commit outcome:** an optional, interoperable structural index, or deletion of this step if production measurements fail the qualification threshold.

**Dependencies:** Steps 5A–5B.1, Decision I, and selected Nano decision.

---

### Step 5D — Implement the selected optional generic region directory

**Class:** BASE, conditional on Step 5A independently justifying logical lookup.

**Goal:** provide direct generic numeric ID/kind-to-range navigation only if Nano/checkpoints cannot meet that requirement and benchmarks justify the additional structure.

**Rationale:** Nano records physical extent topology and candidate spans but contains no generic identity/kind-to-validated-range mapping; a directory is separate and is justified only by consumers that actually need logical lookup. Step 5A must nevertheless measure whether cheap extent spans reduce the directory's incremental value.

**Files/modules likely changed:**

- add `rust/src/directory.rs`
- update `rust/src/{reader,writer}.rs`
- add corresponding checked APIs in `ts/vbuf.ts` and `c/vbuf.h`
- add `rust/tests/directory.rs` and cross-language fixtures

**Relevant existing material:** current linear Key-ID scans; old Nano proposals; Decision B and Step 5A comparative results.

**Existing invariants:** sequential/streaming vBuf remains valid without a directory; Nano presence is independent; generic blocks remain iterable.

**New invariants:** before wire freeze, select one explicit entry target contract—whole canonical block, validated payload range within a canonical block, or another precisely bounded canonical subrange—and do not mix meanings silently; entries contain only accepted generic numeric identity/kind, flags, `u64` offset/length and descriptive alignment; every range is validated against canonical data; any stricter alignment is an integer multiple of BaseStep and cannot redefine block-start geometry; duplicate-ID, ordering, alias/overlap and unknown-entry rules are explicit; directory and canonical scan agree.

**Tests:** direct lookup equals sequential lookup under the selected target contract; reject a directory entry that points outside or inconsistently into canonical data; absent directory fallback; duplicate IDs; overlap/alias policy; malformed count/entry size; unknown flags; alignment/BaseStep contradiction; >4-GiB sparse offsets; Rust/TS/C parity; inconsistency with Nano where both exist.

**Benchmark/qualification:** compare canonical Key-ID/range scan against directory lookup for the identical target contract, including cold/warm page touches, directory bytes/cache footprint and finalization cost. A Nano-assisted header scan/map is a separate combined implementation; Nano alone is not a Key-ID/range lookup competitor. If the selected directory contract enumerates the same independent blocks/ranges as a parallel workload, include it as Step 5A path H and measure partition setup/end-to-end work without adding scheduling fields.

**Non-goals/architectural boundaries:** no names, schemas, tensor fields, metadata, quantization, worker counts, costs, queues, affinity, NUMA policy, or scheduling hints. Do not turn numeric regions into a universal object or scheduler system.

**Downstream impact:** telemetry/column/profile readers gain bounded logical navigation; vBuf-ML maps profile-local role IDs but retains its semantic tensor directory.

**Migration/version implications:** optional v0.6+ finalized artifact. Canonical v0.6 files can gain it through finalization without payload rewrite; legacy-v0.5 files require conversion unless proven to satisfy the complete v0.6 contract.

**Expected commit outcome:** a minimal generic logical directory only if it adds measured value beyond selected structural indexing.

**Dependencies:** Steps 5A–5B.1 and Decision I; independent of Step 5C except coexistence tests.

---

### Step 5E — Add simple rank/select checkpoints only if independently justified

**Class:** BASE, conditional optimization.

**Goal:** bound Nth physical extent navigation—including `select(i)` plus `select(i+1)` candidate span construction—without introducing a complex succinct-data-structure subsystem.

**Rationale:** raw bit-vector select is linear in index words in the worst case; sparse cumulative counts may provide a simple measured bound without replacing vBuf with a sophisticated index library.

**Files/modules likely changed:**

- add checkpoint support to `rust/src/nano_index.rs`
- update selected cross-language readers only after the wire contract is frozen
- add `rust/tests/nano_rank.rs`
- update `spec/finalization.md` and vectors

**Relevant existing material:** Step 5A fixed-slot/fixed-physical-span formulas and raw benchmark reports across BaseSteps.

**Existing invariants:** raw Nano and canonical scans remain valid without checkpoints; checkpoints are derived and optional; mmap consumers can access the small checkpoint array directly; Nano geometry already comes from BaseStep.

**New invariants:** the selected checkpoint rule states whether it is slot-count-based, physical-span-based, or an empirically versioned policy; any slot interval is derived deterministically from BaseStep where needed; interval/count width are fixed/versioned; cumulative counts are monotonic and exactly match preceding Nano bits; size is checked from slot count; lookup documents checkpoint-search and maximum Nano-word bounds for each legal BaseStep; very large files cannot overflow cumulative `u64` counts; checkpoints cannot override or redefine BaseStep.

**Tests:** first/last/exact-boundary Nth extent starts and next-boundary pairs across accepted BaseSteps; final extent uses indexed-region end; extent-count-balanced worker boundaries resolve to the same ordinals as canonical enumeration; fixed-span rounding when span/BaseStep is not integral; malformed/non-monotonic counts; BaseStep/interval/count mismatch; large sparse synthetic files; cross-check every result against raw Nano topology and canonical enumeration.

**Benchmark/qualification:** only policy/interval combinations surviving Step 5A may be implemented. Re-measure exact overhead, physical coverage, cold/warm Nth start lookup, Nth start-plus-next-boundary, sequential extent enumeration, parallel extent-range construction, cache effects, and extent-balanced partition-boundary/end-to-end workload time for each representative BaseStep. Compare against one-time raw POPCNT prefix construction, sparse hints, and dynamic-queue controls. Keep checkpoints only if benefit exceeds raw Nano for declared workloads without hiding added BaseStep padding cost; a win in one synthetic partition operation is insufficient.

**Non-goals/architectural boundaries:** no Elias–Fano, wavelet trees or other sophisticated succinct structures without a new evidence-driven ADR; no Key-ID or ML semantics.

**Downstream impact:** consumers needing block ordinals receive bounded predictable navigation; consumers doing only sequential scan need not touch checkpoint pages.

**Migration/version implications:** an optional artifact associated with a specific Nano version/slot count; readers safely ignore unsupported optional checkpoint versions.

**Expected commit outcome:** a small measured accelerator or an explicit recorded rejection with raw evidence.

**Dependencies:** production Step 5C and Step 5A evidence; it must not block Step 5D.

---

### Step 6 — Add wire-neutral checked range access infrastructure

**Class:** INFRA.

**Goal:** support mmap, `pread` and future range-backed readers with the same validated offsets, without forcing an I/O abstraction into hot tensor access.

**Files/modules likely added:**

- add `rust/vbuf-layout/Cargo.toml`
- add `rust/vbuf-layout/src/{lib,range,align,source}.rs`
- add it to a `[workspace]` in `rust/Cargo.toml` while retaining the root `vbuf-core` package and current `cd rust && cargo ...` workflows
- add `rust/vbuf-layout/tests/`

**Relevant existing code:** `memmap2` ownership in `VBufInstance`, repeated alignment formulas, benchmark mmap usage.

**Existing invariants:** mmap readers can return borrowed mapped bytes with no copy.

**New invariants:** `ByteRange { offset: u64, length: u64 }` is validated before conversion to platform `usize`; a source advertises whether it can lend zero-copy bytes; control-path polymorphism does not appear in numeric kernels.

**Tests/qualification:** >4-GiB sparse-file ranges on a capable filesystem; 32-bit representability tests where CI permits; mmap and pread return equal bytes; overflow/property tests.

**Architectural boundaries:** this crate knows neither vBuf wire fields nor ML semantics. Do not extract string tables, section directories or metadata abstractions until a second real consumer proves reuse.

**Expected commit outcome:** partial/range experiments share safe arithmetic while mmap remains the fast path.

**Dependencies:** Step 3 size policy; can follow Step 4.

---

### Step 7 — Create an empty descendant crate and dependency-boundary tests

**Class:** ML scaffolding only.

**Goal:** establish dependency direction without defining the ML wire format.

**Files/modules likely added/changed:**

- add `rust/vbuf-ml/Cargo.toml`
- add `rust/vbuf-ml/src/lib.rs`
- add `rust/vbuf-ml/tests/dependency_boundary.rs`
- add workspace membership in `rust/Cargo.toml`
- add `docs/vbuf-ml/README.md`

**Relevant existing code:** `vbuf-core` public APIs and crate layout.

**Existing invariants:** existing `vbuf-core` library/C ABI and commands continue to build independently.

**New invariants:** `vbuf-ml -> vbuf-core` and optionally `vbuf-layout`; never the reverse. Building with `cargo build -p vbuf-core` does not compile vBuf-ML.

**Tests/qualification:** dependency graph check in CI; search/lint preventing ML vocabulary and dependency names in base source/spec; all old tests and benchmark validators remain runnable.

**Architectural boundaries:** no parser, writer or speculative model abstraction in this commit.

**Expected commit outcome:** a mechanically enforced architectural boundary.

**Dependencies:** Step 1 framing decision; preferably Steps 3–5.

#### Step 7 implementation recorded

`rust/vbuf-ml` is now a separate workspace crate depending only downward on
`vbuf-core` and `vbuf-layout`. `scripts/verify_architecture.py` checks the
actual Cargo metadata graph and rejects reverse references from generic Rust
sources. Its only public helper accepts an already validated generic range and
reports its byte length; this is a boundary smoke seam, not ML semantics.

No tensor, model, tokenizer, quantization, backend, placement, finalization,
Nano, checkpoint, or directory behavior was added. Future profile semantics
must follow:

```text
raw bytes -> canonical v0.6 validation -> checked ranges -> vbuf-ml semantics
```

Generic promotion still requires independent downstream-neutral utility and
qualification. Building generic packages does not require the descendant.

---

### Step 8 — Specify the vBuf-ML bootstrap and map semantic roles onto generic regions

**Class:** ML.

**Goal:** identify the descendant profile and assign ML-local meaning to directly addressable generic vBuf regions, with bounded reads and no tensor scan.

**Files/modules likely added:**

- add `docs/vbuf-ml/spec-0.1-draft.md`
- add `docs/vbuf-ml/bootstrap.md`
- add `rust/vbuf-ml/src/{bootstrap,region_roles,error}.rs`
- add binary fixtures under `rust/vbuf-ml/tests/fixtures/`

**Relevant existing material:** accepted vBuf parent spec; generic byte/blob blocks; whichever Nano/directory/finalization artifacts survive Steps 5A–5E; GGUF header/tensor information structure only as comparison evidence.

**Existing invariants:** outer bytes are valid generic vBuf; a generic reader can always parse canonical blocks and can locate/skip regions through any selected generic artifact without understanding ML.

**New invariants:** profile-local magic/version; bootstrap has a bounded maximum size; profile role IDs map required/optional ML meanings onto validated generic regions when a region directory exists. If generic qualification rejects a region directory, the bootstrap may carry profile-local role ranges, but it must use the same checked `u64` range rules and document why Nano/scan is insufficient. Unknown optional profile roles are skippable; unknown required roles fail; hot payload regions are directly locatable.

**Tests/qualification:** parse bootstrap through mmap and small range reads; unknown optional/required roles; duplicate required roles; truncation; mismatch between bootstrap declarations and any generic directory/Nano; canonical generic iteration over the same fixture; test the no-artifact fallback selected by Step 5A.

**Architectural boundaries:** vBuf provides only the generic navigation primitives that survived independent qualification; vBuf-ML provides semantic region roles. ML must neither force a rejected base directory back into vBuf nor duplicate selected generic offset/length/alignment framing without a demonstrated missing capability.

**Expected commit outcome:** a vBuf-ML file can be recognized and navigated by composing selected generic primitives, while generic vBuf remains semantically unaware of ML.

**Dependencies:** Decisions A–C/G, Step 5A’s select/reject result, and Step 7. Decision H plus Steps 5B–5B.1 and any selected Steps 5C–5E are dependencies only if at least one generic finalized artifact survives; the no-artifact vBuf-ML path must not depend on a finalization envelope.

#### Step 8 implementation recorded

Added the minimal Rust-only vBuf-ML bootstrap in `rust/vbuf-ml/src/` with
its draft contract in `docs/vbuf-ml/bootstrap.md` and
`docs/vbuf-ml/spec-0.1-draft.md`. Exactly one opaque canonical v0.6 block with
profile-local Key-ID `0xF000` is discovered through generic parsing. Its bounded
little-endian payload maps role IDs to generic Key-ID plus physical occurrence;
it does not duplicate semantic, physical, width, count, alignment, or offset
facts already carried by canonical vBuf.

Profile 0.1 defines required `TensorDirectory` and `ModelMetadata` roles and
optional `TokenizerMetadata`. Unknown optional roles are skipped; unknown
required roles, duplicate known roles, missing required roles, malformed
payloads, unsupported versions, and missing references fail deterministically.
Each resolved role retains its canonical block index and Step 6 checked payload
range. Generic primitive semantics remain authoritative, including when a role
references an `f32[]`, opaque bytes, or another valid generic value.

No tensor-directory schema, model metadata schema, tokenizer schema,
quantization semantics, optional navigation artifact, finalization, or backend
behavior was added.

---

### Step 9 — Derive and specify the tensor directory

**Class:** ML.

**Goal:** represent exactly what a runtime needs to construct tensor views, no more: identity, shape, representation and a canonical generic value reference.

**Files/modules likely added:**

- add `docs/vbuf-ml/tensor-directory.md`
- add `rust/vbuf-ml/src/{tensor_directory,name_table,shape_table,representation}.rs`
- add `rust/vbuf-ml/tests/tensor_directory.rs`

**Relevant existing material:** v0.5 Key-ID/width/count limitations; GGUF tensor info (`name`, dimensions, `ggml_type`, offset); selected llama.cpp tensor construction path from Decision E.

**Existing invariants:** tensor payload bytes need no metadata prefix and remain directly mmap-able.

**New invariants:** unique UTF-8 tensor names; rank and each dimension are explicit `u64`; logical element count is checked against dimensions; the initial representation ID is `CanonicalPrimitive`; generic Key-ID plus physical occurrence resolves the canonical value; offsets, lengths, generic counts, widths, physical types, and alignments are not duplicated; profile 0.1 constrains each tensor to one non-continuing canonical block.

**Tests/qualification:** scalar through maximum accepted rank; duplicate names; invalid UTF-8 policy; dimension product overflow; packed block-size versus byte-length checks; lookup by directory index and exact name; directory-only range read.

**Architectural boundaries:** tensors, shapes and representation registries exist only in vBuf-ML. Do not reinterpret generic vBuf Key-ID as a tensor identifier.

**Expected commit outcome:** a runtime can locate and validate a tensor without scanning payloads or parsing model metadata.

**Dependencies:** Steps 8 and Decision D.

#### Step 9 implementation recorded

Implemented the minimal tensor directory in `rust/vbuf-ml/src/tensor_directory.rs`
with its contract in `docs/vbuf-ml/tensor-directory.md`. The directory is
parsed only from the checked `TensorDirectory` role discovered by the Step 8
bootstrap. Entries contain UTF-8 names, rank/dimensions, a minimal
`CanonicalPrimitive` representation ID, and canonical generic Key-ID plus
physical occurrence.

The wire format uses a 20-byte header and 10-byte fixed per-entry overhead,
followed by inline names and `u64` dimensions. Entries are canonically sorted
by UTF-8 bytes. Rank zero is a scalar; ranks through 16 and nonzero dimensions
are supported. Shape products, canonical count, bit width, and payload byte
length are checked against the canonical descriptor. Exact-name lookup uses
binary search over validated order.

No tensor offsets, lengths, generic primitive facts, quantization layouts,
continuation aggregation, metadata schema, tokenizer schema, backend behavior,
or generic region-directory support was added. Generic vBuf and the v0.6 wire
format remain unchanged.

---

### Step 10 — Add model metadata from runtime requirements

**Class:** ML.

**Goal:** store the minimum typed information required to instantiate the selected architecture, plus safely skippable descriptive metadata.

**Files/modules likely added:**

- add `docs/vbuf-ml/metadata.md`
- add `docs/vbuf-ml/architectures/<selected>.md`
- add `rust/vbuf-ml/src/metadata/{mod,value,keys}.rs`
- add architecture validation tests

**Relevant existing material:** selected llama.cpp loader/model initialization code; GGUF architecture metadata keys as conversion input, not automatic requirements.

**Existing invariants:** tensor access does not require parsing unrelated metadata; unknown optional keys remain skippable.

**New invariants:** metadata values have explicit portable types and lengths; keys are namespaced; required keys are listed per architecture/profile version; duplicate-key policy is deterministic; model identity and architecture are explicit; descriptive fields never alter tensor interpretation unless standardized as required runtime fields.

**Tests/qualification:** derive a requirement matrix mapping every required field to the runtime operation that consumes it; missing/wrong-type/duplicate key tests; unknown-key round trip; compare instantiated selected-model configuration with the GGUF path.

**Architectural boundaries:** architecture names and keys remain in vBuf-ML. A generally reusable typed-value codec may be extracted only in a later independent INFRA commit after proving another consumer.

**Expected commit outcome:** the selected runtime can construct model configuration without external files.

**Dependencies:** Steps 8–9 and Decision E.

#### Step 10 implementation recorded

Added `rust/vbuf-ml/src/metadata.rs` and `tokenizer.rs` with contracts in
`docs/vbuf-ml/metadata.md` and `docs/vbuf-ml/tokenizer.md`.

Model metadata is a sorted semantic index over canonical Key-ID/occurrence
references. Profile 0.1 requires `Architecture`, `ContextLength`,
`EmbeddingLength`, `LayerCount`, and `HeadCount`; it permits
`FeedForwardLength`, `NormalizationEpsilon`, and `RopeTheta`. Domain checks
validate UTF-8, scalar semantic compatibility, positivity, and finite floats
without duplicating generic physical facts.

Tokenizer metadata is optional and currently defines a bounded
`VocabularyOnly` view. It references canonical UTF-8 bytes, a `u64` offset
array, optional scores/types arrays, and optional BOS/EOS/UNK/PAD scalars.
Token IDs are implicit ordinals. Offset ranges, UTF-8, vocabulary counts,
parallel-array lengths, and special-token bounds are checked. Large arrays and
text pools remain direct checked canonical ranges; no token object graph is
materialized.

Unknown optional entries are skipped, unknown required entries fail, known
entries are sorted and unique, and all references resolve through canonical
validated descriptors. No generic crate or v0.6 wire behavior changed.

---

### Step 11 — Qualify future tokenizer algorithms and cold-section loading

**Class:** ML.

**Goal:** qualify tokenizer algorithms and cold-section loading beyond the Step 10 vocabulary-only view, without changing its canonical reference model.

**Files/modules likely added:**

- add `docs/vbuf-ml/tokenizer.md`
- add `rust/vbuf-ml/src/tokenizer/{mod,string_array}.rs`
- add tokenizer fixtures and tests

**Relevant existing material:** selected model’s GGUF tokenizer keys and the selected llama.cpp tokenizer construction path.

**Existing invariants:** tensor lookup does not require tokenizer access; tokenizer data remains outside hot tensor pages.

**New invariants:** tokenizer family/version, tokens, token types/scores where required, merges where required, special token IDs and chat template semantics are explicit; large string arrays use offset/length tables and bounded UTF-8 data; indexes across parallel arrays are validated.

**Tests/qualification:** tokenization/detokenization parity against the same GGUF model on a fixed multilingual/edge-case corpus; malformed offsets; embedded NUL and invalid UTF-8 policy; range-load only tokenizer sections.

**Architectural boundaries:** tokenizer is vBuf-ML-specific and must not become a generic vBuf primitive merely because it is large metadata.

**Expected commit outcome:** self-contained model behavior equivalent to the chosen GGUF fixture.

**Dependencies:** Steps 8 and 10.

#### Step 11 qualification recorded

Added `docs/vbuf-ml/step11-qualification.md`,
`rust/vbuf-ml/examples/step11_qualification.rs`, and
`scripts/run_step11_qualification.sh`. The qualification uses a real read-only
file-backed mmap fixture containing small hot semantic regions, a 4 MiB token
pool, canonical tokenizer arrays, and an 8 MiB unrelated auxiliary region.

The first target is not pinned enough to justify a tokenizer algorithm beyond
the existing `VocabularyOnly` baseline. BPE/merge tables,
SentencePiece-like behavior, WordPiece, chat templates, and tokenizer execution
remain deferred. No merge representation or universal tokenizer framework was
added.

The measured lazy path performs canonical validation, bootstrap discovery, model
metadata parsing, and TensorDirectory parsing without touching tokenizer or
auxiliary payloads. The eager comparison explicitly initializes tokenizer data,
touches token arrays, and then touches the auxiliary range. On the recorded
host/fixture, model-open remained about 0.05 ms with no tokenizer payload access;
tokenizer initialization touched about 4.2 MiB and the auxiliary phase touched
8 MiB. These are warm-cache, host-specific qualification results; process page
fault deltas are recorded but are not claimed as storage-I/O measurements.

The current tokenizer parser eagerly validates the complete UTF-8 pool during
tokenizer initialization. This is documented as the current deterministic
policy, not silently changed for benchmarking.

No portable LayerIndex, PhysicalRangeIndex, Nano dependency, or transfer plan
was selected. The current profile lacks target-qualified layer/group semantics;
a runtime-local ordinal/range index remains possible but not yet justified.
Any future derived index must be constructed from canonical descriptors and
remain tied to its source model instance.

---

### Step 12 — Define exact tensor and quantization representations

**Class:** ML.

**Goal:** map directory representation IDs to exact packed bytes and validation rules, initially preserving selected GGML representations unchanged.

**Files/modules likely added:**

- add `docs/vbuf-ml/representations.md`
- add `docs/vbuf-ml/representations/ggml.md`
- add `rust/vbuf-ml/src/representations/{mod,plain,ggml}.rs`
- add representation fixtures generated from the pinned GGML revision

**Relevant existing material:** selected `ggml_type` definitions, block sizes and llama.cpp kernels; vBuf `PLen` is explicitly insufficient for this task.

**Existing invariants:** payload remains directly consumable; no repacking occurs in the equivalent-kernel qualification path.

**New invariants:** representation namespace, type and version identify block size, bytes per block, scalar interpretation and endian behavior; stored size is derivable and checked; mixed representations per model are supported; unsupported representations fail before tensor view construction.

**Tests/qualification:** byte-for-byte tensor payload equality after GGUF conversion; expected-size tests at block boundaries; kernel result parity for each initial type; reject partial quantization blocks and unknown required versions.

**Architectural boundaries:** no quantization enum enters `vbuf-core`. Generic `PLen` is not extended with ML meanings.

**Expected commit outcome:** vBuf-ML and GGUF can expose identical packed tensors to identical kernels.

**Dependencies:** Step 9 and Decision D.

#### Step 12 implementation recorded

Added `rust/vbuf-ml/src/representations.rs` and
`docs/vbuf-ml/representations.md`. Profile 0.1 now has a small static
representation contract API, but selects only `CanonicalPrimitive`.

Primitive representation validation is centralized in
`validate_tensor_representation`. It derives semantic, physical, bit width,
count, payload length, and alignment from canonical vBuf descriptors without
adding duplicate ML dtype or storage fields.

Packed/quantized representations were deliberately not selected. Decision D
(namespaced local IDs versus external numeric IDs) and Decision E (first real
model plus pinned GGML/llama.cpp revision) remain insufficiently resolved to
provide authoritative Q4/Q5/Q8 layouts, row rules, endianness, and size
functions. No guessed quantized layouts or fake fixtures were added; unknown
representation IDs fail closed.

The existing profile 0.1 single non-continuing canonical-block tensor policy is
unchanged. No v0.6, generic range, backend, Nano, finalization, or converter
changes were required.

---

### Step 13 — Implement canonical layout and alignment policy

**Class:** ML.

**Goal:** write deterministic files with compact cold sections and correctly aligned contiguous tensor payloads, without assuming page alignment is faster.

**Files/modules likely added:**

- add `docs/vbuf-ml/layout-policy.md`
- add `rust/vbuf-ml/src/{planner,writer}.rs`
- add `rust/vbuf-ml/tests/layout.rs`

**Relevant existing material:** current 4096-byte writer default, GGUF default/configurable alignment, representation-specific kernel requirements, benchmark methodology.

**Existing invariants:** outer vBuf remains canonical; tensor bytes are unchanged; payloads are mmap-friendly.

**New invariants:** the parent vBuf BaseStep is already fixed by the generic file and vBuf-ML cannot override it; vBuf-ML may choose only stricter tensor-payload alignment that is a validated integer multiple of BaseStep; the default is the least valid multiple of BaseStep meeting the maximum selected representation/runtime requirement and is recorded explicitly; 4-KiB/page alignment is an optional payload policy variant, not a BaseStep or format assumption; all padding bytes are deterministic; directory offsets match final bytes; metadata order and tensor physical order are declared.

**Tests/qualification:** sizes and offsets for small/large tensors; reject stricter alignments below or not divisible by BaseStep; padding overhead across representative models; compare only legal 32/64/page payload-alignment variants while holding BaseStep fixed, with separate BaseStep trade-offs remaining in Step 5A; identical logical model and payload digests regardless of physical ordering; no performance claim in this commit.

**Architectural boundaries:** execution-aware order is a writer policy and optional metadata hint, not generic vBuf semantics and not yet the default.

**Expected commit outcome:** reproducible vBuf-ML files suitable for objective layout experiments.

**Dependencies:** Steps 8–12.

#### Step 13 implementation recorded

Added `rust/vbuf-ml/src/layout.rs` and
`docs/vbuf-ml/layout-policy.md`. `LayoutPlan` provides a small downstream
placement planner with deterministic control-first ordering:

```text
Bootstrap → ModelMetadata → TensorDirectory → TokenizerControl
→ TokenizerPayload → TensorPayload → Auxiliary
```

Requests provide a unique deterministic source-order key. Key-ID occurrences
are assigned from the final planned order, preserving the existing semantic
reference model without storing offsets or relocation data.

Payload alignment is validated as a power-of-two multiple of the selected
canonical BaseStep and encoded through the existing v0.6 payload-shift
refinement. `CanonicalPrimitive` defaults to BaseStep alignment. Invalid
alignments fail before writing; BaseStep itself is never redefined. Known-size
and indefinite canonical writers are both supported.

This is a writer policy, not a new decoder requirement. No quantized, page,
SIMD, direct-I/O, GPU, Nano, finalization, transfer-plan, or device alignment
was selected. Padding remains canonical writer geometry and is not conflated
with profile control overhead.

---

### Step 14 — Add optional cold integrity information

**Class:** ML unless Decision F separately approves a generic facility.

**Goal:** detect per-tensor corruption without putting checksum bytes or mandatory verification in hot tensor paths.

**Files/modules likely added:**

- add `docs/vbuf-ml/integrity.md`
- add `rust/vbuf-ml/src/integrity.rs`
- add corruption fixtures/tests

**Relevant existing material:** obsolete CRC claims in README/v0.1–v0.4; absent current implementation; GGUF comparison.

**Existing invariants:** files without integrity information remain valid; normal tensor lookup need not read the integrity section.

**New invariants:** optional SHA-256 payload-only records reference canonical Key-ID plus occurrence; canonical validation remains independent; lazy and eager verification produce the same result; absent integrity remains valid; verification is explicit and target-scoped.

**Tests/qualification:** mutate header, directory, padding and tensor bytes separately; verify coverage rules; measure eager/lazy verification I/O and CPU cost outside inference metrics.

**Architectural boundaries:** do not retrofit profile checksum semantics into generic vBuf. Promotion requires a new domain-neutral ADR and base version plan.

**Expected commit outcome:** optional integrity with measurable cost and no hot-data layout penalty.

**Dependencies:** Step 13 and Decision F.

#### Step 14 implementation recorded

Added `rust/vbuf-ml/src/integrity.rs` and `docs/vbuf-ml/integrity.md`.
Bootstrap role `IntegrityMetadata` (optional role ID `4`) identifies a bounded
20-byte-header/40-byte-entry integrity index. Profile 0.1 selects portable
SHA-256 with fixed 32-byte digests.

Records reference canonical Key-ID plus physical occurrence and cover payload
bytes only. Canonical headers, padding, offsets, and lengths are not duplicated.
`IntegrityMetadata::discover` resolves checked canonical ranges without hashing;
`verify_target` hashes only the requested target. Missing integrity metadata is
valid, missing target coverage returns `NoIntegrityAvailable`, and mismatches
return a distinct semantic error.

No whole-file digest, mandatory verification, checksum registry, signatures,
PKI, Nano, finalization, or backend policy was added. Integrity state remains
runtime-local. SHA-256 construction/verification cost is documented; no
universal cold-I/O claim is made.

---

### Step 15 — Specify and test partial/range loading

**Class:** INFRA transport plus ML section selection.

**Goal:** load bootstrap/directory and selected tensors without reading or mapping the whole file.

**Files/modules likely added/changed:**

- add `rust/vbuf-ml/src/range_plan.rs`
- add `rust/vbuf-layout/src/pread.rs`; add HTTP range support only in a separate optional test/tool crate
- add `docs/vbuf-ml/range-loading.md`
- add sparse-file and local HTTP test harness

**Relevant existing material:** current full-file mmap reader; explicit vBuf-ML sections and tensor offsets.

**Existing invariants:** local mmap remains the simplest zero-copy path; transport concerns do not change wire semantics.

**New invariants:** bounded bootstrap reads discover all further ranges; requested tensor set yields deterministic coalesced ranges; no unrequested tensor payload is read by the range path; returned bytes receive the same validation as mmap bytes.

**Tests/qualification:** instrument exact requested ranges and bytes; truncated server responses; reordered requests; selected layers only; local mmap/pread/HTTP results match; report request-count versus byte-count trade-offs.

**Architectural boundaries:** HTTP is tooling/infrastructure, never a `vbuf-core` requirement. “Partial model execution” belongs to the runtime; the format only enables byte selection.

**Expected commit outcome:** evidence that the directory supports real partial loading rather than only theoretical offsets.

**Dependencies:** Steps 6, 8–13.

#### Step 15 implementation recorded

Added `rust/vbuf-ml/src/range_loading.rs`,
`rust/vbuf-ml/tests/range_loading.rs`, and
`docs/vbuf-ml/range-loading.md`. The runtime-local partial-loading foundation
supports semantic tensor selection by name or ordinal, deterministic physical
range planning, exact/limited-gap coalescing, mmap-backed borrowed access, and
positioned file reads.

Semantic targets retain caller order and exact canonical payload ranges. The
physical plan sorts by file offset and may merge duplicate/overlapping ranges
or explicitly permitted adjacent/gapped ranges. Loaded targets remain exact
slices within physical read buffers. Empty selections produce zero reads and
positioned loading never buffers the whole source.

The canonical reader's current behavior was re-audited: it validates the full
canonical block sequence and structural padding through a borrowed mmap but
does not scan payload contents. This permits selective payload reads after
canonical validation without weakening structural safety.

Integrity remains explicit and target-scoped. No Nano, layer semantics,
portable PhysicalRangeIndex, HTTP, backend API, or wire-format read plan was
added.

---

### Step 16 — Qualify real-model tensor placement

**Class:** ML research qualification.

#### Step 16 implementation recorded

Added the research-only `scripts/qualify_step16.py` tool and generated local
Qwen3-0.6B evidence under `benchmark-results/vbuf-ml-step16/`. The exact
ignored Q8_0 and BF16 GGUF artifacts were hash-verified before parsing.

The tool extracts checked GGUF metadata, tensor names/shapes/types/sizes,
absolute payload ranges and alignment, derives conservative Qwen3 layer/role
groupings, and simulates source, name-order, layer-major, role-order and
prefix-friendly placements without copying or repacking payloads.

The artifacts contain 310 and 311 tensors respectively. The logical inventories
match for 310 names/shapes; BF16 additionally contains `output.weight`. Q8_0
source order is already layer-monotonic. BF16 source order is lexicographic
within block names and materially scatters sequential multi-layer prefixes.

The measured recommendation is `LAYER_MAJOR_ROLE_ORDER` as a vBuf-ML writer
policy. It materially reduces BF16 sequential prefix spans and early
multi-layer exact-read extents while changing Q8_0 only negligibly. The policy
is non-normative; readers remain order-independent and GGUF source order is
retained as the baseline.

No converter, representation contract, LayerIndex wire state, sharding, runtime,
backend, llama.cpp source, or generic-crate change was introduced. Full details
are in `docs/vbuf-ml/step16-real-model-placement.md`.

---

### Step 16b — Define sharding without changing generic vBuf

**Class:** ML.

**Goal:** represent models larger than practical single-file/distribution limits while preserving independent validation and direct tensor ranges.

**Files/modules likely added:**

- add `docs/vbuf-ml/sharding.md`
- add `rust/vbuf-ml/src/shard.rs`
- add `rust/vbuf-ml/src/bin/vbuf-ml-shard.rs`
- add multi-shard fixtures/tests

**Relevant existing material:** 32-bit `DataLen` blocker, range infrastructure, GGUF shard naming/metadata as interoperability evidence.

**Existing invariants:** every shard is independently a valid generic vBuf and identifiable vBuf-ML file; single-file models remain valid.

**New invariants:** stable model identity; shard index/count; deterministic tensor ownership; no tensor silently spans shards in v0.1 unless explicitly justified; required global metadata replication/reference policy; missing/duplicate/wrong-model shards fail deterministically.

**Tests/qualification:** split/reassemble logical directory; missing and swapped shards; duplicate tensor; range loading across shards; same tensor digest and runtime output as single file.

**Architectural boundaries:** generic vBuf has no shard concept. Filesystem naming is a convention, not the sole source of identity.

**Expected commit outcome:** large/distributed models work entirely at the profile layer.

**Dependencies:** Steps 13 and 15.

---

### Step 17 — Pin and qualify F32, BF16, and Q8_0 representation semantics

#### Step 17 implementation recorded

Pinned the external `ggml-org/llama.cpp` repository at commit
`4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c` and recorded source-level
provenance for the GGML type enum, BF16 helpers, Q8_0 block definition/type
traits/size functions, and Qwen3 output fallback.

Added profile-local representation contracts:

```text
0 = CanonicalPrimitive
1 = BF16 opaque bytes
2 = GGML_Q8_0 opaque bytes
```

F32 remains the generic canonical primitive. BF16 uses two explicit bytes per
element and Q8_0 uses 32 elements / 34 bytes per block with innermost-row
32-element divisibility. Both packed contracts validate descriptor geometry and
payload length without scanning tensor contents.

The exact contracts match every one of the 621 tensors in the two hash-verified
Qwen3 artifacts. The pinned Qwen3 consumer creates `output.weight` as optional
and duplicates `token_embd.weight` for the output role when it is absent; this
explains the Q8_0/BF16 inventory difference without adding alias wire state.

No converter, inference path, llama.cpp dependency, sharding, or generic-crate
change was introduced. Details and parity evidence are in
`docs/vbuf-ml/step17-upstream-representation-qualification.md` and
`benchmark-results/vbuf-ml-step17/`.

---

### Step 18 — Implement a read-only GGUF inspector and deterministic conversion manifest

#### Step 18 implementation recorded

Added `scripts/build_step18_manifest.py` and generated deterministic manifests
for both hash-verified Qwen3 artifacts under `benchmark-results/vbuf-ml-step18/`.
The builder reuses the Step-16 GGUF parser and Step-17 representation formulas,
verifies source identity, and emits no target model bytes.

Each manifest accounts for every source tensor, maps only F32/BF16/Q8_0,
assigns stable target Key-ID/occurrence identities, selects
`LAYER_MAJOR_ROLE_ORDER`, and separates source offsets from target semantic
identity. Q8_0 records the pinned llama.cpp output fallback as a local
`SHARED_REFERENCE` note without adding alias wire semantics; BF16 copies its
explicit `output.weight` independently.

Both manifests are structurally complete with zero unaccounted tensors, but
both remain blocked for consumer-complete conversion by missing KV-head/head-
dimension ModelMetadata fields and unsupported GPT-2/BPE tokenizer semantics
(model, pre-tokenizer, merges, and chat template). No converter, generic-crate
change, llama.cpp modification, or payload copy was introduced. Full details
are in `docs/vbuf-ml/step18-conversion-manifest.md`.

---

### Step 19A — Resolve pinned Qwen3 model-metadata gaps

#### Step 19A implementation recorded

Pinned llama.cpp commit `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c` separately
loads `attention.head_count_kv`, `attention.key_length`, and
`attention.value_length`. The latter two override the consumer’s fallback of
`embedding_length / head_count`, so none of the three values is safely
reducible for the pinned Qwen3 model.

Added optional profile-local ModelMetadata keys:

```text
9  = KVHeadCount
10 = KeyHeadDimension
11 = ValueHeadDimension
```

Both real artifacts match exactly at `(8, 128, 128)`. The Step-18 manifests now
report no model-metadata blocker and remain blocked only by tokenizer semantics.
No tokenizer, converter, generic-crate, or representation changes were made.
Evidence and provenance are in `docs/vbuf-ml/step19a-qwen3-metadata.md` and
`benchmark-results/vbuf-ml-step19a/`.

---

### Step 19B — Qualify pinned Qwen3 tokenizer semantics

#### Step 19B implementation recorded

Pinned llama.cpp loads `gpt2` as BPE, maps `qwen2` to its Qwen2
pre-tokenizer, preserves vocabulary ordinals, reads merges in source rank order,
and explicitly applies `tokenizer.ggml.add_bos_token`. The Qwen3 artifacts have
151,936 tokens and 151,387 valid merges; every merge component resolves against
the existing vocabulary ordinal namespace.

Added profile-local tokenizer kind `Gpt2BpeQwen2` with GPT2-BPE and Qwen2
identities, exact ranked merge semantics as two u32 ordinal arrays, `add_bos`,
and optional exact UTF-8 chat-template bytes. `VocabularyOnly` remains valid.
No tokenizer execution, BPE implementation, regex/pre-tokenizer code, or chat
template execution was added.

Both manifests now report, while preserving each artifact's exact chat-template
bytes independently:

```text
raw_inference_tokenizer_readiness = READY
chat_template_readiness = READY_FOR_CONSUMER_EXECUTION
conversion_readiness = READY_FOR_CONVERSION
```

Actual token-ID parity remains a later pinned-consumer adapter qualification;
Step 19B proves lossless representation of all required source semantics.
Evidence is in `docs/vbuf-ml/step19b-qwen3-tokenizer.md` and
`benchmark-results/vbuf-ml-step19b/`.

---

### Step 20 — Implement deterministic manifest-driven GGUF → vBuf-ML writer

#### Step 20 implementation recorded

Implemented `scripts/convert_gguf_to_vbuf_ml.py` and the Rust
`vbuf-ml-convert` writer. The writer consumes the validated Step-18/19B
manifest, verifies source identity and qualified policy, creates a compact
execution plan, and writes through the existing `LayoutPlan` and canonical v0.6
writer.

Both research artifacts were converted and independently reopened through the
canonical and vBuf-ML readers:

```text
Q8_0:  310 / 310 tensor payload digests match
BF16:  311 / 311 tensor payload digests match
```

The targets preserve Q8_0 absence of `output.weight`, preserve BF16's distinct
explicit output tensor, emit `Gpt2BpeQwen2`, and materialize
`LAYER_MAJOR_ROLE_ORDER`. Generated `.vbuf` files remain ignored local research
artifacts. Evidence is under `benchmark-results/vbuf-ml-step20/`; details are in
`docs/vbuf-ml/step20-gguf-conversion.md`.

No llama.cpp consumer or inference claim is made.

---

### Step 21 — Prototype a llama.cpp loader adapter

#### Step 21 implementation recorded

Pinned llama.cpp checkout `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c` was
verified and built with the contained research patch
`patches/llama.cpp/0001-user-metadata-tensor-source.patch`. Added the validated
Rust/C ABI and the separated C++ vBuf loader under `integrations/llama.cpp/`.

The adapter calls the public `llama_model_init_from_user` seam, then uses the
existing llama model, GGML, tokenizer, graph, kernel, and decode paths. Real
BF16 and Q8_0 artifacts pass structural load, metadata parity, tokenizer
parity, exact first-token logits parity, and deterministic greedy generation.
Q8_0 absent `output.weight` uses the existing duplicated embedding fallback;
BF16 explicit output remains independent.

```text
BF16 consumer correctness: PASS
Q8_0 consumer correctness: PASS
GPU/performance qualification: DEFERRED
```

Evidence and provenance are in `docs/vbuf-ml/step21-llama-consumer.md` and
`benchmark-results/vbuf-ml-step21/`.

---

**Class:** ML runtime integration, kept outside generic vBuf.

**Goal:** determine whether vBuf-ML can construct the same runtime tensor objects and use the same kernels without repacking.

**Files/modules likely added:**

- add a pinned integration directory such as `integrations/llama.cpp/`
- add `integrations/llama.cpp/README.md`
- add loader source/patch isolated from `vbuf-core`
- add `scripts/build_llama_vbuf_ml.sh`
- add parity tests

**Relevant existing material:** selected pinned llama.cpp revision, existing GGUF loader, vBuf-ML reader and exact GGML representations.

**Existing invariants:** upstream GGUF path remains available; kernel selection is not modified for one format.

**New invariants:** loader exposes identical tensor names, shapes, types and bytes; unsupported features fail before execution; mapping lifetime outlives all tensor views; no hidden payload copy/repack; loader-only timing is separately observable. Backend/runtime-specific transfer plans, placement maps, prepared descriptors and prepacked execution caches are local derived artifacts, never canonical generic vBuf or mandatory portable vBuf-ML state.

**Tests/qualification:** model configuration parity; tokenizer parity; tensor pointer/range audit; logits within the same tolerance as loading the GGUF source; deterministic prompt output under fixed settings; ASan/UBSan where applicable.

**Architectural boundaries:** llama.cpp adapter depends on vBuf-ML. No llama types or headers enter vBuf or reusable wire-neutral infrastructure.

**Expected commit outcome:** feasibility is demonstrated or a concrete runtime blocker is documented with alternatives and a new user decision.

**Dependencies:** Steps 18 and Decision E.

---

### Step 21 — Add execution-order layout experiments

**Class:** ML writer policy and benchmark only.

**Goal:** test whether physical tensor order changes cold-page behavior or startup for the selected runtime.

**Files/modules likely added/changed:**

- add ordering modes to `rust/vbuf-ml/src/planner.rs`
- add `rust/vbuf-ml-tools/src/bin/vbuf-ml-reorder.rs`
- add `benchmarks/vbuf-ml/layout-order/`
- add a methodology document before results

**Relevant existing material:** canonical layout writer, loader tensor-use trace, existing careful benchmark provenance practices under `benchmark-results/`.

**Existing invariants:** directory order/identity and tensor bytes stay unchanged; all variants run identical kernels.

**New invariants:** baseline source order, name order and traced execution order are explicit variants; the trace workload is disclosed; no execution-aware order becomes default from one model/workload.

**Tests/qualification:** compare file size/padding, page faults, bytes read and cold/warm startup; repeat across enough runs; verify equivalent logits and throughput.

**Architectural boundaries:** no execution semantics enter generic vBuf. If no material benefit is measured, retain deterministic source order and remove/keep the optimizer as experimental tooling.

**Expected commit outcome:** evidence for or against execution-aware ordering, not an assumed optimization.

**Dependencies:** Steps 18–19.

---

### Step 22 — Build the controlled GGUF versus vBuf-ML qualification

#### Step 22A implementation recorded

The frozen Step-21 checkpoint is tagged `vbuf-ml-0.1-consumer-parity` at
`73a1f36661482037573a789ad9`. Added the CPU-only in-process benchmark harness
`integrations/llama.cpp/step22_benchmark.cpp`, fixed prompts, and
`scripts/qualify_step22.py`.

The harness verifies the Step-21 gate, pinned llama.cpp commit, adapter patch,
and all four artifact hashes before running balanced GGUF/vBuf repetitions. It
records model-ready, vocab-only metadata control, prompt tokenization, prompt
evaluation, and greedy generation phases together with faults, RSS/PSS,
virtual size, procfs read bytes, and file size.

Results are under `benchmark-results/vbuf-ml-step22/`. Warm runs used three
samples per format; uncached approximations used two. `POSIX_FADV_DONTNEED`
was available but privileged `drop_caches` was not, so no true cold-cache claim
is made. Steady generation was approximately equal; model-ready, metadata, and
first-touch behavior differed. No optimization was introduced.

See `docs/vbuf-ml/step22-baseline-benchmark.md`.

#### Step 22A implementation recorded

Added optional diagnostic-only Rust, C++, and Python attribution tooling under
`rust/vbuf-ml/src/bin/vbuf-ml-diagnose.rs`, `integrations/llama.cpp/`, and
`scripts/qualify_step22a.py`. The immutable Step-22 evidence is untouched.

Ten warm runs per artifact/format show that canonical vBuf validation is small
(about 0.8 ms), while the current Rust consumer materialization and
GGUF-compatible tokenizer metadata synthesis are each about 132 ms. Numeric
merge reconstruction is about 75 ms and performs approximately 910,600 fine-
grained C ABI calls. Tensor projection and payload attachment are sub-
millisecond/tens-of-microseconds costs.

The measured result rejects canonical parsing as the primary cause and
identifies current adapter representation churn as the main optimization
hypothesis. No optimization or native-source seam was implemented.

Evidence is under `benchmark-results/vbuf-ml-step22a/`; details are in
`docs/vbuf-ml/step22a-loader-overhead-attribution.md`.

---

**Class:** benchmark/tooling; no format changes.

**Goal:** test the efficiency hypothesis with the same model representation, runtime revision, kernel selection and workload.

**Files/modules likely added:**

- add `benchmarks/vbuf-ml/README.md`
- add `benchmarks/vbuf-ml/runner/`
- add `scripts/run_vbuf_ml_qualification.sh`
- add `scripts/validate_vbuf_ml_reports.py`
- store immutable raw reports and a manifest under `benchmark-results/vbuf-ml/`

**Relevant existing material:** `scripts/run_vbuf_baseline.sh`, `scripts/validate_vbuf_baseline.py`, benchmark provenance and scope documentation.

**Existing invariants:** no cross-format claim from unmatched work; correctness precedes timing; raw samples remain available.

**New invariants for equivalence:** same source model; per-tensor names/shapes/types and packed-byte digests match; same llama.cpp commit/compiler/flags/CPU affinity/threading/kernel path; same prompts/context/sampling; conversion excluded; outputs validated; each timed category has an exact boundary.

**Required benchmark dimensions:**

1. cold model startup, with cache-clearing method and privileges disclosed;
2. warm model startup;
3. metadata/bootstrap/directory parsing alone;
4. tensor lookup by name and by directory index;
5. mmap time, major/minor page faults, resident pages and readahead behavior;
6. peak and steady RSS/PSS plus mapped virtual size;
7. bytes actually read, not only file size;
8. selected-tensor/selected-layer partial loading and request count;
9. model-open to runtime-ready time, separating metadata/search, host allocation, validation, repacking, transfer-plan construction, fragmented/coalesced transfer calls, transferred bytes and backend/device upload where a supported backend variant exists;
10. cold TTFT split into model loading/preparation, prompt processing and first-token generation;
11. prompt-processing throughput;
12. token-generation throughput after the model is already resident/ready;
13. file size, BaseStep/payload padding, Nano/checkpoint/region-directory bytes where present, metadata, and other finalized-artifact overhead;
14. optional integrity verification cost reported separately.

Backend-specific CUDA/HIP/Metal/shared-memory variants are optional descendant/runtime experiments, not base-format qualification and not promises of faster kernels. Compare steady-state throughput separately; when identical bytes reach identical kernels, a null container-format effect after readiness is the expected control.

**Qualification design:** separate at least (A) metadata only, (B) loader/view construction, (C) first-touch/cold access, and (D) steady inference. Use balanced order, warmups, multiple independent runs, raw samples, CPU migration checks and a predeclared statistic. Cold-cache methods must be validated rather than assumed. Report negative and null results.

**Tests/qualification:** validator recomputes every summary; kernel/tensor equivalence checked before each benchmark set; detect accidental repack/copy; inspect effective compiler invocation; preserve assembly/perf traces where used; do not pool incomparable categories.

**Architectural boundaries:** benchmark findings may motivate a later ADR but do not retroactively justify generic vBuf changes. Inference parity with no startup win is a valid outcome.

**Expected commit outcome:** a technically defensible answer to whether vBuf-ML is more efficient than GGUF, and in which dimensions.

**Dependencies:** Steps 18–20; optional integrity/sharding variants are separate experiment arms.

---

### Step 23 — Source-neutral llama construction seam and direct vBuf source

**Class:** architectural runtime integration.

Added `patches/llama.cpp/0002-source-neutral-model-source.patch` on top of the
frozen Step-21 patch. The pinned llama construction pipeline now accepts a
semantic `llama_model_source` without constructing a `gguf_context` for the
vBuf direct path. GGUF native loading and the Step-21 compatibility adapter
remain intact.

The direct source uses bulk immutable tokenizer, numeric merge, and tensor
views, retains validated mmap ownership through model destruction, preserves
Q8_0 tied-output fallback and BF16 explicit output, and attaches payloads
without copy/repack/reorder.

Three-way BF16/Q8_0 qualification passed metadata/tokenizer parity, zero logit
delta, and deterministic generation parity. Ten warm samples show:

```text
                 GGUF       old compatibility       direct
BF16             249.3 ms   414.1 ms                 247.1 ms
Q8_0             196.0 ms   416.3 ms                 247.0 ms
```

The old approximately 910k ABI calls reduce to 12 coarse calls in the direct
path. Evidence is under `benchmark-results/vbuf-ml-step23/`; full design and
gap report: `docs/vbuf-ml/step23-native-llama-source.md`.

No wire-format, tensor-layout, kernel, scheduler, GPU, prefetch, or inference
optimization was introduced.

#### Step 23A — Nano-backed native-runtime audit

Added a read-only audit of the current benchmark-only Nano reconstruction and
existing Qwen3 vBuf-ML geometry. No Nano wire artifact or native runtime was
implemented.

The qualified files use `BaseStep=8`, with 338/337 canonical blocks but a
hypothetical Nano size of 23.6 MB/10.0 MB. Dense tokenizer arrays are already
direct-viewable through text offsets, scores/types, and numeric merge IDs;
Nano is classified redundant for those structures and for individual tensor
lookup. Nano remains a candidate for future unknown/mixed physical bootstrap,
continuation enumeration, layer readiness, and residency/prefetch planning.

Release-mode audit evidence is under
`benchmark-results/vbuf-ml-nano-runtime-audit/`; the detailed report is
`docs/vbuf-ml/nano-runtime-audit.md`.

#### Step 24 — Borrowed runtime views

Replaced the Step-23 direct path's owned `ConsumerModel` snapshot with a
mmap-owned, validated `BorrowedModel`/`BorrowedModelView` layer. Token text,
offsets, types, optional scores, and numeric merge arrays now cross the ABI as
canonical pointer/count spans; no Rust token or merge object tables are built
by the direct source. The source-neutral `llama_model_source` seam and common
llama/GGML runtime are unchanged.

Required canonical validation remains eager and explicit. Runtime token and
merge indexes remain runtime-local and owned by the final tokenizer. Nano is
not used in dense lookup or normal Step-24 loading; a separate experiment
records canonical-descriptor, Nano-plus-minimal-header, and semantic-directory
physical-index construction costs.

Warm CPU medians over ten samples:

```text
                 GGUF      compatibility  Step-24 direct
BF16             249.9 ms  430.5 ms       152.0 ms
Q8_0             196.0 ms  430.7 ms       149.7 ms
```

BF16/Q8_0 tokenizer, structural, logit, and deterministic-generation parity
passed with zero logit difference. Payload copy/repack/reorder remain zero.
Evidence is under `benchmark-results/vbuf-ml-step24/`; detailed architecture
and limitations are documented in
`docs/vbuf-ml/step24-borrowed-runtime-views.md`.

No v0.6 wire, vBuf-ML semantic, tensor-layout, BaseStep, Nano, kernel,
scheduler, GPU, prefetch, or inference optimization was introduced.

#### Step 25 — Runtime-local tokenizer indexes

Frozen the Step-24 direct-view state with the annotated immutable tag
`vbuf-ml-0.1-direct-view-baseline` targeting `b569f26`. Added reusable
`TokenIndex<'a>`, packed `MergeRankIndex`, and `RuntimeTokenizerIndexes` in
`vbuf-ml`. Token keys borrow validated canonical bytes; merge keys are packed
only after u32 bounds validation. The indexes are runtime-local and are not
persisted or added to the wire format.

Thirty-sample release microbenchmarks selected:

```text
borrowed token map:
BF16 ~7.6 ms, Q8_0 ~7.5 ms

packed merge map:
BF16 ~4.4 ms, Q8_0 ~4.4 ms
```

The reusable C ABI supports lazy index construction and token/merge lookup for
current and future runtimes. The default llama source does not build duplicate
Rust indexes because llama still owns its final vocabulary maps. Existing
Step-24 model-ready behavior and correctness remain the reference; new direct
controls were recorded separately under
`benchmark-results/vbuf-ml-step25-runtime-index/`.

No Nano, LayerView, RuntimeChunk, prefetch, residency, GPU streaming, or other
physical-runtime work was introduced. See
`docs/vbuf-ml/step25-runtime-tokenizer-index.md`.

#### Step 26 — Qwen3-32B placement qualification

Qualified the new dense `Qwen3-32B-Q8_0.gguf` source: 34,817,718,912 bytes,
707 tensors, 64 layers, 151,936 tokens, and 151,387 merges. Existing Qwen3
semantic parsing passed without widening the profile. A deterministic planner
simulated all legal BaseShift values 3..8 from one logical manifest; no payload
copies or six model files were created.

The artifact-qualified selection is:

```text
BaseShift 3
BaseStep 8
```

It minimizes final size and padding while preserving one physical span per
layer. Hypothetical Nano accounting was recorded but Nano remains noncanonical
and unused.

The final artifact is emitted and qualified:

```text
research-models/Qwen3-32B-Q8_0.vbuf
size: 34,816,197,376 bytes
SHA-256: 84597064d5b3530572959345286368b17e891e640b5958bba7f0b67980cd119d
```

The corrected planner includes the two tokenizer identity payloads emitted by
the writer; selected BaseShift/BaseStep remain unchanged and planned versus
actual size and tensor offsets match exactly. Tensor payload parity passed
707/707. Checked-reader, BorrowedModelView, llama MODEL_READY, zero logit
parity, and deterministic generation parity passed. The historical 25.5 GB
disk-capacity blocker and its evidence remain preserved in
`benchmark-results/vbuf-ml-step26-qwen32b-placement/`.

See `docs/vbuf-ml/step26-qwen32b-placement.md`.

#### Step 27 — Controlled loader scaling and first-use attribution

Ran a fresh CPU-only, pinned-llama benchmark over 0.6B and 32B GGUF/vBuf
controls with 10 warm and 3 uncached-approximation samples per case. The
benchmark measured MODEL_READY, first evaluation, TTFUC, TTFT, second
synchronous evaluation, page faults, I/O counters, RSS, and CPU time.

The 32B warm medians were:

```text
             MODEL_READY   FIRST_EVAL   TTFUC
GGUF           20.003 s       1.754 s  21.608 s
vBuf            0.262 s      35.840 s  36.145 s
```

The vBuf readiness advantage is repeatable, but most large-model payload
residency work moves into first evaluation. vBuf RSS was approximately 84 MB
at MODEL_READY and 33.3 GB after first evaluation; GGUF was approximately
33.9 GB at MODEL_READY. Second evaluation was effectively equal (1.071 s vs
1.074 s). Classification: a mixture of genuine readiness/construction
reduction and deferred first-touch work. No optimization was implemented.

See `docs/vbuf-ml/step27-loader-scaling.md` and
`benchmark-results/vbuf-ml-step27-loader-scaling/`.

#### Step 28 — Layer-span prefetch and wave-readiness qualification

Derived a noncanonical runtime-local plan from validated tensor physical
ranges. The Qwen3-32B artifact has 64 layer spans and 3 global spans, with
8,704 gap bytes and 1.0000002500 coverage amplification.

Qualified fault-driven, whole-payload prefault, ordered layer-span prefault,
and `MADV_WILLNEED` variants on 0.6B and 32B vBuf artifacts. In the uncached
approximation, 32B TTFUC was approximately 36.0 s for the baseline, 27.0 s
for either explicit prefault strategy, and 36.1 s for advisory prefetch.
Explicit prefault moved approximately 42k major faults from FIRST_EVAL into
PREPARATION; advisory prefetch did not. Sequential and layer-order preparation
were effectively equivalent because the existing layout is already layer-major.

Preparation and exact layer distributions were measured, but no per-layer
compute timing or actual overlap was available. Track-B wave processing is
therefore **NO/NOT YET**. No runtime, wire, Nano, Nested-vBuf, graph, or kernel
architecture was changed.

See `docs/vbuf-ml/step28-layer-prefetch.md` and
`benchmark-results/vbuf-ml-step28-layer-prefetch/`.

#### Step 29 — Layer compute boundaries and bulk I/O preparation

Used the existing pinned llama/GGML scheduler evaluation callback to observe
`embd`, `l_out-0..63`, `result_norm`, and `result_output` boundaries without
changing graph semantics. Ten resident 32B samples measured approximately
1.061 s median aggregate transformer-layer compute and 27 ms non-layer traced
overhead.

The best uncached bounded bulk-read result on this host was synchronous
`pread`, 16 MiB chunks, QD4: approximately 1.409 GB/s. The page-touch control
was approximately 1.232 GB/s. Median layer preparation was approximately 359
ms versus approximately 16.15 ms resident layer compute; the host-local
prepare/compute ratio was approximately 22.2× and modeled bounded lookahead
still stalled.

This is a host-relative result, not a vBuf bandwidth or universal Wave claim.
Bulk I/O qualification: **YES** for this host. Wave Track-B: **NO / NOT YET**.
The artifact, physical spans, wire format, BaseShift, and runtime architecture
were unchanged.

See `docs/vbuf-ml/step29-layer-io.md` and
`benchmark-results/vbuf-ml-step29-layer-io/`.

#### Step 30 — Reconstructable weight reparameterization qualification

**Status: PARTIAL — 22/22 tensor families tested; hidden-state and deeper group/layer phases remain.**
The previous completion claim is invalid; Step 30 is reopened and Step 31 has
not started.

Tested bounded research-only reparameterizations on selected real Qwen3 Q8_0
tensors: plain low-rank factors, row-centered factors, row/column scaling with
a low-rank core, and column permutations with a low-rank core. All auxiliary
parameters were counted.

For the selected 32B attention-output tensor, rank-16 factors used roughly
0.163 bits per original weight but retained only approximately 2.3% of the
random-probe action signal (approximately 97.7% relative action error). Scaling
and permutation variants did not materially improve this result. The Q8_0
reference remained exact at 8.5 bits/weight.

This is a lossy research result, not a model-compression qualification. No
real hidden-state hook was available, no candidate was emitted, and no wire,
BaseShift, runtime, Nano, or Nested-vBuf changes were made.

See `docs/vbuf-ml/step30-reparameterization.md` and
`benchmark-results/vbuf-ml-step30-reparameterization/`.

The corrected matrix contains 22 families × 7 boundaries, and all 22 families
now have qualifying real-tensor coarse-screen rows. The continuation adds
bounded MPO, Butterfly, generalized Butterfly, genuine displacement-rank, and
Householder candidates plus hierarchical, sign/magnitude/residual, and vector
prototype/residual quantization. Deeper group/layer phases remain gated by
missing hidden-state evidence and poor low-bit action preservation. See
`docs/vbuf-ml/step30-structured-matrix.md` and
`docs/vbuf-ml/step30-continuation.md`.

The bounded 0.6B group pilots found rank-16 joint low-rank **raw factor-array**
byte reductions of 1.40x for QKV and 1.14x for gate/up, with approximately
0.955 and 0.975 random-probe action error. These are not serialized storage
gains. Classification: **F — inconclusive**. No whole-layer candidate was
qualified.

The alignment-slack audit found only 43 bytes in the 0.6B artifact and 44
bytes in the 32B artifact, all in small control regions; every tensor and layer
has zero local slack. Canonical padding must remain zero, so usable correction
capacity is zero. Decision: **E — no meaningful opportunity**. A separate paid
scalar-budget sweep found 2+2 best at four bits, 3+3 at six bits, and 3+5 best
at eight bits. See `docs/vbuf-ml/step30-slack-audit.md`.

A bounded Qwen3-32B CCC Stage-1 assessment sampled 49 real Q8_0 tensors across
layers 0/1/16/32/48/62/63 and seven major roles, producing 3,822 candidate
rows with disjoint 70/15/15 position-hash splits. Learned additive correction
levels beat simplified uniform/affine controls at 2–6 bpw, but zero and
mean anchors tied, row/column context generally hurt, and lane-mod32 context
helped only C2 modestly. Canonical GGML comparators, native direct apply, and
real hidden states remain blockers, so the assessment is **PARTIAL**. See
`benchmark-results/vbuf-ml-step30-ccc-assessment/representation-assessment.md`.

CCC Stage-2 used pinned canonical GGML quantizers and 128 real F32 activation
vectors captured at the exact layer-32 key-projection input. Learned C2 remains
a low-rate Pareto niche but has much worse real W*x error than activation-aware
IQ2 controls at slightly higher rates; learned C3/C4 are dominated. A fixed
C4+0.1% tail remains Pareto. Bounded power alphabets stayed within 1.25x of
learned C3/C4 but those operating points are dominated. Classification at that
historical stage: **CCC_LOW_BIT_NICHE_INTERESTING**, not ready for a native-kernel
gate. See
`benchmark-results/vbuf-ml-step31-ccc-canonical/stage2-report.md`.

#### Final CCC research closure

**Status: CLOSED / REJECTED.** Four independently developed lineages were
preserved with real no-ff merges and reconciled without rewriting their
historical findings. Strong canonical and free-codebook controls reject the
contextual baseline, geometric-alphabet advantage, C3 numerical point, plain C4
point, and claimed C3 runtime advantage. Packed sub-byte direct compute remains
a positive systems result independent of CCC. The Stage-2 fixed 0.1% sparse C4
residual tail remains unreplicated evidence only and is not implementation or
CCC-continuation authorization.

Canonical final classification: **CCC_DIRECTION_REJECTED** and
**CCC_RESEARCH_REMAINS_CLOSED**. See
`docs/vbuf-ml/ccc_research_conclusion.md`. Latest reconciliation before closure:
`59673a1`.

---

### Step 23 — Qualification review and format-freeze decision

**Class:** documentation/release gate.

**Goal:** freeze vBuf-ML 0.1 only if correctness, safety, interoperability and measured behavior justify the current design.

**Files/modules likely changed:**

- promote or revise `docs/vbuf-ml/spec-0.1-draft.md`
- add `docs/vbuf-ml/qualification-0.1.md`
- add `docs/vbuf-ml/compatibility.md`
- update crate versions and changelogs only after approval

**Relevant existing material:** all conformance and benchmark evidence, unresolved issues from converter/loader work.

**Existing invariants:** generic vBuf versioning remains independent of profile versioning.

**New invariants:** frozen fields have compatibility rules; experimental sections/representations are marked; unsupported models are not implied supported; benchmark conclusions name their scope.

**Tests/qualification:** clean checkout reproduction; all fixture digests and validators; independent reader review if possible; security/fuzz report; conversion and llama.cpp parity; benchmark rerun.

**Architectural boundaries:** a profile success does not automatically promote its tensor directory, metadata codec, string tables or integrity policy into base vBuf. Promotion requires a separate proposal with non-ML evidence and migration cost.

**Expected commit outcome:** either a releasable vBuf-ML 0.1 or a documented no-go/redesign decision without destabilizing generic vBuf.

**Dependencies:** all required earlier steps.

---

### Step 31 — Progressive local source qualification

**Class:** documentation/research gate first; source-layer implementation only
after the gates below pass.

**Goal:** determine whether the qualified vBuf-ML Android payload path can use a
same-offset sparse canonical mirror for progressive local reuse, without changing
the v0.6 wire contract, semantic bootstrap, tensor representation, materializer,
lease, residency, scheduler, backend, or inference behavior.

**Current evidence:** the semantic bootstrap is separate from the external
payload, and the qualified Android path carries `TensorRef.source_offset`
directly into the range source. This supports the sparse-mirror hypothesis for
one immutable external payload artifact. It does not establish the design for
arbitrary multi-source models, full offline completeness, or power-loss
durability.

**Non-negotiable ownership:** source identity, source resolution, range coverage,
remote fallback, persistence, materialization inputs, retention, and load
planning remain vBuf-ML runtime/source concerns. No llama.cpp loader or backend
becomes the owner of this policy.

#### Step 31A — Canonical-offset and source-authority audit

- Inventory source identity, declared size, locator, profile/version, hash scope,
  and source bindings for the target model.
- Prove that every target external TensorRef addresses one immutable random-access
  payload artifact directly, with no hidden offset rebasing.
- Record the relationship between semantic bootstrap, payload artifact, v0.6
  headers/control regions, and TensorRef payload ranges.
- Determine whether a local same-offset payload mirror can be consumed by the
  existing semantic bootstrap and native source boundary unchanged.
- Capture requested intervals, bytewise union, repeated overlap, reuse distance,
  requested bytes, remote bytes, and overfetch under the unchanged Android
  configuration.

**Gate:** if canonical offset equality or immutable source identity cannot be
proven, stop the sparse-mirror path and return to a qualified block or
runtime-local envelope design. Do not guess a rebase rule.

**Step 31A result:** **CONDITIONAL PASS for the qualified single-source path.**
The Android audit completed with batched prefill, normal inference, four
generated tokens, `4,287` HTTP range requests, and `4,745,501,920` returned
bytes. The bytewise union was `1,377,067,264` bytes; requested bytes exceeded
that union by `3,368,434,656` bytes through repeated/reloaded ranges. Direct
`PersistentTensorRef.source_offset` to `RangeSource::read_range()` preserves
the canonical payload offset with no rebasing. The detailed evidence is in
`docs/vbuf-ml/step31a-canonical-offset-source-authority-audit.md`.

The result does not authorize persistence. `SourceId(1)` alone is not an
immutable cross-session artifact identity, and the trace covers only the
observed prompt and four-token execution rather than complete model coverage.

#### Step 31B — Persistent representation decision

**Decision:** select **A, a sparse canonical same-offset mirror**, with a
minimal authoritative fixed-chunk coverage map and an immutable artifact
identity binding. The semantic bootstrap remains separate. The detailed
decision record is `docs/vbuf-ml/step31b-persistent-source-representation-decision.md`.

The persistent artifact identity for the qualified source is:

```text
(declared_size, SourceHash.algorithm=1, full 32-byte SHA-256 SourceHash.value)
```

`SourceId(1)` and the source-profile binding select the descriptor but are not
identity by themselves; locator, endpoint, path, tensor identity, and RAM
residency are not identity. The existing semantic-bootstrap source profile
already carries the declared payload size and full-source SHA-256. A thin
future source-boundary seam must expose that existing identity to persistence;
no second hashing subsystem or TensorRef redesign is authorized.

The authoritative coverage unit is a 4 KiB logical source chunk. The
`5,639,819,878`-byte payload requires `1,376,910` chunks and a one-bit bitmap of
`172,114` bytes. Rounding the measured Step 31A ranges to touched chunks would
add a derived `17,924,896` bytes of acquisition, far below the measured
`3,368,434,656` repeated bytes. This is a runtime source-cache granularity,
not a v0.6 format alignment rule.

Only `MISSING` and `VALID` are externally observable. A transient `ACQUIRING`
state is in-memory only. A chunk becomes valid only after exact remote framing,
declared-size containment, a complete canonical write, and coverage-bit
publication. Sparse holes never authorize a hit. The first partial-coverage
strategy fetches the complete touched chunks rather than splitting local and
remote subranges.

The current source profile provides an expected whole-artifact SHA-256, but the
range path has no per-range digest and does not verify a whole-artifact hash
while serving individual ranges. Step 31C may therefore claim structural
validation and source-identity binding under a trusted immutable source
contract, not adversarial remote authenticity. Whole-artifact verification and
power-loss durability remain deferred.

The alternatives were compared as follows:

1. sparse canonical mirror plus minimal explicit coverage;
2. canonical block persistence with sufficient stream/source context;
3. runtime-local envelope/record persistence.

Canonical block persistence is rejected because external TensorRef payload
ranges are already consumed at canonical offsets, while isolated v0.6 blocks
require stream context and would introduce a second persistence representation.
Runtime-local envelopes are rejected because they duplicate payload/model
semantics and lose direct byte-for-byte artifact equivalence. The sparse mirror
has the smallest new state and reuses the existing local/remote RangeSource
boundary.

Step 31B is **DECIDED**. It does not implement any of the selected mechanism.

#### Step 31C — Minimal source-layer mechanism

**Status:** **IMPLEMENTED and HOST-QUALIFIED; physical Android qualification
PARTIAL.** The generic C++ `ProgressiveRangeSource` now composes an
existing remote `RangeSource` with a same-offset sparse payload file and an
identity-bound `.coverage` sidecar. The focused contract test covers cold and
warm reads, final-EOF chunking, partial coverage, reopen, identity mismatch,
remote/truncated failures, and concurrent same-chunk acquisition. The result
report is `research/results/vbuf-android-demo-poc/step31c-progressive-local-source.md`.

The Android direct session obtains the existing profile's declared size and
full SHA-256 through the source-profile C ABI seam and uses a metadata-derived
payload mirror path. A Pixel 7 Pro was subsequently attached and a physical
cold-acquisition smoke run entered batched prefill, but full cold/warm and
generation qualification remains incomplete because serialized 4 KiB
acquisition is request-heavy on the local test transport. The partial result
and the nginx keep-alive boundary diagnosis are recorded in the Step 31C
report.

The bounded implementation follows this contract:

- add local exact-range lookup and explicit coverage lookup at the source layer;
- on an uncovered request, fetch the complete touched 4 KiB canonical chunks
  and use the requested bytes for the current request;
- validate before publishing bytes and coverage at the canonical offset;
- support fully local, fully remote, and qualified partial-coverage reads;
- keep stream-only as a persistence-bypass policy;
- keep persistent source coverage separate from `TensorResidencyStore` RAM
  accounting and leases;
- make the coverage accelerator rebuildable and non-authoritative.

No semantic execution or materialization contract changes are permitted in this
step. A request touching any missing 4 KiB chunk fetches the complete touched
chunk set; local/missing subrange splitting is deferred.

The smallest 31C acceptance slice is one qualified source identity, a same-size
sparse payload mirror, a 4 KiB authoritative bitmap, local containment checks,
remote fallback, canonical writes followed by bit publication, and a cold-then-
warm source qualification. It includes no eviction, offline completion,
prefetch, multi-source policy, residency change, backend change, or inference
semantic change.

#### Step 31D — Bounded demand-driven missing-chunk acquisition

**Status:** **IMPLEMENTED and HOST/ANDROID-QUALIFIED.**
Physical Android source smoke showed that 4 KiB coverage publication is correct
but that acquiring every missing 4 KiB chunk as a separate upstream HTTP
request causes request amplification. This does not invalidate the 4 KiB
authoritative bitmap.

The implementation preserves 4 KiB validity/publication units, detects
contiguous missing chunks within the current consumer request, and splits runs
at a measured maximum `1 MiB` window. It writes canonical same-offset bytes,
publishes only complete individual 4 KiB bits, avoids future-range prefetch,
reduces request count, and returns exactly the original consumer slice. The
offline 4,287-range replay estimates 1,850 upstream requests versus 336,990
one-chunk requests, with `0.236%` chunk-boundary overhead over the bytewise
union. The implementation remains below the `RangeSource` consumer contract;
TensorRef, materialization, residency, GGML, and inference semantics remain
unaware. See `research/results/vbuf-android-demo-poc/step31d-bounded-acquisition.md`.

The Pixel 7 Pro cold control completed with 1,850 upstream windows and
`1,380,311,040` remote bytes. The identical warm run completed after process
restart with zero upstream requests and zero remote bytes. Both runs preserved
the 4 KiB coverage representation, four-token output, and the
`268,435,456`-byte residency cap. Cold/warm prefill was `179,800`/`67,329` ms;
decode was `109,632`/`102,465` ms. These are inclusive existing timers, not
exclusive source-latency measurements.

#### Step 31E — Physical residency curve qualification

**Status:** **MEASUREMENT COMPLETE FOR THE VALID DEVICE RANGE; LARGER POINTS
PHYSICALLY CONSTRAINED.** The focused report is
`research/results/vbuf-android-demo-poc/step31e-residency-curve.md`.

The experiment varied only the active `TensorResidencyStore` budget at exact
`268,435,456`, `536,870,912`, `1,073,741,824`, and `2,147,483,648` byte caps.
The sparse payload mirror, authoritative 4 KiB coverage, 1 MiB acquisition
window, source identity, materializer, backend, thread/runtime configuration,
prompt, batched prefill, and four-token workload remained fixed. Each point
used a fresh app process while retaining the mirror and coverage sidecar.

The 256 MiB point completed with zero remote bytes, `69,699` ms prefill,
`25,279` ms mean decode, `134,436` ms FFN, `4,217` evictions, and `3,025`
reacquisitions. The 512 MiB, 1 GiB, and 2 GiB points also remained fully local
and reduced partial-run evictions/reacquisitions, but each failed at the first
decode before producing a comparable token result. No working-set threshold or
successful larger-cap speedup is claimed. The result is device-constrained and
inconclusive for A-versus-B attribution; no production residency default is
changed.

#### Step 31F — Crash, corruption, and recovery qualification

**First-decode failure attribution completed diagnostically; root cause
unresolved.** The fixed 512 MiB Android workload remained fully local, reached
the first decode after `67,419` ms batched prefill, and failed in `block 7` at
`expert_down_matmul` while constructing `blk.1.ffn_down_shexp.weight`. The
reported rank-2 view had dimensions
`12970367413557264240,0` and `1,486,848` payload bytes. The source/materializer
and Android boundary now preserve operation, tensor, geometry, and invariant
details. This is descriptor-corruption/lifetime evidence, not a proven OOM or
backend allocation failure; no workaround or runtime behavior change was made.

Qualify interrupted writes, process death during tee, concurrent same-range
requests, sparse holes, corruption, source mismatch, disk full, deletion,
coverage-index loss, and model replacement. Define separately whether process
crash publication and power-loss durability are required before reporting a
range as retained. Readers must see uncovered, previously valid, or newly valid
bytes, never partially published coverage.

#### Step 31G — Extended cold/warm attribution

**First-invalid descriptor boundary qualified.** The canonical 512 MiB trace
shows valid `executor_persistent` geometry followed by malformed geometry at
`resident_ready`; the payload remains valid and the malformed view propagates
through `executor_ready` and `pre_ggml`. The result proves a shape-owner
lifetime mismatch/dangling borrowed descriptor view exposed by the residency
cache hit; the exact construction or reuse event that invalidated the shape
storage remains unresolved. No runtime repair was made. Evidence:
`research/results/vbuf-android-demo-poc/step31g-descriptor-provenance.md`.

The existing 256 MiB successful qualification remains the comparison control
from Step 31E. A fresh control rerun requires restoring the local model artifact
and range server; no new 256 MiB result is claimed by Step 31G.

#### Step 31H — Own cached materialized tensor geometry

**Host-qualified ownership repair completed; exact inputs restored, physical confirmation is blocked by ADB.**
`MaterializedTensor` now owns its rank and inline geometry while preserving the
existing leased payload pointer and residency behavior. The residency contract
verifies geometry after source-scope destruction, cache-hit retrieval, copy,
move, and vector relocation. Host materializer, residency, striped,
tensor-wave, adapter, source-fallback, Rust, native Release CTest, neutrality,
and Android ARM64/`assembleDebug` checks pass. The exact IQ2_XXS source and
regenerated payload are restored with an exact payload identity match and the
range server returns verified 206 responses. The Pixel ADB endpoint is refusing
connections, so no physical 512 MiB result is claimed.
Evidence: `research/results/vbuf-android-demo-poc/step31h-owned-materialized-tensor-geometry.md`.

Do not resume the residency curve in Step 31H.

#### Step 31I — Retention and offline completion

- add stream-only, cache-as-used, bounded retention, and keep-model policy above
  the source mechanism;
- choose storage budgets from measured reuse and coverage simulation, not the
  256 MiB RAM setting;
- use the same mirror for missing-range acquisition and offline completion;
- distinguish encountered TensorRef coverage from complete declared source
  coverage, including headers, padding/control regions, and never-routed MoE
  ranges;
- report exact incompleteness rather than implying offline readiness.

#### Step 31J — Deferred experiments

Keep idle warmup, next-use prefetch, compute/prefetch overlap, advanced
replacement, model-aware policy tuning, mobile-data UX, residency tuning,
backend tuning, and Android-specific inference changes independent of this
qualification. Revisit them only after the source mechanism and cold/warm
attribution are correct.

**Expected outcome:** a measured sparse-mirror qualification, a measured
alternative selection, or a documented no-go. None changes the generic vBuf
format or transfers runtime ownership to llama.cpp.

---

## 5. Runtime-derived requirement checklist

The following questions must have concrete answers before the corresponding profile field is frozen:

### Tensor construction

- How does the runtime identify each required tensor?
- What rank/dimensions and dimension ordering does it require?
- Which packed representation and block constraints does the selected kernel require?
- Does the runtime require mutable storage, repacking or device upload? Those costs must not be attributed to the container without evidence.
- Can a tensor view point directly into a mapped/range buffer for its full lifetime?

### Model construction

- Which architecture values select graph topology and dimensions?
- Which values are derivable from tensor shapes, and is derivation unambiguous? Store rather than derive only when required for correctness or compatibility.
- Which metadata is descriptive only and can be skipped?
- Which optional features change tensor interpretation or graph construction?

### Tokenization

- Is an embedded tokenizer required for self-contained deployment?
- Which tokenizer arrays and special IDs are actually consumed?
- Can a tensor-only application omit loading tokenizer sections?

### Loading and distribution

- What is the smallest bootstrap range needed to locate a named tensor?
- Can requested ranges be coalesced without loading unrelated payloads?
- What model/shard identity prevents accidental shard mixing?
- Which integrity level is required: accidental-corruption detection, cryptographic identity, or none?

A field with no consuming operation, validation purpose, compatibility purpose or measured optimization should not enter the first frozen profile.

---

## 6. Explicit non-goals for the first qualification

- Designing a universal ML framework or graph serialization format.
- Storing runtime execution graphs unless the selected runtime proves tensors plus metadata insufficient.
- Inventing a new quantizer or kernel.
- Repacking tensors to claim container speed.
- Making all metadata fixed-width; only hot lookup tables need that hypothesis tested.
- Requiring 4-KiB alignment for every tensor.
- Claiming CRC repairs data.
- Adding HTTP, sharding, tokenizer or tensor concepts to generic vBuf.
- Claiming faster token generation when both formats feed identical bytes to identical kernels; a null result is expected and informative.
- Preserving unsafe arbitrary native-struct zero-copy as a portable format guarantee.

---

## 7. Definition of success

The plan succeeds architecturally if:

1. generic vBuf remains independently specified, tested and useful, retaining only independently justified bounded-region/navigation primitives; optional Nano, checkpoints, directory, and finalization envelope may be absent if qualification rejects them;
2. a generic reader can safely locate/navigate/skip regions in a vBuf-ML file without assigning ML meaning to them;
3. vBuf-ML can locate a named tensor from bounded cold metadata reads;
4. mapped compatible tensors reach the same kernel without copy or repack;
5. model configuration, tokenization and outputs match the selected GGUF path;
6. large-file, malformed-file, extension and shard behavior is explicit and tested;
7. benchmark categories isolate parsing, mapping, page faults and inference;
8. results can show a win, loss or no difference without changing methodology;
9. no vBuf-ML requirement has leaked upward into generic vBuf without an independently approved general-purpose rationale.

The performance hypothesis succeeds only for dimensions where controlled measurements show a reproducible material improvement. Architectural success does not require outperforming GGUF.
