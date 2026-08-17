# vBuf (Vector-Buffer) ⚡

**Current milestone:** [`model-is-working`](https://github.com/eugen252009/vBuf/tree/model-is-working)

vBuf is a generic binary block format for checked, mmap-friendly, direct native consumption.

## vBuf-ML research overview

The vBuf-ML work investigates a different way to run models whose logical weight
set is larger than the memory that should be active at one time. The artifact
keeps persistent tensor identity, representation, and exact byte ranges, while
the runtime decides what execution currently needs and when those bytes should
be materialized. Demand-driven materialization matters because routing, graph
dependencies, and the current execution wave can exclude most of a model from
the immediate work; it is not necessary to read, transform, or execute an
unselected tensor merely because it is reachable in the full model.

The design therefore treats logical model size, resident cache, and active
execution memory as different quantities. Its working relationship is
`TOTAL MODEL >> RESIDENT CACHE >> ACTIVE EXECUTION WAVE`, but the ratio is a
runtime model, not a universal memory promise. ggml currently supplies useful
tensor operations, graph execution, scheduling, and backend kernels; it is a
compute substrate rather than the owner of vBuf model identity or lifecycle.
The pinned llama.cpp path is primarily a semantic/reference oracle and
compatibility consumer. The final runtime boundaries are being promoted from
measured POCs rather than assumed from an existing loader's ownership model.

### Runtime lifecycle at a glance

```text
Persistent vBuf artifact
        |
        v
PersistentTensorRef
        |
  execution demand
        |
        v
      Source
        |
        v
  Materialization
        |
        v
    Residency  <---- cache budget / replacement policy
        |
        v
 ExecutionLease ---- execution lifetime
        |
        v
      Compute

RuntimeState and TensorValue cross execution boundaries;
Placement is selected at runtime rather than persisted as model identity.
```

The separation between `PersistentTensorRef`, `Source`, `Residency`,
`ExecutionLease`, `RuntimeState`, `TensorValue`, and `Placement` is deliberate:
the entire model need not be loaded before execution, logically reachable
tensors need not all be resident, a residency lifetime need not equal an
execution lifetime, and one model tensor need not have one permanent physical
placement.

## Current milestone / status

The current qualified native path is a functional model-forward pipeline for
the real `DeepSeek-V2-Lite.IQ1_S.vbuf` artifact on x86:

```text
real token IDs
  -> quantized token embedding
  -> blk.0 dense transformer block
  -> blk.1..blk.26 sparse MoE transformer blocks
  -> final RMSNorm
  -> quantized LM head/output projection
  -> logits
  -> greedy next token
```

POC21 reports exact parity for token IDs 0 and 1 at the embedding, all 27
transformer blocks, final normalization/output logits, and greedy token `86711`
for both tested inputs. Activation handoff is zero-copy. This is a functional
native model-forward qualification, not a claim of production chat inference;
tokenizer, sampler, chat-serving, and deployment completeness remain separate
concerns. See the [`POC21 report`](research/results/vbuf-functional-pipeline-poc21-x86/report.md)
for the qualified result and scope guards.

## What we learned

The table summarizes measured findings, with scope kept next to each number.

| Finding | Evidence | Architectural consequence | Details |
|---|---|---|---|
| Structural readiness can avoid whole-payload work | The fresh 32B control measured vBuf `MODEL_READY` at about `0.262 s` versus GGUF at about `20.003 s`; vBuf first useful compute was about `36.145 s` | Opening/navigation and first-use materialization must be measured separately | [`Step 27 loader scaling`](docs/vbuf-ml/step27-loader-scaling.md) |
| The readiness delta is not storage bandwidth | The same qualification reports vBuf RSS near `84 MB` at `MODEL_READY` and about `33.3 GB` after first evaluation; `TTFUC` was about `36.145 s` | A structural open may be fast because payload pages were not read yet | [`Step 27 attribution`](docs/vbuf-ml/step27-loader-scaling.md) |
| Structure is self-navigable | POC13 visited `377` structural records for the real artifact; cold and warm structural opens were `2.134805224 s` and `6.588236 ms`; exact payload-to-consumer timing was explicitly not instrumented | Metadata/ranges can be opened and validated before demand-driven payload reads | [`POC13 structural open`](research/results/vbuf-full-moe-layer-poc13-x86/qualification-report.md) |
| Materialization is tensor/range-driven | POC19 records `928/928` exact identity/range matches and no whole-model, whole-block, or whole-packed-expert loading; preparation copies, repacks, and transcodes were `0` | Persistent identity is independent of the buffer currently resident | [`POC19 report`](research/results/vbuf-deep-stack-poc19-x86/report.md) |
| Sparse routing reaches storage | In the qualified MoE traces, unselected expert graphs, acquisitions, source reads, and materializations were all `0` | Router decisions can prune graph construction and backing reads, not only arithmetic | [`POC13 MoE qualification`](research/results/vbuf-full-moe-layer-poc13-x86/qualification-report.md) |
| Active execution stays local while logical demand grows | For POC19, logical persistent bytes were `1,462,931,456`, peak active persistent bytes `1,892,352`, ratio `773.076`, active fraction `0.00129353` | Active wave memory, resident cache, activations, state, and backend workspace need separate accounting | [`POC19 active/logical scaling`](research/results/vbuf-deep-stack-poc19-x86/report.md) |
| Execution lease and residency are different lifetimes | POC19 retained resident entries after lease release and used them for replay; `140,943,360` token-to-token reusable bytes were retained at `100%` at `256 MiB` but `3.95%` at `8 MiB` | Releasing the last consumer lease must not imply evicting the resident cache entry | [`POC19 capacity and reuse`](research/results/vbuf-deep-stack-poc19-x86/report.md) |
| Capacity can dominate replacement quality | At `8 MiB`, source bytes were `750,665,728` and reload bytes `516,276,224`; at `256 MiB`, source bytes were `234,389,504` and reload bytes `0` | Cache budget and replacement policy are separate questions; high residency can be productive reuse | [`POC19 capacity sweep`](research/results/vbuf-deep-stack-poc19-x86/report.md) |
| Replacement quality matters in the middle regime | COST_AWARE saved `82,542,592` reload bytes at `64 MiB` and `228,050,944` at `128 MiB`, capturing `41.90%` and `57.26%` of MIN headroom | Policy should be evaluated at several capacities; COST_AWARE is not treated as an optimum | [`POC20 policy comparison`](research/results/vbuf-cost-aware-residency-poc20-x86/report.md) |
| Tiering was a useful negative result | The tested `8 MiB HOT + 32 MiB WARM` policy did not reduce backing bytes versus an equal-total `40 MiB` flat cache and added `259,117,056` bytes of inter-tier movement | Do not add movement or tiers without a measured benefit for the target trace | [`POC18 tiering result`](research/results/vbuf-tiered-residency-poc18-x86/poc18-report.md) |
| Native forward execution is now functionally closed for the qualified path | POC21 reached all `27` blocks, final normalization, quantized output projection, `102400` logits, and greedy token `86711` with exact parity | The current milestone is model-forward execution, not a complete serving stack | [`POC21 functional pipeline`](research/results/vbuf-functional-pipeline-poc21-x86/report.md) |

## What we deliberately did not conclude

- About `0.262 s` structural readiness does not mean the complete model payload was read in that time; first useful compute and payload access were later.
- About `1.9 MiB` peak active persistent memory does not mean the model requires only `1.9 MiB` total RAM. Resident weights, activations, runtime state, mappings, and backend workspace are separate measurements.
- `256 MiB` is not a universal cache requirement. It was the observed reload-free knee for one POC19 artifact and trace.
- COST_AWARE is not claimed to be an optimal replacement policy. MIN is an offline oracle/reference bound, not a runtime policy.
- ggml is not bad or obsolete. It currently carries valuable compute and backend functionality, while its full-model ownership assumptions are not the vBuf runtime boundary.
- ggml is not a permanent architectural upper bound either. Components should be replaced only if measurements show that their assumptions constrain the required runtime.
- llama.cpp is not the vBuf-native model lifecycle. It is used as a semantic/reference baseline and compatibility consumer where that is useful.
- POC21 does not imply production-ready chat serving, tokenizer completeness, sampler completeness, or chat-template/runtime completeness.
- x86 qualification does not imply complete ARM32 or RISC-V compute support. The pinned ggml RVV FP16 issue still blocks the qualified RV2 compute path.
- No claim is made that SBCs outperform GPUs in raw compute efficiency; the repository does not establish that comparison.

## Architecture and research method

The runtime boundary follows the responsibility split described by the native
region audit: vBuf owns validated model identity, source/range selection,
materialization, residency, execution lifetime, state, placement, and region
planning; ggml supplies tensor descriptors, operations, graph allocation,
scheduling, transfers, and backend kernels. This is a vBuf runtime on ggml, not
a modified llama model lifecycle and not a claim that ggml must eventually be
removed. [`vbuf-native-region-runtime-audit.md`](research/vbuf-native-region-runtime-audit.md)
records the boundary and the remaining full-model assumptions.

The research method is evidence-driven engineering: observation -> hypothesis
-> bounded POC -> reference comparison -> invariant -> next question. A runtime
abstraction is promoted when a measured experiment shows why it is needed.
Negative results remain part of the history: POC17 found an 8 MiB trace that was
primarily capacity-bound, POC18 found no backing-traffic benefit from the tested
tier split, and Step 29 found that host-local dense preparation could not feed
the measured resident compute window. Later POCs then tested the narrower
questions that remained, including replacement quality and the native forward
pipeline.

## Research and evidence

The detailed history remains in [`research/`](research/) and the broader ML
qualification notes in [`docs/vbuf-ml/`](docs/vbuf-ml/). The shortest path through
the current evidence is:

- [`POC13: full real MoE layer`](research/results/vbuf-full-moe-layer-poc13-x86/qualification-report.md)
- [`POC19: deep stack, capacity, active/logical memory`](research/results/vbuf-deep-stack-poc19-x86/report.md)
- [`POC20: COST_AWARE residency`](research/results/vbuf-cost-aware-residency-poc20-x86/report.md)
- [`POC21: functional native model pipeline`](research/results/vbuf-functional-pipeline-poc21-x86/report.md)
- [`Step 27: structural readiness and first-use attribution`](docs/vbuf-ml/step27-loader-scaling.md)
- [`Step 29: layer I/O and host-relative preparation`](docs/vbuf-ml/step29-layer-io.md)
- [`Native region runtime audit`](research/vbuf-native-region-runtime-audit.md)

## Architectural intent

`BaseStep` is vBuf's hardware-neutral physical granularity. It provides predictable block starts and can support naturally aligned scalar loads, SIMD-friendly payload positions, simple physical address calculation, vectorized traversal, and favorable cache behavior. The format does not hard-code one contemporary SIMD, cache-line, or page width.

The v0.6 wire contract permits BaseStep values from 8 through 256 bytes. Choosing a writer default is a whole-system trade-off among native access, cache/traversal behavior, padding, and packing density—especially for small or composite multi-block representations. Optional indexes, if qualified later, derive their geometry from BaseStep and do not select it.

### v0.6 base properties

- exact little-endian magic/version and checked 64-bit ranges;
- power-of-two canonical block geometry;
- orthogonal payload alignment as a multiple of BaseStep;
- portable selected primitive encodings and opaque bytes;
- known-size and indefinite canonical streams;
- deterministic next-block calculation with no required final tail padding;
- no mandatory or currently selected index, checksum, directory, or finalization artifact.

The existing Rust, TypeScript, and C implementations still represent legacy v0.5-style behavior until the v0.6 safety/writer steps are implemented. Do not infer implementation conformance from publication of the specification.

---

## 🏗️ Canonical memory layout

```text
minimum global header
alignment padding to BaseStep
canonical block anchor [+ optional extended count]
padding to PayloadAlignment
payload bytes
padding to the next BaseStep block start (only when another block follows)
```

Canonical headers and checked ranges remain authoritative. See the normative specification for exact fields and formulas.

---

## 🛠️ Roadmap

1. **Current:** normative v0.6 base specification and immutable legacy evidence.
2. **Next:** checked v0.6 Rust, TypeScript, and C readers/writers with cross-language conformance.
3. **Qualification:** measure BaseStep and optional generic navigation structures before selecting defaults or artifacts.

## 💻 Legacy TypeScript prototype usage

> This example uses the pre-v0.6 prototype API and does not claim v0.6 wire conformance.

```typescript
import { VBufWriter } from "./src/vbuf";

const writer = new VBufWriter();
writer.add("user_id", "550e8400-e29b-11d4-a716-446655440000"); // UUID
writer.add("balance", 41234); // SMI

const buffer = writer.finish();
// Now ready to be written to disk or sent over the wire.
```

## 📜 Specification

- **Normative generic wire contract:** [`spec/spec_0.6.md`](spec/spec_0.6.md)
- **Compatibility and historical status:** [`spec/compatibility.md`](spec/compatibility.md)
- **Execution and qualification plan:** [`step-by-step.md`](step-by-step.md)

Specifications v0.1 through v0.5 are retained as historical evidence, not alternate definitions of v0.6.

## ⚖️ License

MIT
