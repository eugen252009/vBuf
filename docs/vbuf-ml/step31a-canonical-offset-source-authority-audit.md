# Step 31A Canonical-Offset and Source-Authority Audit

Status: **CONDITIONAL PASS for the qualified single-source path**

This audit is evidence only. It does not implement persistence, coverage,
source selection, or a sparse mirror.

## Configuration

- Baseline commit: `80409a4`
- Device: Pixel 7 Pro, Android 17, `arm64-v8a`
- Model: DeepSeek-V2-Lite IQ2_XXS
- Runtime: normal inference, batched prefill, single-position decode
- Prompt tokens: 7
- Generated tokens: 4
- Residency budget: `268,435,456` bytes
- Payload: `/models/DeepSeek-V2-Lite.IQ2_XXS.vbuf`
- Declared/served payload size: `5,639,819,878` bytes

The run used a temporary audit-only range ledger. Because the local nginx
keep-alive limit closed the connection after 1,000 requests, the diagnostic
run reopened the HTTP connection after each successful range. Requested
offsets and lengths, semantic bootstrap, runtime mode, batching, model,
prompt, and residency budget were unchanged.

## Authority and Offsets

The semantic bootstrap is a separate app-private discovery artifact. Its
external bindings use `SourceId(1)` and the original payload source size. The
Android path carries each `PersistentTensorRef.source_offset` directly into
`HttpRangeSource::read_range(offset, length, ...)`. No source-relative rebasing
or backend-specific offset translation was found.

For this qualified one-source deployment, the canonical model is therefore:

```text
semantic bootstrap: local discovery artifact
external source:   one random-access canonical payload artifact
TensorRef range:   payload-artifact offset and length
mirror range:      the same offset and length
```

This is not a complete immutable artifact identity contract. A numeric
`SourceId` is not sufficient across sessions or model replacements. A future
persistent mirror still requires an immutable source binding, declared size,
and a suitable digest/manifest contract before reuse is authorized.

## Physical Trace

Trace file: temporary `vbuf-ml-source-range-audit-v1` ledger from the Android
app-private semantic bootstrap path.

| Phase | Requests | Requested bytes | Bytewise union |
|---|---:|---:|---:|
| Batched prefill | 1,499 | 1,480,411,744 | 1,277,582,944 |
| Decode token 1 | 665 | 819,151,520 | 819,151,520 |
| Decode token 2 | 664 | 806,544,032 | 806,544,032 |
| Decode token 3 | 767 | 879,644,320 | 879,644,320 |
| Decode token 4 | 692 | 759,750,304 | 759,750,304 |
| **Total** | **4,287** | **4,745,501,920** | **1,377,067,264** |

The HTTP source returned exactly the requested range lengths. Transport
overfetch was therefore zero. Requested bytes exceeded the global bytewise
union by `3,368,434,656` bytes, representing repeated/reloaded source ranges.
The decode union intersected the prefill union by `907,623,424` bytes.

New bytes at each decode boundary, relative to all prior phases, were:

| Token | New bytes | Reused bytes from prior phases |
|---|---:|---:|
| 1 | 34,198,176 | 784,953,344 |
| 2 | 18,653,184 | 787,890,848 |
| 3 | 24,870,912 | 854,773,408 |
| 4 | 21,762,048 | 737,988,256 |

Most reuse was one decode step old: `769,237,664` bytes for token 2,
`759,911,072` bytes for token 3, and `706,899,616` bytes for token 4.

## Residency Boundary

The physical run reported:

```text
peak resident bytes:       267,452,416 <= 268,435,456
residency hits/misses:     6,107 / 7,898
evictions:                 4,217
reload bytes:              2,791,685,088
decode total:              261,266 ms
total generation:          399,146 ms
```

Each token's fetched interval union was between approximately 724.6 MiB and
838.9 MiB, so a full per-token working set cannot fit in the 256 MiB RAM
residency budget. This confirms that RAM residency and any future persistent
source retention budget must remain separate decisions.

## Gate Result

**Conditional pass.** Same-offset consumption is established for the
qualified single external payload and the measured trace provides sufficient
range geometry for later retention simulation. The sparse-mirror path remains
blocked from implementation until the persistent artifact identity contract,
coverage publication/recovery contract, and local/remote byte-equivalence
qualification are defined.

The trace does not establish complete offline model coverage. It only records
the ranges encountered by this prompt and four-token execution; headers,
padding, control regions, and never-routed expert ranges remain outside the
observed coverage claim.
