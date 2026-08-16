# RV2 Dual-Path Multi-Source Striping

## Source Configuration

Source A was configured as:

```text
URL: http://192.168.188.2/models/DeepSeek-V2-Lite.IQ1_S.vbuf
local_source_ip: 192.168.188.50
observed socket: 192.168.188.50:* -> 192.168.188.2:80
```

Source B was configured as:

```text
URL: http://192.168.188.40/models/DeepSeek-V2-Lite.IQ1_S.vbuf
local_source_ip: 192.168.188.42
observed socket: 192.168.188.42:* -> 192.168.188.40:80
```

The Pi's current kernel mapping is `end1=192.168.188.50` and
`end0=192.168.188.42`, which is reversed relative to the interface labels in
the requested topology. Qualification binds and verifies the IP addresses,
not interface names. The two addresses are on distinct physical interfaces.

## Large Direct-to-RAM Qualification

The transport executable used the existing `HttpRangeSource`,
`RangeStripingPlan`, and `ParallelRangeMaterializer`. It allocated one
contiguous 2 GiB destination and streamed each response directly into its
assigned destination interval.

```text
total:  [0, 2147483648)
A:      [0, 1073741824), destination [0, 1073741824)
B:      [1073741824, 2147483648), destination [1073741824, 2147483648)
```

Every repetition returned exactly 1 GiB from each server. No full artifact was
downloaded and no full-stripe response buffer was used by the HTTP source.

Three repetitions were run per path:

| Path | Run 1 MiB/s | Run 2 MiB/s | Run 3 MiB/s | Median MiB/s |
|---|---:|---:|---:|---:|
| A only | 62.024 | 64.183 | 64.303 | 64.183 |
| B only | 64.368 | 60.784 | 63.942 | 63.942 |
| A+B striped | 147.733 | 152.210 | 149.843 | 149.843 |

Dual-path median speedup over the fastest single path was `2.335x`.

The dual request intervals overlapped in all repetitions. Median overlap was
`12.975 s`; median completion skew was approximately `10.213 ms`.

## Real Tensor Qualification

Tensor: `blk.0.ffn_down.weight`, offset `83518776`, length `12607488`.

```text
A: [83518776, 89822520), 6303744 bytes
B: [89822520, 96126264), 6303744 bytes
```

Both ranges completed with HTTP 206. The reconstructed payload hash was
`0c7bdf85b162206b`. The warm re-request reported `warm_hit=PASS` and
`warm_stripe_operations=0`. The transport-only path did not execute ggml.

## Teardown and Scope

- Large A-only teardown: `0` resources.
- Large B-only teardown: `0` resources.
- Large dual teardown: `0` resources.
- Real tensor teardown: `0` resources.
- RV2 ggml compute: `NOT_EXECUTED`; the existing pinned ggml RVV FP16 blocker remains unchanged.
- Bandwidth weighting: not justified; simultaneous dual measurements were sufficiently balanced for this POC, and 50/50 remains in use.
