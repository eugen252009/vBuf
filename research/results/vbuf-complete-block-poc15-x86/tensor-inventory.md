# Complete Block Tensor Inventory

All persistent tensors are direct, independently addressable ranges from the
real artifact. Attention bytes are `2,960,384`; FFN always-required bytes are
`4,677,632`.

| Tensor group | Persistent bytes | Layout |
|---|---:|---|
| POC14 attention, six tensors | 2,960,384 | DIRECT |
| `ffn_norm`, router, shared gate/up/down | 4,677,632 | DIRECT |
| Eight unique selected routed-expert slices across positions | 21,987,328 | DIRECT |
| Complete logical block working set | 29,625,344 | DIRECT |

Selected expert IDs:

```text
token 0: 56, 11, 50, 34, 3, 47
token 1: 56, 29, 11, 3, 50, 26
intersection: 56, 11, 3, 50
new at token 1: 29, 26
evicted from token 0 set: 34, 47
```

No packed expert tensor or whole block buffer was materialized.
