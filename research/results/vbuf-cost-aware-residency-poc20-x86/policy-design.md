# POC20 Policy Design

## Seam

`ResidencyReplacementPolicy::choose()` receives generic eviction candidates and the current request ordinal. `LRU` and `COST_AWARE` are selectable through the harness policy argument; LRU remains the default.

## COST_AWARE

For each eligible candidate:

```text
REACQUIRE_COST = reacquire_cost_bytes = tensor_bytes
REUSE_DENSITY = observed_request_count * REACQUIRE_COST / resident_bytes
RECENCY = 1 / (request_age + 1)
RETENTION_SCORE = 0.9 * normalized(REUSE_DENSITY)
                 + 0.1 * normalized(RECENCY)
```

The lowest score is evicted. Stable tensor identity is the deterministic tie-breaker. The byte cost is intentionally simple: the POC19 HTTP range source has no stable per-identity latency difference, so precise milliseconds would be fabricated. The contract can accept richer source cost later.

The policy uses only past observations, current resident state, size, lease state, and generic byte cost. It does not use model names, tensor names, layer numbers, expert IDs, the completed trace, or MIN future knowledge. Asynchronous overlap-aware scoring is deferred.

No admission change, tiering, payload duplication, or execution scheduling change was introduced.
