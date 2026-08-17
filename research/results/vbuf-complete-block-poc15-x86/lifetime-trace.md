# Cross-Sublayer Lifetime Trace

The complete-block log shows the dependency-driven sequence for each position:

```text
attention_timing / payload ready
→ attention consumer starts
→ attention residual TensorValue produced
→ ffn_timing router payload ready and consumer starts
→ ffn_timing routed expert payload ready and consumer starts
→ ffn_timing shared expert payload ready and consumer starts
→ FFN residual and complete block output
```

The attention graph and each FFN graph use independent execution leases. The
block orchestration does not acquire a complete-block weight lease. Attention
weights are released after their last consumers while the residual activation
survives as a transient value into the FFN graph.
