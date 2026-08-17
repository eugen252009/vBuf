# POC7 Boundary Check

```text
SourceSelectionPolicy → chooses eligible source ordering
PrefetchPlanner      → schedules dependency-driven requests
Materializer         → performs primary/fallback range reads
TensorResidencyStore → owns resident payloads and leases
TensorDependencyExecutor → executes consumers
vBuf                 → remains persistent artifact format
```

The current generic source-selection implementation contains no model-specific
branches and does not perform prefetch, residency, execution, or format
transformation.
