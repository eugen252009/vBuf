# Step 5A generic navigation qualification

This benchmark is evidence-only. It generates canonical v0.6 streams with the
stable Rust writer for BaseStep values 8, 16, 32, 64, 128, and 256 bytes, then
measures generic physical traversal and experimental in-memory accelerators.
Nano, checkpoints, and the directory are not v0.6 wire artifacts.

## Reproduction

From the repository root:

```sh
cargo run --release --manifest-path rust/Cargo.toml --bin v06_navigation_bench -- \
  benchmark-results/vbuf-navigation/step5a-raw.csv
python3 scripts/validate_v06_navigation.py \
  benchmark-results/vbuf-navigation/step5a-raw.csv
```

The runner uses three warmups and 20 measured samples per layout/BaseStep/
operation. Layouts include many tiny blocks, few large blocks, mixed blocks,
AoS-style and SoA-style compositions, continuation composites, opaque ranges,
and zero/partial-final cases. The CSV records exact payload, file, padding
(indirectly through file minus payload), Nano, checkpoint, and directory sizes.

The benchmark compares canonical validation/traversal, physical-start
enumeration, simple checkpointed select, linear Key-ID lookup versus a sorted
binary-search directory, and Rayon parallel controls. Construction and
embedded/reconstructed-cache load are separate operations. It does not claim
cold-cache, page-fault, SIMD, or native-alignment results; those require a
separate instrumented run on a controlled host.

## Interpretation boundary

The generated Markdown distinguishes measured facts, inferences, and decisions.
This benchmark cannot promote an artifact by itself: any candidate must survive
end-to-end construction, canonical validation, persistence/cache invalidation,
artifact-byte, and equivalent-work comparisons before a later finalization step.
