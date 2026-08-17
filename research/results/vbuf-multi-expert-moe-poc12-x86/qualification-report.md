# POC12 Real Multi-Expert MoE Execution

## Artifact And Router

- Artifact: `DeepSeek-V2-Lite.IQ1_S.vbuf`
- SHA256: `780a55b77d2730705a93622338d9747149624d72e558e26182176868210fafcc`
- Router: `blk.1.ffn_gate_inp.weight`
- Router representation: `CanonicalPrimitive` / float32, dimensions `[2048,64]`
- Router range: offset `99094944`, length `524288`
- Expert count: `64`; top-k: `6`
- Activation source: `DETERMINISTIC_FIXTURE`, one-hot hidden vectors at indices 0 and 1.

## Weighting Semantics

The pinned reference path uses softmax router probabilities, selects the top-k
experts, and normalizes the selected probabilities by their selected sum
(`norm_topk_prob`). POC12 applies exactly:

```text
p_i = exp(logit_i - max(logits)) / sum_j exp(logit_j - max(logits))
w_i = p_i / sum_selected p_i
```

The generic merge computes `sum_i(w_i * expert_output_i)` and rejects empty,
incompatible, mismatched, or non-finite inputs.

## Router Results

- Activation A IDs: `[37,21,31,54,6,33]`
- Activation A logits: `[0.130859,0.125,0.124512,0.115723,0.104492,0.0996094]`
- Activation A normalized weights: `[0.169033,0.168045,0.167963,0.166493,0.164634,0.163832]`
- Activation B IDs: `[3,53,47,5,20,35]`
- Activation B logits: `[0.0776367,0.0708008,0.0566406,0.0532227,0.0449219,0.0397949]`
- Activation B normalized weights: `[0.170098,0.168939,0.166564,0.165995,0.164623,0.163781]`
- Router score parity: PASS, max abs/rel/mean abs `0`.

## Dynamic Expert Execution

Six graphs are created after top-k selection for each activation. Each graph is
the unchanged POC10 graph: local graph refs `gate:0,up:1,down:2`; storage refs
are offset only inside an observational materializer adapter. No unselected
graph is created.

Per selected expert, each of gate/up/down is a direct 2D view of the real
3D packed tensor. Every slice is `563200` bytes for gate/up and `1622016`
bytes for down. The complete exact offsets and traces are in `execution.log`.

- Selected graphs created: `6/6` for A and B.
- Selected experts executed: `6/6` for A and B.
- Unselected graphs created: `0`.
- Unselected TensorRefs acquired: `0`.
- Unselected source reads/materializations: `0`.
- Execution order: router rank order.

## Residency And Wave Metrics

The shared tensor residency store uses an 8 MiB budget and global slice refs
(`expert_id * 3 + local_ref`). No expert-level residency state is introduced.
Gate slices for alternating cold experts are locally preloaded; remaining
selected slices use the normal HTTP materializer.

- Selected total persistent bytes: `16490496`.
- Peak active persistent bytes: `1622016`.
- Peak active fraction: `0.0983607`.
- Peak resident bytes: `8245248`.
- POC10 tensor-level acquire/release and last-consumer semantics: preserved.
- Prefetcher: existing `PrefetchPlanner`; no MoE-specific planner added.
- Source striping: none on x86; selected slices are single-source HTTP ranges.

Per-run counters are recorded in `execution.log`:

- A cold: hits `39`, misses `39`, source reads `15`, materialization requests `15`, policy calls `15`, evictions `9`.
- A warm: hits `36`, misses `24`, source reads `2`, materialization requests `2`, policy calls `11`, evictions `2`.
- B transition: hits `36`, misses `42`, source reads `18`, materialization requests `18`, policy calls `18`, evictions `18`.
- A replay: hits `2`, misses `42`, source reads `1`, materialization requests `1`, policy calls `18`, evictions `1`.

Source policy is called observationally only for tensor residency misses; hit
paths bypass source policy and source reads.

## Correctness

- Per-expert reference parity, A: PASS for all six, max abs/rel `0`.
- Per-expert reference parity, B: PASS for all six, max abs/rel `0`.
- Weighted merge parity, A cold/warm/replay: PASS, max abs/rel `0`.
- Weighted merge parity, B: PASS, max abs/rel `0`.
- A→B working-set transition: PASS; sets are disjoint and B causes real evictions.
- A→B→A replay: PASS; A reloads after B under the constrained budget.
- Invalid merge inputs: PASS via `vbuf_weighted_merge_contract`.

## Failure Isolation

With the router endpoint available and the selected expert endpoint unavailable,
the first required selected tensor reaches `FAILED`; its consumer operation and
final merge do not execute, incomplete materialization is discarded, and
resources after teardown are `0`. The failure identifies selected expert `37`
and local TensorRef `1`.

## Layout And Architecture Guards

- Router persistent layout: `DIRECT`.
- Expert persistent layout: `DIRECT` views; no repack or transcode.
- vBuf layout change required: `NO`.
- vBuf format change required: `NO`.
- Architecture-specific runtime logic: `NO`.
- Full shared-expert/residual transformer path: outside scope.
