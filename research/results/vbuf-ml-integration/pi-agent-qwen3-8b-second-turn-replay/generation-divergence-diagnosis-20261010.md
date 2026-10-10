# Qwen3-8B fixed-prefix generation-divergence diagnosis

**Status: diagnostic only. No specific implementation defect was demonstrated, so no runtime fix or production regression was made.** Qwen3-8B remains experimental and identity-gated; its autonomous Snake task remains failed. This follow-up does not alter the earlier parser safety fix or the stopped Qwen3-14B Native AV matrix.

## Findings

The 6,144-token vBuf completion enters an exact period-3 suffix at **generated token index 1,288 (zero-based)**, not at the generation cap. The repeated IDs are `[364, 67, 516]`; all 4,856 remaining tokens follow that period (1,618 complete cycles and two tokens). The suffix is in the generated `char moves[MAX_MOVES]` initializer. The earlier 12,206-byte raw text and all 6,144 token IDs remain preserved in `raw/ctx12-6144-vbuf-01/` and are copied into the diagnosis evidence directory.

The model-visible request is exactly recoverable: the rendered prompt is 3,998 bytes with SHA-256 `39c32c329cd2683ed946e1903282ffe3afb5f54c4179de436cd15dcda018e99c`; its 916 token IDs have SHA-256 `feaf88943eafee31328b6ed0c9a3fc7db5ceb82b393f8a6bde5d9d5637be2236`. The exact request, rendered prompt, and token IDs are preserved in the raw evidence.

A pinned llama.cpp fixed-prefix sweep found the **first greedy top-1 mismatch at generated index 60**: vBuf's saved token is `17211`, while llama.cpp ranks `2207` first and the vBuf token second. The llama top-1 margin is only `0.250389` logit. At that same prefix, incremental vBuf ranks token `17211` first by `0.116997`; a fresh vBuf prefix replay ranks it first by `0.024250`. Across the full vocabulary at this boundary, incremental-vBuf/llama relative RMS is `0.018526`, cosine is `0.999838`, and softmax TV is `0.07775`; the small top-pair rank flip occurs amid otherwise highly similar logits. Across targets 0–1,348, there are four fixed-prefix top-1 mismatches (indices 60, 623, 816, 1049); the vBuf token is rank 2 in llama.cpp at each. Thus the actual greedy trajectories first separate on an early near-tie, well before the repetitive suffix.

At the repetitive suffix itself, llama.cpp agrees with vBuf **when conditioned on the same vBuf-generated prefix**: at targets 1,287, 1,288, 1,318, and 1,348, both top-1 predictions match (`314`, then `364` repeatedly). At target 1,288 the vBuf/llama logit cosine is `0.999822`, softmax total-variation distance is `0.001772`, and top-1 margins are `2.522` / `2.534`. This is local conditional evidence only: after index 60, that forced prefix is no longer llama.cpp's own trajectory, so it does **not** show that llama.cpp would autonomously enter the same loop.

The evidence therefore supports this bounded explanation: a small backend-sensitive near-tie at token 60 changes the greedy path; the vBuf path later reaches a highly self-reinforcing period-3 continuation. It does not identify which low-level arithmetic or execution difference flips that near-tie, nor prove that this alone causes the eventual loop.

## Prefix and generation settings

All artifacts were identity-checked:

- GGUF SHA-256: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`
- vBuf payload SHA-256: `cc85fa7afd90808484485de0e0a09e88ff5b98b58d1fb69c7083f64916417cb5`
- Semantic sidecar SHA-256: `99f2892dbe58d427605457729a6edcfa037d67f8ec1ff7f325953345b5d11212`
- llama.cpp commit: `a97123e497968f3440264c0464a7adc7c999c027`

The captured vBuf request has context capacity 12,288, 916 prompt tokens (11,372 slots remain), a 6,144 completion-token cap, greedy argmax with no request-level sampler, thinking off, 36 model layers, 32-token prefill chunks, and single-token decode. Its generation sequence through token 1,349 was re-run with diagnostic logit capture and matched the saved sequence exactly.

The pinned llama.cpp reference used the same 916 token IDs, context 12,288, batch 512 / ubatch 128, flash attention off, and all layers on the RTX 3060. The matched server configuration was temperature 0, top-k 1, top-p 1, repeat penalty 1, seed 42, thinking off. The fixed-prefix harness reads raw logits (no sampler); those settings reduce to greedy argmax. The reference was an offline fixed-prefix evaluation, not a replacement inference path.

For each target index `j`, the harness evaluates the 916-token prompt followed by the saved vBuf-generated tokens `[0, j)`, then captures logits predicting token `j`; the target token is not included in its own prefix. The llama sweep records top-1 predictions for all 1,349 target positions. Full 151,936-element F32 logit vectors are retained at indices `0, 58–61, 1287, 1288, 1318, 1348` for vBuf incremental, vBuf fresh-prefix, and llama.cpp paths.

## KV-cache check and evidence limits

Because the first cross-backend mismatch occurs during decode, a small fresh-prefix control was warranted. A new vBuf session re-evaluated the known prompt-plus-generated prefix at selected positions, contrasting that with the original incremental session. At index 60 both vBuf paths still choose `17211`; their full-logit cosine is `0.999903` and softmax TV is `0.02780`. They differ in margin (`0.116997` incremental versus `0.024250` fresh), but do not change the selected token. Both also select the same repeated token at the sampled later positions.

This is not a cache-only experiment: the fresh path reprocesses the combined prefix in prefill chunks, while the incremental path reaches it through prompt prefill followed by one-token decode. Those shapes can change arithmetic. No retained K/V tensors were compared, and no same-topology cache reset/replay was performed. The limited captures show no top-1 evidence of a stale or corrupted KV cache; they cannot rule out every cache or execution defect. The first mismatch is also near-tied, so the observed backend-sensitive rank flip is not, by itself, proof of a bug.

## Decision and scope

No production execution, tokenizer, sampler, admission, numerical contract, threshold, or parser change is justified by this evidence. In particular:

- Preserve the `a6f3f05` incomplete-tool-call safety behavior unchanged.
- Keep Qwen3-8B experimental, identity-gated, and not coding-agent-qualified.
- Do not switch the final inference path to llama.cpp.
- Keep production defaults, Numerical Contracts, and Native AV status unchanged (`Candidate_NOT_Valid`; canonical packed-V remains authoritative).
- Do not backfill or reinterpret the stopped Native AV matrix.

The logit-capture modification was applied only to a temporary diagnostic build. Repository changes in this follow-up are research report and raw evidence only; no focused production regression was added because no specific defect was demonstrated.

## Evidence and validation

The self-contained raw capture, F32 logits, commands/source, corrected top-1 sweep, checksums, and offline analyzer are in [`raw/fixed-prefix-logit-diagnosis-20261010/`](raw/fixed-prefix-logit-diagnosis-20261010/). Re-run the offline analysis with:

```bash
python3 raw/fixed-prefix-logit-diagnosis-20261010/harness/analyze_fixed_prefix_logits.py \
  raw/fixed-prefix-logit-diagnosis-20261010
```

The analysis found the period-3 suffix at 1,288, 1,349 tested fixed-prefix targets, and the four llama top-1 mismatches listed above. These are diagnostic observations, not qualification or performance evidence. The earlier Pi Snake task remains unsuccessful, and the three sequential tool-cycle sessions remain protocol/workflow evidence only.
