# Qwen3 Higher-Precision Reference Qualification

**Disposition:** BF16 from the same immutable Qwen3-14B GGUF repository snapshot is a practical higher-precision reference and loaded successfully in the pinned llama.cpp CPU runtime. Three-way common-token numerical comparisons and 16-token greedy trajectories are recorded. The seed-0 divergence is explained by a narrow BF16 top-1/top-2 margin: BF16 agrees with vBuf Q4_K_M at the first divergent token, while llama.cpp Q4_K_M chooses the other near-tied token. This does **not** establish that vBuf is globally closer: across selected checkpoints the two Q4 paths alternate, and one seed-0 prefix (25) shows substantially larger vBuf error. A 77-target-token teacher-forced NLL probe shows a small aggregate vBuf disadvantage versus llama.cpp Q4. No broad task-quality conclusion is justified because the requested behavioral/coding/structured-output suite was not run.

Production Qwen3 remains **DISABLED**. The fixed `1e-5` llama.cpp Q4 comparison remains a **FAIL** and was not changed.

## Reference artifact and checkpoint identity

- **Reference representation:** BF16 GGUF, preferred over Q8/Q6 because it was available, fit in RAM, and executed end-to-end on the CPU.
- **Repository:** `bartowski/Qwen_Qwen3-14B-GGUF`; its card identifies base model `Qwen/Qwen3-14B`.
- **Immutable repository revision:** `bd080f768a6401c2d5a7fa53a2e50cd8218a9ce2`.
- **Filename:** `Qwen_Qwen3-14B-bf16.gguf`.
- **Size:** 29,543,423,776 bytes.
- **SHA-256:** `9677a58b0fa8da7771a4d8cc8080208ce02a32ab21305ded10a3798766132d3a`.
- **Tensor storage:** 282 BF16 tensors and 161 F32 tensors; 443 tensors total.
- **Parameter count:** 14,768,307,200.
- **Architecture/geometry:** Qwen3 dense; 40 layers, embedding 5,120, 40 query heads, 8 KV heads, FFN 17,408, vocabulary 151,936, context 32,768, RoPE frequency base 1,000,000.
- **Tokenizer:** GGUF model `gpt2`, pre-tokenizer `qwen2`, 151,936 tokens; BOS 151643, EOS 151645, padding 151643. Token, merge, and token-type canonical hashes match the Q4 artifact: `40d4ea40d5a19d60f50b7acfd7c66026cfca9c0186b7b52a69e757fca65af538`, `24fa2ae2a398e50784a1fff678482094af4f63e6783d35686726abacda8dc371`, and `536de815c96740a876dbaaa0b181fc54b71e958de2063d4d1a7f425f8692cc16`.
- **Chat template:** present and byte-content equivalent after canonical serialization; SHA-256 `f83d4f1bec129eea16be98cf41a88c8d61f18b3f169dd5435e05d67c30625a39`.
- **Q4 comparison artifact:** `Qwen_Qwen3-14B-Q4_K_M.gguf`, 9,001,753,632 bytes, SHA-256 `915913e22399475dbe6c968ac014d9f1fbe08975e489279aede9d5c7b2c98eb6`; same repository revision, architecture, tensor inventory/shapes, tokenizer and chat template. Its tensors are 241 Q4_K, 41 Q6_K, and 161 F32.
- **Identity conclusion:** no artifact metadata mismatch was found. Both files are sibling representations in the same immutable repository commit and identify the same published base model. The GGUFs do not encode a separate upstream `Qwen/Qwen3-14B` source commit, so upstream weight lineage is supported by the repository snapshot/card and matching metadata, not independently proven by an upstream-file hash comparison.
- **Reason BF16 is practical:** the local CPU-only llama.cpp run loaded the mapped weights (reported model buffer 28,169 MiB) and completed a 32-token teacher-forced decode. No GPU offload was used. Q8/Q6 were therefore unnecessary fallback choices.

## Compared paths and method

A. **Higher-precision reference:** BF16 GGUF through llama.cpp/GGML commit `a97123e497968f3440264c0464a7adc7c999c027`, CPU, flash attention off, no GPU layers.

B. **Q4 reference:** the same llama.cpp/GGML build and GGUF repository revision, Q4_K_M artifact.

C. **vBuf Q4:** the existing isolated 40-layer Qwen3 qualification graph, canonical per-query extent grouping, pinned GGML commit `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`, CPU AVX2+FMA, granularity 8, 8 threads. No production session/runtime was used.

The fixed inputs are exact token IDs, with causal positions starting at zero. The three paths received identical token sequences for every numerical row. Rows after a generated trajectory had diverged are explicitly **teacher-forced comparisons** on the named sequence, not comparisons of the three models' generated histories. Final hidden means the final transformer-block output (`l_out-39`, before final RMSNorm); final-normalized hidden and logits were also captured.

Reference and Q4 llama.cpp ran at 16 threads for teacher-forced batches; the generation runs used 8 threads. vBuf used 8 threads. The llama.cpp and vBuf paths also use different GGML revisions, so the vBuf-versus-reference distance includes runtime/kernel-version effects as well as representation and graph differences. It is not a pure quantizer-only experiment.

## Common-prefix numerical results

Each table row is the last-position output after consuming the stated number of tokens in one fixed teacher-forced sequence. RMS values are absolute F32 RMS errors against BF16; ratios are `vBuf_Q4_error / llama_Q4_error`. The full per-prefix max absolute, mean absolute, relative RMS, cosine, norm ratio, top-k, and per-token NLL data are in:

- [`sequence-A32.json`](qwen3-higher-precision/sequence-A32.json)
- [`sequence-B32.json`](qwen3-higher-precision/sequence-B32.json)
- [`sequence-C16.json`](qwen3-higher-precision/sequence-C16.json)

| Sequence | Prefix | Hidden RMS llama / vBuf | Hidden ratio | Logit RMS llama / vBuf | Logit ratio | Top-1 BF16 / llama / vBuf | Top-5 overlap llama / vBuf | Top-10 overlap llama / vBuf |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| A | 1 | 7.168 / 7.168 | 1.000 | 0.2305 / 0.2305 | 1.000 | 25 / 25 / 25 | 4/5 / 4/5 | 8/10 / 8/10 |
| A | 4 | 2.113 / 2.084 | 0.986 | 0.3230 / 0.2607 | 0.807 | 13 / 13 / 13 | 5/5 / 5/5 | 9/10 / 9/10 |
| A | 8 | 2.476 / 2.605 | 1.052 | 0.6642 / 0.7206 | 1.085 | 198 / 198 / 198 | 4/5 / 4/5 | 9/10 / 9/10 |
| A | 9 | 1.841 / 1.842 | 1.000 | 0.2219 / 0.2289 | 1.032 | 220 / 262 / 220 | 5/5 / 5/5 | 10/10 / 10/10 |
| A | 16 | 2.270 / 2.107 | 0.928 | 0.2117 / 0.2061 | 0.974 | 13 / 13 / 13 | 5/5 / 5/5 | 9/10 / 10/10 |
| A | 17 | 2.047 / 2.052 | 1.002 | 0.2091 / 0.1814 | 0.868 | 15 / 15 / 15 | 5/5 / 5/5 | 10/10 / 10/10 |
| A | 24 | 2.277 / 2.330 | 1.024 | 0.2697 / 0.2783 | 1.032 | 13 / 13 / 13 | 5/5 / 5/5 | 10/10 / 10/10 |
| A | 25 | 2.401 / 6.308 | 2.627 | 0.3637 / 0.9315 | 2.561 | 15 / 15 / 15 | 5/5 / 5/5 | 10/10 / 10/10 |
| A | 32 | 1.912 / 1.897 | 0.992 | 0.2176 / 0.2095 | 0.963 | 220 / 220 / 220 | 4/5 / 4/5 | 8/10 / 9/10 |
| B | 1 | 8.799 / 8.799 | 1.000 | 0.3554 / 0.3554 | 1.000 | 25 / 25 / 25 | 4/5 / 4/5 | 8/10 / 8/10 |
| B | 4 | 2.148 / 1.858 | 0.865 | 0.2523 / 0.2662 | 1.055 | 13 / 13 / 13 | 5/5 / 5/5 | 9/10 / 10/10 |
| B | 8 | 2.269 / 2.436 | 1.074 | 0.3078 / 0.4042 | 1.313 | 13 / 13 / 13 | 5/5 / 5/5 | 9/10 / 9/10 |
| B | 9 | 2.296 / 2.394 | 1.043 | 0.2288 / 0.2058 | 0.899 | 16 / 16 / 16 | 4/5 / 4/5 | 10/10 / 10/10 |
| B | 16 | 2.370 / 2.290 | 0.966 | 0.4061 / 0.3937 | 0.969 | 13 / 13 / 13 | 5/5 / 5/5 | 10/10 / 10/10 |
| B | 17 | 2.648 / 2.514 | 0.950 | 0.4951 / 0.4816 | 0.973 | 16 / 16 / 16 | 4/5 / 4/5 | 10/10 / 9/10 |
| B | 24 | 1.982 / 1.993 | 1.006 | 0.1705 / 0.2159 | 1.266 | 13 / 13 / 13 | 5/5 / 5/5 | 10/10 / 9/10 |
| B | 25 | 2.118 / 2.044 | 0.965 | 0.3177 / 0.2881 | 0.907 | 16 / 16 / 16 | 5/5 / 5/5 | 9/10 / 9/10 |
| B | 32 | 1.999 / 1.992 | 0.997 | 0.2417 / 0.2237 | 0.925 | 13 / 13 / 13 | 5/5 / 5/5 | 10/10 / 10/10 |
| C | 1 | 8.056 / 8.056 | 1.000 | 0.2259 / 0.2259 | 1.000 | 25 / 25 / 25 | 5/5 / 5/5 | 10/10 / 10/10 |
| C | 4 | 2.377 / 2.248 | 0.946 | 0.2950 / 0.3159 | 1.071 | 24 / 24 / 24 | 4/5 / 4/5 | 10/10 / 10/10 |
| C | 8 | 1.742 / 1.792 | 1.029 | 0.2151 / 0.2146 | 0.998 | 23790 / 23790 / 23790 | 4/5 / 4/5 | 9/10 / 9/10 |
| C | 9 | 1.728 / 1.723 | 0.997 | 0.2348 / 0.2046 | 0.871 | 17553 / 2953 / 2953 | 5/5 / 5/5 | 8/10 / 8/10 |
| C | 16 | 2.596 / 2.578 | 0.993 | 0.5077 / 0.5619 | 1.107 | 11 / 11 / 11 | 3/5 / 4/5 | 8/10 / 8/10 |

Over all 80 teacher-forced token positions (32+32+16), the vBuf/llama RMS-error ratio had median 1.000 (hidden), 1.009 (normalized hidden), and 1.000 (logits). The vBuf Q4 error was lower at 30/80 hidden, 28/80 normalized-hidden, and 32/80 logit positions; llama Q4 was lower at 33/80, 40/80, and 38/80 respectively; the rest were within 1%. Mean ratios were 1.028 (hidden), 1.041 (normalized hidden), and 1.062 (logits). This is an overall slight llama-Q4 numerical edge in this sample, not a uniform per-prefix ordering.

Across those same teacher-forced rows, BF16 top-1 was matched at 74/80 by llama Q4 and 76/80 by vBuf Q4. Mean top-5 overlap was 4.675/5 and 4.663/5; mean top-10 overlap 9.425/10 and 9.413/10. These rankings are separate from direct numerical error.

### Seed-0 first divergence

The last common input prefix is nine tokens:

```text
0,25,220,16,13,15,13,15,198
```

At that exact prefix the next-token top-10s are:

| Rank | BF16 reference | llama Q4_K_M | vBuf Q4_K_M |
|---:|---|---|---|
| 1 | 220: 11.870283 | 262: 11.553518 | 220: 11.616869 |
| 2 | 262: 11.762486 | 220: 11.520546 | 262: 11.472177 |
| 3 | 353: 11.666457 | 2791: 11.246319 | 2791: 11.245657 |
| 4 | 13874: 11.519739 | 353: 11.042898 | 353: 11.237369 |
| 5 | 2791: 11.454535 | 13874: 11.038836 | 13874: 11.014476 |
| 6 | 2: 10.935736 | 2: 10.563581 | 198: 10.637022 |
| 7 | 198: 10.894167 | 198: 10.550009 | 2: 10.529964 |
| 8 | 0: 10.750675 | 256: 10.412968 | 256: 10.377691 |
| 9 | 256: 10.565045 | 0: 10.376014 | 0: 10.353373 |
| 10 | 606: 10.382340 | 606: 10.349030 | 606: 10.131055 |

BF16 top-1/top-2 margin: **0.107798**. The llama Q4 top-1/top-2 margin is **0.032972**; vBuf Q4 margin **0.144691**. BF16 chooses **220**, llama Q4 chooses **262**, and vBuf Q4 chooses **220**. This is outcome **B: high precision agrees with vBuf** at the first divergence. The BF16 margin is narrow, and all three top-10s contain the same tokens; this is consistent with an expected Q4 argmax flip, not evidence by itself of a broad vBuf quality regression. At this prefix vBuf's logit RMS error is slightly larger than llama Q4's (0.22890 vs 0.22190), showing that matching argmax is not the same as being numerically closer.

### Additional deterministic sequence C divergence

Seed 1234, 16 total tokens (seed plus 15 greedy outputs):

- BF16 trajectory: `1234,25,220,16,24,23,15,82,23790,17553,26208,11,1379,220,19,15`
- llama Q4 and vBuf Q4 trajectory: `1234,25,220,16,24,17,15,82,5166,3714,78,24334,59363,21525,448,220`
- First divergence: after the common five-token input `1234,25,220,16,24`; BF16 selects 23, both Q4 paths select 17.
- At that prefix BF16 top-1/top-2 are 23/24 with margin **0.050131**; llama Q4 top-1/top-2 margin is **0.018409**; vBuf Q4 margin is **0.174535**. All three top-10 sets have 10/10 overlap with BF16 and all top-5 sets have 5/5 overlap.

The later C numerical table rows are fixed teacher-forced BF16-sequence inputs. They are **not** post-divergence generated-trajectory comparisons.

## Greedy trajectories

All runs were greedy (temperature 0 / direct argmax), CPU-only. Fixed token counts include the initial seed.

| Seed | BF16 | llama Q4_K_M | vBuf Q4_K_M | Result |
|---:|---|---|---|---|
| 0 | `0,25,220,16,13,15,13,15,198,220,829,25,330,2408,33696,698` | `0,25,220,16,13,15,13,15,198,262,549,0,220,16,13,15` | `0,25,220,16,13,15,13,15,198,220,829,25,330,2408,33696,698` | BF16 and vBuf agree for 16 total tokens; llama first differs on the token after common prefix 9. |
| 42 | `42,25,220,16,13,16,13,16,13,16,13,16,13,16,13,16` | same | same | **TOKEN-PARITY QUALIFIED** through 16 across all three. Existing vBuf/llama Q4 record extends through 32. |
| 1234 | `1234,25,220,16,24,23,15,82,23790,17553,26208,11,1379,220,19,15` | `1234,25,220,16,24,17,15,82,5166,3714,78,24334,59363,21525,448,220` | same as llama Q4 | Q4 paths agree with each other through 16; both diverge from BF16 after common prefix 5. |

After each sequence's first divergence, later token positions are reported as separate greedy trajectories only. No later logits were compared across nonidentical inputs.

## Teacher-forced NLL / perplexity probe

NLL scores the next fixed token from logits at each preceding position. The 77 targets are the same across the three paths per sequence, but the three short sequences were generated by Q4 llama (A), the previously qualified common seed-42 path (B), and BF16 (C); this is not a held-out corpus or a general perplexity benchmark.

| Path | Target tokens | Mean NLL | Perplexity |
|---|---:|---:|---:|
| BF16 reference | 77 | 0.966701 | 2.62926 |
| llama Q4_K_M | 77 | 0.973978 | 2.64846 |
| vBuf Q4_K_M | 77 | 0.980049 | 2.66459 |

On this small mixed fixed-token sample, vBuf is **0.006072 nats/token** worse than llama Q4 (about 0.61% higher perplexity), and both are close to BF16. Per-sequence ordering differs: vBuf is slightly better on B and C; llama Q4 is better on A. This is a measured small distribution-quality difference, not evidence of a statistically established task-quality regression.

## Numerical outlier requiring follow-up

Sequence A at teacher-forced prefix 25 is a material local outlier: vBuf final-hidden RMS error is **6.308** vs llama Q4 **2.401** (ratio **2.627**); logit RMS is **0.9315** vs **0.3637** (ratio **2.561**). Cosines are 0.99075 and 0.99521 respectively. All three paths still choose token 15, with the BF16 reference margin 3.622, and both Q4 top-10s overlap BF16 10/10.

A diagnostic export of each block output shows the difference is small at layer 0 (both about 0.005475 RMS) and grows late: at `l_out-22`, llama/vBuf RMS errors are 0.258/0.376; `l_out-23`, 0.297/0.530; final `l_out-39`, 2.401/6.308. Sequence B at prefix 25 does not show the same effect: vBuf is closer than llama Q4 on both hidden and logits there. The source of the sequence-A outlier is unresolved. It may include the differing GGML revisions/runtime execution path; it must not be averaged away or silently attributed to quantization. This is the strongest concrete reason not to claim global vBuf superiority or declare the quality gate complete.

## Behavioral quality

- Factual QA: **NOT TESTED**.
- Instruction following: **NOT TESTED**.
- Simple reasoning: **NOT TESTED**.
- Coding task: **NOT TESTED**.
- Bug-fix task: **NOT TESTED**.
- Structured JSON/schema adherence: **NOT TESTED**.
- Executable code checks: **NOT TESTED**.

The numerical and short-trajectory qualification does not establish assistant-task correctness or structured-output robustness. The evaluated artifact is the base `Qwen/Qwen3-14B` checkpoint, not a separate Instruct checkpoint. No prompt/task score is inferred from greedy token parity or NLL. The isolated vBuf CPU greedy path took about 26–29 minutes for only 15 generated tokens, making a meaningful multi-task suite impractical on this execution path; one-token/truncated answers would not be valid substitutes.

## Interpretation and production gate

- **vBuf numerically worse than llama?** Not uniformly. Across the 80 selected common-input rows the two paths alternate; llama is closer on somewhat more rows and the mean RMS ratios slightly favor llama. The seed-A prefix-25 outlier is significant and unresolved.
- **llama numerically worse than vBuf?** Yes, on many individual rows; at the known seed-0 divergence BF16 chooses the vBuf token, and at that point the argmax margin is narrow.
- **Comparable Q4 approximations?** Broadly similar on most tested rows and top-k sets, but not interchangeable: trajectories differ, and a few prefix-conditioned metrics have larger vBuf error.
- **Actual vBuf model-quality regression?** **NOT ESTABLISHED.** A small NLL disadvantage is measured on this 77-token probe, but broad task quality was not measured. The prefix-25 numerical outlier warrants investigation before any production decision.
- **Expected quantization/runtime variation?** Supported for near-tie argmax flips (seed 0 and C), but insufficient to explain every numerical difference or the prefix-25 outlier.
- **Is llama.cpp Q4 mathematical ground truth?** No. The BF16 reference shows Q4 Q4 trajectories can differ on a narrow margin, and the BF16 top-1 may agree with vBuf rather than llama Q4.
- **Should strict llama.cpp Q4 parity remain a required production gate?** This experiment shows it is not a sound *sole correctness oracle* for Q4 model quality. Keep the existing `1e-5` result visible and **FAIL**; do not widen or remove it. Because the task-quality suite is absent and a numerical outlier remains unresolved, retain production Qwen3 as blocked pending those gates rather than waiving parity now.

## Status and remaining gates

- BF16 same-snapshot reference artifact: **QUALIFIED** for CPU execution and the recorded loader.
- Checkpoint/tokenizer metadata compatibility: **QUALIFIED** at the shared immutable GGUF repository revision; no mismatch found.
- Three-way common-token comparisons: **QUALIFIED** for sequences A/B (32 tokens) and C (16 tokens), selected prefixes; all files finite and dimensions matched.
- Seed-0 interpretation: BF16 agrees with vBuf at the first divergence; **narrow-margin divergence explained**, not a general correctness result.
- Seed 42: **TOKEN-PARITY QUALIFIED** through 16 across all three; prior vBuf/llama Q4 comparison through 32 remains recorded.
- Sequence C: BF16 diverges from both Q4 paths after common prefix 5; llama Q4 and vBuf Q4 agree through 16.
- NLL: **MEASURED, DIAGNOSTIC ONLY** on 77 generated tokens; no corpus perplexity claim.
- Behavioral task/coding/JSON suite: **NOT TESTED**.
- Strict llama Q4 `1e-5` gate: **FAIL**, unchanged.
- Production Qwen3 / `VbufGenerationSession` / HTTP / native tools / Pi: **DISABLED / NOT TESTED**.

## Performance observations

CPU-only, qualification-harness timings; not comparable serving benchmarks:

- llama.cpp Q4 full teacher-forced A32: about 44.6 s; BF16 A32: about 14.1 s in a warm-cache run.
- llama.cpp greedy generation of 15 new tokens (including repeated full-prefix dump work): Q4 about 14.5–17.2 s; BF16 about 20.7–27.5 s.
- vBuf Q4 isolated persistent generation of 15 new tokens: seed 0 about 1,717 s; seed 1234 about 1,599 s. This path includes the qualification runtime's materialization and per-layer execution and is not comparable to llama.cpp timing.
- vBuf full teacher-forced 32-token qualification: about 57–71 s in recorded runs.

## Verification

- BF16 model load and 32-position llama.cpp capture: **PASS**.
- Q4/BF16 llama.cpp exact-token capture: A32, B32, C16 **PASS**.
- vBuf Q4 full-sequence output export: A32, B32, C16 **PASS**; selected arrays finite and correctly shaped.
- vBuf Q4 isolated greedy runs: seed 0 and 1234, 16 total tokens **PASS**; seed 42 32-token parity remains prior recorded evidence.
- Pinned-GGML Qwen3 qualification target build: **PASS**.
- Pinned-GGML CTest: **33/33 PASS**.
- Reference dumper compiled with `-Wall -Wextra` without warnings: **PASS**.
- Python comparison-tool syntax and all three result JSON parses/finite-value checks: **PASS**.
- `git diff --check`: **PASS**.

## Artifacts and implementation scope

Model artifacts are local under `${QWEN3_ARTIFACT_DIR}` and are not added to the repository. Raw activation/logit dumps and run logs are under `/tmp/qwen3-*` and are not included. This report and the three compact JSON result summaries preserve the analyzed values. The only code changes are to the isolated Qwen3 reference dumper/qualification tool plus `qwen3_precision_compare.py`; no shared runtime, production attention, format, or session behavior was changed.
