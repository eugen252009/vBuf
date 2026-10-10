# Post-stop native-AV follow-up (2026-10-10)

These are separate diagnostic runs after the original 20-fixture matrix stopped on its repeated common-token numeric-gate failure. They do not amend that matrix, change its `NOT_RUN_AFTER_STOP` rows, or qualify/promote native AV. The canonical packed-V path remains authoritative and the candidate remains `Candidate_NOT_Valid`.

## Fixed setup

- Qwen3-14B Q4_K_M source payload: SHA-256 `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`; semantic artifact SHA-256 `cc20b816a4cfdd192d4870b853354c51dd1b1402b83d01fac1e8ea98f2a9fea7`.
- GGML commit `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`; devices and placement remained RTX 3060 / RTX 2080 SUPER, `multi:0x26,1x14;emb=0;norm=1;head=1`.
- CUDA build used `CMAKE_CUDA_ARCHITECTURES=75;86`, `VBUF_ENABLE_CUDA=ON`, and `VBUF_ENABLE_UNVALIDATED_QUALIFICATION_TRIALS=ON` for this qualification-only executable. The runner explicitly reports `production_selection=disabled; candidate=unvalidated_trial`. No production path, default, optimizer mode, admission rule, guard, threshold, or numerical-contract scope changed.
- The active final-output contract v2 is scoped to capacities 512–32768. Policy digest: `1b8349e77a9494103260856e2a19a392152c16b51bfbaad2b7b750242817c067`.

Per-run environment, hardware, executable hashes, range-server request logs, runner logs, CSV evidence, and exit statuses are kept beside each run. Exact one-row manifests are under `manifests/`.

## Results

### In-scope 32-row natural prompt

`natural-repeat-p32-g1-cap512/` ran the planned 32-token natural-repeat prompt plus one generated token twice at capacity 512. Both repetitions passed the unchanged gate and repeated bitwise capture/token hashes. Canonical and candidate selected the same token (`504`; token hash `914c4dfbfdd42e07`); the candidate selected/executed two native steps (52 eligible layer graphs). Final hidden relative RMS/cosine: `0.01281547` / `0.99992673`; logits: `0.01796165` / `0.99984705`. This reproduces the previously observed guarded prefill-plus-decode endpoint pass; it does not negate the separate prefill-only/common-history failures.

### In-scope synthetic repeated-token prompt

`repeated-token-id-p8-g25-cap512/` used eight copies of valid token ID 785 (explicitly synthetic, not natural language), then 25 generated tokens. Both repetitions passed the unchanged gate with repeatable captures and identical canonical/candidate token sequence hash `060de772e20d22a2`. All 33 candidate steps selected native AV on 26 eligible layers (858 native layer-steps). Final hidden relative RMS/cosine: `0.00451797` / `0.99999032`; logits: `0.01041327` / `0.99994585`. This is one synthetic trajectory, not broad prompt qualification.

### Capacity-64 scope handling

`short-natural-reviewer-p1-g2-cap64/` measured the one-token natural prefix plus two generated tokens twice. Tokens/captures repeated; session cancellation/re-entry also passed. Final metrics were hidden `0.00642701` / `0.99998113` and logits `0.00673264` / `0.99998790`. However, the active final-output contract is not applicable below capacity 512. These values are diagnostic only and are **not** a formal gate pass or failure.

The first run in that directory used the previous runner and emitted `NUMERIC_GATE_FAIL` because it treated contract `NOT_APPLICABLE` as a failed gate. Its raw output is preserved. The qualification-only runner was corrected to distinguish `NUMERIC_NOT_APPLICABLE` from an actual `NUMERIC_GATE_FAIL`, keep the CSV `numeric_gate` field as `NOT_APPLICABLE`, continue diagnostic collection, and finish with `COMPLETED_WITH_NOT_APPLICABLE`. `contract-scope-control/` reran this capacity-64 case followed by the in-scope capacity-512 case: the former is explicitly not applicable, the latter passes, and the final matrix status remains `Candidate_NOT_Valid`. No contract scope or tolerance was extended.

## Reproduction

The source fixture rows are copied verbatim (with provenance comments) from `qwen3-native-av-fixture-matrix-v1.tsv`. Configure/build the qualification executable with the options recorded above, then for each manifest run:

```sh
python3 scripts/range_server.py \
  --file /home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.vbuf \
  --host 127.0.0.1 --port 18786 --log

CUDA_VISIBLE_DEVICES=0,1 \
  /tmp/vbuf-qwen-native-matrix-build/vbuf_qwen3_native_av_sequence_qualification \
  /home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf \
  http://127.0.0.1:18786 --matrix <manifest.tsv> <new-output-directory>
```

The range server was stopped after each recorded run. The original external GGML checkout’s pre-existing modification was left untouched; the build preparation script materialized the pinned commit with the repository’s native-AV patch into `/tmp`.

## Qualification boundary

The original matrix remains stopped at its first new in-scope common-token failure, and its other 19 rows remain unrun in the original evidence. The follow-ups add two repeatable passes within the existing capacity-512 scope plus one capacity-64 diagnostic outside policy scope. They do not resolve the prefill-only and common-token failures, do not authorize thresholds or candidate admission, and do not establish performance.
