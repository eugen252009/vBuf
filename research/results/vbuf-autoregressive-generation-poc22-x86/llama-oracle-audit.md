# POC22 Pinned llama.cpp/GGUF Oracle Audit

## Reproduction

The canonical repository harness is `scripts/qualify_step21_runtime.py`. Its
default external checkout is `/tmp/llama.cpp-step21`; it validates that checkout
at commit `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`, compiles
`integrations/llama.cpp/step21_qualification.cpp`, and links the pinned llama
and ggml libraries. Reproduction before any change failed at the first git
lookup because that checkout did not exist. No model load, decode, logits
extraction, or recurrence stage was reached. The captured exit code was `1`.

The pinned checkout was then restored in `/tmp` at the exact commit. A fresh
checkout confirms the pin is obtainable, but the repository's documented patch
stack is not a self-contained fresh-checkout command: applying patch 0001 first
then patch 0002 fails on overlapping `llama-model-loader.cpp` context. The
existing script assumes an already prepared external worktree and only checks
patch 0001 in reverse. This is a second setup reproducibility gap, not evidence
of a DeepSeek recurrence defect.

## Reproducible Preparation Result

The canonical consolidated patch applies cleanly to two independent fresh
worktrees at the exact pin. Both prepared source deltas hash to
`b365448a51b9e2801d8b86269317e9396a975f19c9562d9534ed34de25e7cb38`. The pinned
`llama` target builds. The existing Qwen3 BF16/Q8_0 single-position, logits, and
autoregressive regressions pass unchanged.

The durable preparation is now:

```text
fresh checkout at 4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
  -> scripts/prepare_step21_llama.py --upstream-root <checkout>
  -> canonical consolidated patch 0000-pinned-step21-canonical.patch
  -> verify source-delta SHA-256
```

The qualification harness also required two setup corrections: the pinned
`src` include directory must be present when compiling the adapter, and
`vbuf_direct_source.cpp` must be linked because `llama_vbuf_loader.cpp` contains
the direct-source symbols even for the compatibility qualification.

## Existing Capability

| Capability | Existing support | Evidence |
|---|---|---|
| Model load | PASS for the pinned Step 21 consumer path | `scripts/qualify_step21_runtime.py:18-22`, `benchmark-results/vbuf-ml-step21/adapter-provenance.json` |
| Token input | PASS for the qualified Qwen3 artifacts | `step21_qualification.cpp:14-18, 21-24` |
| Single-position decode | PASS | `step21_qualification.cpp:22-24` |
| Logits extraction | PASS | `step21_qualification.cpp:27-32` |
| Raw greedy argmax | PASS; direct `>` scan over logits | `step21_qualification.cpp:24` |
| KV/state preservation | PASS in the generation helper; one context is reused | `step21_qualification.cpp:20-25` |
| Multi-position decode | PASS for the existing Qwen3 generation qualification | `benchmark-results/vbuf-ml-step21/generation-parity.json` |
| Generated-token feedback | PASS for that helper; argmax is assigned to `one[0]` before the next decode | `step21_qualification.cpp:24` |
| DeepSeek POC22 GGUF sequence | PARTIAL | direct-token mode now loads and decodes two positions; sequence diverges at position 0 |

The earlier Step 21 `generation parity` result is therefore a true llama-owned
autoregressive loop for Qwen3 BF16/Q8_0, not a one-shot or fixed-token fixture.
It does not by itself qualify the DeepSeek four-position sequence required by
POC22.

## API Contract

At the pinned source revision, the existing helper uses:

```text
llama_model_load / llama_model_load_vbuf
llama_init_from_model
llama_batch_get_one
llama_decode
llama_get_logits_ith
llama_vocab_n_tokens
llama_free / llama_model_free[_vbuf]
```

The helper decodes the initial token vector, reads the active logits row, uses
raw argmax, creates a one-token batch for the generated token, and calls
`llama_decode` on the same context. The llama context therefore owns the KV
cache across positions. The helper does not explicitly populate `pos` or `seq_id`
fields because `llama_batch_get_one` supplies the standard single-sequence
position progression at this pin.

## DeepSeek Result

The direct-token mode loads the DeepSeek GGUF at the pinned revision and performs
two decode positions on one persistent context. It feeds the generated token
back into the next batch:

```text
token 0 -> 19304
token 19304 -> 19304
```

This diverges at position 0 from the existing native evidence's expected
`86711`. The four-position run was stopped after that first divergence. This is
a new semantic oracle mismatch, not a preparation failure.

The diagnostic vBuf consumer path cannot resolve the mismatch: its compatibility
metadata projection identifies the DeepSeek artifact as `qwen3` and fails with
`wrong number of tensors; expected 377, got 300`. That path is not the
independent oracle and was not used as one.

## Position and Fixture

The vBuf-native POC22 uses position `0` for seed token `0` and increments one
position per generated token. The existing llama helper starts with a prompt
token vector and lets the same context advance positions; a direct token-ID
DeepSeek mode has not yet been added. The DeepSeek artifact reports vocabulary
size `102400` through the pinned GGUF load, so IDs `0` and `86711` are in the
vocabulary domain.

## Classification

## CPU_REPACK Contract Capture

The pinned x86 llama run now captures the block-0 `blk.0.ffn_down.weight`
contract without executing a vBuf repacked intervention. The selected buffer is
`CPU_REPACK`, with 32-byte alignment and a requested allocation of
`2711642112` bytes. The logical public descriptor remains IQ4_NL with
`ne=[10944,2048,1,1]` and `nb=[18,6156,12607488,12607488]`, while the
CPU-specific execution trait is `iq4_nl_8x8_q8_0`.

The captured packed storage is `12607488` bytes at data offset zero, with
storage hash `3d8b9543543e6e57`; the source payload hash is
`0c7bdf85b162206b`. The first block-0 down-projection GEMV dispatch is
`n=10944, nr=1, nc=256, bs=2048`. Its kernel-side Q8_0 operand is
`11628` bytes with hash `d2fd48314bc644ab`, derived from the captured F32
SwiGLU input hash `c6ee5d5d36e9c861`.

This establishes the llama-side storage, descriptor, dispatch, and Q8 input
facts needed for a future comparison. It does not establish vBuf parity: the
vBuf public descriptor remains raw IQ4_NL-shaped, and no new vBuf repacked
matmul was run under the current task constraint.

```text
BLOCKER_CLASS: N (model/oracle semantic mismatch), after resolving the M/L setup blocker
PIN_UPDATE_REQUIRED: NO
FIX_SCOPE: ENVIRONMENT_ONLY / ORACLE_HARNESS_ONLY
DEEPSEEK_MULTI_POSITION_SUPPORTED_AT_PIN: PARTIAL; two positions execute, sequence diverges at position 0
ORACLE_STATE_PERSISTS_ACROSS_POSITIONS: YES
ORACLE_GENERATED_TOKEN_FEEDBACK: YES
ORACLE_INDEPENDENT_FROM_VBUF_EXECUTION: YES by design
```

## Smallest Safe Workaround

The reproducibility workaround is complete: use the canonical preparation
script and consolidated patch. The remaining POC22 work is now a model/oracle
semantic investigation: explain why pinned GGUF logits for direct token `0`
produce `19304` while the independent native path produces `86711`. No vBuf
runtime, cache policy, ggml pin, llama pin, or `model-is-working` tag needs to
change for that investigation.
