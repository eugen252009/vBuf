# POC15 Complete Real Transformer Block Closure

## Correctness

- Attention normalized-input parity, attention residual parity, FFN input
  parity, FFN normalized-input parity, router logits, routed aggregate, shared
  expert, FFN pre-residual, and final block output all passed at tolerance
  `1e-5`; exact output max absolute errors were `0`.
- `BLOCK_REFERENCE_PARITY_TOKEN_0`: PASS.
- `BLOCK_REFERENCE_PARITY_TOKEN_1`: PASS.
- Attention-to-FFN boundary parity: PASS.
- Full-block state influence without token-0 state: max absolute delta
  `1.44486`, PASS.
- Runtime state bytes: `20,480` after token 0 and `40,960` after token 1.
- Runtime state append/update count: `4`; read count: `96`.

## Routing And Residency

- Token 0 selected IDs: `56,11,50,34,3,47`.
- Token 1 selected IDs: `56,29,11,3,50,26`.
- All selected experts executed: PASS.
- Unselected graphs created/acquired/source reads/materializations: `0/0/0/0`.
- Logical routed-expert bytes: `21,987,328` for eight unique slices.
- Peak active persistent bytes: `1,892,352`.
- Peak active fraction: `0.0638761`.
- Peak resident bytes: `8,363,008`.
- Cold block source reads/bytes: `58 / 48,257,024`.
- Warm block source reads/bytes: `58 / 48,257,024`.
- Warm token 0/token 1 parity: PASS.

The natural 8 MiB residency budget evicts and reloads working-set ranges; warm
execution is not falsely reported as zero-read reuse. Token 1 adds two expert
slices and evicts two token-0-only slices. The backing materializer reload
contract was fixed and covered by `vbuf_materializer_contract`.

## Timing And Preparation

Payload-to-first-consumer instrumentation is valid for attention, router,
routed expert gate, and shared expert representatives. All observed gaps obey
`first_consumer_start_ns >= payload_ready_ns`.

| Representative | Payload ready ns | First consumer ns | Gap ns |
|---|---:|---:|---:|
| `blk.1.attn_q.weight` | 9336148985620 | 9336218954904 | 69969284 |
| `blk.1.ffn_gate_inp.weight` | 9337205684568 | 9337275214721 | 69530153 |
| `blk.1.ffn_gate_exps.weight` | 9337381216490 | 9337451033754 | 69817264 |
| `blk.1.ffn_gate_shexp.weight` | 9340569674876 | 9340640397353 | 70722477 |

Attention and FFN timing records are also retained in `execution.log` as
`attention_timing` and `ffn_timing` events.

- Copied for execution preparation: `0` bytes.
- Repacked: `0` bytes.
- Transcoded: `0` bytes.
- Cross-sublayer last-consumer release: PASS.
- No complete-block weight lease was introduced; the attention residual
  `TensorValue` crosses the sublayer boundary while weight leases are released.

## Failures

- Required attention weight failure: PASS, FFN not executed, block invalid,
  state positions `0`, resources after teardown `0`.
- Selected expert failure: PASS, routed merge/shared final composition not
  executed, block invalid, resources after teardown `0`.
- Invalid state capacity/position: PASS, attention failed closed, FFN not
  executed, block invalid, resources after teardown `0`.

## Guards

- Whole model load: `NO`.
- Whole packed expert tensor load: `NO`.
- Whole block weight buffer: `NO`.
- Model artifact mutated: `NO`.
- Architecture-specific generic runtime logic: `NO`.
- vBuf layout change required: `NO`.
- vBuf format change required: `NO`.
- POC10, POC11, POC12, POC13, and POC14 standalone qualifications remain
  preserved; the shared materializer reload contract is additionally tested.
