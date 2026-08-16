# vBuf Region Chain POC2: x86 Qualification

## Scope

This qualification executes the layer-0 dense FFN as two sequential
`ExecutionRegion` calls:

- Region A acquires `ffn_norm`, `ffn_gate`, and `ffn_up`, then executes
  RMSNorm, both matmuls, and SwiGLU.
- Region B acquires `ffn_down` and consumes the owned Region A activation.
- Region A's release callback occurs before Region B's acquisition callback.

The production path uses the pinned direct ggml substrate at commit
`2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`, with CUDA disabled and
`CPU_REPACK=OFF`.

## Result

PASS on x86-64.

The complete runtime trace is in `region-poc2.log`:

```text
region=A phase=released active_weight_bytes=0 active_weight_leases=0
region=B phase=weights_acquired active_weight_bytes=12607488 active_weight_leases=1
```

Region A acquired 8,763,392 bytes across 3 weights. Region B acquired
12,607,488 bytes across 1 weight. The combined selected and acquired weight
bytes were 21,370,880, with 6 graph nodes across both regions.

Intermediate activation parity against `ffn_swiglu-0`:

- elements: 21,888
- max absolute error: `4.47035e-08`
- mean absolute error: `7.10122e-10`

Final output parity against `ffn_out-0`:

- elements: 4,096
- max absolute error: `1.78814e-07`
- max relative error: `1.96622e-03`
- mean absolute error: `1.39676e-08`

The model source mapping remained ~4.99 GB while the region weight leases
were bounded to the selected region weights. RSS was 28,096 KiB before
execution, 50,720 KiB after execution, and 7,440 KiB after release.

The existing substrate CTest suite also passed: 3/3 tests.

## RV2 Status

The same source was sent to the RV2 qualification workspace. Rebuilding the
remote ggml CPU backend failed before the POC2 tool compiled because the
remote compiler headers do not provide the RVV FP16 intrinsic types and
functions used by the pinned ggml CPU backend, including `vfloat16m2_t` and
`__riscv_vle16_v_f16m2`. This is an environment/toolchain blocker; no RV2
POC2 result is claimed.

## Artifacts

- `metadata.txt`: reference capture metadata
- `ffn_inp.f32`: input activation
- `ffn_swiglu.f32`: architecture-specific Region A reference
- `ffn_out.f32`: final reference output
- `region_a_activation.f32`: production Region A activation
- `region_output.f32`: production Region B output
- `region-poc2.log`: lifecycle, memory, and parity trace
