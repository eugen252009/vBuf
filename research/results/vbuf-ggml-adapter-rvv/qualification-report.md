# Native RV2 vBuf to ggml Adapter Qualification

## Identity

```text
native platform: Orange Pi RV2 / Ky(R) X1 / riscv64
compiler: gcc-14/g++-14 14.2.0
cmake: 3.28.3
native ggml revision: 2d191b5dee1a591c41ee8a653ce42bfcd9c8716d
historical llama reference: 4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
```

The historical llama checkout and preserved benchmark evidence were not
modified. The build used a fresh native CMake directory and the existing
`integrations/ggml` entry point with CUDA disabled and `GGML_CPU_REPACK=OFF`.
Configure selected:

```text
GGML_SYSTEM_ARCH: riscv64
Adding CPU backend variant ggml-cpu: -march=rv64gcv_zfh_zvfh_zicbop_zihintpause
```

## Results

The complete native CTest set passed:

```text
vbuf_ggml_smoke: PASS
vbuf_borrowed_cpu_buffer_probe: PASS
vbuf_tensor_adapter_qualification: PASS
100% tests passed, 0 tests failed out of 3
```

The adapter qualification reported:

```text
descriptors=1 malformed=1 f32=1 q2=1
```

## Q2_K Native Binding

The Q2_K probe executed on the RVV host through the ordinary CPU backend:

```text
Q2_K source=0x2aab76fe40 tensor_data=0x2aab76fe40 type=10 shape=[256,1]
ne=[256,1,1,1] nb=[84,84,84,84]
expected_bytes=84 actual_bytes=84 result=0 reference=0
```

This proves the canonical 256-value/84-byte Q2_K block was bound without a
payload transform. The source pointer and ggml tensor pointer are identical.
The test uses `ggml_backend_tensor_alloc()` over an external CPU buffer and
does not call `ggml_backend_tensor_set()` for the weight payload.

## Ownership And Lifetime

The bounded probe verified:

- source bytes remain unchanged after execution;
- `tensor->data` resolves to the source backing range;
- the ggml CPU wrapper does not free the source allocation;
- the source lease remains held by the adapter object through backend execution;
- malformed and out-of-range storage spans fail closed.

No tensor-sized destination weight allocation is created by the adapter path.

## Initial Native Failure

The first attempt used the host-default GCC 13.3.0 and failed in the pinned
ggml RVV backend with errors such as:

```text
error: unknown type name ‘vfloat16m2_t’
error: implicit declaration of function ‘__riscv_vle16_v_f16m2’
```

This was classified as `BUILD_CONFIGURATION`: GCC 14.2.0 was installed on the
host and is required for this RVV intrinsic path. Reconfiguring from a fresh
directory with explicit `gcc-14`/`g++-14` fixed the issue. No adapter, ggml
revision, vBuf format, CPU_REPACK setting, or payload behavior was changed.

## Architecture Audit

The production adapter contains only representation mapping, shape/block
validation, descriptor derivation, storage binding, and lifetime management.
No model-architecture branching was introduced. The source search found no
DeepSeek, Qwen, Phi, Llama, layer-ID, or expert-ID logic in the adapter
implementation.

## Evidence Files

- `environment.md`
- `vbuf-ml-adapter-rvv-gcc14-configure.log`
- `vbuf-ml-adapter-rvv-gcc14-build.log`
- `vbuf-ml-adapter-rvv-gcc14-ctest.log`
- `vbuf-ml-adapter-rvv-gcc14-adapter.log`
- `vbuf-ml-adapter-rvv-gcc14-borrowed.log`
- `vbuf-ml-adapter-rvv-gcc14-smoke.log`

## Final Status

```text
NATIVE_PLATFORM:
Orange Pi RV2 / Ky(R) X1 / riscv64 / RVV

NATIVE_GGML_REVISION:
2d191b5dee1a591c41ee8a653ce42bfcd9c8716d

RVV_NATIVE_BUILD:
PASS

GGML_SMOKE:
PASS

BORROWED_CPU_BUFFER:
PASS

F32_BORROWED_EXECUTION:
PASS

Q2_K_BORROWED_EXECUTION:
PASS

POINTER_IDENTITY_NO_COPY:
PASS

SOURCE_LIFETIME:
PASS

VBUF_FORMAT_CHANGE_REQUIRED:
NO

ARCHITECTURE_SPECIFIC_ADAPTER_LOGIC:
NO

READY_FOR_REAL_LAYER_POC:
YES
```
