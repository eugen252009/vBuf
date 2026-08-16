# Region POC 1 Qualification

## Selected Region

The selected region is the dense leading-layer FFN subregion of the qualified
DeepSeek-V2-Lite artifact:

```text
boundary input:  ffn_inp-0       [2048, 2] F32
boundary output: ffn_out-0       [2048, 2] F32
```

This is a dense region from layer 0. Layer 1 and later MoE data are not part of
the region. The full-model reference was used only to capture the boundary
activation pair and validate the lowering recipe.

## Generic Region Contract

The production region description contains only:

- numeric tensor IDs and transient vBuf tensor views;
- input, temporary, and output slots;
- generic operation kinds;
- operation slot dependencies;
- a storage-provider callback for borrowed ranges.

The operation sequence is:

```text
RMS_NORM(input, ffn_norm.weight, 1e-6)
MUL_MAT(ffn_gate.weight, normalized)
MUL_MAT(ffn_up.weight, normalized)
SWIGLU_SPLIT(gate_output, up_output)
MUL_MAT(ffn_down.weight, swiglu_output)
```

The architecture-specific source recipe was used only by the reference capture
tool. The production `ExecutionRegion` contains no model-family or tensor-name
branching.

## Tensor Acquisition

Exactly four vBuf weight views were requested and acquired:

| Tensor role in lowering | Representation | Shape | Bytes |
|---|---|---:|---:|
| norm | F32 | `[2048]` | 8,192 |
| gate | IQ1_S | `[2048,10944]` | 4,377,600 |
| up | IQ1_S | `[2048,10944]` | 4,377,600 |
| down | IQ4_NL | `[10944,2048]` | 12,607,488 |
| **Total** |  |  | **21,370,880** |

```text
requested_tensor_count=4
acquired_tensor_count=4
selected_weight_bytes=21370880
acquired_weight_bytes=21370880
```

No other layer weight descriptor or backend binding was created.

## Reference Capture

The reference-only tool used the historical llama.cpp revision
`4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`, loaded the real vBuf artifact, and
captured `ffn_inp-0` and `ffn_out-0` around the leading dense FFN. It used the
prompt `Hello world`, producing two activation columns. Capture metadata and
raw F32 files are preserved beside this report.

The reference lowering recipe follows the pinned generic `build_ffn` behavior:
parallel gate/up projections, SwiGLU split, and down projection. RMSNorm is
included at the region input boundary using the qualified epsilon `1e-6`.

## Execution And Parity

Native target:

```text
Orange Pi RV2 / Ky(R) X1 / riscv64 / RVV
gcc-14/g++-14 14.2.0
ggml 2d191b5dee1a591c41ee8a653ce42bfcd9c8716d
CPU_REPACK=OFF
CUDA=OFF
```

Runtime output:

```text
output_elements=4096
max_abs=0
max_rel=0
mean_abs=0
actual_hash=00edbc8c56555043
reference_hash=00edbc8c56555043
output_shape=[2048,2]
graph_nodes=6
```

The generic graph therefore matched the reference output bit-for-bit on the
native RVV machine.

## Memory And Lifetime

The POC mapped the full vBuf source through the existing validated Rust
consumer, but acquired only the four selected payload ranges through the
region adapter. The source file size was 4,993,331,814 bytes; this is a mapped
address range, not a claim of resident pages.

```text
rss_before_kib=26432
rss_after_execute_kib=48780
rss_after_release_kib=6044
anonymous_before_kib=1400
anonymous_after_execute_kib=1992
anonymous_after_release_kib=1732
private_clean_before_kib=23576
shared_clean_before_kib=1028
```

The region graph synchronizes before releasing its compute allocation and
borrowed tensor wrappers. The source lease remains alive through execution and
is released only after the region provider is destroyed. The source bytes are
never copied into a full-model weight destination.

## Platform Results

The exact source was built and run on both platforms. The x86 run used a
temporary copy of the existing 4.7 GiB artifact under `/tmp`; no model artifact
was added to the repository.

The x86 CTest set passed and the real region output met the explicit tolerance:

```text
max_abs=1.78814e-07
max_rel=0.00196622
mean_abs=1.39676e-08
actual_hash=4a6526893cc2f652
reference_hash=00edbc8c56555043
```

The x86 hash differs from the reference/RVV hash because CPU floating-point
ordering differs. The absolute error is small and the relative threshold used
by the POC is `2e-3`; the result passed without weakening that threshold.
The RVV run matched the reference bit-for-bit.

## Evidence Files

- `metadata.txt`
- `ffn_inp.f32`
- `ffn_out.f32`
- `region_output.f32`
- `region-poc1.log`
- `vbuf-ml-adapter-rvv-gcc14-region-configure.log`
- `vbuf-ml-adapter-rvv-gcc14-region-build.log`
- `vbuf-ml-adapter-rvv-gcc14-region-ctest.log`
- `reference-capture.log`
- `x86-configure.log`
- `x86-build.log`
- `x86-ctest.log`
- `x86-region-poc1.log`
- `x86-region_output.f32`

## Final Status

```text
REAL_REGION_EXECUTION:
PASS

REFERENCE_ACTIVATION_PARITY:
PASS

ARCHITECTURE_SPECIFIC_RUNTIME_LOGIC:
NO

FULL_MODEL_WEIGHT_ALLOCATION:
NO

OUT_OF_REGION_WEIGHT_ACQUISITION:
NO

CPU_BORROWED_VBUF_WEIGHTS:
PASS

VBUF_FORMAT_CHANGE_REQUIRED:
NO

READY_FOR_REGION_CHAIN_POC:
YES
```
