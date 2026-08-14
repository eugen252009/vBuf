# Local vBuf-ML research artifacts

The `.gguf` and `.vbuf` files in this directory are ignored local research
artifacts. They are not committed.

Step 20 conversion configuration:

```text
profile:       vbuf-ml-0.1
BaseShift:     3
placement:     LAYER_MAJOR_ROLE_ORDER
integrity:     none
consumer:      llama.cpp 4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
```

| Source | Source SHA-256 | Target | Target SHA-256 | Target size |
|---|---|---|---|---:|
| `Qwen3-0.6B-Q8_0.gguf` | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | `Qwen3-0.6B-Q8_0.vbuf` | `2982cedd0ddc12d762d12ff3426bcc105b2cee1c675ca13bbb5ca4c3cce9a998` | 637,925,504 |
| `Qwen3-0.6B-BF16.gguf` | `65a16246f5814dc0587acadcf0328186b17febf6dcaeb1b13efa9243b551d38e` | `Qwen3-0.6B-BF16.vbuf` | `6ec3db0bb8a26914be7312cc26c3ec0fb6945202659c46b26b4506f4de75a806` | 1,507,825,912 |

The converted files were reopened and validated through the canonical v0.6,
vBuf-ML Bootstrap, ModelMetadata, TensorDirectory, and tokenizer readers.
Tensor payload parity evidence is under
`benchmark-results/vbuf-ml-step20/`.
