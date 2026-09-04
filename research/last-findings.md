# Last Findings

> Living snapshot: keep only the latest 2–3 highest-value findings here. When a
> larger achievement is qualified, replace an older item instead of appending.
> Detailed evidence remains in the linked research reports.

- **112G GLM-4.5-Air-FP8 model runs on a workstation with 64 GiB RAM and 20 GiB aggregate GPU VRAM.** The vBuf-ML portable CPU runtime imported the model directly from remote Safetensors ranges, executed all 46 layers, retained KV state, and completed 8-token greedy generation with independent token, routing, argmax, and top-10 parity. Peak converted working set was approximately **1.06 GiB**; no complete source copy was created. **Edge:** this is CPU F32 qualification, not full-stack GPU, GGML, or throughput qualification. ([Step 32C](results/vbuf-ml-integration/step32c-real-fp8-oversubscription.md), [Step 32J](results/vbuf-ml-integration/step32j-repeated-autoregressive-generation.md))

- **The same 112G model executes consecutive real layers on an RTX 3060 without full-model or full-expert-bank preload.** The progressive 1/2/4/8-layer CUDA gates passed with selected-expert-only transfers, zero unselected-expert bytes, direct device activation handoff, and a bounded approximately **203 MiB logical device residency**. **Edge:** full 46-layer CUDA prefill, GPU decode/generation, native FP8 device execution, and multi-GPU execution remain unqualified. ([Step 32K-B](results/vbuf-ml-integration/step32k-b-progressive-cuda-layers.md))

- **A 5.6 GB DeepSeek-V2-Lite IQ2_XXS model runs on a Pixel 7 Pro (Android 17, arm64-v8a) with a 256 MiB vBuf-ML residency cap.** Remote bounded-range materialization, real execution, prompt prefill, and bounded generation were qualified without loading the full payload into the cap. **Edge:** the Pixel qualification is ARM64, not ARM32; ARM32 has format/runtime evidence but is not qualified for this DeepSeek model-compute path. ([Android qualification](results/vbuf-android-demo-poc/step31i-residency-curve-after-geometry-fix.md), [current state](results/vbuf-ml-current-state-summary.md))
