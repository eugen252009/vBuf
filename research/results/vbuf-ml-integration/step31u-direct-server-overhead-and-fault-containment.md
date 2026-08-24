# Step 31U Direct/Server Overhead And Fault Containment

Status: **HOST QUALIFIED; MATCHED SERVER OVERHEAD SMALL; EARLY/MID/LATE
SOURCE-FAILURE CONTAINMENT QUALIFIED**.

Target: Linux x86_64 workstation. This result does not claim Android, ADB,
Orange Pi, physical RISC-V, or external-device evidence.

The qualification compared the direct `VbufGenerationSession::run()` control
with `vbuf_compat_server` using the same semantic model bootstrap, HTTP Range
source, prompt, ten prompt tokens, two blocks, `NormalInference`, 256 MiB
residency capacity, and one generated token. The prompt token hash was
`232a1eb1cb807d1b`; both paths returned the same output bytes
`20446570656e6473` (`" Depends"`). The direct control and server each owned
their own fresh process, but both were warmed against the same source artifact
before measured requests. No llama.cpp loader or llama-server runtime was used.

## Matched Overhead

The run used three independent direct/server repetitions. Each process had two
warm-up requests followed by seven measured requests. The reported medians are
the p50 over 21 measured requests per path. `total_ns` is the process-local
end-to-end interval for the direct control or server request; client-side HTTP
wall time was not used in the subtraction.

| Measurement | Result |
|---|---:|
| direct runtime p50 | `67.620 ms` |
| server request p50 | `67.784 ms` |
| server minus direct | `0.164 ms` |
| relative overhead | `0.242%` |
| classification | **SMALL** |

The cold first warm-up fetched `231130368` bytes and materialized
`62606336` bytes. Subsequent warm-up and measured requests fetched and
materialized zero additional bytes, with `231130368` bytes resident. All
measured requests completed with one generated token and identical prompt hash
and output.

## Fault Matrix

The fault matrix used one persistent server process, a 64 MiB residency cap,
qualification faults enabled, and the request field
`vbuf_source_failure_after_requests`. Each failed request was immediately
followed by a normal request in the same process.

| Stage | Failure threshold | HTTP result | Successful source reads | Completed layers | Completed positions | Follow-up |
|---|---:|---:|---:|---:|---:|---|
| early | `1` | `500` | `1` | `1` | `0` | `200`, one token |
| mid | `30` | `500` | `30` | `1` | `4` | `200`, one token |
| late | `60` | `500` | `60` | `2` | `9` | `200`, one token |

Every injected failure logged `source=controlled-failure`. The failed late
request reached the final-logit boundary and returned the expected
`server_logits failed` error rather than a successful response. Failed and
recovery requests returned zero active leases and zero active inflight bytes;
post-request residency remained below the 64 MiB cap. Recovery reacquired
needed ranges and completed normally. The HTTP Range source has no progressive
coverage cache, so progressive-cache insertion is not applicable to this
qualification; failed materializations remained failed and were reacquired on
the following request.

The qualification fault mechanism is disabled by default and is only enabled
with `--enable-qualification-faults`. Normal serving behavior does not inject
faults.

## Reproduction

The disposable local setup used:

```text
semantic model: /tmp/opencode/DeepSeek-V2-Lite.IQ1_S.semantic.vbuf
source: http://127.0.0.1:18124/models/DeepSeek-V2-Lite.IQ1_S.vbuf
ggml source: /home/eugen/projekte/llama.cpp/ggml
ggml commit: a97123e497968f3440264c0464a7adc7c999c027
Rust library: rust/target/debug/libvbuf_ml.so
```

The canonical ggml pin remains `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`;
that revision was unavailable locally and upstream during this qualification.
The disposable build therefore used the available local checkout without
changing the canonical pin or claiming pinned-ggml equivalence.

The reproducible harness is
`integrations/ggml/tests/vbuf_step31u_qualification_test.py`. Its generated
raw JSON evidence was `/tmp/opencode/vbuf-step31u-qualification-final.json`.
