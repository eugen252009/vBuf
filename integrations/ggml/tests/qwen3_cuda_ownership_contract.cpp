#include "qwen3_cuda_core.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

using namespace vbuf_ggml;

namespace {
void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

bool throws(const std::function<void()> & action) {
    try { action(); } catch (const std::exception &) { return true; }
    return false;
}

void prepare_decode_scratch(const std::shared_ptr<QwenCudaSessionState> & session) {
    require(ggml_new_tensor_1d(session->graph_context(), GGML_TYPE_F32, 32) != nullptr,
        "session decode graph fixture tensor creation failed");
    require(session->allocate_decode_scratch() != nullptr, "session decode scratch allocation failed");
}

QwenCudaRuntimeConfig test_config() {
    QwenCudaRuntimeConfig config;
    config.prefill_scratch_bytes = 4096;
    config.decode_scratch_bytes = 2048;
    return config;
}

void validated_placement_map() {
    Qwen3Model model;
    model.layer_count = 40;
    for (const char * name : {"token_embd.weight", "output_norm.weight", "output.weight"})
        model.tensors.emplace(name, Qwen3Tensor{});
    const char * suffixes[] = {"attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight", "ffn_norm.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};
    for (uint32_t layer = 0; layer < model.layer_count; ++layer)
        for (const char * suffix : suffixes)
            model.tensors.emplace("blk." + std::to_string(layer) + "." + suffix, Qwen3Tensor{});

    const auto split = QwenCudaPlacement::contiguous_split(0, 1, 26);
    split.validate(model);
    require(split.owner_for_tensor("token_embd.weight") == 0, "embedding placement is incorrect");
    require(split.owner_for_tensor("blk.25.ffn_down.weight") == 0, "block 25 placement is incorrect");
    require(split.owner_for_tensor("blk.26.attn_q.weight") == 1, "block 26 placement is incorrect");
    require(split.owner_for_tensor("output_norm.weight") == 1 && split.owner_for_tensor("output.weight") == 1,
        "final head placement is incorrect");
    require(split.device_ids() == std::vector<uint32_t>({0, 1}), "split device identity list is incorrect");

    const auto single = QwenCudaPlacement::single_device(1);
    single.validate(model);
    require(single.owner_for_tensor("blk.39.ffn_down.weight") == 1, "single-device placement is incorrect");
    auto noncontiguous = split;
    noncontiguous.block_device_ids[10] = 1;
    require(throws([&] { noncontiguous.validate(model); }), "noncontiguous block placement was accepted");
    auto wrong_head = split;
    wrong_head.output_head_device_id = 0;
    require(throws([&] { wrong_head.validate(model); }), "misplaced output head was accepted");
    require(throws([&] { (void) split.owner_for_tensor("blk.bad.attn_q.weight"); }),
        "malformed block tensor name was accepted");
    require(throws([&] { (void) split.owner_for_tensor("unknown.weight"); }),
        "unknown model tensor name was accepted");
}

void runtime_lifecycle_and_sessions() {
    auto runtime = QwenCudaRuntimeState::create_for_testing(test_config());
    require(runtime->backend() != nullptr, "runtime backend was not created");
    require(runtime->embedding() != nullptr, "runtime embedding was not created");
    require(runtime->resident_tensor_count() == 1, "runtime residency is not model-global");
    require(runtime->resident_model_bytes() == 8 * 4 * sizeof(float), "runtime model byte count mismatch");

    std::weak_ptr<QwenCudaRuntimeState> runtime_weak = runtime;
    auto first = runtime->create_session(8);
    require(first->runtime().get() == runtime.get(), "session does not retain its runtime");
    require(first->current_length() == 0 && first->capacity() == 8, "new session logical state mismatch");
    require(first->key_cache(0) != nullptr && first->value_cache(0) != nullptr,
        "session KV tensors were not allocated");
    require(first->prefill_scratch() != nullptr, "session prefill scratch was not allocated");
    prepare_decode_scratch(first);

    first->commit_tokens(3);
    require(first->current_length() == 3, "session length did not advance");
    require(throws([&] { first->commit_tokens(6); }), "session capacity overflow was accepted");
    const uint64_t generation = first->reset_generation();
    ggml_tensor * first_key = first->key_cache(0);
    ggml_backend_buffer_t first_decode_scratch = first->decode_scratch();
    first->reset();
    require(first->current_length() == 0, "reset did not clear logical length");
    require(first->reset_generation() == generation + 1, "reset generation did not advance");
    require(first->key_cache(0) == first_key && first->decode_scratch() == first_decode_scratch,
        "reset unexpectedly replaced persistent session allocations");

    auto second = runtime->create_session(8);
    prepare_decode_scratch(second);
    require(second->key_cache(0) != first->key_cache(0), "sessions alias their K cache");
    require(second->value_cache(0) != first->value_cache(0), "sessions alias their V cache");
    require(second->packed_value_scratch() != first->packed_value_scratch(), "sessions alias packed-V scratch");
    require(second->prefill_scratch() != first->prefill_scratch(), "sessions alias prefill scratch");
    require(second->decode_scratch() != first->decode_scratch(), "sessions alias decode scratch");
    second->commit_tokens(1);
    require(first->current_length() == 0 && second->current_length() == 1,
        "session logical lengths are not isolated");

    first.reset();
    require(runtime->resident_tensor_count() == 1 && runtime->embedding() != nullptr,
        "destroying a session damaged model-global residency");
    second.reset();
    runtime.reset();
    require(runtime_weak.expired(), "runtime remained alive after its final session/runtime owner was released");
}

void sequential_sessions_reuse_runtime() {
    auto runtime = QwenCudaRuntimeState::create_for_testing(test_config());
    const ggml_backend_t backend = runtime->backend();
    ggml_tensor * embedding = runtime->embedding();
    {
        auto a = runtime->create_session(4);
        a->commit_tokens(2);
    }
    require(runtime->backend() == backend && runtime->embedding() == embedding &&
        runtime->resident_tensor_count() == 1, "session A destruction changed runtime resources");
    auto b = runtime->create_session(4);
    prepare_decode_scratch(b);
    require(b->runtime().get() == runtime.get(), "session B did not reuse runtime");
}

void exact_artifact_cuda_smoke(const char * semantic_path, const char * source_url) {
    Qwen3Model admitted;
    open_qwen3_model(semantic_path, source_url, &admitted, true);
    require(admitted.artifact_identity == std::string("sha256:") + QWEN3_14B_Q4_K_M_SHA256,
        "real CUDA smoke artifact SHA-256 mismatch");
    auto runtime = QwenCudaRuntimeState::create(admitted);
    require(runtime->backend() != nullptr && runtime->embedding() != nullptr,
        "real CUDA runtime backend/embedding is missing");
    require(runtime->resident_tensor_count() == admitted.count,
        "real CUDA runtime did not retain every admitted tensor");
    auto session = runtime->create_session(40);
    require(ggml_new_tensor_1d(session->graph_context(), GGML_TYPE_F32, 32) != nullptr,
        "real CUDA decode-scratch smoke tensor creation failed");
    require(session->allocate_decode_scratch() != nullptr, "real CUDA decode scratch allocation failed");
    require(session->key_cache(0) != nullptr && session->value_cache(39) != nullptr &&
        session->prefill_scratch() != nullptr,
        "real CUDA session allocations are incomplete");
    std::printf("qwen3_cuda_real_smoke=PASS artifact=%s backend=%s resident_tensors=%zu resident_bytes=%llu "
        "embedding=RESIDENT capacity=%u kv=persistent scratch=ALLOCATED inference=NOT_RUN\n",
        runtime->artifact_identity().c_str(), runtime->backend_name().c_str(), runtime->resident_tensor_count(),
        static_cast<unsigned long long>(runtime->resident_model_bytes()), session->capacity());
    session.reset();
    require(runtime->resident_tensor_count() == admitted.count,
        "real CUDA session destruction released model residency");
    runtime.reset();
    for (const auto & entry : admitted.tensors)
        require(entry.second.ggml == nullptr, "real CUDA runtime teardown left a dangling model tensor binding");
}

void partial_creation_failures() {
    for (QwenCudaFailurePoint point : {QwenCudaFailurePoint::RuntimeAfterBackend,
             QwenCudaFailurePoint::RuntimeAfterModelAllocation,
             QwenCudaFailurePoint::RuntimeAfterResidency}) {
        QwenCudaRuntimeConfig config = test_config();
        config.inject_failure = point;
        require(throws([&] { (void)QwenCudaRuntimeState::create_for_testing(config); }),
            "runtime failure injection did not throw");
    }
    auto runtime = QwenCudaRuntimeState::create_for_testing(test_config());
    for (QwenCudaFailurePoint point : {QwenCudaFailurePoint::SessionAfterKvAllocation,
             QwenCudaFailurePoint::SessionAfterPrefillScratch}) {
        require(throws([&] { (void)runtime->create_session(4, point); }),
            "session failure injection did not throw");
    }
    auto failing_session = runtime->create_session(4, QwenCudaFailurePoint::SessionAfterDecodeScratch);
    require(ggml_new_tensor_1d(failing_session->graph_context(), GGML_TYPE_F32, 32) != nullptr,
        "failure-injection decode graph fixture tensor creation failed");
    require(throws([&] { (void)failing_session->allocate_decode_scratch(); }),
        "decode scratch failure injection did not throw");
    failing_session.reset();
    require(runtime->backend() != nullptr && runtime->resident_tensor_count() == 1,
        "partial session failure damaged runtime resources");
    require(throws([&] { (void)runtime->create_session(0); }), "zero-capacity session was accepted");
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 3) {
            exact_artifact_cuda_smoke(argv[1], argv[2]);
            return 0;
        }
        if (argc != 1) throw std::invalid_argument("usage: qwen3_cuda_ownership_contract [SEMANTIC_ARTIFACT SOURCE_URL]");
        validated_placement_map();
        runtime_lifecycle_and_sessions();
        sequential_sessions_reuse_runtime();
        partial_creation_failures();
        std::puts("qwen3_cuda_ownership_contract=PASS runtime_lifetime=session_retained model_resources=shared session_KV_scratch=isolated reset=logical_only partial_failure_cleanup=RAII");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "qwen3_cuda_ownership_contract=FAIL: %s\n", error.what());
        return 1;
    }
}
