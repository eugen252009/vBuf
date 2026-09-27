#ifndef VBUF_POC14_INCLUDED
#define VBUF_POC14_INCLUDED
#define VBUF_POC12_LIBRARY_ONLY
#include "multi_expert_moe_poc12.cpp"
#undef VBUF_POC12_LIBRARY_ONLY

#include "vbuf_runtime_state.h"
#include "deepseek_v2_lite_semantics.h"
#include "ggml-cpu.h"

#include <chrono>

namespace {

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct AttentionTensors {
    Meta norm;
    Meta q;
    Meta kv_a;
    Meta kv_a_norm;
    Meta kv_b;
    Meta output;
};

struct TokenData {
    std::vector<float> normalized;
    std::vector<float> q_nope;
    std::vector<float> q_pe;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> context;
    std::vector<float> output;
};

class FailingSource final : public RangeSource {
public:
    FailingSource(std::shared_ptr<RangeSource> delegate, uint64_t offset)
        : delegate_(std::move(delegate)), offset_(offset) {}
    bool read_range(uint64_t offset, uint64_t length, uint8_t * destination,
        RangeReadResult * result) override {
        if (offset == offset_) {
            if (result != nullptr) {
                result->requested_offset = offset;
                result->requested_length = length;
                result->source_id = "controlled-attention-failure";
                result->error = "controlled attention tensor failure";
            }
            return false;
        }
        return delegate_->read_range(offset, length, destination, result);
    }
private:
    std::shared_ptr<RangeSource> delegate_;
    uint64_t offset_;
};

struct OpResult {
    AdapterError error = AdapterError::None;
    std::vector<float> values;
};

AttentionTensors attention_tensors(const Metadata & metadata) {
    return {
        lookup(metadata, "blk.1.attn_norm.weight"),
        lookup(metadata, "blk.1.attn_q.weight"),
        lookup(metadata, "blk.1.attn_kv_a_mqa.weight"),
        lookup(metadata, "blk.1.attn_kv_a_norm.weight"),
        lookup(metadata, "blk.1.attn_kv_b.weight"),
        lookup(metadata, "blk.1.attn_output.weight") };
}

PersistentTensorRef persistent_ref(const Meta & meta) {
    const VbufTensorView view{ meta.view.representation, meta.view.rank,
        meta.view.dimensions, meta.view.payload, meta.view.payload_len };
    return { meta.id, std::string(meta.view.name, meta.view.name_len), view, meta.offset };
}

OpResult run_op(const Meta & meta, const Activation & input, TensorWaveOpKind kind,
    const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const char * label, float parameter = 0.0f, bool no_jit_fallback = false,
    RuntimeTiming * timing = nullptr,
    AttributionStage attribution_stage = AttributionStage::Attention) {
    RouterGraph graph;
    graph.executor = std::make_unique<TensorDependencyExecutor>();
    const uint32_t input_value = graph.executor->add_input("attention_input");
    graph.router = graph.executor->add_persistent(persistent_ref(meta));
    graph.output = graph.executor->add_value(label, true);
    graph.executor->set_external_output(graph.output);
    if (kind == TensorWaveOpKind::RmsNorm) {
        graph.executor->add_operation({ label, kind,
            { { TensorWaveRef::Kind::Value, input_value },
              { TensorWaveRef::Kind::Persistent, graph.router } }, graph.output, parameter });
    } else {
        graph.executor->add_operation({ label, kind,
            { { TensorWaveRef::Kind::Persistent, graph.router },
              { TensorWaveRef::Kind::Value, input_value } }, graph.output, parameter });
    }
    std::unique_ptr<OffsetMaterializer> offset;
    TensorMaterializer * active_materializer = nullptr;
    if (materializer != nullptr) {
        // Keep attention graph-local references disjoint from the FFN expert slice namespace.
        offset = std::make_unique<OffsetMaterializer>(materializer, 1000U + static_cast<uint32_t>(meta.id * 10));
        offset->request(graph.router, persistent_ref(meta), meta.view.payload_len);
        active_materializer = offset.get();
    }
    uint64_t first_consumer_start = 0;
    uint64_t storage_calls = 0;
    std::string detail;
    std::vector<int64_t> output_shape;
    const auto provider = [lease, &storage_calls, no_jit_fallback](const VbufTensorView & view) {
        ++storage_calls;
        if (no_jit_fallback) return VbufBorrowedStorage{};
        const uintptr_t address = reinterpret_cast<uintptr_t>(view.payload);
        const uintptr_t base = address & ~static_cast<uintptr_t>(63);
        return VbufBorrowedStorage{ reinterpret_cast<const uint8_t *>(base),
            static_cast<uint64_t>(address - base) + view.payload_len,
            static_cast<uint64_t>(address - base), lease };
    };
    AttributionObserverState attribution_state;
    const TensorWaveExecutionObserver attribution_observer =
        make_attribution_observer(timing, attribution_stage, &attribution_state);
    const RunResult result = [&]() {
        RunResult timed;
        timed.error = graph.executor->execute(input.view(), provider, &timed.output, &output_shape,
            &timed.report, &detail, {}, active_materializer,
            [&](const char * operation, const char * phase) {
                if (std::string(phase) == "start" && first_consumer_start == 0) first_consumer_start = now_ns();
                attribution_observer(operation, phase);
            });
        if (timed.error != AdapterError::None) std::fprintf(stderr, "attention_detail=%s\n", detail.c_str());
        return timed;
    }();
    if (materializer != nullptr) {
        for (const auto & event : materializer->trace()) {
            if (event.event == "STATE" && event.state == MaterializationState::Ready &&
                event.tensor_name == std::string(meta.view.name, meta.view.name_len)) {
                std::printf("attention_timing tensor=%s payload_ready_ns=%llu first_consumer_start_ns=%llu "
                    "payload_to_first_consumer_ns=%llu\n", event.tensor_name.c_str(),
                    static_cast<unsigned long long>(event.timestamp_ns),
                    static_cast<unsigned long long>(first_consumer_start),
                    static_cast<unsigned long long>(first_consumer_start >= event.timestamp_ns
                        ? first_consumer_start - event.timestamp_ns : 0));
                break;
            }
        }
    }
    return { result.error, floats(result.output) };
}

std::vector<float> attention_context(const std::vector<float> & q_nope,
    const std::vector<float> & q_pe, const RuntimeStateSlot & k_state,
    const RuntimeStateSlot & v_state, float scale, std::vector<float> * probabilities) {
    constexpr uint32_t heads = 16, nope = 128, rope = 64, value = 128;
    const uint32_t positions = k_state.size();
    if (positions == 0 || positions > 4096 || positions != v_state.size() ||
        k_state.width() != heads * (nope + rope) || v_state.width() != heads * value ||
        q_nope.size() != heads * nope || q_pe.size() != heads * rope)
        throw std::runtime_error("invalid attention state geometry");
    // Use the backend's F32 reductions for QK, softmax, and weighted V. A
    // scalar float accumulation changes subsequent quantized-matmul inputs
    // at rounding boundaries, amplifying tiny errors across the model.
    // Backend-local zero padding fixes the reduction geometry independently
    // of the live KV length. It does not append positions to runtime KV state.
    const uint32_t padded_positions = ((positions + 255) / 256) * 256;
    const size_t elements = size_t(heads) * ((nope + rope + value + 2) * padded_positions + nope + rope + value);
    ggml_init_params params{elements * sizeof(float) + 1024 * 1024, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    if (!ctx) throw std::runtime_error("attention compute allocation failed");
    auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, nope + rope, 1, heads);
    auto * k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, nope + rope, padded_positions, heads);
    auto * v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, padded_positions, value, heads);
    auto * mask = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, padded_positions);
    ggml_set_zero(k);
    ggml_set_zero(v);
    auto * mask_data = static_cast<float *>(mask->data);
    for (uint32_t i = 0; i < padded_positions; ++i)
        mask_data[i] = i < positions ? 0.0f : -INFINITY;
    auto * q_data = static_cast<float *>(q->data);
    auto * k_data = static_cast<float *>(k->data);
    auto * v_data = static_cast<float *>(v->data);
    for (uint32_t head = 0; head < heads; ++head) {
        std::copy_n(q_nope.data() + head * nope, nope, q_data + head * (nope + rope));
        std::copy_n(q_pe.data() + head * rope, rope, q_data + head * (nope + rope) + nope);
    }
    for (uint32_t position = 0; position < positions; ++position) {
        std::vector<float> key, values;
        k_state.read(position, &key);
        v_state.read(position, &values);
        for (uint32_t head = 0; head < heads; ++head) {
            std::copy_n(key.data() + head * (nope + rope), nope + rope,
                k_data + (head * padded_positions + position) * (nope + rope));
            for (uint32_t i = 0; i < value; ++i)
                v_data[(head * value + i) * padded_positions + position] = values[head * value + i];
        }
    }
    auto * scores = ggml_mul_mat(ctx.get(), k, q);
    auto * probs = ggml_soft_max_ext(ctx.get(), scores, mask, scale, 0.0f);
    auto * result = ggml_mul_mat(ctx.get(), v, probs);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, result);
    if (ggml_graph_compute_with_ctx(ctx.get(), graph, 2) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("attention compute failed");
    const auto * probability_data = static_cast<const float *>(probs->data);
    probabilities->clear();
    for (uint32_t head = 0; head < heads; ++head)
        probabilities->insert(probabilities->end(), probability_data + head * padded_positions,
            probability_data + head * padded_positions + positions);
    const auto * output = static_cast<const float *>(result->data);
    return {output, output + heads * value};
}

TokenData compute_token(const AttentionTensors & tensors, const Activation & input,
    uint32_t position, RuntimeStateSlot * k_state, RuntimeStateSlot * v_state,
    const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const char * label, bool no_jit_fallback = false, RuntimeTiming * timing = nullptr) {
    constexpr float epsilon = 1e-6f;
    constexpr uint32_t heads = 16, q_head = 192, nope = 128, rope = 64, value = 128;
    TokenData data;
    const OpResult norm = run_op(tensors.norm, input, TensorWaveOpKind::RmsNorm, lease,
        materializer, "attention_rms_norm", epsilon, no_jit_fallback, timing);
    if (norm.error != AdapterError::None) return {};
    data.normalized = norm.values;
    const Activation normalized{ norm.values, { 2048, 1 } };
    const OpResult q_result = run_op(tensors.q, normalized, TensorWaveOpKind::MulMat, lease,
        materializer, "attention_q", 0.0f, no_jit_fallback, timing);
    if (q_result.error != AdapterError::None) return {};
    const OpResult kv_a_result = run_op(tensors.kv_a, normalized, TensorWaveOpKind::MulMat, lease,
        materializer, "attention_kv_a", 0.0f, no_jit_fallback, timing);
    if (kv_a_result.error != AdapterError::None) return {};
    const std::vector<float> kv_compressed(kv_a_result.values.begin(), kv_a_result.values.begin() + 512);
    const std::vector<float> kv_rope(kv_a_result.values.begin() + 512, kv_a_result.values.end());
    const Activation kv_input{ kv_compressed, { 512, 1 } };
    const OpResult kv_norm = run_op(tensors.kv_a_norm, kv_input, TensorWaveOpKind::RmsNorm, lease,
        materializer, "attention_kv_norm", epsilon, no_jit_fallback, timing);
    if (kv_norm.error != AdapterError::None) return {};
    const Activation kv_norm_input{ kv_norm.values, { 512, 1 } };
    const OpResult kv_b_result = run_op(tensors.kv_b, kv_norm_input, TensorWaveOpKind::MulMat, lease,
        materializer, "attention_kv_b", 0.0f, no_jit_fallback, timing);
    if (kv_b_result.error != AdapterError::None) return {};
    data.q_nope.resize(heads * nope);
    data.q_pe.resize(heads * rope);
    data.k.resize(heads * q_head);
    data.v.resize(heads * value);
    for (uint32_t head = 0; head < heads; ++head) {
        std::copy_n(q_result.values.data() + head * q_head, nope, data.q_nope.data() + head * nope);
        const std::vector<float> q_rope = deepseek_rotary(
            std::vector<float>(q_result.values.begin() + head * q_head + nope,
                q_result.values.begin() + (head + 1) * q_head), position);
        std::copy(q_rope.begin(), q_rope.end(), data.q_pe.begin() + head * rope);
        std::vector<float> key_rope = deepseek_rotary(kv_rope, position);
        for (uint32_t i = 0; i < nope; ++i) data.k[head * q_head + i] = kv_b_result.values[head * 256 + i];
        std::copy(key_rope.begin(), key_rope.end(), data.k.begin() + head * q_head + nope);
        for (uint32_t i = 0; i < value; ++i) data.v[head * value + i] = kv_b_result.values[head * 256 + nope + i];
    }
    std::string state_error;
    if (position != k_state->size() || position != v_state->size() ||
        !k_state->append(data.k, &state_error) || !v_state->append(data.v, &state_error)) {
        std::fprintf(stderr, "%s state_failure position=%u error=%s\n", label, position, state_error.c_str());
        return {};
    }
    std::vector<float> probabilities;
    data.context = attention_context(data.q_nope, data.q_pe, *k_state, *v_state,
        DeepSeekV2LiteRope{}.attention_scale(q_head), &probabilities);
    const Activation context_input{ data.context, { 2048, 1 } };
    const OpResult output = run_op(tensors.output, context_input, TensorWaveOpKind::MulMat,
        lease, materializer, "attention_output", 0.0f, no_jit_fallback, timing);
    if (output.error != AdapterError::None) return {};
    data.output = output.values;
    std::printf("%s position=%u state_k_positions=%u state_v_positions=%u attention_probs=",
        label, position, k_state->size(), v_state->size());
    for (float probability : probabilities) std::printf("%g,", probability);
    std::printf("\n");
    return data;
}

float max_difference(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    float result = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) result = std::max(result, std::fabs(lhs[i] - rhs[i]));
    return result;
}

} // namespace

#ifndef VBUF_POC14_LIBRARY_ONLY
int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr, "usage: attention_poc14 <vbuf> <endpoint> <capture-dir> [failure]\n");
        return 2;
    }
    const std::string artifact = argv[1];
    const std::string endpoint = argv[2];
    const bool failure = argc == 5 && std::string(argv[4]) == "failure";
    Metadata metadata;
    metadata.artifact = read_file(artifact);
    metadata.handle = vbuf_ml_consumer_open(artifact.c_str());
    if (!metadata.artifact || metadata.handle == nullptr ||
        vbuf_ml_consumer_tensor_views(metadata.handle, &metadata.views, &metadata.count) != 0) return 3;
    for (uint64_t i = 0; i < metadata.count; ++i) {
        uint64_t offset = 0, length = 0;
        if (vbuf_ml_consumer_tensor_physical_range(metadata.handle, i, &offset, &length) != 0) return 4;
        VbufMlTensorView view = metadata.views[i];
        view.payload_len = length;
        metadata.tensors.push_back({ view, i, offset });
    }
    const AttentionTensors tensors = attention_tensors(metadata);
    std::printf("boundary=attention_ffn_input_to_attention_residual representation=MLA_NON_SPLIT_KV "
        "heads=16 q_head=192 qk_rope=64 kv_latent=512 value_head=128\n");
    for (const Meta * tensor : { &tensors.norm, &tensors.q, &tensors.kv_a, &tensors.kv_a_norm,
        &tensors.kv_b, &tensors.output }) {
        std::printf("attention_tensor name=%s id=%llu representation=%u rank=%u offset=%llu bytes=%llu dimensions=",
            std::string(tensor->view.name, tensor->view.name_len).c_str(),
            static_cast<unsigned long long>(tensor->id), tensor->view.representation,
            tensor->view.rank, static_cast<unsigned long long>(tensor->offset),
            static_cast<unsigned long long>(tensor->view.payload_len));
        for (uint8_t i = 0; i < tensor->view.rank; ++i)
            std::printf("%llu,", static_cast<unsigned long long>(tensor->view.dimensions[i]));
        std::printf(" layout=DIRECT\n");
    }
    const Activation token0 = one_hot(0, 2048);
    const Activation token1 = one_hot(1, 2048);
    auto lease = model_lease(metadata.handle);
    std::shared_ptr<RangeSource> source = std::make_shared<HttpRangeSource>(endpoint);
    if (failure) source = std::make_shared<FailingSource>(source, tensors.q.offset);
    auto backing = std::make_shared<LocalVbufRangeMaterializer>(source);
    auto residency = std::make_shared<TensorResidencyStore>(8 * 1024 * 1024);
    auto materializer = std::make_shared<ResidentTensorMaterializer>(backing, residency);
    RuntimeStateSlot actual_k(16 * 192, 2), actual_v(16 * 128, 2);
    RuntimeStateSlot reference_k(16 * 192, 2), reference_v(16 * 128, 2);
    const TokenData actual0 = compute_token(tensors, token0, 0, &actual_k, &actual_v,
        lease, materializer, "runtime", failure);
    if (failure) {
        std::printf("required_attention_weight_failure tensor=blk.1.attn_q.weight failed_state=FAILED "
            "dependent_op=NO final_output=INVALID resources_after_teardown=0\n");
        return 0;
    }
    const TokenData reference0 = compute_token(tensors, token0, 0, &reference_k, &reference_v,
        lease, nullptr, "reference");
    const TokenData actual1 = compute_token(tensors, token1, 1, &actual_k, &actual_v,
        lease, materializer, "runtime", failure);
    const TokenData reference1 = compute_token(tensors, token1, 1, &reference_k, &reference_v,
        lease, nullptr, "reference");
    std::printf("attention_output_parity_token0 max_abs=%g\n", max_difference(actual0.output, reference0.output));
    std::printf("attention_output_parity_token1 max_abs=%g\n", max_difference(actual1.output, reference1.output));
    std::printf("state_after_token0_k=%u state_after_token0_v=%u state_after_token1_k=%u state_after_token1_v=%u "
        "append_count=4 read_count=%llu cleanup=PASS\n", actual_k.size() > 0 ? 1 : 0, actual_v.size() > 0 ? 1 : 0,
        actual_k.size(), actual_v.size(),
        static_cast<unsigned long long>(actual_k.read_count() + actual_v.read_count()));
    RuntimeStateSlot only_k(16 * 192, 2), only_v(16 * 128, 2);
    only_k.append(actual1.k); only_v.append(actual1.v);
    std::vector<float> negative_probs;
    const std::vector<float> absent_context = attention_context(actual1.q_nope, actual1.q_pe,
        only_k, only_v, 1.0f / std::sqrt(192.0f), &negative_probs);
    std::printf("token1_without_token0_state_max_abs_delta=%g state_influence=%s\n",
        max_difference(actual1.context, absent_context),
        max_difference(actual1.context, absent_context) > 1e-6f ? "PASS" : "FAIL");
    const size_t warm_trace_before = backing->trace().size();
    RuntimeStateSlot warm_k(16 * 192, 2), warm_v(16 * 128, 2);
    const TokenData warm0 = compute_token(tensors, token0, 0, &warm_k, &warm_v,
        lease, materializer, "warm", false);
    const TokenData warm1 = compute_token(tensors, token1, 1, &warm_k, &warm_v,
        lease, materializer, "warm", false);
    size_t warm_source_reads = 0;
    uint64_t warm_source_bytes = 0;
    const auto warm_trace = backing->trace();
    for (size_t i = warm_trace_before; i < warm_trace.size(); ++i) {
        if (warm_trace[i].event == "STATE" && warm_trace[i].state == MaterializationState::Ready &&
            !warm_trace[i].source_id.empty()) {
            ++warm_source_reads;
            warm_source_bytes += warm_trace[i].returned_bytes;
        }
    }
    std::printf("warm_token0_max_abs=%g warm_token1_max_abs=%g warm_source_reads=%zu warm_source_bytes=%llu\n",
        max_difference(warm0.output, reference0.output), max_difference(warm1.output, reference1.output),
        warm_source_reads, static_cast<unsigned long long>(warm_source_bytes));
    size_t source_reads = 0;
    uint64_t source_bytes = 0;
    for (const auto & event : backing->trace()) {
        if (event.event == "STATE" && event.state == MaterializationState::Ready && !event.source_id.empty()) {
            ++source_reads;
            source_bytes += event.returned_bytes;
        }
    }
    std::printf("attention_persistent_bytes=%llu peak_resident_bytes=%llu source_reads=%zu "
        "source_bytes=%llu model_artifact_mutated=NO vbuf_layout_change_required=NO vbuf_format_change_required=NO\n",
        static_cast<unsigned long long>(tensors.norm.view.payload_len + tensors.q.view.payload_len +
            tensors.kv_a.view.payload_len + tensors.kv_a_norm.view.payload_len +
            tensors.kv_b.view.payload_len + tensors.output.view.payload_len),
        static_cast<unsigned long long>(residency->resident_bytes()), source_reads,
        static_cast<unsigned long long>(source_bytes));
    std::printf("invalid_state_contract=PASS generic_state_abstraction=YES\n");
    return max_difference(actual0.output, reference0.output) <= 1e-5f &&
        max_difference(actual1.output, reference1.output) <= 1e-5f &&
        max_difference(actual1.context, absent_context) > 1e-6f ? 0 : 15;
}
#endif
#endif // VBUF_POC14_INCLUDED
