#include "qwen3_execution_plan.h"
#include "qwen3_generation.h"
#include "qwen3_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace vbuf_ggml;
namespace {
constexpr uint32_t rows = 32;
constexpr uint32_t dim = 128;
constexpr uint32_t kv_heads = 8;
constexpr uint32_t query_heads = 40;
constexpr uint32_t early_layers = 26;
constexpr uint64_t expected_source_size = 9000232144ULL;
constexpr const char * expected_identity =
    "sha256:f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31";

void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

std::vector<uint32_t> read_tokens(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open token IDs: " + path);
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<uint32_t> result;
    std::string number;
    for (char c : text) {
        if (c >= '0' && c <= '9') number.push_back(c);
        else if (!number.empty()) {
            const unsigned long long value = std::stoull(number);
            if (value > UINT32_MAX) throw std::out_of_range("token ID exceeds u32");
            result.push_back(static_cast<uint32_t>(value));
            number.clear();
        }
    }
    if (!number.empty()) {
        const unsigned long long value = std::stoull(number);
        if (value > UINT32_MAX) throw std::out_of_range("token ID exceeds u32");
        result.push_back(static_cast<uint32_t>(value));
    }
    if (result.size() < rows) throw std::invalid_argument("token fixture has fewer than 32 tokens");
    result.resize(rows);
    return result;
}

struct Metrics {
    double max_abs = 0.0;
    double rms = 0.0;
    double reference_rms = 0.0;
    double candidate_rms = 0.0;
    double relative_rms = 0.0;
    double cosine = 1.0;
    uint64_t changed = 0;
};

Metrics compare(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size() && !a.empty(), "comparison vector geometry mismatch");
    double error2 = 0.0, a2 = 0.0, b2 = 0.0, dot = 0.0;
    Metrics m;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "non-finite AV diagnostic value");
        const double delta = static_cast<double>(b[i]) - a[i];
        m.max_abs = std::max(m.max_abs, std::abs(delta));
        error2 += delta * delta;
        a2 += static_cast<double>(a[i]) * a[i];
        b2 += static_cast<double>(b[i]) * b[i];
        dot += static_cast<double>(a[i]) * b[i];
        m.changed += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
    }
    m.rms = std::sqrt(error2 / static_cast<double>(a.size()));
    m.reference_rms = std::sqrt(a2 / static_cast<double>(a.size()));
    m.candidate_rms = std::sqrt(b2 / static_cast<double>(a.size()));
    m.relative_rms = std::sqrt(error2 / std::max(a2, 1e-300));
    m.cosine = dot / std::max(std::sqrt(a2 * b2), 1e-300);
    return m;
}

float value_as_float(const uint8_t * ptr, ggml_type type) {
    if (type == GGML_TYPE_F16) {
        ggml_fp16_t half;
        std::memcpy(&half, ptr, sizeof(half));
        return ggml_fp16_to_fp32(half);
    }
    if (type == GGML_TYPE_F32) {
        float value;
        std::memcpy(&value, ptr, sizeof(value));
        return value;
    }
    throw std::runtime_error("AV boundary oracle received an unsupported V type");
}

struct CompactBoundary {
    uint32_t layer = 0;
    uint32_t device_id = 0;
    uint32_t capacity = 0;
    uint32_t query_rows = 0;
    uint32_t visible_context = 0;
    bool prefill = false;
    bool native_intervention = false;
    std::array<int64_t, 4> value_ne{};
    std::array<size_t, 4> value_nb{};
    std::array<int64_t, 4> probability_ne{};
    std::array<size_t, 4> probability_nb{};
    std::array<int64_t, 4> output_ne{};
    std::array<size_t, 4> output_nb{};
    std::array<int64_t, 4> score_ne{};
    std::array<size_t, 4> score_nb{};
    std::array<int64_t, 4> query_ne{};
    std::array<size_t, 4> query_nb{};
    std::array<int64_t, 4> key_ne{};
    std::array<size_t, 4> key_nb{};
    ggml_type value_type = GGML_TYPE_COUNT;
    std::vector<uint8_t> values_f16;
    std::vector<uint8_t> keys_f16;
    std::vector<float> scores;
    std::vector<float> query;
    std::vector<float> qk_oracle;
    std::vector<float> softmax_from_scores;
    std::vector<float> softmax_from_oracle_scores;
    std::vector<float> probabilities;
    std::vector<int32_t> positions;
    std::vector<float> canonical;
    std::vector<float> native;
    std::vector<float> oracle;
};

CompactBoundary compact_capture(const Qwen3AttentionAVBoundaryCapture & capture) {
    require((capture.prefill && capture.query_rows == rows) || (!capture.prefill && capture.query_rows == 1),
        "unexpected AV diagnostic phase or query geometry");
    require(capture.capacity >= rows + 1, "AV diagnostic capacity is below the captured context");
    require(!capture.native_intervention,
        "canonical AV compaction cannot reinterpret a native-only intervention capture");
    require(capture.value_type == GGML_TYPE_F16 && capture.probability_type == GGML_TYPE_F32 &&
        capture.position_type == GGML_TYPE_I32, "unexpected real-model AV input types");
    require(capture.value_ne[0] == dim && capture.value_ne[1] == static_cast<int64_t>(kv_heads) * capture.capacity &&
        capture.probability_ne[0] == capture.capacity && capture.probability_ne[1] == capture.query_rows &&
        capture.probability_ne[2] == query_heads && capture.output_ne[0] == dim &&
        capture.output_ne[1] == capture.query_rows && capture.output_ne[2] == query_heads,
        "real-model AV boundary tensor geometry mismatch");
    require(capture.value_nb[0] == sizeof(ggml_fp16_t) && capture.value_nb[1] == dim * sizeof(ggml_fp16_t) &&
        capture.probability_nb[0] == sizeof(float) &&
        capture.probability_nb[1] == capture.capacity * sizeof(float) &&
        capture.probability_nb[2] == capture.capacity * capture.query_rows * sizeof(float) &&
        capture.output_nb[0] == sizeof(float) && capture.output_nb[1] == dim * sizeof(float) &&
        capture.output_nb[2] == dim * capture.query_rows * sizeof(float),
        "AV boundary tensors are not in the expected contiguous native layouts");
    require(capture.value_bytes.size() >= static_cast<size_t>(capture.value_nb[1]) * kv_heads * capture.capacity &&
        capture.probabilities.size() == static_cast<size_t>(capture.capacity) * capture.query_rows * query_heads &&
        capture.canonical_output.size() == static_cast<size_t>(dim) * capture.query_rows * query_heads &&
        capture.native_output.size() == capture.canonical_output.size() && capture.positions.size() == capture.query_rows,
        "AV boundary capture byte count does not match tensor metadata");

    CompactBoundary result;
    result.layer = capture.layer;
    result.device_id = capture.device_id;
    result.capacity = capture.capacity;
    result.query_rows = capture.query_rows;
    result.prefill = capture.prefill;
    result.native_intervention = capture.native_intervention;
    result.visible_context = static_cast<uint32_t>(*std::max_element(capture.positions.begin(), capture.positions.end()) + 1);
    result.value_ne = capture.value_ne;
    result.value_nb = capture.value_nb;
    result.probability_ne = capture.probability_ne;
    result.probability_nb = capture.probability_nb;
    result.output_ne = capture.output_ne;
    result.output_nb = capture.output_nb;
    result.value_type = capture.value_type;
    result.values_f16.resize(static_cast<size_t>(dim) * result.visible_context * kv_heads * sizeof(ggml_fp16_t));
    for (uint32_t d = 0; d < dim; ++d) for (uint32_t p = 0; p < result.visible_context; ++p)
        for (uint32_t kh = 0; kh < kv_heads; ++kh) {
            const size_t src = static_cast<size_t>(d) * capture.value_nb[0] +
                static_cast<size_t>(p * kv_heads + kh) * capture.value_nb[1];
            const size_t dst = static_cast<size_t>(d + dim * (p * kv_heads + kh)) * sizeof(ggml_fp16_t);
            std::memcpy(result.values_f16.data() + dst, capture.value_bytes.data() + src, sizeof(ggml_fp16_t));
        }
    result.probabilities.resize(static_cast<size_t>(result.visible_context) * capture.query_rows * query_heads, 0.0f);
    result.positions = capture.positions;
    for (uint32_t h = 0; h < query_heads; ++h) for (uint32_t q = 0; q < capture.query_rows; ++q)
        for (uint32_t p = 0; p < result.visible_context; ++p) {
            const size_t src = (static_cast<size_t>(p) * capture.probability_nb[0] +
                static_cast<size_t>(q) * capture.probability_nb[1] +
                static_cast<size_t>(h) * capture.probability_nb[2]) / sizeof(float);
            result.probabilities[p + result.visible_context * (q + capture.query_rows * h)] = capture.probabilities[src];
        }
    result.canonical = capture.canonical_output;
    result.native = capture.native_output;
    result.oracle.resize(result.canonical.size());
    if (capture.layer == 0) {
        require(capture.score_type == GGML_TYPE_F32 && capture.query_type == GGML_TYPE_F32 &&
            capture.key_type == GGML_TYPE_F16 && capture.score_ne[0] == capture.capacity &&
            capture.score_ne[1] == capture.query_rows && capture.score_ne[2] == query_heads &&
            capture.query_ne[0] == dim && capture.query_ne[1] == query_heads &&
            capture.query_ne[2] == capture.query_rows && capture.key_ne[0] == dim &&
            capture.key_ne[1] == static_cast<int64_t>(kv_heads) * capture.capacity,
            "layer-zero QK boundary geometry/type mismatch");
        require(capture.score_nb[0] == sizeof(float) && capture.score_nb[1] == capture.capacity * sizeof(float) &&
            capture.score_nb[2] == capture.capacity * capture.query_rows * sizeof(float) &&
            capture.query_nb[0] == sizeof(float) && capture.query_nb[1] == dim * sizeof(float) &&
            capture.query_nb[2] == dim * query_heads * sizeof(float) && capture.key_nb[0] == sizeof(ggml_fp16_t) &&
            capture.key_nb[1] == dim * sizeof(ggml_fp16_t), "layer-zero QK tensors are not contiguous");
        require(capture.scores.size() == static_cast<size_t>(capture.capacity) * capture.query_rows * query_heads &&
            capture.query.size() == static_cast<size_t>(dim) * query_heads * capture.query_rows &&
            capture.key_bytes.size() >= static_cast<size_t>(capture.key_nb[1]) * kv_heads * capture.capacity,
            "layer-zero QK tensor capture byte count mismatch");
        result.score_ne = capture.score_ne;
        result.score_nb = capture.score_nb;
        result.query_ne = capture.query_ne;
        result.query_nb = capture.query_nb;
        result.key_ne = capture.key_ne;
        result.key_nb = capture.key_nb;
        result.query = capture.query;
        result.scores.resize(static_cast<size_t>(result.visible_context) * capture.query_rows * query_heads);
        result.keys_f16.resize(static_cast<size_t>(dim) * result.visible_context * kv_heads * sizeof(ggml_fp16_t));
        for (uint32_t d = 0; d < dim; ++d) for (uint32_t p = 0; p < result.visible_context; ++p)
            for (uint32_t kh = 0; kh < kv_heads; ++kh) {
                const size_t src = static_cast<size_t>(d) * capture.key_nb[0] +
                    static_cast<size_t>(p * kv_heads + kh) * capture.key_nb[1];
                const size_t dst = static_cast<size_t>(d + dim * (p * kv_heads + kh)) * sizeof(ggml_fp16_t);
                std::memcpy(result.keys_f16.data() + dst, capture.key_bytes.data() + src, sizeof(ggml_fp16_t));
            }
        result.qk_oracle.resize(result.scores.size());
        for (uint32_t h = 0; h < query_heads; ++h) {
            const uint32_t kh = h / (query_heads / kv_heads);
            for (uint32_t q = 0; q < capture.query_rows; ++q) {
                for (uint32_t p = 0; p < result.visible_context; ++p) {
                    double sum = 0.0;
                    for (uint32_t d = 0; d < dim; ++d) {
                        const size_t q_index = d + static_cast<size_t>(dim) * (h + query_heads * q);
                        const size_t k_index = d + static_cast<size_t>(dim) * (p * kv_heads + kh);
                        const float key = value_as_float(result.keys_f16.data() + k_index * sizeof(ggml_fp16_t), GGML_TYPE_F16);
                        sum += static_cast<double>(result.query[q_index]) * static_cast<double>(key);
                    }
                    const size_t compact = p + static_cast<size_t>(result.visible_context) * (q +
                        static_cast<size_t>(capture.query_rows) * h);
                    const size_t source = (static_cast<size_t>(p) * capture.score_nb[0] +
                        static_cast<size_t>(q) * capture.score_nb[1] +
                        static_cast<size_t>(h) * capture.score_nb[2]) / sizeof(float);
                    result.scores[compact] = capture.scores[source];
                    result.qk_oracle[compact] = static_cast<float>(sum);
                }
            }
        }
        result.softmax_from_scores.assign(result.probabilities.size(), 0.0f);
        result.softmax_from_oracle_scores.assign(result.probabilities.size(), 0.0f);
        const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
        for (uint32_t h = 0; h < query_heads; ++h) for (uint32_t q = 0; q < capture.query_rows; ++q) {
            const uint32_t last = static_cast<uint32_t>(capture.positions[q]);
            for (int which = 0; which < 2; ++which) {
                const auto & score_source = which == 0 ? result.scores : result.qk_oracle;
                double maximum = -std::numeric_limits<double>::infinity();
                for (uint32_t p = 0; p <= last; ++p) {
                    const size_t index = p + static_cast<size_t>(result.visible_context) *
                        (q + static_cast<size_t>(capture.query_rows) * h);
                    maximum = std::max(maximum, static_cast<double>(score_source[index]) * scale);
                }
                double denominator = 0.0;
                std::vector<double> exponentials(last + 1);
                for (uint32_t p = 0; p <= last; ++p) {
                    const size_t index = p + static_cast<size_t>(result.visible_context) *
                        (q + static_cast<size_t>(capture.query_rows) * h);
                    exponentials[p] = std::exp(static_cast<double>(score_source[index]) * scale - maximum);
                    denominator += exponentials[p];
                }
                auto & destination = which == 0 ? result.softmax_from_scores : result.softmax_from_oracle_scores;
                for (uint32_t p = 0; p <= last; ++p) {
                    const size_t index = p + static_cast<size_t>(result.visible_context) *
                        (q + static_cast<size_t>(capture.query_rows) * h);
                    destination[index] = static_cast<float>(exponentials[p] / denominator);
                }
            }
        }
    }
    for (uint32_t h = 0; h < query_heads; ++h) {
        const uint32_t kv_head = h / (query_heads / kv_heads);
        for (uint32_t q = 0; q < capture.query_rows; ++q) {
            const int32_t last = capture.positions[q];
            require(last >= 0 && last < static_cast<int32_t>(result.visible_context), "AV diagnostic position is out of range");
            for (uint32_t d = 0; d < dim; ++d) {
                double sum = 0.0;
                for (int32_t p = 0; p <= last; ++p) {
                    const size_t v_element = static_cast<size_t>(d + dim * (static_cast<uint32_t>(p) * kv_heads + kv_head));
                    const float value = value_as_float(result.values_f16.data() + v_element * sizeof(ggml_fp16_t),
                        capture.value_type);
                    const float weight = result.probabilities[static_cast<size_t>(p) + result.visible_context *
                        (q + static_cast<size_t>(capture.query_rows) * h)];
                    sum += static_cast<double>(weight) * static_cast<double>(value);
                }
                const size_t output = d + static_cast<size_t>(dim) * (q + static_cast<size_t>(capture.query_rows) * h);
                result.oracle[output] = static_cast<float>(sum);
            }
        }
    }
    return result;
}

std::vector<float> active_values_as_float(const CompactBoundary & capture) {
    const size_t count = static_cast<size_t>(dim) * capture.visible_context * kv_heads;
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i)
        result[i] = value_as_float(capture.values_f16.data() + i * sizeof(ggml_fp16_t), capture.value_type);
    return result;
}

std::vector<float> active_keys_as_float(const CompactBoundary & capture) {
    const size_t count = static_cast<size_t>(dim) * capture.visible_context * kv_heads;
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i)
        result[i] = value_as_float(capture.keys_f16.data() + i * sizeof(ggml_fp16_t), GGML_TYPE_F16);
    return result;
}

std::vector<float> visible_matrix(const CompactBoundary & capture, const std::vector<float> & matrix) {
    std::vector<float> result;
    for (uint32_t h = 0; h < query_heads; ++h) for (uint32_t q = 0; q < capture.query_rows; ++q)
        for (int32_t p = 0; p <= capture.positions[q]; ++p)
            result.push_back(matrix[static_cast<size_t>(p) + capture.visible_context *
                (q + static_cast<size_t>(capture.query_rows) * h)]);
    return result;
}

std::vector<float> visible_probabilities(const CompactBoundary & capture) {
    return visible_matrix(capture, capture.probabilities);
}

void write_vector(std::ofstream & out, const void * data, size_t bytes) {
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    if (!out) throw std::runtime_error("failed writing compact AV boundary capture");
}

void save_capture(const std::filesystem::path & directory, const CompactBoundary & capture) {
    std::ostringstream stem;
    stem << "boundary-capacity-" << capture.capacity << '-' << (capture.prefill ? "prefill" : "decode")
         << "-layer-" << std::setw(2) << std::setfill('0') << capture.layer;
    const auto base = directory / stem.str();
    {
        std::ofstream meta(base.string() + ".meta");
        if (!meta) throw std::runtime_error("cannot create AV boundary metadata");
        meta << "format=qwen3-native-av-boundary-v1\n"
             << "capacity=" << capture.capacity << "\nlayer=" << capture.layer
             << "\ndevice_id=" << capture.device_id << "\nphase=" << (capture.prefill ? "prefill" : "decode")
             << "\nquery_rows=" << capture.query_rows << "\nvisible_context=" << capture.visible_context << "\n"
             << "query_heads=40\nkv_heads=8\nhead_dim=128\nvalue_type=F16\nprobability_type=F32\n"
             << "canonical_op=GGML_OP_MUL_MAT\nnative_op=GGML_OP_ATTENTION_AV\n"
             << "value_ne=" << capture.value_ne[0] << ',' << capture.value_ne[1]
             << "\nvalue_nb=" << capture.value_nb[0] << ',' << capture.value_nb[1]
             << "\nprobability_ne=" << capture.probability_ne[0] << ',' << capture.probability_ne[1]
             << ',' << capture.probability_ne[2] << "\nprobability_nb=" << capture.probability_nb[0]
             << ',' << capture.probability_nb[1] << ',' << capture.probability_nb[2]
             << "\noutput_ne=" << capture.output_ne[0] << ',' << capture.output_ne[1]
             << ',' << capture.output_ne[2] << "\noutput_nb=" << capture.output_nb[0]
             << ',' << capture.output_nb[1] << ',' << capture.output_nb[2]
             << "\ncompact_value_shape=[128," << capture.visible_context << ",8] compact_probability_shape=["
             << capture.visible_context << ',' << capture.query_rows << ",40]\n"
             << "qk_capture=" << (!capture.scores.empty() ? "yes" : "no") << "\n"
             << "oracle=ascending-visible-position-FP64-accumulator-output-F32\n";
    }
    {
        std::ofstream out(base.string() + ".bin", std::ios::binary);
        if (!out) throw std::runtime_error("cannot create AV boundary payload capture");
        write_vector(out, capture.values_f16.data(), capture.values_f16.size());
        write_vector(out, capture.probabilities.data(), capture.probabilities.size() * sizeof(float));
        write_vector(out, capture.positions.data(), capture.positions.size() * sizeof(int32_t));
        write_vector(out, capture.canonical.data(), capture.canonical.size() * sizeof(float));
        write_vector(out, capture.native.data(), capture.native.size() * sizeof(float));
        write_vector(out, capture.oracle.data(), capture.oracle.size() * sizeof(float));
        if (!capture.scores.empty()) {
            write_vector(out, capture.keys_f16.data(), capture.keys_f16.size());
            write_vector(out, capture.query.data(), capture.query.size() * sizeof(float));
            write_vector(out, capture.scores.data(), capture.scores.size() * sizeof(float));
            write_vector(out, capture.qk_oracle.data(), capture.qk_oracle.size() * sizeof(float));
            write_vector(out, capture.softmax_from_scores.data(), capture.softmax_from_scores.size() * sizeof(float));
            write_vector(out, capture.softmax_from_oracle_scores.data(),
                capture.softmax_from_oracle_scores.size() * sizeof(float));
        }
    }
    size_t bytes = capture.values_f16.size() + capture.probabilities.size() * sizeof(float) +
        capture.positions.size() * sizeof(int32_t) + capture.canonical.size() * sizeof(float) * 3;
    if (!capture.scores.empty()) bytes += capture.keys_f16.size() + capture.query.size() * sizeof(float) +
        capture.scores.size() * sizeof(float) * 4;
    std::printf("saved_boundary_capture=%s.bin bytes=%zu section_order=V_F16,P_F32,positions_I32,canonical_F32,native_F32,oracle_F32%s\n",
        base.string().c_str(), bytes, capture.scores.empty() ? "" :
        ",K_F16,Q_F32,scores_F32,qk_oracle_F32,softmax_from_scores_F32,softmax_from_qk_oracle_F32");
}

struct CapacityRun {
    uint32_t capacity = 0;
    std::vector<CompactBoundary> layers;
    std::vector<float> final_hidden;
    std::vector<float> final_logits;
    uint32_t next_token = 0;
};

CapacityRun run_capacity(Qwen3Model & model, const std::shared_ptr<QwenCudaRuntimeState> & runtime,
        const std::vector<uint32_t> & prompt, uint32_t capacity) {
    CapacityRun run;
    run.capacity = capacity;
    {
        auto session = runtime->create_session(capacity);
        Qwen3MultiDeviceGenerationExecutor executor(model, runtime, session, false, false, true);
        const auto execution = executor.run(prompt, 1);
        require(execution.completed && session->current_length() == rows + 1,
            "canonical diagnostic run did not complete the expected 32-row prefill plus decode");
        require(execution.native_av_steps == 0, "boundary diagnostic unexpectedly selected native AV candidate execution");
        require(execution.attention_av_boundary_captures.size() == early_layers * 2,
            "incomplete prefill/decode real-model AV boundary capture set");
        run.final_hidden = execution.final_hidden;
        run.final_logits = execution.final_logits;
        run.next_token = execution.tokens.empty() ? UINT32_MAX : execution.tokens.front();
        run.layers.reserve(early_layers * 2);
        for (size_t index = 0; index < execution.attention_av_boundary_captures.size(); ++index) {
            const auto & capture = execution.attention_av_boundary_captures[index];
            const uint32_t expected_layer = static_cast<uint32_t>(index % early_layers);
            const bool expected_prefill = index < early_layers;
            require(capture.layer == expected_layer && capture.capacity == capacity &&
                capture.prefill == expected_prefill,
                "AV boundary captures are not in expected prefill/decode layer order");
            run.layers.push_back(compact_capture(capture));
        }
    }
    if (capacity == 512) {
        auto baseline_session = runtime->create_session(capacity);
        Qwen3MultiDeviceGenerationExecutor baseline_executor(model, runtime, baseline_session);
        const auto baseline = baseline_executor.run(prompt, 1);
        require(baseline.completed && baseline.tokens.size() == 1 && baseline.tokens[0] == run.next_token &&
            baseline.final_hidden.size() == run.final_hidden.size() && baseline.final_logits.size() == run.final_logits.size() &&
            std::memcmp(baseline.final_hidden.data(), run.final_hidden.data(), run.final_hidden.size() * sizeof(float)) == 0 &&
            std::memcmp(baseline.final_logits.data(), run.final_logits.data(), run.final_logits.size() * sizeof(float)) == 0,
            "diagnostic side branch changed canonical final hidden/logits at capacity 512");
        std::puts("diagnostic_noninterference capacity=512 canonical_hidden_logits=BITWISE_EQUAL PASS");
    }
    return run;
}

struct PropagationWorkload {
    uint32_t position = 0;
    uint32_t prompt_rows = 0;
    bool prefill_capture = false;
    std::vector<uint32_t> tokens;
    uint32_t final_top1 = UINT32_MAX;
    std::vector<float> final_hidden;
    std::vector<float> final_logits;
    std::vector<Qwen3LayerActivationCapture> layer_outputs;
    std::vector<CompactBoundary> local_av;
    std::optional<Qwen3AttentionAVBoundaryCapture> native_capture;
    uint64_t diagnostic_native_av_interventions = 0;
};

PropagationWorkload execute_propagation_workload(Qwen3MultiDeviceGenerationExecutor & executor,
        QwenCudaSessionState & session, const std::vector<uint32_t> & fixture, uint32_t position,
        bool prefill_capture, bool capture_local_av, int32_t native_intervention_layer) {
    const uint32_t prompt_rows = prefill_capture ? rows : position;
    require(prompt_rows > 0 && prompt_rows <= fixture.size(), "invalid propagation prompt length");
    std::vector<uint32_t> input(fixture.begin(), fixture.begin() + prompt_rows);
    executor.configure_attention_av_diagnostic(position, capture_local_av, native_intervention_layer);
    Qwen3GenerationExecution execution = executor.run(input, 1);
    require(execution.completed && session.current_length() == prompt_rows + 1 && execution.tokens.size() == 1,
        "propagation workload did not complete one isolated target decode");
    require(execution.layer_activation_captures.size() == 40,
        "propagation workload did not capture all 40 transformer-block outputs at its target step");
    for (uint32_t layer = 0; layer < 40; ++layer) {
        const auto & capture = execution.layer_activation_captures[layer];
        require(capture.layer == layer && capture.position == position && capture.rows == (prefill_capture ? rows : 1) &&
            capture.prefill == prefill_capture && capture.hidden.size() ==
                static_cast<size_t>(capture.rows) * 5120,
            "layer activation captures are not in target-step layer order/geometry");
    }
    if (capture_local_av) {
        require(execution.attention_av_boundary_captures.size() == early_layers,
            "propagation workload did not capture every early-device AV boundary");
    } else {
        require(execution.attention_av_boundary_captures.empty(),
            "disabled local AV capture unexpectedly retained AV boundary tensors");
    }
    PropagationWorkload result;
    result.position = position;
    result.prompt_rows = prompt_rows;
    result.prefill_capture = prefill_capture;
    result.tokens = execution.tokens;
    result.final_top1 = static_cast<uint32_t>(std::max_element(execution.final_logits.begin(),
        execution.final_logits.end()) - execution.final_logits.begin());
    result.final_hidden = std::move(execution.final_hidden);
    result.final_logits = std::move(execution.final_logits);
    result.layer_outputs = std::move(execution.layer_activation_captures);
    result.diagnostic_native_av_interventions = execution.diagnostic_native_av_interventions;
    for (auto & capture : execution.attention_av_boundary_captures) {
        if (capture.native_intervention) {
            require(native_intervention_layer >= 0 && capture.layer == static_cast<uint32_t>(native_intervention_layer) &&
                !result.native_capture.has_value(), "unexpected or duplicate native AV intervention capture");
            result.native_capture = std::move(capture);
        } else {
            result.local_av.push_back(compact_capture(capture));
        }
    }
    if (native_intervention_layer >= 0)
        require(result.diagnostic_native_av_interventions == 1 && result.native_capture.has_value(),
            "requested single-layer AV intervention was not observed exactly once");
    else
        require(result.diagnostic_native_av_interventions == 0 && !result.native_capture.has_value(),
            "canonical propagation workload unexpectedly selected a native AV intervention");
    if (capture_local_av)
        require(result.local_av.size() + (result.native_capture ? 1U : 0U) == early_layers,
            "local AV captures do not cover all early-device blocks");
    return result;
}

std::vector<float> gather_indices(const std::vector<float> & source, const std::vector<size_t> & indices) {
    std::vector<float> result;
    result.reserve(indices.size());
    for (size_t index : indices) result.push_back(source.at(index));
    return result;
}

void write_local_metric_row(std::ofstream & out, const std::string & workload,
        const CompactBoundary & capture, const char * scope, int32_t group,
        const std::vector<size_t> & indices) {
    const auto canonical = gather_indices(capture.canonical, indices);
    const auto native = gather_indices(capture.native, indices);
    const auto oracle = gather_indices(capture.oracle, indices);
    const Metrics cn = compare(canonical, native);
    const Metrics co = compare(oracle, canonical);
    const Metrics no = compare(oracle, native);
    out << workload << ',' << (capture.prefill ? "prefill" : "decode") << ','
        << capture.positions.front() << ',' << capture.visible_context << ',' << capture.layer << ','
        << scope << ',' << group << ',' << indices.size() << ','
        << cn.max_abs << ',' << cn.rms << ',' << cn.relative_rms << ',' << cn.cosine << ','
        << co.max_abs << ',' << co.rms << ',' << co.relative_rms << ',' << co.cosine << ','
        << no.max_abs << ',' << no.rms << ',' << no.relative_rms << ',' << no.cosine << '\n';
}

void write_local_metrics(std::ofstream & out, const std::string & workload,
        const CompactBoundary & capture) {
    std::vector<size_t> indices(capture.canonical.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    write_local_metric_row(out, workload, capture, "all", -1, indices);
    for (uint32_t h = 0; h < query_heads; ++h) {
        indices.clear();
        const size_t begin = static_cast<size_t>(dim) * capture.query_rows * h;
        for (size_t i = 0; i < static_cast<size_t>(dim) * capture.query_rows; ++i) indices.push_back(begin + i);
        write_local_metric_row(out, workload, capture, "query_head", static_cast<int32_t>(h), indices);
    }
    for (uint32_t kh = 0; kh < kv_heads; ++kh) {
        indices.clear();
        for (uint32_t h = kh * 5; h < kh * 5 + 5; ++h) {
            const size_t begin = static_cast<size_t>(dim) * capture.query_rows * h;
            for (size_t i = 0; i < static_cast<size_t>(dim) * capture.query_rows; ++i) indices.push_back(begin + i);
        }
        write_local_metric_row(out, workload, capture, "kv_head_group", static_cast<int32_t>(kh), indices);
    }
    for (uint32_t d = 0; d < dim; ++d) {
        indices.clear();
        for (uint32_t h = 0; h < query_heads; ++h) for (uint32_t q = 0; q < capture.query_rows; ++q)
            indices.push_back(d + static_cast<size_t>(dim) * (q + capture.query_rows * h));
        write_local_metric_row(out, workload, capture, "output_dimension", static_cast<int32_t>(d), indices);
    }
}

const Qwen3LayerActivationCapture & activation_at(const PropagationWorkload & run, uint32_t layer) {
    require(layer < run.layer_outputs.size() && run.layer_outputs[layer].layer == layer,
        "layer activation lookup is not aligned");
    return run.layer_outputs[layer];
}

const CompactBoundary & av_at(const PropagationWorkload & run, uint32_t layer) {
    const auto found = std::find_if(run.local_av.begin(), run.local_av.end(),
        [layer](const CompactBoundary & item) { return item.layer == layer; });
    if (found == run.local_av.end()) throw std::runtime_error("missing canonical local AV capture");
    return *found;
}

bool intervention_inputs_match(const CompactBoundary & baseline,
        const Qwen3AttentionAVBoundaryCapture & intervention, Metrics * native_output_metrics) {
    if (baseline.layer != intervention.layer || baseline.capacity != intervention.capacity ||
        baseline.query_rows != intervention.query_rows || baseline.visible_context == 0 ||
        intervention.native_intervention == false || baseline.positions != intervention.positions ||
        baseline.value_type != intervention.value_type || intervention.value_type != GGML_TYPE_F16 ||
        baseline.value_ne != intervention.value_ne || baseline.value_nb != intervention.value_nb ||
        baseline.probability_ne[0] != intervention.probability_ne[0] ||
        baseline.probability_ne[1] != intervention.probability_ne[1] ||
        baseline.probability_ne[2] != intervention.probability_ne[2] ||
        baseline.probability_nb != intervention.probability_nb) return false;
    for (uint32_t d = 0; d < dim; ++d) for (uint32_t p = 0; p < baseline.visible_context; ++p)
        for (uint32_t kh = 0; kh < kv_heads; ++kh) {
            const size_t compact = static_cast<size_t>(d) + dim * (p * kv_heads + kh);
            const size_t offset = static_cast<size_t>(d) * intervention.value_nb[0] +
                static_cast<size_t>(p * kv_heads + kh) * intervention.value_nb[1];
            if (offset + sizeof(ggml_fp16_t) > intervention.value_bytes.size() ||
                std::memcmp(baseline.values_f16.data() + compact * sizeof(ggml_fp16_t),
                    intervention.value_bytes.data() + offset, sizeof(ggml_fp16_t)) != 0) return false;
        }
    for (uint32_t h = 0; h < query_heads; ++h) for (uint32_t q = 0; q < baseline.query_rows; ++q)
        for (uint32_t p = 0; p < static_cast<uint32_t>(baseline.positions[q] + 1); ++p) {
            const size_t compact = p + static_cast<size_t>(baseline.visible_context) *
                (q + static_cast<size_t>(baseline.query_rows) * h);
            const size_t offset = (static_cast<size_t>(p) * intervention.probability_nb[0] +
                static_cast<size_t>(q) * intervention.probability_nb[1] +
                static_cast<size_t>(h) * intervention.probability_nb[2]);
            if (offset + sizeof(float) > intervention.probabilities.size() * sizeof(float) ||
                std::memcmp(baseline.probabilities.data() + compact,
                    reinterpret_cast<const uint8_t *>(intervention.probabilities.data()) + offset,
                    sizeof(float)) != 0) return false;
        }
    if (baseline.layer == 0) {
        if (intervention.query.size() != baseline.query.size() ||
            std::memcmp(intervention.query.data(), baseline.query.data(), baseline.query.size() * sizeof(float)) != 0)
            return false;
        for (uint32_t d = 0; d < dim; ++d) for (uint32_t p = 0; p < baseline.visible_context; ++p)
            for (uint32_t kh = 0; kh < kv_heads; ++kh) {
                const size_t compact = static_cast<size_t>(d) + dim * (p * kv_heads + kh);
                const size_t offset = static_cast<size_t>(d) * intervention.key_nb[0] +
                    static_cast<size_t>(p * kv_heads + kh) * intervention.key_nb[1];
                if (offset + sizeof(ggml_fp16_t) > intervention.key_bytes.size() ||
                    std::memcmp(baseline.keys_f16.data() + compact * sizeof(ggml_fp16_t),
                        intervention.key_bytes.data() + offset, sizeof(ggml_fp16_t)) != 0) return false;
            }
        for (uint32_t h = 0; h < query_heads; ++h) for (uint32_t q = 0; q < baseline.query_rows; ++q)
            for (uint32_t p = 0; p < static_cast<uint32_t>(baseline.positions[q] + 1); ++p) {
                const size_t compact = p + static_cast<size_t>(baseline.visible_context) *
                    (q + static_cast<size_t>(baseline.query_rows) * h);
                const size_t offset = (static_cast<size_t>(p) * intervention.score_nb[0] +
                    static_cast<size_t>(q) * intervention.score_nb[1] +
                    static_cast<size_t>(h) * intervention.score_nb[2]);
                if (offset + sizeof(float) > intervention.scores.size() * sizeof(float) ||
                    std::memcmp(baseline.scores.data() + compact,
                        reinterpret_cast<const uint8_t *>(intervention.scores.data()) + offset,
                        sizeof(float)) != 0) return false;
            }
    }
    if (intervention.native_output.size() != baseline.native.size()) return false;
    *native_output_metrics = compare(baseline.native, intervention.native_output);
    return true;
}

void save_intervention_capture(const std::filesystem::path & directory,
        const Qwen3AttentionAVBoundaryCapture & capture, bool inputs_equal, bool output_equal) {
    require(capture.native_intervention && capture.canonical_output.empty(),
        "intervention artifact must contain only the executed native AV output");
    std::filesystem::create_directories(directory);
    std::ostringstream stem;
    stem << "native-layer-" << std::setw(2) << std::setfill('0') << capture.layer;
    const auto base = directory / stem.str();
    std::ofstream meta(base.string() + ".meta");
    require(static_cast<bool>(meta), "cannot write native intervention metadata");
    meta << "format=qwen3-native-av-intervention-input-v1\n"
         << "layer=" << capture.layer << "\nposition=" << capture.positions.front()
         << "\ncapacity=" << capture.capacity << "\nquery_rows=" << capture.query_rows
         << "\nvisible_context=" << capture.positions.front() + 1
         << "\nvalue_ne=" << capture.value_ne[0] << ',' << capture.value_ne[1]
         << "\nvalue_nb=" << capture.value_nb[0] << ',' << capture.value_nb[1]
         << "\nprobability_ne=" << capture.probability_ne[0] << ',' << capture.probability_ne[1]
         << ',' << capture.probability_ne[2]
         << "\nprobability_nb=" << capture.probability_nb[0] << ',' << capture.probability_nb[1]
         << ',' << capture.probability_nb[2]
         << "\ninputs_bitwise_equal=" << (inputs_equal ? "yes" : "no")
         << "\nnative_output_matches_local_side_branch=" << (output_equal ? "yes" : "no")
         << "\nsection_order=V_F16_full,P_F32_full,positions_I32,native_output_F32"
         << (capture.layer == 0 ? ",K_F16_full,Q_F32,scores_F32" : "") << '\n';
    std::ofstream out(base.string() + ".bin", std::ios::binary);
    require(static_cast<bool>(out), "cannot write native intervention payload");
    write_vector(out, capture.value_bytes.data(), capture.value_bytes.size());
    write_vector(out, capture.probabilities.data(), capture.probabilities.size() * sizeof(float));
    write_vector(out, capture.positions.data(), capture.positions.size() * sizeof(int32_t));
    write_vector(out, capture.native_output.data(), capture.native_output.size() * sizeof(float));
    if (capture.layer == 0) {
        write_vector(out, capture.key_bytes.data(), capture.key_bytes.size());
        write_vector(out, capture.query.data(), capture.query.size() * sizeof(float));
        write_vector(out, capture.scores.data(), capture.scores.size() * sizeof(float));
    }
}

std::string workload_name(uint32_t position, bool prefill_capture) {
    if (prefill_capture) return "prefill32";
    return "decode" + std::to_string(position);
}

void save_local_captures(const std::filesystem::path & directory, const PropagationWorkload & run,
        bool all_layers) {
    std::filesystem::create_directories(directory);
    for (const auto & capture : run.local_av) {
        if (all_layers || capture.layer == 0) save_capture(directory, capture);
    }
}

void write_layer_drift(std::ofstream & out, const PropagationWorkload & reference,
        const PropagationWorkload & candidate, int32_t intervention_layer,
        bool inputs_equal, bool native_output_equal, const Metrics & local_av) {
    for (uint32_t layer = 0; layer < 40; ++layer) {
        const auto & a = activation_at(reference, layer);
        const auto & b = activation_at(candidate, layer);
        const Metrics m = compare(a.hidden, b.hidden);
        out << reference.position << ',' << intervention_layer << ',' << layer << ',' << a.rows << ','
            << m.max_abs << ',' << m.rms << ',' << m.relative_rms << ',' << m.cosine << ','
            << m.reference_rms << ',' << m.candidate_rms << ',' << m.changed << ','
            << (inputs_equal ? "yes" : "no") << ',' << (native_output_equal ? "yes" : "no") << ','
            << local_av.rms << ',' << local_av.relative_rms << '\n';
    }
}

int run_propagation(const std::string & semantic, const std::string & source,
        const std::string & token_file, const std::string & output_dir, bool smoke) {
    constexpr uint32_t capacity = 512;
    const std::vector<uint32_t> prompt = read_tokens(token_file);
    Qwen3Model model;
    open_qwen3_model(semantic, source, &model, true);
    require(model.artifact_identity == expected_identity && model.source_size == expected_source_size &&
        model.layer_count == 40 && model.count == 443,
        "propagation diagnostic requires the admitted Qwen3-14B Q4_K_M artifact");
    QwenCudaRuntimeConfig config;
    config.placement = QwenCudaPlacement::contiguous_split(0, 1, early_layers);
    auto runtime = QwenCudaRuntimeState::create(model, config);
    require(runtime->device_count() == 2, "propagation diagnostic requires the qualified two-GPU runtime");
    const auto root = std::filesystem::path(output_dir);
    if (std::filesystem::exists(root) && !std::filesystem::is_empty(root))
        throw std::invalid_argument("propagation output directory must be absent or empty: " + root.string());
    std::filesystem::create_directories(root);

    std::ofstream local_csv(root / "local_av_metrics.csv");
    std::ofstream capacity_csv(root / "capacity_layer_hidden.csv");
    std::ofstream qk_csv(root / "capacity_qk.csv");
    std::ofstream intervention_csv(root / "single_layer_interventions.csv");
    std::ofstream intervention_summary(root / "single_layer_summary.csv");
    std::ofstream all_native_csv(root / "all_native_decode_layers.csv");
    std::ofstream native_candidate_summary(root / "native_candidate_summary.csv");
    std::ofstream native_prefill_layers(root / "all_native_prefill_layers.csv");
    require(local_csv && capacity_csv && qk_csv && intervention_csv && intervention_summary && all_native_csv &&
        native_candidate_summary && native_prefill_layers, "cannot create propagation CSV outputs");
    local_csv << "workload,phase,position,visible_context,layer,scope,group,count,"
        << "canonical_native_max_abs,canonical_native_rms,canonical_native_relative_rms,canonical_native_cosine,"
        << "canonical_oracle_max_abs,canonical_oracle_rms,canonical_oracle_relative_rms,canonical_oracle_cosine,"
        << "native_oracle_max_abs,native_oracle_rms,native_oracle_relative_rms,native_oracle_cosine\n";
    capacity_csv << "from_capacity,to_capacity,layer,rows,max_abs,rms,relative_rms,cosine,reference_rms,candidate_rms,changed\n";
    qk_csv << "from_capacity,to_capacity,phase,query_max_abs,query_changed,active_key_max_abs,active_key_changed,"
        << "score_max_abs,score_rms,score_relative_rms,score_changed,high_capacity_qk_fp64_relative_rms,"
        << "probability_max_abs,probability_relative_rms,softmax_actual_score_relative_rms,"
        << "softmax_fp64_qk_relative_rms\n";
    intervention_csv << "position,intervention_layer,evaluated_layer,rows,hidden_max_abs,hidden_rms,hidden_relative_rms,"
        << "hidden_cosine,baseline_rms,candidate_rms,changed,inputs_bitwise_equal,native_same_input_bitwise,"
        << "local_av_rms,local_av_relative_rms\n";
    intervention_summary << "position,intervention_layer,fed_token_equal,final_top1_equal,final_hidden_max_abs,"
        << "final_hidden_rms,final_hidden_relative_rms,final_hidden_cosine,final_logits_max_abs,final_logits_rms,"
        << "final_logits_relative_rms,final_logits_cosine,native_av_operations,inputs_bitwise_equal,"
        << "native_same_input_bitwise,existing_numeric_gate\n";
    all_native_csv << "position,from_variant,to_layer,rows,max_abs,rms,relative_rms,cosine,reference_rms,candidate_rms,changed,packed_v_copy_bytes_avoided_total\n";
    native_candidate_summary << "case,position,prompt_rows,decode_steps,native_av_steps,native_av_layers,"
        << "packed_v_copy_bytes_avoided,fed_token_equal,top1_equal,final_hidden_relative_rms,final_hidden_cosine,"
        << "final_logits_relative_rms,final_logits_cosine,numeric_gate\n";
    native_prefill_layers << "phase,position,from_variant,to_layer,rows,max_abs,rms,relative_rms,cosine,"
        << "reference_rms,candidate_rms,changed,packed_v_copy_bytes_avoided_total\n";

    auto session512 = runtime->create_session(capacity);
    auto executor512 = std::make_unique<Qwen3MultiDeviceGenerationExecutor>(
        model, runtime, session512, false, true, true);
    auto baseline512 = execute_propagation_workload(*executor512, *session512, prompt, 32,
        false, true, -1);
    require(baseline512.local_av.size() == early_layers && baseline512.local_av[0].capacity == capacity,
        "capacity-512 baseline missing the fixed-context local AV boundaries");
    for (const auto & capture : baseline512.local_av) write_local_metrics(local_csv, "decode32-cap512", capture);
    save_local_captures(root / "raw" / "capacity" / "cap-512-decode32", baseline512, true);

    std::map<uint32_t, PropagationWorkload> capacity_runs;
    capacity_runs.emplace(512, baseline512);
    if (!smoke) for (uint32_t cap : {1024U, 1032U}) {
        auto session = runtime->create_session(cap);
        Qwen3MultiDeviceGenerationExecutor executor(model, runtime, session, false, false, true);
        auto run = execute_propagation_workload(executor, *session, prompt, 32, false, true, -1);
        require(run.local_av.size() == early_layers, "high-capacity local AV sweep was incomplete");
        for (const auto & capture : run.local_av) write_local_metrics(local_csv, "decode32-cap" + std::to_string(cap), capture);
        save_local_captures(root / "raw" / "capacity" / ("cap-" + std::to_string(cap) + "-decode32"), run, true);
        capacity_runs.emplace(cap, std::move(run));
    }

    const auto & cap512_av0 = av_at(capacity_runs.at(512), 0);
    for (uint32_t cap : {512U, 1024U, 1032U}) {
        if (smoke && cap != 512) continue;
        const auto & candidate = capacity_runs.at(cap);
        require(capacity_runs.at(512).tokens == candidate.tokens,
            "capacity-dependent QK comparison changed the generated token before the target decode");
        const Metrics hidden = compare(capacity_runs.at(512).final_hidden, candidate.final_hidden);
        const Metrics logits = compare(capacity_runs.at(512).final_logits, candidate.final_logits);
        std::printf("capacity_propagation cap=512->%u hidden_rel_rms=%.9g logits_rel_rms=%.9g top1_equal=%s\n",
            cap, hidden.relative_rms, logits.relative_rms,
            capacity_runs.at(512).final_top1 == candidate.final_top1 ? "yes" : "no");
        for (uint32_t layer = 0; layer < 40; ++layer) {
            const auto & a = activation_at(capacity_runs.at(512), layer);
            const auto & b = activation_at(candidate, layer);
            const Metrics m = compare(a.hidden, b.hidden);
            capacity_csv << 512 << ',' << cap << ',' << layer << ',' << a.rows << ',' << m.max_abs << ','
                << m.rms << ',' << m.relative_rms << ',' << m.cosine << ',' << m.reference_rms << ','
                << m.candidate_rms << ',' << m.changed << '\n';
        }
        if (cap == 512) continue;
        const auto & high_av0 = av_at(candidate, 0);
        const Metrics q = compare(cap512_av0.query, high_av0.query);
        const Metrics k = compare(active_keys_as_float(cap512_av0), active_keys_as_float(high_av0));
        const Metrics score = compare(visible_matrix(cap512_av0, cap512_av0.scores),
            visible_matrix(high_av0, high_av0.scores));
        const Metrics score_oracle = compare(visible_matrix(high_av0, high_av0.scores),
            visible_matrix(high_av0, high_av0.qk_oracle));
        const Metrics p = compare(visible_probabilities(cap512_av0), visible_probabilities(high_av0));
        const Metrics softmax_actual = compare(visible_probabilities(high_av0),
            visible_matrix(high_av0, high_av0.softmax_from_scores));
        const Metrics softmax_oracle = compare(visible_probabilities(high_av0),
            visible_matrix(high_av0, high_av0.softmax_from_oracle_scores));
        qk_csv << 512 << ',' << cap << ",decode," << q.max_abs << ',' << q.changed << ',' << k.max_abs << ','
            << k.changed << ',' << score.max_abs << ',' << score.rms << ',' << score.relative_rms << ','
            << score.changed << ',' << score_oracle.relative_rms << ',' << p.max_abs << ',' << p.relative_rms << ','
            << softmax_actual.relative_rms << ',' << softmax_oracle.relative_rms << '\n';
    }

    std::map<uint32_t, PropagationWorkload> position_baselines;
    std::optional<PropagationWorkload> prefill32_baseline;
    std::optional<PropagationWorkload> prefill_output_baseline;
    position_baselines.emplace(32, baseline512);
    {
        auto prefill32 = execute_propagation_workload(*executor512, *session512, prompt, 0, true, true, -1);
        for (const auto & capture : prefill32.local_av) write_local_metrics(local_csv, "prefill32-cap512", capture);
        std::filesystem::create_directories(root / "raw" / "local" / "prefill32");
        save_capture(root / "raw" / "local" / "prefill32", *std::find_if(prefill32.local_av.begin(), prefill32.local_av.end(),
            [](const CompactBoundary & item) { return item.layer == 0; }));
        prefill32_baseline = std::move(prefill32);
        require(prefill32_baseline->tokens.size() == 1,
            "32-row canonical prefill baseline did not retain its sampled decode token");
        executor512->configure_attention_av_diagnostic(0, false, -1);
        auto prefill_only = executor512->run(prompt, 1, prefill32_baseline->tokens.front());
        require(prefill_only.completed && prefill_only.tokens.empty() && session512->current_length() == rows &&
            prefill_only.layer_activation_captures.size() == 40 &&
            prefill_only.attention_av_boundary_captures.empty(),
            "canonical 32-row prefill stop-token control did not stop before decode");
        PropagationWorkload prefill_output;
        prefill_output.position = 0;
        prefill_output.prompt_rows = rows;
        prefill_output.prefill_capture = true;
        prefill_output.final_top1 = static_cast<uint32_t>(std::max_element(prefill_only.final_logits.begin(),
            prefill_only.final_logits.end()) - prefill_only.final_logits.begin());
        prefill_output.final_hidden = std::move(prefill_only.final_hidden);
        prefill_output.final_logits = std::move(prefill_only.final_logits);
        prefill_output.layer_outputs = std::move(prefill_only.layer_activation_captures);
        prefill_output_baseline = std::move(prefill_output);
        if (!smoke) for (uint32_t position : {1U, 8U, 16U}) {
            auto baseline = execute_propagation_workload(*executor512, *session512, prompt, position, false, true, -1);
            require(baseline.local_av.size() == early_layers, "decode context sweep lacks local AV captures");
            for (const auto & capture : baseline.local_av)
                write_local_metrics(local_csv, "decode" + std::to_string(position) + "-cap512", capture);
            save_local_captures(root / "raw" / "local" / ("decode-context-" + std::to_string(position)), baseline, true);
            position_baselines.emplace(position, std::move(baseline));
        }
    }
    local_csv.flush();

    uint32_t min_layer = 0, max_layer = 0;
    double min_error = std::numeric_limits<double>::infinity(), max_error = -1.0;
    for (const auto & capture : baseline512.local_av) {
        const double error = compare(capture.canonical, capture.native).relative_rms;
        if (error < min_error) { min_error = error; min_layer = capture.layer; }
        if (error > max_error) { max_error = error; max_layer = capture.layer; }
    }
    std::vector<uint32_t> intervention_layers = smoke ? std::vector<uint32_t>{8} :
        std::vector<uint32_t>{0, 8, 16, 25, min_layer, max_layer};
    std::sort(intervention_layers.begin(), intervention_layers.end());
    intervention_layers.erase(std::unique(intervention_layers.begin(), intervention_layers.end()), intervention_layers.end());
    std::printf("intervention_layer_selection fixed=[0,8,16,25] local_error_min=%u(%.9g) max=%u(%.9g) selected=",
        min_layer, min_error, max_layer, max_error);
    for (uint32_t layer : intervention_layers) std::printf("%u%s", layer,
        layer == intervention_layers.back() ? "\n" : ",");

    const std::vector<uint32_t> positions = smoke ? std::vector<uint32_t>{32} :
        std::vector<uint32_t>{1, 8, 16, 32};
    for (uint32_t position : positions) {
        const auto & baseline = position_baselines.at(position);
        for (uint32_t layer : intervention_layers) {
            auto intervention = execute_propagation_workload(*executor512, *session512, prompt,
                position, false, true, static_cast<int32_t>(layer));
            require(intervention.native_capture.has_value(),
                "native intervention did not retain its exact AV input/output boundary");
            Metrics actual_native_output;
            const bool inputs_equal = intervention_inputs_match(av_at(baseline, layer),
                *intervention.native_capture, &actual_native_output);
            require(inputs_equal,
                "single-layer intervention changed local V/probability inputs versus canonical replay");
            const Metrics local = compare(av_at(baseline, layer).canonical, av_at(baseline, layer).native);
            const bool native_output_equal = actual_native_output.changed == 0;
            const bool fed_token_equal = baseline.tokens == intervention.tokens;
            require(fed_token_equal, "single-layer intervention changed the input token before its target decode");
            for (uint32_t evaluated_layer = 0; evaluated_layer < layer; ++evaluated_layer)
                require(activation_at(baseline, evaluated_layer).hidden ==
                    activation_at(intervention, evaluated_layer).hidden,
                    "single-layer AV intervention changed an upstream transformer block");
            write_layer_drift(intervention_csv, baseline, intervention, static_cast<int32_t>(layer),
                inputs_equal, native_output_equal, local);
            const Metrics hidden = compare(baseline.final_hidden, intervention.final_hidden);
            const Metrics logits = compare(baseline.final_logits, intervention.final_logits);
            const bool numeric_gate = hidden.relative_rms <= 0.02 && hidden.cosine >= 0.9998 &&
                logits.relative_rms <= 0.02 && logits.cosine >= 0.9998;
            intervention_summary << position << ',' << layer << ',' << (fed_token_equal ? "yes" : "no") << ','
                << (baseline.final_top1 == intervention.final_top1 ? "yes" : "no") << ','
                << hidden.max_abs << ',' << hidden.rms << ',' << hidden.relative_rms << ',' << hidden.cosine << ','
                << logits.max_abs << ',' << logits.rms << ',' << logits.relative_rms << ',' << logits.cosine << ','
                << intervention.diagnostic_native_av_interventions << ',' << (inputs_equal ? "yes" : "no") << ','
                << (native_output_equal ? "yes" : "no") << ',' << (numeric_gate ? "PASS" : "FAIL") << '\n';
            std::filesystem::create_directories(root / "raw" / "interventions" /
                ("context-" + std::to_string(position)));
            save_intervention_capture(root / "raw" / "interventions" /
                ("context-" + std::to_string(position)), *intervention.native_capture,
                inputs_equal, native_output_equal);
            if (position == 32 && layer == 8) {
                auto repeat = execute_propagation_workload(*executor512, *session512, prompt,
                    position, false, false, -1);
                require(repeat.tokens == baseline.tokens && repeat.final_hidden == baseline.final_hidden &&
                    repeat.final_logits == baseline.final_logits,
                    "A/B/A canonical replay changed final outputs after a diagnostic intervention");
                for (uint32_t index = 0; index < 40; ++index)
                    require(activation_at(repeat, index).hidden == activation_at(baseline, index).hidden,
                        "A/B/A canonical replay changed a transformer-block activation");
                std::puts("diagnostic_A_B_A context=32 intervention_layer=8 canonical_replay=BITWISE_EQUAL PASS");
            }
        }
    }

    auto & optimizer = runtime->execution_optimizer();
    const QwenExecutionPlan plan = build_qwen_execution_plan(model, *runtime, *session512,
        QwenExecutionPlanPath::MultiGpu);
    const auto decode_candidate = make_qwen3_native_attention_av_candidate(plan, QwenExecutionPhase::Decode);
    require(optimizer.register_candidate(decode_candidate), "could not register decode-only all-native diagnostic candidate");
    optimizer.set_unvalidated_trial_for_testing(true);
    optimizer.set_mode(QwenOptimizerMode::Enabled);
    const uint64_t copy_bytes_per_native_step = static_cast<uint64_t>(capacity) * kv_heads * dim *
        sizeof(ggml_fp16_t) * early_layers;
    auto append_candidate_summary = [&](const char * name, uint32_t position, uint32_t prompt_rows,
            uint32_t decode_steps, uint64_t native_steps, uint64_t native_layers, uint64_t copy_bytes,
            bool fed_token_equal, bool fed_token_known, uint32_t reference_top1,
            const std::vector<float> & reference_hidden, const std::vector<float> & reference_logits,
            const Qwen3GenerationExecution & candidate) {
        const Metrics hidden = compare(reference_hidden, candidate.final_hidden);
        const Metrics logits = compare(reference_logits, candidate.final_logits);
        const uint32_t candidate_top1 = static_cast<uint32_t>(std::max_element(candidate.final_logits.begin(),
            candidate.final_logits.end()) - candidate.final_logits.begin());
        const bool top1_equal = reference_top1 == candidate_top1;
        const bool numeric_gate = hidden.relative_rms <= 0.02 && hidden.cosine >= 0.9998 &&
            logits.relative_rms <= 0.02 && logits.cosine >= 0.9998;
        native_candidate_summary << name << ',' << position << ',' << prompt_rows << ',' << decode_steps << ','
            << native_steps << ',' << native_layers << ',' << copy_bytes << ','
            << (fed_token_known ? (fed_token_equal ? "yes" : "no") : "n/a") << ','
            << (top1_equal ? "yes" : "no") << ',' << hidden.relative_rms << ',' << hidden.cosine << ','
            << logits.relative_rms << ',' << logits.cosine << ',' << (numeric_gate ? "PASS" : "FAIL") << '\n';
        std::printf("native_candidate case=%s position=%u native_steps=%llu native_layers=%llu "
            "packed_v_copy_bytes_avoided=%llu fed_token_equal=%s top1_equal=%s hidden_rel_rms=%.9g "
            "logits_rel_rms=%.9g numeric_gate=%s\n", name, position,
            static_cast<unsigned long long>(native_steps), static_cast<unsigned long long>(native_layers),
            static_cast<unsigned long long>(copy_bytes), fed_token_known ? (fed_token_equal ? "yes" : "no") : "n/a",
            top1_equal ? "yes" : "no", hidden.relative_rms, logits.relative_rms,
            numeric_gate ? "PASS" : "FAIL");
    };

    // First isolate native decode at each guarded context while prefill remains canonical.
    const std::vector<uint32_t> native_decode_positions{32};
    for (uint32_t position : native_decode_positions) {
        const auto & baseline = position_baselines.at(position);
        std::vector<uint32_t> input(prompt.begin(), prompt.begin() + position);
        executor512->configure_attention_av_diagnostic(position, false, -1, false, true);
        auto native_decode = executor512->run(input, 1);
        require(native_decode.completed && session512->current_length() == position + 1 &&
            native_decode.tokens == baseline.tokens && native_decode.native_av_steps == 1 &&
            native_decode.native_av_layers == early_layers && native_decode.packed_v_copy_bytes_avoided == copy_bytes_per_native_step &&
            native_decode.layer_activation_captures.size() == 40 && native_decode.attention_av_boundary_captures.empty(),
            "decode-only native candidate did not select exactly the eligible early-device AV blocks");
        const auto & decision = optimizer.snapshot().last_decision;
        require(decision.candidate_selected && decision.qualification_trial &&
            decision.strategy == QwenCandidateStrategy::NativeLayoutAttentionAV,
            "decode-only diagnostic did not use explicit ExecutionPlan qualification-trial selection");
        for (uint32_t layer = 0; layer < 40; ++layer) {
            const auto & reference_layer = activation_at(baseline, layer);
            const auto & candidate_layer = native_decode.layer_activation_captures[layer];
            require(candidate_layer.layer == layer && candidate_layer.rows == reference_layer.rows,
                "decode-only native candidate activation capture geometry mismatch");
            const Metrics m = compare(reference_layer.hidden, candidate_layer.hidden);
            all_native_csv << position << ",all-native-decode-only," << layer << ',' << reference_layer.rows << ','
                << m.max_abs << ',' << m.rms << ',' << m.relative_rms << ',' << m.cosine << ','
                << m.reference_rms << ',' << m.candidate_rms << ',' << m.changed << ','
                << native_decode.packed_v_copy_bytes_avoided << '\n';
        }
        append_candidate_summary("decode-only", position, position, 1, native_decode.native_av_steps,
            native_decode.native_av_layers, native_decode.packed_v_copy_bytes_avoided, true, true,
            baseline.final_top1, baseline.final_hidden, baseline.final_logits, native_decode);
    }

    // Qualify the exact guarded 32-row prefill separately, then qualify prefill + decode together.
    require(prefill32_baseline.has_value() && prefill_output_baseline.has_value(),
        "propagation run is missing its isolated 32-row prefill baseline");
    const auto prefill_candidate = make_qwen3_native_attention_av_candidate(plan, QwenExecutionPhase::Prefill);
    require(optimizer.register_candidate(prefill_candidate), "could not register all-native prefill diagnostic candidate");
    executor512->configure_attention_av_diagnostic(0, false, -1, true, false);
    auto native_prefill = executor512->run(prompt, 1, prefill32_baseline->tokens.front());
    const bool prefill_stopped_before_decode = native_prefill.tokens.empty();
    require(native_prefill.completed &&
        session512->current_length() == rows + (prefill_stopped_before_decode ? 0U : 1U) &&
        native_prefill.native_av_steps == 1 && native_prefill.native_av_layers == early_layers &&
        native_prefill.packed_v_copy_bytes_avoided == copy_bytes_per_native_step &&
        native_prefill.layer_activation_captures.size() == 40 &&
        native_prefill.attention_av_boundary_captures.empty(),
        "32-row native prefill trial did not execute exactly its eligible early-device AV blocks");
    const auto & prefill_reference = *prefill32_baseline;
    for (uint32_t layer = 0; layer < 40; ++layer) {
        const auto & reference_layer = activation_at(prefill_reference, layer);
        const auto & candidate_layer = native_prefill.layer_activation_captures[layer];
        require(candidate_layer.layer == layer && candidate_layer.rows == rows,
            "native prefill trial activation capture geometry mismatch");
        const Metrics m = compare(reference_layer.hidden, candidate_layer.hidden);
        native_prefill_layers << "prefill," << 0 << ",native-prefill32," << layer << ',' << rows << ','
            << m.max_abs << ',' << m.rms << ',' << m.relative_rms << ',' << m.cosine << ','
            << m.reference_rms << ',' << m.candidate_rms << ',' << m.changed << ','
            << native_prefill.packed_v_copy_bytes_avoided << '\n';
    }
    const auto & prefill_final_reference = prefill_stopped_before_decode ?
        *prefill_output_baseline : position_baselines.at(32);
    const bool prefill_fed_token_equal = prefill_stopped_before_decode ||
        native_prefill.tokens == prefill32_baseline->tokens;
    append_candidate_summary(prefill_stopped_before_decode ? "prefill32-only" : "prefill32-plus-canonical-decode32",
        prefill_stopped_before_decode ? 0 : 32, rows, prefill_stopped_before_decode ? 0 : 1,
        native_prefill.native_av_steps, native_prefill.native_av_layers,
        native_prefill.packed_v_copy_bytes_avoided, prefill_fed_token_equal,
        !prefill_stopped_before_decode, prefill_final_reference.final_top1,
        prefill_final_reference.final_hidden, prefill_final_reference.final_logits, native_prefill);

    executor512->configure_attention_av_diagnostic(32, false, -1, true, true);
    auto native_prefill_decode = executor512->run(prompt, 1);
    require(native_prefill_decode.completed && session512->current_length() == rows + 1 &&
        native_prefill_decode.native_av_steps == 2 && native_prefill_decode.native_av_layers == 2 * early_layers &&
        native_prefill_decode.packed_v_copy_bytes_avoided == 2 * copy_bytes_per_native_step &&
        native_prefill_decode.layer_activation_captures.size() == 40 &&
        native_prefill_decode.attention_av_boundary_captures.empty(),
        "guarded native 32-row prefill plus decode did not select both eligible AV candidates");
    const auto & combined_decision = optimizer.snapshot().last_decision;
    require(combined_decision.candidate_selected && combined_decision.qualification_trial &&
        combined_decision.strategy == QwenCandidateStrategy::NativeLayoutAttentionAV,
        "combined prefill/decode diagnostic did not use explicit ExecutionPlan qualification-trial selection");
    const auto & baseline32 = position_baselines.at(32);
    const bool combined_fed_token_equal = native_prefill_decode.tokens == baseline32.tokens;
    for (uint32_t layer = 0; layer < 40; ++layer) {
        const auto & reference_layer = activation_at(baseline32, layer);
        const auto & candidate_layer = native_prefill_decode.layer_activation_captures[layer];
        require(candidate_layer.layer == layer && candidate_layer.rows == reference_layer.rows,
            "combined native prefill/decode activation capture geometry mismatch");
        const Metrics m = compare(reference_layer.hidden, candidate_layer.hidden);
        all_native_csv << 32 << ",native-prefill32-plus-decode," << layer << ',' << reference_layer.rows << ','
            << m.max_abs << ',' << m.rms << ',' << m.relative_rms << ',' << m.cosine << ','
            << m.reference_rms << ',' << m.candidate_rms << ',' << m.changed << ','
            << native_prefill_decode.packed_v_copy_bytes_avoided << '\n';
    }
    append_candidate_summary("prefill32-plus-decode32", 32, rows, 1, native_prefill_decode.native_av_steps,
        native_prefill_decode.native_av_layers, native_prefill_decode.packed_v_copy_bytes_avoided,
        combined_fed_token_equal, true, baseline32.final_top1, baseline32.final_hidden, baseline32.final_logits,
        native_prefill_decode);

    optimizer.set_mode(QwenOptimizerMode::Shadow);
    optimizer.set_unvalidated_trial_for_testing(false);
    executor512->configure_attention_av_diagnostic(32, false, -1);
    auto canonical_replay = executor512->run(prompt, 1);
    require(canonical_replay.completed && canonical_replay.native_av_steps == 0 &&
        canonical_replay.tokens == baseline512.tokens && canonical_replay.final_hidden == baseline512.final_hidden &&
        canonical_replay.final_logits == baseline512.final_logits,
        "canonical ExecutionOptimizer replay changed after native candidate trials");
    std::puts("native_candidate_lifecycle_reset=canonical-replay-BITWISE_EQUAL PASS");

    std::ofstream manifest(root / "manifest.txt");
    manifest << "experiment=qwen3-14b-native-av-numerical-propagation-v1\n"
        << "smoke=" << (smoke ? "yes" : "no") << '\n'
        << "model=" << expected_identity << "\nsource_size=" << expected_source_size
        << "\narchitecture=40-blocks,40-query-heads,8-kv-heads,head-dim-128\n"
        << "placement=CUDA0-SM86-blocks-0..25;CUDA1-SM75-blocks-26..39,norm,head\n"
        << "canonical_gemm_commit=2d191b5dee1a591c41ee8a653ce42bfcd9c8716d\n"
        << "native_av_patch_sha256=6fe680adb0e659ef0f7f5bdc7eaa1efa64c879380026e74c45327e694e477813\n"
        << "capacity=512-for-AV-interventions;canonical-capacity-sweep=512,1024,1032\n"
        << "decode_positions=1,8,16,32;prefill=32-rows;native-eligible-blocks=0..25\n"
        << "native_guard=capacity<=512,decode-context<=32,SM86;production-guard-unchanged=yes\n"
        << "intervention_layers=";
    for (size_t i = 0; i < intervention_layers.size(); ++i)
        manifest << (i ? "," : "") << intervention_layers[i];
    manifest << "\nnative_candidate_runs=decode-only-at-32;prefill32-only;prefill32-plus-decode32\n"
        << "native_eligible_layers_per_candidate=early26-native,late14-canonical\n"
        << "packed_v_copy_avoided_bytes_per_decode_step="
        << static_cast<uint64_t>(512) * 8 * 128 * sizeof(ggml_fp16_t) * early_layers << '\n'
        << "canonical_av_fallback=unchanged\nproduction_defaults=unchanged\n";
    require(static_cast<bool>(manifest), "failed writing propagation experiment manifest");
    std::printf("QWEN_NUMERICAL_PROPAGATION=%s capacity_axis=%s "
        "AV_axis=512-positions-%s interventions=%zu all_native_early_layers=%u output=%s\n",
        smoke ? "SMOKE" : "COMPLETE", smoke ? "512" : "512,1024,1032",
        smoke ? "32" : "1,8,16,32", intervention_layers.size() * positions.size(),
        early_layers, root.string().c_str());
    return 0;
}

int run(const std::string & semantic, const std::string & source, const std::string & token_file,
        const std::string & output_dir, const std::vector<uint32_t> & capacities) {
    const std::vector<uint32_t> prompt = read_tokens(token_file);
    Qwen3Model model;
    open_qwen3_model(semantic, source, &model, true);
    require(model.artifact_identity == expected_identity && model.source_size == expected_source_size &&
        model.layer_count == 40 && model.count == 443, "AV diagnostic requires the admitted Qwen3-14B Q4_K_M artifact");
    QwenCudaRuntimeConfig config;
    config.placement = QwenCudaPlacement::contiguous_split(0, 1, early_layers);
    auto runtime = QwenCudaRuntimeState::create(model, config);
    require(runtime->device_count() == 2, "AV boundary diagnostic requires the qualified two-GPU runtime");
    std::filesystem::create_directories(output_dir);

    std::vector<CapacityRun> runs;
    runs.reserve(capacities.size());
    for (uint32_t capacity : capacities) {
        require(capacity >= rows + 1 && capacity <= 1032, "diagnostic capacity must be in 33..1032");
        std::printf("boundary_run_start capacity=%u candidate_selection=DISABLED diagnostic_native_side_branch=ENABLED\n", capacity);
        runs.push_back(run_capacity(model, runtime, prompt, capacity));
        const CapacityRun & current = runs.back();
        double max_same_input_rel = 0.0;
        uint32_t first_prefill_divergence = UINT32_MAX, first_decode_divergence = UINT32_MAX;
        for (const auto & layer : current.layers) {
            const Metrics canonical_native = compare(layer.canonical, layer.native);
            const Metrics canonical_oracle = compare(layer.oracle, layer.canonical);
            const Metrics native_oracle = compare(layer.oracle, layer.native);
            max_same_input_rel = std::max(max_same_input_rel, canonical_native.relative_rms);
            uint32_t & first_divergence = layer.prefill ? first_prefill_divergence : first_decode_divergence;
            if (first_divergence == UINT32_MAX && canonical_native.changed != 0)
                first_divergence = layer.layer;
            std::printf("av_same_input capacity=%u phase=%s layer=%u device=%u rows=%u visible_context=%u "
                "graph=[canonical=GGML_OP_MUL_MAT,native=GGML_OP_ATTENTION_AV] "
                "value_ne=[%lld,%lld] value_nb=[%zu,%zu] probability_ne=[%lld,%lld,%lld] "
                "probability_nb=[%zu,%zu,%zu] canonical_native_max_abs=%.9g canonical_native_rel_rms=%.9g "
                "canonical_native_changed=%llu canonical_oracle_rel_rms=%.9g native_oracle_rel_rms=%.9g "
                "canonical_oracle_cosine=%.12g native_oracle_cosine=%.12g\n",
                capacity, layer.prefill ? "prefill" : "decode", layer.layer, layer.device_id,
                layer.query_rows, layer.visible_context,
                static_cast<long long>(layer.value_ne[0]), static_cast<long long>(layer.value_ne[1]),
                layer.value_nb[0], layer.value_nb[1],
                static_cast<long long>(layer.probability_ne[0]), static_cast<long long>(layer.probability_ne[1]),
                static_cast<long long>(layer.probability_ne[2]), layer.probability_nb[0],
                layer.probability_nb[1], layer.probability_nb[2],
                canonical_native.max_abs, canonical_native.relative_rms,
                static_cast<unsigned long long>(canonical_native.changed),
                canonical_oracle.relative_rms, native_oracle.relative_rms,
                canonical_oracle.cosine, native_oracle.cosine);
            if (layer.layer == 0) {
                const Metrics qk = compare(visible_matrix(layer, layer.scores),
                    visible_matrix(layer, layer.qk_oracle));
                const Metrics softmax_actual = compare(visible_probabilities(layer),
                    visible_matrix(layer, layer.softmax_from_scores));
                const Metrics softmax_oracle = compare(visible_probabilities(layer),
                    visible_matrix(layer, layer.softmax_from_oracle_scores));
                std::printf("qk_softmax_oracle capacity=%u phase=%s layer=0 score_op=GGML_OP_MUL_MAT "
                    "q_ne=[%lld,%lld,%lld] k_ne=[%lld,%lld] score_ne=[%lld,%lld,%lld] "
                    "qk_fp64_rel_rms=%.9g qk_fp64_max_abs=%.9g actual_softmax_vs_actual_scores_rel_rms=%.9g "
                    "actual_softmax_vs_fp64_qk_rel_rms=%.9g\n",
                    capacity, layer.prefill ? "prefill" : "decode",
                    static_cast<long long>(layer.query_ne[0]), static_cast<long long>(layer.query_ne[1]),
                    static_cast<long long>(layer.query_ne[2]), static_cast<long long>(layer.key_ne[0]),
                    static_cast<long long>(layer.key_ne[1]), static_cast<long long>(layer.score_ne[0]),
                    static_cast<long long>(layer.score_ne[1]), static_cast<long long>(layer.score_ne[2]),
                    qk.relative_rms, qk.max_abs, softmax_actual.relative_rms, softmax_oracle.relative_rms);
            }
        }
        std::printf("boundary_run_summary capacity=%u captures=%zu first_prefill_av_difference=%s "
            "first_decode_av_difference=%s max_same_input_rel_rms=%.9g canonical_next_token=%u "
            "final_hidden_values=%zu final_logits_values=%zu\n",
            capacity, current.layers.size(), first_prefill_divergence == UINT32_MAX ? "none" :
                std::to_string(first_prefill_divergence).c_str(), first_decode_divergence == UINT32_MAX ? "none" :
                std::to_string(first_decode_divergence).c_str(), max_same_input_rel,
            current.next_token, current.final_hidden.size(), current.final_logits.size());
    }

    auto reference = std::find_if(runs.begin(), runs.end(), [](const CapacityRun & item) { return item.capacity == 512; });
    require(reference != runs.end(), "capacity list must include 512 as the cross-capacity control");
    std::vector<std::tuple<uint32_t, bool, uint32_t>> save_layers;
    for (const auto & current : runs) {
        uint32_t first_prefill_changed = UINT32_MAX, first_decode_changed = UINT32_MAX;
        if (current.capacity != 512) {
            for (size_t index = 0; index < current.layers.size(); ++index) {
                const auto & base = reference->layers[index];
                const auto & candidate = current.layers[index];
                require(base.layer == candidate.layer && base.prefill == candidate.prefill &&
                    base.query_rows == candidate.query_rows && base.visible_context == candidate.visible_context,
                    "cross-capacity AV phase/geometry mismatch");
                const Metrics v = compare(active_values_as_float(base), active_values_as_float(candidate));
                const Metrics p = compare(visible_probabilities(base), visible_probabilities(candidate));
                uint32_t & first_changed = candidate.prefill ? first_prefill_changed : first_decode_changed;
                if (first_changed == UINT32_MAX && (v.changed != 0 || p.changed != 0)) first_changed = candidate.layer;
                const Metrics canonical = compare(base.canonical, candidate.canonical);
                const Metrics native = compare(base.native, candidate.native);
                if (candidate.layer == 0) {
                    const Metrics q = compare(base.query, candidate.query);
                    const Metrics k = compare(active_keys_as_float(base), active_keys_as_float(candidate));
                    const Metrics score = compare(visible_matrix(base, base.scores),
                        visible_matrix(candidate, candidate.scores));
                    const Metrics score_oracle = compare(visible_matrix(candidate, candidate.scores),
                        visible_matrix(candidate, candidate.qk_oracle));
                    const Metrics probability_oracle = compare(visible_probabilities(candidate),
                        visible_matrix(candidate, candidate.softmax_from_scores));
                    std::printf("cross_capacity_qk_root from=%u to=%u phase=%s layer=0 q_max_abs=%.9g q_changed=%llu "
                        "k_active_max_abs=%.9g k_active_changed=%llu score_max_abs=%.9g score_rel_rms=%.9g "
                        "score_changed=%llu high_capacity_qk_fp64_rel_rms=%.9g "
                        "softmax_actual_scores_rel_rms=%.9g\n",
                        reference->capacity, current.capacity, candidate.prefill ? "prefill" : "decode",
                        q.max_abs, static_cast<unsigned long long>(q.changed), k.max_abs,
                        static_cast<unsigned long long>(k.changed), score.max_abs, score.relative_rms,
                        static_cast<unsigned long long>(score.changed), score_oracle.relative_rms,
                        probability_oracle.relative_rms);
                }
                std::printf("cross_capacity from=%u to=%u phase=%s layer=%u active_v_max_abs=%.9g active_v_changed=%llu "
                    "visible_p_max_abs=%.9g visible_p_rel_rms=%.9g visible_p_changed=%llu "
                    "canonical_av_rel_rms=%.9g native_av_rel_rms=%.9g\n",
                    reference->capacity, current.capacity, candidate.prefill ? "prefill" : "decode", candidate.layer,
                    v.max_abs, static_cast<unsigned long long>(v.changed), p.max_abs, p.relative_rms,
                    static_cast<unsigned long long>(p.changed), canonical.relative_rms, native.relative_rms);
            }
            std::printf("cross_capacity_first_input_drift from=%u to=%u phase=prefill layer=%s\n",
                reference->capacity, current.capacity, first_prefill_changed == UINT32_MAX ? "none" :
                    std::to_string(first_prefill_changed).c_str());
            std::printf("cross_capacity_first_input_drift from=%u to=%u phase=decode layer=%s\n",
                reference->capacity, current.capacity, first_decode_changed == UINT32_MAX ? "none" :
                    std::to_string(first_decode_changed).c_str());
            if (first_prefill_changed != UINT32_MAX)
                save_layers.emplace_back(current.capacity, true, first_prefill_changed);
            if (first_decode_changed != UINT32_MAX)
                save_layers.emplace_back(current.capacity, false, first_decode_changed);
        }
        save_layers.emplace_back(current.capacity, true, 0);
        save_layers.emplace_back(current.capacity, false, 0);
    }
    std::sort(save_layers.begin(), save_layers.end());
    save_layers.erase(std::unique(save_layers.begin(), save_layers.end()), save_layers.end());
    for (const auto & key : save_layers) {
        const uint32_t capacity = std::get<0>(key);
        const bool prefill = std::get<1>(key);
        const uint32_t layer = std::get<2>(key);
        const auto run_it = std::find_if(runs.begin(), runs.end(), [&](const CapacityRun & item) {
            return item.capacity == capacity;
        });
        require(run_it != runs.end(), "selected boundary capacity not found");
        const auto capture = std::find_if(run_it->layers.begin(), run_it->layers.end(), [&](const CompactBoundary & item) {
            return item.prefill == prefill && item.layer == layer;
        });
        require(capture != run_it->layers.end(), "selected boundary capture not found");
        save_capture(output_dir, *capture);
    }

    for (const auto & current : runs) {
        const Metrics hidden = compare(reference->final_hidden, current.final_hidden);
        const Metrics logits = compare(reference->final_logits, current.final_logits);
        std::printf("canonical_cross_capacity from=%u to=%u token_equal=%s hidden_rel_rms=%.9g logits_rel_rms=%.9g "
            "hidden_cosine=%.12g logits_cosine=%.12g\n", reference->capacity, current.capacity,
            reference->next_token == current.next_token ? "yes" : "no", hidden.relative_rms,
            logits.relative_rms, hidden.cosine, logits.cosine);
    }
    std::printf("AV_BOUNDARY_DIAGNOSTIC=COMPLETE candidates=DISABLED native_side_branch=DIAGNOSTIC_ONLY oracle=FP64\n");
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 6 && std::string(argv[5]) == "--propagation")
            return run_propagation(argv[1], argv[2], argv[3], argv[4], false);
        if (argc == 6 && std::string(argv[5]) == "--propagation-smoke")
            return run_propagation(argv[1], argv[2], argv[3], argv[4], true);
        if (argc < 6) throw std::invalid_argument(
            "usage: qwen3_native_av_boundary_diagnostic SEMANTIC.vbuf SOURCE_URL TOKEN_IDS_FILE OUTPUT_DIR CAPACITY [CAPACITY ...]");
        std::vector<uint32_t> capacities;
        for (int i = 5; i < argc; ++i) capacities.push_back(static_cast<uint32_t>(std::stoul(argv[i])));
        return run(argv[1], argv[2], argv[3], argv[4], capacities);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "qwen3_native_av_boundary_diagnostic=FAIL: %s\n", error.what());
        return 1;
    }
}
