#include "vbuf_generation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace vbuf_ggml;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<uint32_t> prompt(size_t count) {
    static const std::vector<uint32_t> pattern{
        0,25,220,16,13,15,13,15,198,262,549,0,220,16,13,15,13,15,198,262,549,
        0,220,16,13,15,13,15,198,262,549,0 };
    std::vector<uint32_t> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) result.push_back(pattern[i % pattern.size()]);
    return result;
}

VbufGenerationConfig request(const std::string & semantic, const std::string & endpoint,
    uint32_t capacity, size_t prompt_count, uint32_t generated) {
    VbufGenerationConfig config;
    config.semantic_model = semantic;
    config.source_endpoint = endpoint;
    config.block_count = 40;
    config.context_capacity = capacity;
    config.prompt_tokens = prompt(prompt_count);
    config.max_new_tokens = generated;
    return config;
}

void require_finite(const std::vector<float> & logits) {
    require(!logits.empty(), "production result did not return final logits");
    require(std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); }),
        "production final logits contain non-finite values");
}

void report_workload(const char * label, const VbufGenerationResult & result,
    uint32_t expected_length) {
    require(result.error.empty() && result.completed, std::string(label) + " production generation failed: " + result.error);
    require(result.completed_positions > 0, std::string(label) + " did not report completed positions");
    require_finite(result.final_logits);
    const double tok_s = result.decode_ns == 0 ? 0.0 :
        static_cast<double>(result.tokens.size()) * 1e9 / result.decode_ns;
    std::cout << "production_workload=" << label
        << " generated=" << result.tokens.size()
        << " final_length=" << expected_length
        << " prefill_ms=" << result.prefill_ns / 1e6
        << " decode_ms=" << result.decode_ns / 1e6
        << " decode_tok_s=" << tok_s
        << " peak_vram_bytes=" << result.peak_vram_bytes
        << " post_run_free_vram_bytes=" << result.post_run_free_vram_bytes
        << " model_upload_bytes=" << result.qwen_model_upload_bytes
        << " session_H2D_calls=" << result.qwen_session_h2d_calls
        << " session_H2D_bytes=" << result.qwen_session_h2d_bytes
        << " session_D2H_calls=" << result.qwen_session_d2h_calls
        << " session_D2H_bytes=" << result.qwen_session_d2h_bytes << '\n';
}
}

int main(int argc, char ** argv) {
    try {
        require(argc == 4, "usage: qwen3_production_session_qualification SEMANTIC_BOOTSTRAP SOURCE_URL FINAL_LOGITS_OUTPUT");
        const std::string semantic = argv[1];
        const std::string endpoint = argv[2];
        auto runtime = std::make_shared<VbufModelRuntime>(semantic, 40);
        auto session_a = runtime->create_session(40);
        auto session_b = runtime->create_session(40);
        VbufGenerationConfig request_a = request(semantic, endpoint, 40, 32, 8);
        const std::vector<uint32_t> qualification_tokens{220,16,13,15,13,15,198,262};

        const VbufGenerationResult first = session_a->run(request_a);
        report_workload("32+8_first", first, 40);
        require(first.tokens == qualification_tokens,
            "production 32+8 token trajectory differs from the migrated qualification runner");
        require(session_a->current_context_length() == 40, "production 32+8 logical length mismatch");
        const auto first_logits = first.final_logits;
        {
            std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
            require(output.good(), "could not create final-logits qualification output");
            output.write(reinterpret_cast<const char *>(first_logits.data()),
                static_cast<std::streamsize>(first_logits.size() * sizeof(float)));
            require(output.good(), "could not write final-logits qualification output");
        }
        VbufGenerationConfig overflowing = request_a;
        overflowing.max_new_tokens = 9;
        const auto failed_lifecycle = session_a->run(overflowing);
        require(!failed_lifecycle.error.empty() && session_a->current_context_length() == 0,
            "over-capacity request did not fail closed and reset logical session state");
        const auto loaded_snapshot = runtime->snapshot();
        require(loaded_snapshot.qwen_model_upload_tensors == 443 &&
            loaded_snapshot.qwen_model_upload_bytes == 8995793920ULL,
            "canonical runtime did not retain the exact one-time model residency");

        session_a->reset();
        require(session_a->current_context_length() == 0, "production reset did not clear logical length");
        require(runtime->snapshot().qwen_model_upload_bytes == loaded_snapshot.qwen_model_upload_bytes,
            "production reset discarded runtime weight residency");
        const VbufGenerationResult replay = session_a->run(request_a);
        report_workload("32+8_reset_replay", replay, 40);
        require(replay.tokens == first.tokens && replay.final_logits.size() == first_logits.size() &&
            std::memcmp(replay.final_logits.data(), first_logits.data(), first_logits.size() * sizeof(float)) == 0,
            "production reset replay changed tokens or final logits");

        VbufGenerationConfig request_b = request(semantic, endpoint, 40, 32, 8);
        request_b.prompt_tokens[0] = 25;
        const VbufGenerationResult result_b = session_b->run(request_b);
        report_workload("session-B", result_b, 40);
        require(session_b->current_context_length() == 40 && session_a->current_context_length() == 40,
            "production sessions contaminated each other's logical lengths");
        session_b->reset();
        require(session_b->current_context_length() == 0 && session_a->current_context_length() == 40,
            "reset of production session B changed session A");
        session_b.reset();
        require(session_a->current_context_length() == 40,
            "destroying production session B changed session A");
        session_a->reset();
        const auto isolated_replay = session_a->run(request_a);
        report_workload("session-A-after-B", isolated_replay, 40);
        require(isolated_replay.tokens == first.tokens &&
            isolated_replay.final_logits.size() == first_logits.size() &&
            std::memcmp(isolated_replay.final_logits.data(), first_logits.data(), first_logits.size() * sizeof(float)) == 0,
            "session B execution/reset/destruction contaminated session A output");

        const auto before_compat = runtime->snapshot();
        const VbufGenerationResult compatibility = runtime->run(request_a);
        report_workload("runtime.run-32+8", compatibility, 40);
        require(compatibility.tokens == first.tokens && compatibility.final_logits.size() == first_logits.size() &&
            std::memcmp(compatibility.final_logits.data(), first_logits.data(), first_logits.size() * sizeof(float)) == 0,
            "whole-request runtime.run differs from canonical session execution");
        const auto after_compat = runtime->snapshot();
        require(after_compat.qwen_model_upload_bytes == before_compat.qwen_model_upload_bytes &&
            after_compat.qwen_model_upload_tensors == before_compat.qwen_model_upload_tensors,
            "runtime.run recreated or reuploaded the Qwen model backend");

        auto session_128 = runtime->create_session(160);
        const auto result_128 = session_128->run(request(semantic, endpoint, 160, 128, 32));
        report_workload("128+32", result_128, 160);
        require(session_128->current_context_length() == 160, "128+32 final session length mismatch");
        session_128.reset();

        auto oversized_session = runtime->create_session(1033);
        const auto oversized_result = oversized_session->run(request(semantic, endpoint, 1033, 32, 8));
        require(!oversized_result.error.empty() && oversized_session->current_context_length() == 0,
            "unsupported capacity above 1032 did not fail closed");
        oversized_session.reset();

        auto session_512 = runtime->create_session(520);
        const auto result_512 = session_512->run(request(semantic, endpoint, 520, 512, 8));
        report_workload("512+8", result_512, 520);
        require(session_512->current_context_length() == 520, "512+8 final session length mismatch");
        session_512.reset();

        auto session_1024 = runtime->create_session(1032);
        const auto result_1024 = session_1024->run(request(semantic, endpoint, 1032, 1024, 8));
        report_workload("1024+8", result_1024, 1032);
        require(session_1024->current_context_length() == 1032, "1024+8 final session length mismatch");

        const auto final_snapshot = runtime->snapshot();
        require(final_snapshot.qwen_model_upload_tensors == 443 &&
            final_snapshot.qwen_model_upload_bytes == 8995793920ULL,
            "production workloads caused model reupload");
        require(final_snapshot.resident_count == 443,
            "production runtime did not retain all model tensors");
        std::cout << "qwen3_canonical_production_qualification=PASS exact_artifact=YES "
            << "same_shape_tokens=PASS reset=PASS sequential_sessions=PASS runtime_run=PASS "
            << "128+32=PASS 512+8=PASS 1024+8=PASS true_concurrency=UNSUPPORTED\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qwen3_production_session_qualification: " << error.what() << '\n';
        return 1;
    }
}
