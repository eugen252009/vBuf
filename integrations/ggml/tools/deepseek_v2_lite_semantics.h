#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vbuf_ggml {

// Execution profile of the direct DeepSeek-V2-Lite graph (2048 hidden, 27
// layers). These are model semantics, not generic vBuf or tensor-adapter rules.
// The pinned source uses interleaved RoPE, YaRN factor 40, original context
// 4096, beta_fast/slow 32/1, and exported YaRN log multiplier 0.0707. Its converted artifact
// predates optional YaRN metadata; keep this explicit with the other fixed
// model geometry, rather than silently infer scaling from a file size/context.
struct DeepSeekV2LiteRope {
    float base = 10000.0f;
    float frequency_scale = 1.0f / 40.0f;
    uint32_t original_context = 4096;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
    float all_dimension_log_multiplier = 0.0707f / 0.1f;

    float attention_scale(uint32_t head_width) const {
        const float magnitude = 1.0f + 0.1f * all_dimension_log_multiplier *
            std::log(1.0f / frequency_scale);
        return magnitude * magnitude / std::sqrt(static_cast<float>(head_width));
    }
};

inline std::vector<float> deepseek_rotary(const std::vector<float> & input,
    uint32_t position, const DeepSeekV2LiteRope & parameters = {}) {
    if (input.empty() || input.size() % 2 != 0)
        throw std::invalid_argument("rotary vector must contain complete pairs");
    const float dimension = static_cast<float>(input.size());
    constexpr float pi = 3.14159265358979323846f;
    const auto correction = [&](float rotations) {
        return dimension * std::log(parameters.original_context / (rotations * 2.0f * pi)) /
            (2.0f * std::log(parameters.base));
    };
    const float low = std::max(0.0f, std::floor(correction(parameters.beta_fast)));
    const float high = std::min(dimension - 1.0f, std::ceil(correction(parameters.beta_slow)));
    const float theta_scale = std::pow(parameters.base, -2.0f / dimension);
    float theta = static_cast<float>(position);
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); i += 2) {
        const float ramp = 1.0f - std::clamp((static_cast<float>(i / 2) - low) /
            std::max(0.001f, high - low), 0.0f, 1.0f);
        const float interpolated = parameters.frequency_scale * theta;
        const float angle = std::fma(interpolated, 1.0f - ramp, theta * ramp);
        // DeepSeek cancels YaRN's rotary magnitude adjustment; the magnitude
        // correction belongs to the attention score scale instead.
        const float cosine = std::cos(angle), sine = std::sin(angle);
        output[i] = std::fma(input[i], cosine, -input[i + 1] * sine);
        output[i + 1] = std::fma(input[i], sine, input[i + 1] * cosine);
        theta *= theta_scale;
    }
    return output;
}

} // namespace vbuf_ggml
