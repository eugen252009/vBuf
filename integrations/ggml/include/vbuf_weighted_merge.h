#pragma once

#include <string>
#include <vector>

namespace vbuf_ggml {

// Merges compatible float value vectors using finite scalar weights.
bool weighted_merge(const std::vector<std::vector<float>> & values,
    const std::vector<float> & weights, std::vector<float> * output,
    std::string * error = nullptr);

} // namespace vbuf_ggml
