#include "vbuf_weighted_merge.h"

#include <cmath>

namespace vbuf_ggml {

bool weighted_merge(const std::vector<std::vector<float>> & values,
    const std::vector<float> & weights, std::vector<float> * output,
    std::string * error) {
    if (output == nullptr) {
        if (error != nullptr) *error = "merge output is null";
        return false;
    }
    output->clear();
    if (values.empty()) {
        if (error != nullptr) *error = "merge requires at least one value";
        return false;
    }
    if (values.size() != weights.size()) {
        if (error != nullptr) *error = "merge value and weight counts differ";
        return false;
    }
    const size_t width = values.front().size();
    if (width == 0) {
        if (error != nullptr) *error = "merge values must be non-empty";
        return false;
    }
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i].size() != width) {
            if (error != nullptr) *error = "merge value shapes differ";
            return false;
        }
        if (!std::isfinite(weights[i])) {
            if (error != nullptr) *error = "merge weight is non-finite";
            return false;
        }
    }
    output->assign(width, 0.0f);
    for (size_t i = 0; i < values.size(); ++i)
        for (size_t j = 0; j < width; ++j) (*output)[j] += weights[i] * values[i][j];
    return true;
}

} // namespace vbuf_ggml
