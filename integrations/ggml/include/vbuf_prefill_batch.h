#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vbuf_ggml {

// A prompt batch describes ordered rows; it does not imply independent
// execution. State-dependent operations must consume rows in position order.
struct PromptBatch {
    std::vector<uint32_t> token_ids;
    uint32_t first_position = 0;

    size_t size() const { return token_ids.size(); }

    uint32_t position(size_t row) const {
        if (row >= token_ids.size()) throw std::out_of_range("prompt batch row");
        return first_position + static_cast<uint32_t>(row);
    }

    bool causally_visible(size_t query_row, size_t key_row) const {
        return key_row <= query_row;
    }

    void validate() const {
        if (token_ids.empty()) throw std::invalid_argument("prompt batch is empty");
        if (token_ids.size() > UINT32_MAX - first_position)
            throw std::invalid_argument("prompt batch position overflow");
    }
};

} // namespace vbuf_ggml
