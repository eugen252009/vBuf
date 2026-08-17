#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vbuf_ggml {

// Generic append-only float state used by qualification paths that carry
// position-indexed runtime values across executions. It owns no model data.
class RuntimeStateSlot final {
public:
    RuntimeStateSlot(uint32_t width, uint32_t max_positions);

    bool append(const std::vector<float> & value, std::string * error = nullptr);
    bool read(uint32_t position, std::vector<float> * value,
        std::string * error = nullptr) const;
    uint32_t width() const { return width_; }
    uint32_t size() const { return static_cast<uint32_t>(values_.size()); }
    uint64_t read_count() const { return read_count_; }
    void clear() { values_.clear(); }

private:
    uint32_t width_;
    uint32_t max_positions_;
    std::vector<std::vector<float>> values_;
    mutable uint64_t read_count_ = 0;
};

} // namespace vbuf_ggml
