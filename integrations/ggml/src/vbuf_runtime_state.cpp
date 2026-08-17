#include "vbuf_runtime_state.h"

namespace vbuf_ggml {

RuntimeStateSlot::RuntimeStateSlot(uint32_t width, uint32_t max_positions)
    : width_(width), max_positions_(max_positions) {}

bool RuntimeStateSlot::append(const std::vector<float> & value, std::string * error) {
    if (value.size() != width_) {
        if (error != nullptr) *error = "state value width mismatch";
        return false;
    }
    if (values_.size() >= max_positions_) {
        if (error != nullptr) *error = "state position capacity exceeded";
        return false;
    }
    values_.push_back(value);
    return true;
}

bool RuntimeStateSlot::read(uint32_t position, std::vector<float> * value,
    std::string * error) const {
    if (value == nullptr) {
        if (error != nullptr) *error = "state output is null";
        return false;
    }
    if (position >= values_.size()) {
        if (error != nullptr) *error = "state position is unavailable";
        return false;
    }
    ++read_count_;
    *value = values_[position];
    return true;
}

} // namespace vbuf_ggml
