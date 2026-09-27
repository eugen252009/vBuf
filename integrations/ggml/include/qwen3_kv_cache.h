#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vbuf_ggml {

struct Qwen3KvGeometry {
    uint32_t layers = 0;
    uint32_t query_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t max_positions = 0;
};

// Isolated Qwen3 append-only KV storage. Entries are opaque F16 bit patterns
// in token-major [position, kv_head, head_component] order. The caller supplies
// already-RoPE-transformed K and unrotated V bytes.
class Qwen3KvCache {
public:
    explicit Qwen3KvCache(Qwen3KvGeometry geometry) : geometry_(geometry) {
        if (geometry_.layers == 0 || geometry_.query_heads == 0 ||
            geometry_.kv_heads == 0 || geometry_.head_dim == 0 ||
            geometry_.max_positions == 0 ||
            geometry_.query_heads % geometry_.kv_heads != 0) {
            throw std::invalid_argument("invalid Qwen3 KV geometry");
        }
        if (geometry_.kv_heads > std::numeric_limits<size_t>::max() / geometry_.head_dim / 2) {
            throw std::length_error("Qwen3 KV token bytes overflow");
        }
        token_bytes_ = static_cast<size_t>(geometry_.kv_heads) * geometry_.head_dim * 2;
        keys_.resize(geometry_.layers);
        values_.resize(geometry_.layers);
        layer_lengths_.resize(geometry_.layers, 0);
    }

    const Qwen3KvGeometry & geometry() const noexcept { return geometry_; }
    uint32_t completed_positions() const noexcept { return completed_positions_; }
    uint32_t layer_length(uint32_t layer) const { return layer_lengths_.at(layer); }
    size_t bytes() const noexcept {
        size_t total = 0;
        for (size_t layer = 0; layer < keys_.size(); ++layer)
            total += keys_[layer].size() + values_[layer].size();
        return total;
    }
    size_t bytes_per_token() const noexcept { return token_bytes_ * geometry_.layers * 2; }
    uint32_t kv_head_for_query(uint32_t query_head) const {
        if (query_head >= geometry_.query_heads) throw std::out_of_range("Qwen3 query head");
        return query_head / (geometry_.query_heads / geometry_.kv_heads);
    }
    size_t element_byte_offset(uint32_t position, uint32_t kv_head, uint32_t component) const {
        if (position >= geometry_.max_positions || kv_head >= geometry_.kv_heads ||
            component >= geometry_.head_dim) throw std::out_of_range("Qwen3 KV element");
        return ((static_cast<size_t>(position) * geometry_.kv_heads + kv_head) *
            geometry_.head_dim + component) * sizeof(uint16_t);
    }

    void append(uint32_t layer, uint32_t position,
        const uint8_t * key_f16, size_t key_bytes,
        const uint8_t * value_f16, size_t value_bytes) {
        if (layer >= geometry_.layers || position >= geometry_.max_positions ||
            position != completed_positions_ || layer_lengths_[layer] != position ||
            key_f16 == nullptr || value_f16 == nullptr ||
            key_bytes != token_bytes_ || value_bytes != token_bytes_) {
            throw std::invalid_argument("Qwen3 KV append contract violation");
        }
        keys_[layer].insert(keys_[layer].end(), key_f16, key_f16 + key_bytes);
        values_[layer].insert(values_[layer].end(), value_f16, value_f16 + value_bytes);
        ++layer_lengths_[layer];
    }

    const uint8_t * key_bytes(uint32_t layer) const { return keys_.at(layer).data(); }
    const uint8_t * value_bytes(uint32_t layer) const { return values_.at(layer).data(); }
    size_t layer_bytes(uint32_t layer) const { return keys_.at(layer).size(); }

    void complete_position(uint32_t position) {
        if (position != completed_positions_ || position >= geometry_.max_positions ||
            std::any_of(layer_lengths_.begin(), layer_lengths_.end(),
                [position](uint32_t length) { return length != position + 1; })) {
            throw std::invalid_argument("Qwen3 KV position cannot be completed");
        }
        ++completed_positions_;
    }

    // Zero before releasing logical lengths so a subsequent request cannot
    // observe data left in retained vector capacity.
    void reset() noexcept {
        for (auto * tensors : { &keys_, &values_ })
            for (auto & layer : *tensors) {
                std::fill(layer.begin(), layer.end(), uint8_t{0});
                layer.clear();
            }
        std::fill(layer_lengths_.begin(), layer_lengths_.end(), 0);
        completed_positions_ = 0;
    }

private:
    Qwen3KvGeometry geometry_;
    size_t token_bytes_ = 0;
    uint32_t completed_positions_ = 0;
    std::vector<std::vector<uint8_t>> keys_;
    std::vector<std::vector<uint8_t>> values_;
    std::vector<uint32_t> layer_lengths_;
};

} // namespace vbuf_ggml
