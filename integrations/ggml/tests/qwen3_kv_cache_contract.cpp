#include "qwen3_kv_cache.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

using vbuf_ggml::Qwen3KvCache;
using vbuf_ggml::Qwen3KvGeometry;

int main() {
    constexpr size_t row_bytes = 8u * 128u * 2u;
    Qwen3KvCache cache({40, 40, 8, 128, 32});
    assert(cache.geometry().query_heads / cache.geometry().kv_heads == 5);
    assert(cache.kv_head_for_query(0) == 0);
    assert(cache.kv_head_for_query(4) == 0);
    assert(cache.kv_head_for_query(5) == 1);
    assert(cache.kv_head_for_query(39) == 7);
    assert(cache.element_byte_offset(0, 0, 0) == 0);
    assert(cache.element_byte_offset(1, 0, 0) == row_bytes);
    assert(cache.element_byte_offset(1, 1, 0) == row_bytes + 128u * 2u);
    assert(cache.element_byte_offset(7, 7, 127) == 8u * row_bytes - 2u);
    assert(cache.bytes_per_token() == 40u * 8u * 128u * 2u * 2u);
    assert(cache.bytes() == 0);
    std::array<uint8_t, row_bytes> key{}, value{};

    for (uint32_t position = 0; position < 32; ++position) {
        for (uint32_t layer = 0; layer < 40; ++layer) {
            key.fill(static_cast<uint8_t>(position + layer));
            value.fill(static_cast<uint8_t>(0x80u + position + layer));
            cache.append(layer, position, key.data(), key.size(), value.data(), value.size());
            assert(cache.layer_length(layer) == position + 1);
        }
        cache.complete_position(position);
        assert(cache.completed_positions() == position + 1);
    }
    assert(cache.layer_bytes(39) == 32u * row_bytes);
    assert(cache.bytes() == 32u * cache.bytes_per_token());
    for (uint32_t layer : {0u, 17u, 39u}) {
        for (uint32_t position : {0u, 1u, 7u, 8u, 15u, 16u, 23u, 24u, 31u}) {
            const size_t offset = static_cast<size_t>(position) * row_bytes;
            const uint8_t expected_key = static_cast<uint8_t>(position + layer);
            const uint8_t expected_value = static_cast<uint8_t>(0x80u + position + layer);
            assert(cache.key_bytes(layer)[offset] == expected_key);
            assert(cache.key_bytes(layer)[offset + row_bytes - 1] == expected_key);
            assert(cache.value_bytes(layer)[offset] == expected_value);
        }
    }

    bool rejected = false;
    try { cache.append(0, 32, key.data(), key.size(), value.data(), value.size()); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);

    cache.reset();
    assert(cache.completed_positions() == 0 && cache.bytes() == 0);
    for (uint32_t layer = 0; layer < 40; ++layer) assert(cache.layer_length(layer) == 0);
    cache.append(0, 0, key.data(), key.size(), value.data(), value.size());
    assert(cache.key_bytes(0)[0] == key[0]);
    cache.reset();
    assert(cache.bytes() == 0 && cache.completed_positions() == 0);

    rejected = false;
    try { Qwen3KvCache invalid({40, 40, 0, 128, 8}); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { Qwen3KvCache invalid({40, 40, 6, 128, 8}); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    return 0;
}
