#include "vbuf_range_source.h"

#include <cassert>
#include <cstdint>
#include <cstring>

int main() {
    const uint8_t artifact[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    vbuf_ggml::LocalVbufRangeSource source(artifact, sizeof(artifact));
    uint8_t destination[3] = {};
    vbuf_ggml::RangeReadResult result;
    assert(source.read_range(2, sizeof(destination), destination, &result));
    assert(std::memcmp(destination, artifact + 2, sizeof(destination)) == 0);
    assert(result.requested_offset == 2);
    assert(result.returned_bytes == sizeof(destination));
    assert(!source.read_range(7, 2, destination, &result));
    return 0;
}
