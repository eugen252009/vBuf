#include "vbuf_runtime_state.h"

#include <cassert>
#include <vector>

int main() {
    using vbuf_ggml::RuntimeStateSlot;
    RuntimeStateSlot block_a(4, 2), block_b(4, 2);
    const std::vector<float> a{ 1, 2, 3, 4 };
    const std::vector<float> b{ 5, 6, 7, 8 };
    std::vector<float> value;
    std::string error;
    assert(block_a.append(a, &error));
    assert(!block_b.read(0, &value, &error));
    assert(block_b.append(b, &error));
    assert(block_a.read(0, &value, &error));
    assert(value == a);
    assert(block_b.read(0, &value, &error));
    assert(value == b);
    return 0;
}
