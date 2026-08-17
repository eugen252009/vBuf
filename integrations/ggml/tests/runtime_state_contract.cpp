#include "vbuf_runtime_state.h"

#include <cassert>
#include <string>
#include <vector>

using namespace vbuf_ggml;

int main() {
    RuntimeStateSlot state(2, 2);
    std::string error;
    assert(state.append({ 1.0f, 2.0f }, &error));
    assert(state.append({ 3.0f, 4.0f }, &error));
    std::vector<float> value;
    assert(state.read(0, &value, &error) && value[1] == 2.0f);
    assert(!state.read(2, &value, &error));
    assert(!state.append({ 5.0f }, &error));
    assert(!state.append({ 5.0f, 6.0f }, &error));
    return 0;
}
