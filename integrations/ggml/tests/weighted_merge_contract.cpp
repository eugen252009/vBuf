#include "vbuf_weighted_merge.h"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

using namespace vbuf_ggml;

int main() {
    std::vector<float> output;
    std::string error;
    assert(weighted_merge({ { 1.0f, 2.0f }, { 3.0f, 4.0f } },
        { 0.25f, 0.75f }, &output, &error));
    assert(output.size() == 2 && output[0] == 2.5f && output[1] == 3.5f);
    assert(!weighted_merge({}, {}, &output, &error));
    assert(!weighted_merge({ { 1.0f } }, {}, &output, &error));
    assert(!weighted_merge({ { 1.0f }, { 2.0f, 3.0f } }, { 0.5f, 0.5f }, &output, &error));
    assert(!weighted_merge({ { 1.0f } }, { NAN }, &output, &error));
    return 0;
}
