#include "vbuf_topk.h"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

using namespace vbuf_ggml;

int main() {
    TopKSelection selection;
    std::string error;
    assert(deterministic_top_k({ 1.0f, 3.0f, 2.0f, 3.0f }, 4, 3, &selection, &error));
    assert((selection.ids == std::vector<uint32_t>{ 1, 3, 2 }));
    assert(selection.scores[0] == 3.0f && selection.scores[1] == 3.0f);

    assert(!deterministic_top_k({ 1.0f, 2.0f }, 3, 1, &selection, &error));
    assert(!deterministic_top_k({ 1.0f, 2.0f }, 2, 0, &selection, &error));
    assert(!deterministic_top_k({ 1.0f, 2.0f }, 2, 3, &selection, &error));
    assert(!deterministic_top_k({ 1.0f, NAN }, 2, 1, &selection, &error));
    assert(selection.ids.empty());
    return 0;
}
