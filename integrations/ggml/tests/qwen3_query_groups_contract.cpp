#include "qwen3_query_groups.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

using vbuf_ggml::Qwen3QueryGroup;
using vbuf_ggml::qwen3_query_groups;

int main() {
    struct Expected { uint32_t length, group_count; };
    const std::array<Expected, 7> tests{{
        {8, 1}, {9, 2}, {16, 2}, {17, 3}, {24, 3}, {25, 4}, {32, 4},
    }};
    for (const Expected test : tests) {
        const auto groups = qwen3_query_groups(test.length, 8);
        assert(groups.size() == test.group_count);
        std::vector<uint32_t> assigned(test.length, 0);
        for (const auto & group : groups) {
            assert(group.compute_extent % 8 == 0);
            for (uint32_t row = 0; row < group.query_count; ++row) {
                const uint32_t query = group.first_query + row;
                const uint32_t logical = query + 1;
                assert((logical + 7) / 8 * 8 == group.compute_extent);
                const uint32_t causal_masked = group.compute_extent - logical;
                const uint32_t synthetic_padding = group.compute_extent > test.length
                    ? group.compute_extent - test.length : 0;
                assert(causal_masked >= synthetic_padding);
                ++assigned[query];
            }
        }
        for (uint32_t count : assigned) assert(count == 1);
    }
    const auto groups = qwen3_query_groups(32, 8);
    assert(groups[0].first_query == 0 && groups[0].query_count == 8 && groups[0].compute_extent == 8);
    assert(groups[1].first_query == 8 && groups[1].query_count == 8 && groups[1].compute_extent == 16);
    assert(groups[2].first_query == 16 && groups[2].query_count == 8 && groups[2].compute_extent == 24);
    assert(groups[3].first_query == 24 && groups[3].query_count == 8 && groups[3].compute_extent == 32);

    bool rejected = false;
    try { (void) qwen3_query_groups(0, 8); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { (void) qwen3_query_groups(8, 0); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    return 0;
}
