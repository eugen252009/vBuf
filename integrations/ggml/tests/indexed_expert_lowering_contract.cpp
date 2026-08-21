#include "vbuf_indexed_expert.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

int main() {
    constexpr uint32_t expert_count = 5;
    constexpr uint64_t member_bytes = 12;
    alignas(64) std::array<uint8_t, expert_count * member_bytes> payload{};
    for (size_t index = 0; index < payload.size(); ++index) payload[index] = static_cast<uint8_t>(index);
    uint64_t dimensions[3] = { 4, 3, expert_count };
    const vbuf_ggml::PersistentTensorRef bank{
        91, "fixture.experts", { 7, 3, dimensions, payload.data(), payload.size() }, 1000 };
    const vbuf_ggml::TopKSelection selection{ { 4, 1, 3 }, { 0.9f, 0.7f, 0.2f } };
    const std::vector<float> weights{ 0.5f, 0.3f, 0.2f };

    vbuf_ggml::IndexedExpertLowering fallback;
    std::string error;
    assert(vbuf_ggml::lower_indexed_expert_bank(bank, selection, weights,
        vbuf_ggml::IndexedExpertExecutionCapability::Rank2Only, &fallback, &error));
    assert(fallback.path == vbuf_ggml::IndexedExpertExecutionPath::Rank2Fallback);
    assert(fallback.bank.tensor_id == bank.tensor_id);
    assert(fallback.scores == selection.scores);
    assert(fallback.weights == weights);
    assert(fallback.members.size() == selection.ids.size());
    for (size_t rank = 0; rank < fallback.members.size(); ++rank) {
        const auto & member = fallback.members[rank];
        const uint64_t expected_offset = selection.ids[rank] * member_bytes;
        assert(member.bank_tensor_id == bank.tensor_id);
        assert(member.bank_name == bank.name);
        assert(member.expert_id == selection.ids[rank]);
        assert(member.topk_rank == rank);
        assert(member.source_offset == bank.source_offset + expected_offset);
        assert(member.payload_offset == expected_offset);
        assert(member.view().rank == 2);
        assert(member.view().dimensions[0] == dimensions[0]);
        assert(member.view().dimensions[1] == dimensions[1]);
        assert(member.view().payload == payload.data() + expected_offset);
        assert(member.view().payload_len == member_bytes);
    }

    vbuf_ggml::IndexedExpertLowering indexed;
    assert(vbuf_ggml::lower_indexed_expert_bank(bank, selection, weights,
        vbuf_ggml::IndexedExpertExecutionCapability::IndexedBank, &indexed, &error));
    assert(indexed.path == vbuf_ggml::IndexedExpertExecutionPath::IndexedBank);
    assert(indexed.members.size() == fallback.members.size());
    for (size_t rank = 0; rank < indexed.members.size(); ++rank) {
        assert(indexed.members[rank].expert_id == fallback.members[rank].expert_id);
        assert(indexed.members[rank].topk_rank == fallback.members[rank].topk_rank);
        assert(indexed.members[rank].source_offset == fallback.members[rank].source_offset);
    }

    const vbuf_ggml::PersistentTensorRef metadata_only{
        bank.tensor_id, bank.name, { bank.view.representation, bank.view.rank,
            dimensions, nullptr, bank.view.payload_len }, bank.source_offset };
    vbuf_ggml::IndexedExpertLowering metadata_plan;
    assert(vbuf_ggml::lower_indexed_expert_bank(metadata_only, selection, weights,
        vbuf_ggml::IndexedExpertExecutionCapability::IndexedBank, &metadata_plan, &error));
    assert(metadata_plan.members[0].payload == nullptr);
    assert(metadata_plan.members[0].source_offset == indexed.members[0].source_offset);

    const vbuf_ggml::TopKSelection invalid_id{ { 5 }, { 1.0f } };
    assert(!vbuf_ggml::lower_indexed_expert_bank(bank, invalid_id, { 1.0f },
        vbuf_ggml::IndexedExpertExecutionCapability::Rank2Only, &fallback, &error));
    assert(error == "indexed expert ID is outside the bank");
    const vbuf_ggml::TopKSelection invalid_score{ { 1 }, { 1.0f } };
    assert(!vbuf_ggml::lower_indexed_expert_bank(bank, invalid_score,
        { 0.0f / 0.0f }, vbuf_ggml::IndexedExpertExecutionCapability::Rank2Only,
        &fallback, &error));
    assert(error == "indexed expert score or weight is non-finite");
    const vbuf_ggml::PersistentTensorRef overflowing_bank{
        bank.tensor_id, bank.name, bank.view, UINT64_MAX - member_bytes + 1 };
    assert(!vbuf_ggml::lower_indexed_expert_bank(overflowing_bank, selection, weights,
        vbuf_ggml::IndexedExpertExecutionCapability::Rank2Only, &fallback, &error));
    assert(error == "indexed expert source offset overflows");
    assert(std::memcmp(payload.data(), indexed.bank.view.payload, payload.size()) == 0);
    return 0;
}
