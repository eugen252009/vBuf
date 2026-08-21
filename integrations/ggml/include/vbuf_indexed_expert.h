#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "vbuf_tensor_wave.h"
#include "vbuf_topk.h"

namespace vbuf_ggml {

// Backend-neutral capability: the backend may consume the full bank directly,
// or the lowerer must expose one rank-2 member per TopK rank.
enum class IndexedExpertExecutionCapability {
    Rank2Only,
    IndexedBank,
};

enum class IndexedExpertExecutionPath {
    Rank2Fallback,
    IndexedBank,
};

struct IndexedExpertMember {
    uint64_t bank_tensor_id = 0;
    std::string bank_name;
    uint32_t expert_id = 0;
    uint32_t topk_rank = 0;
    uint64_t source_offset = 0;
    uint64_t payload_offset = 0;
    uint8_t representation = 0;
    std::array<uint64_t, GGML_MAX_DIMS> dimensions{};
    const uint8_t * payload = nullptr;
    uint64_t payload_len = 0;

    VbufTensorView view() const {
        return { representation, 2, dimensions.data(), payload, payload_len };
    }
};

struct IndexedExpertLowering {
    IndexedExpertExecutionPath path = IndexedExpertExecutionPath::Rank2Fallback;
    PersistentTensorRef bank{};
    std::vector<IndexedExpertMember> members;
    std::vector<float> scores;
    std::vector<float> weights;
};

// Derives ordered rank-2 members from one canonical rank-3 bank. The bank
// bytes are never copied; a null bank payload is allowed for metadata-only
// planning and is resolved by the owning materializer before execution.
bool lower_indexed_expert_bank(
    const PersistentTensorRef & bank,
    const TopKSelection & selection,
    const std::vector<float> & weights,
    IndexedExpertExecutionCapability capability,
    IndexedExpertLowering * lowering,
    std::string * error = nullptr);

} // namespace vbuf_ggml
