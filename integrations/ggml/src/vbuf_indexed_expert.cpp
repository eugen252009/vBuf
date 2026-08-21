#include "vbuf_indexed_expert.h"

#include <cmath>
#include <limits>
#include <utility>

namespace vbuf_ggml {
namespace {

void set_error(std::string * error, const char * message) {
    if (error != nullptr) *error = message;
}

bool add_overflows(uint64_t left, uint64_t right) {
    return right > std::numeric_limits<uint64_t>::max() - left;
}

} // namespace

bool lower_indexed_expert_bank(
    const PersistentTensorRef & bank,
    const TopKSelection & selection,
    const std::vector<float> & weights,
    IndexedExpertExecutionCapability capability,
    IndexedExpertLowering * lowering,
    std::string * error) {
    if (lowering == nullptr) {
        set_error(error, "indexed expert lowering output is null");
        return false;
    }
    *lowering = {};
    if (bank.view.rank != 3 || bank.view.dimensions == nullptr) {
        set_error(error, "expert bank must be a rank-three tensor");
        return false;
    }
    const uint64_t expert_count = bank.view.dimensions[2];
    if (expert_count == 0 || expert_count > std::numeric_limits<uint32_t>::max()) {
        set_error(error, "expert bank has an invalid expert count");
        return false;
    }
    if (bank.view.dimensions[0] == 0 || bank.view.dimensions[1] == 0) {
        set_error(error, "expert bank has an invalid member shape");
        return false;
    }
    if (bank.view.payload_len == 0 || bank.view.payload_len % expert_count != 0) {
        set_error(error, "expert bank payload is not evenly divisible by expert count");
        return false;
    }
    if (selection.ids.empty() || selection.ids.size() != selection.scores.size() ||
        selection.ids.size() != weights.size()) {
        set_error(error, "indexed expert selection and weights differ");
        return false;
    }

    const uint64_t member_bytes = bank.view.payload_len / expert_count;
    IndexedExpertLowering result;
    result.path = capability == IndexedExpertExecutionCapability::IndexedBank
        ? IndexedExpertExecutionPath::IndexedBank : IndexedExpertExecutionPath::Rank2Fallback;
    result.bank = bank;
    result.scores = selection.scores;
    result.weights = weights;
    result.members.reserve(selection.ids.size());
    for (size_t rank = 0; rank < selection.ids.size(); ++rank) {
        const uint32_t expert_id = selection.ids[rank];
        if (expert_id >= expert_count) {
            set_error(error, "indexed expert ID is outside the bank");
            return false;
        }
        if (!std::isfinite(selection.scores[rank]) || !std::isfinite(weights[rank])) {
            set_error(error, "indexed expert score or weight is non-finite");
            return false;
        }
        if (member_bytes != 0 && expert_id >
            std::numeric_limits<uint64_t>::max() / member_bytes) {
            set_error(error, "indexed expert payload offset overflows");
            return false;
        }
        const uint64_t payload_offset = static_cast<uint64_t>(expert_id) * member_bytes;
        if (add_overflows(bank.source_offset, payload_offset) ||
            payload_offset > bank.view.payload_len - member_bytes) {
            set_error(error, "indexed expert source offset overflows");
            return false;
        }
        IndexedExpertMember member;
        member.bank_tensor_id = bank.tensor_id;
        member.bank_name = bank.name;
        member.expert_id = expert_id;
        member.topk_rank = static_cast<uint32_t>(rank);
        member.source_offset = bank.source_offset + payload_offset;
        member.payload_offset = payload_offset;
        member.representation = bank.view.representation;
        member.dimensions[0] = bank.view.dimensions[0];
        member.dimensions[1] = bank.view.dimensions[1];
        member.payload_len = member_bytes;
        if (bank.view.payload != nullptr) member.payload = bank.view.payload + payload_offset;
        result.members.push_back(std::move(member));
    }
    *lowering = std::move(result);
    return true;
}

} // namespace vbuf_ggml
