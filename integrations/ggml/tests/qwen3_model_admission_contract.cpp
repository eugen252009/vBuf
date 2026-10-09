#include "qwen3_model.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char ** argv) {
    try {
        using namespace vbuf_ggml;
        require(qwen3_artifact_identity_is_qualified(QWEN3_14B_Q4_K_M_SHA256),
            "exact qualified Qwen3 identity was rejected");
        require(!qwen3_artifact_identity_is_qualified("sha256:other"),
            "unsupported Qwen3 identity was admitted");
        require(qwen3_artifact_identity_is_experimental_8b(QWEN3_8B_VBUF_SOURCE_SHA256),
            "exact experimental Qwen3-8B source identity was rejected");
        require(!qwen3_artifact_identity_is_qualified(QWEN3_8B_VBUF_SOURCE_SHA256),
            "experimental Qwen3-8B identity entered default production admission");
        require(!qwen3_artifact_identity_is_experimental_8b(QWEN3_14B_Q4_K_M_SHA256) &&
            !qwen3_artifact_identity_is_experimental_8b("sha256:other"),
            "non-Qwen3-8B identity entered experimental admission");
        if (argc == 1) return 0;
        Qwen3Model model;
        open_qwen3_model(argv[1], "http://127.0.0.1:1", &model, true);
        require(model.layer_count == 40 && model.metadata.embedding_length == 5120 &&
            model.metadata.head_count == 40 && model.metadata.kv_head_count == 8 &&
            model.metadata.key_head_dimension == 128 && model.metadata.value_head_dimension == 128 &&
            model.metadata.feed_forward_length == 17408 && model.metadata.vocabulary_size == 151936,
            "qualified Qwen3 dimensions differ from pinned contract");
        require(model.catalog.tensors.size() == 443 &&
            model.artifact_identity == std::string("sha256:") + QWEN3_14B_Q4_K_M_SHA256,
            "qualified Qwen3 tensor catalog or source identity mismatch");
        require(model.special_token_present[0] && model.special_token_present[1] &&
            model.special_tokens[0] < model.metadata.vocabulary_size &&
            model.special_tokens[1] < model.metadata.vocabulary_size,
            "Qwen3 tokenizer BOS/EOS metadata was not admitted");
        require(model.catalog.tensors.at("token_embd.weight").dimensions ==
            model.catalog.tensors.at("output.weight").dimensions,
            "embedding/output vocabulary dimensions differ");
        std::cout << "qwen3_model_admission=PASS sha256=" << QWEN3_14B_Q4_K_M_SHA256 << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qwen3_model_admission_contract: " << error.what() << '\n';
        return 1;
    }
}
