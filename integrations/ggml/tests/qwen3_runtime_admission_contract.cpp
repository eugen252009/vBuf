#include "vbuf_generation.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: qwen3_runtime_admission_contract <semantic-model>");
        auto runtime = std::make_shared<vbuf_ggml::VbufModelRuntime>(argv[1], 40);
        auto first = runtime->create_session();
        auto second = runtime->create_session();
        vbuf_ggml::VbufGenerationConfig config;
        config.semantic_model = argv[1];
        config.block_count = 40;
        config.prompt_tokens = {1, 2};
        config.max_new_tokens = 1;
        const auto rejected = first->run(config);
        const std::string expected = "canonical Qwen execution requires a payload source endpoint";
        if (rejected.error != expected || first->current_context_length() != 0 ||
            second->current_context_length() != 0)
            throw std::runtime_error("exact Qwen3 variant did not remain safely disabled in canonical dispatch");
        const auto wrapped = runtime->run(config);
        if (wrapped.error != expected || runtime->snapshot().request_count != 2)
            throw std::runtime_error("whole-request Qwen3 wrapper bypassed canonical unsupported dispatch");
        std::cout << "qwen3_runtime_metadata_admission=PASS missing_source_fails_closed=YES\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qwen3_runtime_admission_contract: " << error.what() << '\n';
        return 1;
    }
}
