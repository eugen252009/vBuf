#define VBUF_COMPAT_SERVER_LIBRARY_ONLY
#include "vbuf_compat_server.cpp"
#undef VBUF_COMPAT_SERVER_LIBRARY_ONLY

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

struct DirectConfig {
    std::string semantic_model;
    std::string source_url;
    std::string model_alias = "vbuf-direct-control";
    std::string prompt = "Say hi";
    uint32_t blocks = 2;
    uint64_t capacity = 268435456;
    uint32_t max_new_tokens = 1;
    uint32_t warmup = 1;
    uint32_t requests = 3;
};

static std::string required_value(int * index, int argc, char ** argv) {
    if (*index + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[*index]);
    return argv[++*index];
}

static DirectConfig parse_direct_args(int argc, char ** argv) {
    DirectConfig config;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--semantic-model") config.semantic_model = required_value(&index, argc, argv);
        else if (arg == "--source-url") config.source_url = required_value(&index, argc, argv);
        else if (arg == "--model-alias") config.model_alias = required_value(&index, argc, argv);
        else if (arg == "--prompt") config.prompt = required_value(&index, argc, argv);
        else if (arg == "--blocks") config.blocks = static_cast<uint32_t>(std::stoul(required_value(&index, argc, argv)));
        else if (arg == "--capacity") config.capacity = std::stoull(required_value(&index, argc, argv));
        else if (arg == "--max-new-tokens") config.max_new_tokens = static_cast<uint32_t>(std::stoul(required_value(&index, argc, argv)));
        else if (arg == "--warmup") config.warmup = static_cast<uint32_t>(std::stoul(required_value(&index, argc, argv)));
        else if (arg == "--requests") config.requests = static_cast<uint32_t>(std::stoul(required_value(&index, argc, argv)));
        else throw std::runtime_error("unknown direct-control option: " + arg);
    }
    if (config.semantic_model.empty() || config.source_url.empty() || config.blocks == 0 ||
        config.max_new_tokens == 0 || config.requests == 0)
        throw std::runtime_error("usage: vbuf_direct_control --semantic-model PATH --source-url URL "
            "[--prompt TEXT --blocks N --capacity BYTES --max-new-tokens N --warmup N --requests N]");
    return config;
}

static std::string hex_text(const std::string & value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2);
    for (const unsigned char byte : value) {
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 0x0f]);
    }
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const DirectConfig direct = parse_direct_args(argc, argv);
        ServerConfig server;
        server.semantic_model = direct.semantic_model;
        server.source_url = direct.source_url;
        server.model_alias = direct.model_alias;
        server.blocks = direct.blocks;
        server.capacity = direct.capacity;
        server.max_new_tokens = direct.max_new_tokens;
        server.mode = vbuf_ggml::RuntimeMode::NormalInference;
        ServerRuntime runtime(server);
        const std::vector<Message> messages = {{"user", direct.prompt}};

        for (uint32_t index = 0; index < direct.warmup + direct.requests; ++index) {
            const bool warmup = index < direct.warmup;
            const uint64_t total_start = steady_now_ns();
            const uint64_t tokenize_start = steady_now_ns();
            const std::vector<uint32_t> prompt_tokens = runtime.tokenizer.encode_chat(messages);
            const uint64_t tokenize_ns = steady_now_ns() - tokenize_start;
            vbuf_ggml::VbufGenerationConfig generation;
            generation.semantic_model = server.semantic_model;
            generation.source_endpoint = server.source_url;
            generation.block_count = server.blocks;
            generation.residency_capacity = server.capacity;
            generation.max_new_tokens = server.max_new_tokens;
            generation.mode = server.mode;
            generation.prompt_tokens = prompt_tokens;
            generation.stop_token = runtime.tokenizer.eos();
            vbuf_ggml::VbufGenerationResult result;
            std::fflush(stdout);
            {
                StdoutSilencer silence;
                result = runtime.session->run(generation);
            }
            if (!result.error.empty() || !result.completed)
                throw std::runtime_error("direct generation failed: " + result.error);
            const std::string output = runtime.tokenizer.decode(result.tokens);
            const uint64_t total_ns = steady_now_ns() - total_start;
            std::printf("vbuf_direct_request request_index=%u phase=%s model=%s prompt_tokens=%llu "
                "prompt_token_hash=%016llx generated_tokens=%zu output=%s source_bytes=%llu "
                "materialized_bytes=%llu peak_residency_bytes=%llu resident_bytes_after=%llu "
                "active_leases_after=%u active_inflight_bytes_after=%llu source_successful_requests=%llu "
                "prefill_ns=%llu decode_ns=%llu runtime_ns=%llu tokenize_ns=%llu total_ns=%llu\n",
                index + 1, warmup ? "warmup" : "measure", server.model_alias.c_str(),
                static_cast<unsigned long long>(prompt_tokens.size()),
                static_cast<unsigned long long>(token_hash(prompt_tokens)), result.tokens.size(),
                hex_text(output).c_str(), static_cast<unsigned long long>(result.source_bytes),
                static_cast<unsigned long long>(result.materialized_bytes),
                static_cast<unsigned long long>(result.peak_resident_bytes),
                static_cast<unsigned long long>(result.resident_bytes_after),
                result.active_lease_count_after,
                static_cast<unsigned long long>(result.active_inflight_bytes_after),
                static_cast<unsigned long long>(result.source_successful_requests),
                static_cast<unsigned long long>(result.prefill_ns),
                static_cast<unsigned long long>(result.decode_ns),
                static_cast<unsigned long long>(result.elapsed_ns),
                static_cast<unsigned long long>(tokenize_ns),
                static_cast<unsigned long long>(total_ns));
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "vbuf-direct-control: %s\n", error.what());
        return 1;
    }
}
