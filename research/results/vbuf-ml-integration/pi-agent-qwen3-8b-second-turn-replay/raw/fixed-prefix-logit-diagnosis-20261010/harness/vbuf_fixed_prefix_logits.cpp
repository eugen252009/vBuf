#define VBUF_COMPAT_SERVER_LIBRARY_ONLY
#include "/home/eugen/projekte/vBuf/integrations/ggml/tools/vbuf_compat_server.cpp"
#undef VBUF_COMPAT_SERVER_LIBRARY_ONLY

#include <filesystem>
#include <fstream>
#include <sstream>

static std::vector<uint32_t> read_ids(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open token IDs: " + path);
    std::string csv;
    std::getline(input, csv);
    std::stringstream stream(csv);
    std::vector<uint32_t> ids;
    std::string field;
    while (std::getline(stream, field, ',')) ids.push_back(static_cast<uint32_t>(std::stoul(field)));
    if (ids.empty()) throw std::runtime_error("empty token IDs: " + path);
    return ids;
}

static void write_ids(const std::string & path, const std::vector<uint32_t> & ids) {
    std::ofstream output(path);
    for (size_t i = 0; i < ids.size(); ++i) output << (i ? "," : "") << ids[i];
    output << '\n';
    if (!output) throw std::runtime_error("cannot write token IDs: " + path);
}

static vbuf_ggml::VbufGenerationResult run_prefix(ServerRuntime & runtime,
    const std::vector<uint32_t> & prompt, uint32_t max_new_tokens) {
    vbuf_ggml::VbufGenerationConfig generation;
    generation.semantic_model = runtime.config.semantic_model;
    generation.source_endpoint = runtime.config.source_url;
    generation.block_count = runtime.config.blocks;
    generation.context_capacity = runtime.context_capacity;
    generation.residency_capacity = runtime.config.capacity;
    generation.max_new_tokens = max_new_tokens;
    generation.mode = vbuf_ggml::RuntimeMode::NormalInference;
    generation.prompt_tokens = prompt;
    generation.stop_token = runtime.tokenizer.eos();
    generation.on_token = [](uint32_t, uint32_t) { return true; };
    auto session = runtime.model_runtime->create_session(runtime.context_capacity);
    auto result = session->run(generation);
    if (!result.error.empty() || !result.completed)
        throw std::runtime_error("generation failed: " + result.error);
    return result;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 9 && argc != 10) {
            std::cerr << "usage: vbuf_fixed_prefix_logits SEMANTIC SOURCE_URL PROMPT_IDS GENERATED_IDS OUTDIR CONTEXT INCREMENTAL_LIMIT POSITIONS_CSV\n";
            return 2;
        }
        const std::string semantic = argv[1];
        const std::string source_url = argv[2];
        const auto prompt_ids = read_ids(argv[3]);
        const auto expected_ids = read_ids(argv[4]);
        const std::filesystem::path output_root = argv[5];
        const uint32_t context = static_cast<uint32_t>(std::stoul(argv[6]));
        const uint32_t incremental_limit = static_cast<uint32_t>(std::stoul(argv[7]));
        const auto positions = read_ids(argv[8]);
        const bool incremental_only = argc == 10 && std::string(argv[9]) == "incremental-only";
        if (argc == 10 && !incremental_only) throw std::runtime_error("unknown optional mode");
        std::string capture_steps;
        for (size_t i = 0; i < positions.size(); ++i)
            capture_steps += (i ? "," : "") + std::to_string(positions[i]);
        std::filesystem::create_directories(output_root);
        ServerConfig config;
        config.semantic_model = semantic;
        config.source_url = source_url;
        config.model_alias = "qwen3-8b-fixed-prefix-diagnostic";
        config.blocks = 36;
        config.max_new_tokens = incremental_limit;
        config.qwen_context_capacity = context;
        config.experimental_qwen3_8b = true;
        config.capacity = 268435456;
        config.mode = vbuf_ggml::RuntimeMode::NormalInference;
        ServerRuntime runtime(config);

        const auto incremental_dir = output_root / "incremental";
        std::filesystem::create_directories(incremental_dir);
        setenv("VBUF_LOGIT_CAPTURE_STEPS", capture_steps.c_str(), 1);
        setenv("VBUF_LOGIT_CAPTURE_DIR", incremental_dir.c_str(), 1);
        const auto incremental = run_prefix(runtime, prompt_ids, incremental_limit);
        if (incremental.tokens.size() != incremental_limit || expected_ids.size() < incremental_limit)
            throw std::runtime_error("unexpected incremental output length");
        write_ids((output_root / "incremental-generated-token-ids.csv").string(), incremental.tokens);
        size_t first_mismatch = incremental.tokens.size();
        for (size_t i = 0; i < incremental.tokens.size(); ++i) {
            if (incremental.tokens[i] != expected_ids[i]) { first_mismatch = i; break; }
        }
        std::cout << "incremental prompt_tokens=" << prompt_ids.size()
            << " generated_tokens=" << incremental.tokens.size()
            << " exact_prefix_match=" << (first_mismatch == incremental.tokens.size() ? "YES" : "NO")
            << " first_mismatch=" << (first_mismatch == incremental.tokens.size() ? -1 : static_cast<long long>(first_mismatch))
            << "\n";
        if (first_mismatch != incremental.tokens.size()) return 5;
        if (incremental_only) return 0;

        setenv("VBUF_LOGIT_CAPTURE_STEPS", "0", 1);
        for (uint32_t position : positions) {
            if (position >= expected_ids.size()) throw std::runtime_error("requested target position outside saved generation");
            std::vector<uint32_t> prefix = prompt_ids;
            prefix.insert(prefix.end(), expected_ids.begin(), expected_ids.begin() + position);
            const auto fresh_dir = output_root / ("fresh-target-" + std::to_string(position));
            std::filesystem::create_directories(fresh_dir);
            setenv("VBUF_LOGIT_CAPTURE_DIR", fresh_dir.c_str(), 1);
            const auto fresh = run_prefix(runtime, prefix, 1);
            std::cout << "fresh target_index=" << position
                << " input_prefix_tokens=" << prefix.size()
                << " expected=" << expected_ids[position]
                << " top1=" << fresh.tokens.at(0)
                << " top1_match=" << (fresh.tokens.at(0) == expected_ids[position] ? "YES" : "NO")
                << "\n";
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "vbuf_fixed_prefix_logits: " << error.what() << "\n";
        return 1;
    }
}
