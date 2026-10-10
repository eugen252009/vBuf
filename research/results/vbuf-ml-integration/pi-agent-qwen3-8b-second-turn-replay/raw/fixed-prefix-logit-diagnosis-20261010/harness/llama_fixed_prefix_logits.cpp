#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<uint32_t> read_ids(const char * path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error(std::string("cannot open token IDs: ") + path);
    std::string csv;
    std::getline(input, csv);
    std::stringstream stream(csv);
    std::vector<uint32_t> ids;
    std::string field;
    while (std::getline(stream, field, ',')) ids.push_back(static_cast<uint32_t>(std::stoul(field)));
    if (ids.empty()) throw std::runtime_error(std::string("empty token IDs: ") + path);
    return ids;
}

static void decode_tokens(llama_context * context, const std::vector<llama_token> & tokens,
    uint32_t start_position, bool emit_last_logits) {
    if (tokens.empty()) throw std::runtime_error("empty decode batch");
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(start_position + i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = emit_last_logits && i + 1 == tokens.size() ? 1 : 0;
    }
    const int32_t status = llama_decode(context, batch);
    llama_batch_free(batch);
    if (status != 0) throw std::runtime_error("llama_decode failed with status " + std::to_string(status));
}

static void write_logits(const std::filesystem::path & outdir, uint32_t target_index,
    uint32_t expected, const float * logits, uint32_t vocab) {
    if (logits == nullptr) throw std::runtime_error("llama returned no logits");
    std::ofstream output(outdir / ("llama-target-" + std::to_string(target_index) + ".f32"), std::ios::binary);
    output.write(reinterpret_cast<const char *>(logits), static_cast<std::streamsize>(vocab * sizeof(float)));
    if (!output) throw std::runtime_error("cannot write llama logits");
    std::vector<uint32_t> ranked(vocab);
    for (uint32_t i = 0; i < vocab; ++i) ranked[i] = i;
    const size_t top_count = std::min<size_t>(10, ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + top_count, ranked.end(),
        [&](uint32_t a, uint32_t b) { return logits[a] == logits[b] ? a < b : logits[a] > logits[b]; });
    std::printf("llama target_index=%u expected=%u top1=%u top10=", target_index, expected, ranked[0]);
    for (size_t i = 0; i < top_count; ++i)
        std::printf("%s%u:%.9g", i == 0 ? "" : ",", ranked[i], logits[ranked[i]]);
    std::printf("\n");
}

int main(int argc, char ** argv) {
    try {
        if (argc != 6) {
            std::fprintf(stderr, "usage: llama_fixed_prefix_logits MODEL.gguf PROMPT_IDS GENERATED_IDS OUTDIR POSITIONS_CSV\n");
            return 2;
        }
        const std::filesystem::path outdir(argv[4]);
        std::filesystem::create_directories(outdir);
        const auto prompt_ids = read_ids(argv[2]);
        const auto generated_ids = read_ids(argv[3]);
        const auto positions = read_ids(argv[5]);
        if (positions.back() >= generated_ids.size()) throw std::runtime_error("target index outside generated sequence");
        std::vector<bool> selected(static_cast<size_t>(positions.back()) + 1, false);
        for (uint32_t position : positions) selected.at(position) = true;

        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = -1;
        model_params.main_gpu = 0;
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        llama_model * model = llama_model_load_from_file(argv[1], model_params);
        if (model == nullptr) throw std::runtime_error("llama model load failed");
        const uint32_t vocab = static_cast<uint32_t>(llama_vocab_n_tokens(llama_model_get_vocab(model)));
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 12288;
        context_params.n_batch = 512;
        context_params.n_ubatch = 128;
        context_params.n_seq_max = 1;
        context_params.n_outputs_max = 1;
        context_params.n_threads = 8;
        context_params.n_threads_batch = 8;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        llama_context * context = llama_init_from_model(model, context_params);
        if (context == nullptr) throw std::runtime_error("llama context creation failed");

        std::vector<llama_token> prompt(prompt_ids.begin(), prompt_ids.end());
        uint32_t cursor = 0;
        while (cursor < prompt.size()) {
            const uint32_t count = std::min<uint32_t>(512, static_cast<uint32_t>(prompt.size()) - cursor);
            std::vector<llama_token> chunk(prompt.begin() + cursor, prompt.begin() + cursor + count);
            const bool final_prompt_chunk = cursor + count == prompt.size();
            decode_tokens(context, chunk, cursor, final_prompt_chunk);
            cursor += count;
        }
        if (selected[0]) write_logits(outdir, 0, generated_ids[0], llama_get_logits_ith(context, -1), vocab);

        for (uint32_t target_index = 1; target_index <= positions.back(); ++target_index) {
            const llama_token token = static_cast<llama_token>(generated_ids[target_index - 1]);
            decode_tokens(context, std::vector<llama_token>{token},
                static_cast<uint32_t>(prompt.size()) + target_index - 1, true);
            if (selected[target_index])
                write_logits(outdir, target_index, generated_ids[target_index],
                    llama_get_logits_ith(context, -1), vocab);
        }
        std::ofstream manifest(outdir / "reference.meta");
        manifest << "llama_commit=a97123e497968f3440264c0464a7adc7c999c027\n"
            << "model=" << argv[1] << "\nmodel_layers=" << llama_model_n_layer(model)
            << "\nvocabulary=" << vocab << "\nprompt_tokens=" << prompt.size()
            << "\ncontext=12288\nbatch=512\nubatch=128\nthreads=8\ngpu_layers=all"
            << "\nflash_attention=off\nkv_cache_default=llama.cpp\npositions=";
        for (size_t i = 0; i < positions.size(); ++i) manifest << (i ? "," : "") << positions[i];
        manifest << "\n";
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return manifest ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "llama_fixed_prefix_logits: %s\n", error.what());
        return 1;
    }
}
