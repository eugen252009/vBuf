// Opt-in observation harness for the pinned llama.cpp Qwen3 graph. This calls
// one reference decode to export named checkpoints; it is not a vBuf runtime.
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

struct Capture {
    std::filesystem::path directory;
    uint32_t block_count = 1;
    uint32_t model_layers = 0;
    bool compact = true;
    bool final_only = false;
    bool layers_only = false;
    std::unordered_map<std::string, uint32_t> serials;
    bool failed = false;
};

bool selected(const char * name, const Capture & capture) {
    if (std::strcmp(name, "embd") == 0 || std::strcmp(name, "result_norm") == 0 ||
        std::strcmp(name, "result_output") == 0) return true;
    if (capture.final_only) {
        const std::string final_output = "l_out-" + std::to_string(capture.model_layers - 1);
        return std::strcmp(name, final_output.c_str()) == 0;
    }
    if (capture.layers_only) return std::strncmp(name, "l_out-", 6) == 0;
    static constexpr const char * full_names[] = {
        "attn_norm", "Qcur", "Kcur", "Vcur", "Qcur_normed", "Kcur_normed",
        "kq", "kq_soft_max", "kqv", "kqv_out", "ffn_inp", "ffn_norm", "ffn_up",
        "ffn_gate", "ffn_swiglu", "ffn_out", "l_out",
    };
    if (capture.compact) {
        const uint32_t selected_layers[] = {0, capture.model_layers / 2, capture.model_layers - 1};
        static constexpr const char * compact_names[] = {
            "Qcur", "Kcur", "Vcur", "Qcur_normed", "Kcur_normed", "kq_soft_max",
        };
        for (uint32_t layer : selected_layers) {
            for (const char * candidate : compact_names) {
                const std::string layer_name = std::string(candidate) + "-" + std::to_string(layer);
                if (std::strcmp(name, layer_name.c_str()) == 0) return true;
            }
        }
        const std::string final_output = "l_out-" + std::to_string(capture.model_layers - 1);
        return std::strcmp(name, final_output.c_str()) == 0;
    }
    for (uint32_t layer = 0; layer < capture.block_count; ++layer)
        for (const char * candidate : full_names) {
            const std::string layer_name = std::string(candidate) + "-" + std::to_string(layer);
            if (std::strcmp(name, layer_name.c_str()) == 0) return true;
        }
    return false;
}

bool write_checkpoint(ggml_tensor * tensor, Capture * capture) {
    if (tensor->type != GGML_TYPE_F32 || tensor->data == nullptr ||
        tensor->nb[0] != sizeof(float)) {
        std::fprintf(stderr, "unsupported reference checkpoint: %s type=%s data=%p\n",
            ggml_get_name(tensor), ggml_type_name(tensor->type), tensor->data);
        return false;
    }
    const std::string name = ggml_get_name(tensor);
    const uint32_t serial = capture->serials[name]++;
    const std::string stem = name + "-" + std::to_string(serial);
    uint64_t element_count = 1;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (tensor->ne[dim] < 1 ||
            element_count > UINT64_MAX / static_cast<uint64_t>(tensor->ne[dim])) return false;
        element_count *= static_cast<uint64_t>(tensor->ne[dim]);
    }
    std::vector<float> dense(element_count);
    size_t cursor = 0;
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
        const auto * value = reinterpret_cast<const float *>(
            static_cast<const uint8_t *>(tensor->data) + i0 * tensor->nb[0] +
            i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3]);
        dense[cursor++] = *value;
    }
    std::ofstream output(capture->directory / (stem + ".f32"), std::ios::binary);
    output.write(reinterpret_cast<const char *>(dense.data()),
        static_cast<std::streamsize>(dense.size() * sizeof(float)));
    output.close();
    if (!output) return false;
    std::ofstream metadata(capture->directory / (stem + ".meta"));
    metadata << "name=" << name << "\nserial=" << serial << "\ntype=F32\nshape=";
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim)
        metadata << (dim == 0 ? "" : ",") << tensor->ne[dim];
    metadata << "\nvalues=" << dense.size() << "\nop=" << ggml_op_name(tensor->op) << "\n";
    for (int source = 0; source < 2; ++source) {
        const ggml_tensor * operand = tensor->src[source];
        metadata << "src" << source << "=";
        if (operand == nullptr) {
            metadata << "none\n";
            continue;
        }
        metadata << "type:" << ggml_type_name(operand->type) << ";shape:";
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim)
            metadata << (dim == 0 ? "" : ",") << operand->ne[dim];
        metadata << ";nb:";
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim)
            metadata << (dim == 0 ? "" : ",") << operand->nb[dim];
        metadata << "\n";
    }
    return static_cast<bool>(metadata);
}

bool eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * capture = static_cast<Capture *>(user_data);
    if (!selected(ggml_get_name(tensor), *capture)) return true;
    if (ask) return true;
    if (!write_checkpoint(tensor, capture)) {
        capture->failed = true;
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 9) {
        std::fprintf(stderr, "usage: %s MODEL.gguf OUTPUT_DIR POSITION_COUNT(1..32) [THREADS] [BLOCK_COUNT] [generate:1..32|tokens:ID,...] [seed:TOKEN_ID] [compact|full|final|layers]\n", argv[0]);
        return 2;
    }
    const uint32_t positions = static_cast<uint32_t>(std::stoul(argv[3]));
    const int32_t threads = argc >= 5 ? std::stoi(argv[4]) : 8;
    const uint32_t block_count = argc >= 6 ? static_cast<uint32_t>(std::stoul(argv[5])) : 1;
    uint32_t generation_steps = 0;
    uint32_t seed_token = 0;
    bool have_seed = false;
    bool compact = true;
    bool final_only = false;
    bool layers_only = false;
    std::vector<llama_token> explicit_tokens;
    if (argc >= 7) {
        const std::string mode = argv[6];
        if (mode.rfind("generate:", 0) == 0) {
            generation_steps = static_cast<uint32_t>(std::stoul(mode.substr(9)));
        } else if (mode.rfind("tokens:", 0) == 0) {
            std::stringstream values(mode.substr(7));
            std::string item;
            while (std::getline(values, item, ',')) {
                if (item.empty()) return 2;
                explicit_tokens.push_back(static_cast<llama_token>(std::stol(item)));
            }
        } else return 2;
    }
    if (argc >= 8) {
        const std::string option = argv[7];
        if (explicit_tokens.empty()) {
            if (option.rfind("seed:", 0) != 0) return 2;
            seed_token = static_cast<uint32_t>(std::stoul(option.substr(5)));
            have_seed = true;
        } else if (option == "compact" || option == "full" || option == "final" || option == "layers") {
            compact = option == "compact";
            final_only = option == "final";
            layers_only = option == "layers";
        } else return 2;
    }
    if (argc >= 9) {
        const std::string profile = argv[8];
        if (profile != "compact" && profile != "full" && profile != "final" && profile != "layers") return 2;
        compact = profile == "compact";
        final_only = profile == "final";
        layers_only = profile == "layers";
    }
    if (positions == 0 || positions > 32 || threads < 1 || threads > 128 || block_count == 0 ||
        (generation_steps != 0 && (positions != 1 || generation_steps > 32)) ||
        (have_seed && positions != 1) ||
        (!explicit_tokens.empty() && (generation_steps != 0 || have_seed || explicit_tokens.size() != positions))) {
        std::fprintf(stderr, "position count must be 1..32; generation steps 1..32; threads 1..128; block count > 0\n");
        return 2;
    }
    const std::filesystem::path output_directory(argv[2]);
    std::filesystem::create_directories(output_directory);
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) return 1;
    const uint32_t model_layers = static_cast<uint32_t>(llama_model_n_layer(model));
    const uint32_t vocabulary = static_cast<uint32_t>(llama_vocab_n_tokens(llama_model_get_vocab(model)));
    if (block_count > model_layers || (have_seed && seed_token >= vocabulary) ||
        std::any_of(explicit_tokens.begin(), explicit_tokens.end(), [vocabulary](llama_token token) {
            return token < 0 || static_cast<uint32_t>(token) >= vocabulary;
        })) {
        std::fprintf(stderr, "block count or seed token exceeds model geometry\n");
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    std::vector<llama_token> tokens(positions);
    if (!explicit_tokens.empty()) tokens = explicit_tokens;
    else if (have_seed) tokens[0] = static_cast<llama_token>(seed_token);
    else for (uint32_t index = 0; index < positions; ++index) tokens[index] = static_cast<llama_token>(index);
    std::ofstream generation(output_directory / "generation.meta");
    bool failed = false;
    const uint32_t passes = generation_steps == 0 ? 1 : generation_steps;
    for (uint32_t step = 0; step < passes; ++step) {
        const std::filesystem::path pass_directory = generation_steps == 0 ? output_directory :
            output_directory / ("prefix-" + std::to_string(step));
        std::filesystem::create_directories(pass_directory);
        Capture capture{ pass_directory, block_count, model_layers, compact, final_only, layers_only, {} };
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 256;
        context_params.n_batch = static_cast<uint32_t>(tokens.size());
        context_params.n_ubatch = static_cast<uint32_t>(tokens.size());
        context_params.n_threads = threads;
        context_params.n_threads_batch = threads;
        context_params.n_outputs_max = generation_steps == 0 ? static_cast<uint32_t>(tokens.size()) : 0;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context_params.offload_kqv = false;
        context_params.op_offload = false;
        context_params.cb_eval = eval_callback;
        context_params.cb_eval_user_data = &capture;
        llama_context * context = llama_init_from_model(model, context_params);
        if (context == nullptr) { failed = true; break; }
        llama_batch batch{};
        if (generation_steps == 0) {
            batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
            batch.n_tokens = static_cast<int32_t>(tokens.size());
            for (size_t index = 0; index < tokens.size(); ++index) {
                batch.token[index] = tokens[index];
                batch.pos[index] = static_cast<llama_pos>(index);
                batch.n_seq_id[index] = 1;
                batch.seq_id[index][0] = 0;
                batch.logits[index] = 1;
            }
        } else {
            batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        }
        const int32_t status = llama_decode(context, batch);
        std::ofstream manifest(pass_directory / "reference.meta");
        manifest << "model=" << argv[1] << "\nrevision=a97123e497968f3440264c0464a7adc7c999c027\n"
            << "ggml_revision=a97123e497968f3440264c0464a7adc7c999c027\n"
            << "positions=" << tokens.size() << "\nthreads=" << threads << "\nblock_count=" << block_count << "\ntokens=";
        for (size_t index = 0; index < tokens.size(); ++index)
            manifest << (index == 0 ? "" : ",") << tokens[index];
        manifest << "\ncheckpoint_profile=" << (final_only ? "final-only" : compact ? "compact-selected" : "full")
            << "\nselected_attention_layers=0," << model_layers / 2 << "," << model_layers - 1
            << "\nllama_decode_status=" << status << "\n";
        if (!manifest || capture.failed || status != 0) failed = true;
        if (!failed && generation_steps != 0) {
            const float * logits = llama_get_logits_ith(context, -1);
            if (logits == nullptr) failed = true;
            else {
                const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
                llama_token next = 0;
                for (int32_t id = 1; id < vocab; ++id)
                    if (logits[id] > logits[next]) next = id;
                generation << "step=" << step << " input_tokens=";
                for (size_t index = 0; index < tokens.size(); ++index)
                    generation << (index == 0 ? "" : ",") << tokens[index];
                generation << " reference_next_token=" << next << "\n";
                tokens.push_back(next);
            }
        }
        if (generation_steps == 0) llama_batch_free(batch);
        llama_free(context);
        if (failed) break;
    }
    llama_model_free(model);
    llama_backend_free();
    if (!generation || failed) return 1;
    return 0;
}
