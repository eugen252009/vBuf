// Independent llama.cpp public-API oracle; never linked into vBuf production.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

struct Capture { std::string directory; int position = 0; };

static bool capture(ggml_tensor * tensor, bool ask, void * opaque) {
    const std::string name = tensor->name;
    const bool selected = tensor->type == GGML_TYPE_F32 && tensor->op != GGML_OP_NONE &&
        ggml_nelements(tensor) <= 102400 && name.find('/') == std::string::npos;
    if (ask) return selected;
    if (!selected) return true;
    auto & state = *static_cast<Capture *>(opaque);
    std::cout << std::setprecision(9);
    if (state.position == 0 && tensor->op == GGML_OP_SOFT_MAX && name == "kq_soft_max-0") {
        float scale;
        std::memcpy(&scale, &tensor->op_params[0], 4);
        std::cout << "attention_score_scale=" << scale << " softmax_shape="
                  << tensor->ne[0] << ',' << tensor->ne[1] << ',' << tensor->ne[2] << '\n';
    }
    if (state.position == 0 && tensor->op == GGML_OP_ROPE && name == "q_pe-0") {
        std::cout << "rope integer parameters:";
        for (int i = 0; i < 5; ++i) std::cout << ' ' << tensor->op_params[i];
        std::cout << " float parameters:";
        for (int i = 5; i < 11; ++i) {
            float value;
            std::memcpy(&value, &tensor->op_params[i], 4);
            std::cout << ' ' << value;
        }
        std::cout << '\n';
    }
    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    std::vector<float> values;
    for (int64_t l = 0; l < tensor->ne[3]; ++l)
        for (int64_t k = 0; k < tensor->ne[2]; ++k)
            for (int64_t j = 0; j < tensor->ne[1]; ++j)
                for (int64_t i = 0; i < tensor->ne[0]; ++i) {
                    float value;
                    std::memcpy(&value, raw.data() + i*tensor->nb[0] + j*tensor->nb[1] +
                                k*tensor->nb[2] + l*tensor->nb[3], sizeof(value));
                    values.push_back(value);
                }
    const auto path = state.directory + "/p" + std::to_string(state.position) + "-" + name + ".f32";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(values.data()), values.size()*4);
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) return 2;
    Capture state{argv[2]};
    std::filesystem::create_directories(state.directory);
    llama_backend_init();
    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = {nullptr};
    mp.devices = devices;
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 1; cp.n_ubatch = 1;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.offload_kqv = false; cp.op_offload = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = capture; cp.cb_eval_user_data = &state;
    auto * context = llama_init_from_model(model, cp);
    if (!context) return 1;
    std::vector<llama_token> tokens{100000, 549, 6077, 280, 7239, 317};
    const size_t positions = 5 + (argc == 4 ? std::stoul(argv[3]) : 1);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> generated;
    for (state.position = 0; state.position < static_cast<int>(tokens.size()); ++state.position) {
        auto token = tokens[state.position];
        if (llama_decode(context, llama_batch_get_one(&token, 1))) return 1;
        const auto * logits = llama_get_logits_ith(context, -1);
        if (state.position >= 5) generated.push_back(std::max_element(logits, logits + vocab) - logits);
        std::cout << "position=" << state.position << " token=" << token << " argmax="
                  << std::max_element(logits, logits + vocab) - logits << '\n';
        if (state.position >= 5 && tokens.size() < positions)
            tokens.push_back(std::max_element(logits, logits + vocab) - logits);
    }
    std::ofstream text(state.directory + "/generation.txt", std::ios::binary);
    std::ofstream ids(state.directory + "/generation.tokens");
    for (auto token : generated) {
        char piece[1024];
        const int length = llama_token_to_piece(llama_model_get_vocab(model), token, piece, sizeof(piece), 0, false);
        if (length < 0) return 1;
        text.write(piece, length);
        ids << token << '\n';
    }
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
}
