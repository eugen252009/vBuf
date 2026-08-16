// Reference-only activation capture for Region POC 1.
// This is an architecture-specific lowering/oracle tool, not runtime code.

#include "llama_vbuf_loader.h"
#include "llama-context.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static void quiet_log(enum ggml_log_level, const char *, void *) {}

struct Capture {
    std::string pending;
    std::vector<uint8_t> ffn_inp;
    std::vector<uint8_t> ffn_out;
    std::vector<uint8_t> ffn_swiglu;
    int64_t ffn_inp_ne[4]{};
    int64_t ffn_out_ne[4]{};
    int64_t ffn_swiglu_ne[4]{};
    enum ggml_type ffn_inp_type = GGML_TYPE_COUNT;
    enum ggml_type ffn_out_type = GGML_TYPE_COUNT;
    enum ggml_type ffn_swiglu_type = GGML_TYPE_COUNT;
};

static bool target_name(const char * name) {
    return std::strcmp(name, "ffn_inp-0") == 0 ||
        std::strcmp(name, "ffn_swiglu-0") == 0 ||
        std::strcmp(name, "ffn_out-0") == 0;
}

static void capture_tensor(ggml_tensor * tensor, Capture * capture) {
    const char * name = ggml_get_name(tensor);
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    int64_t * destination = nullptr;
    enum ggml_type * type = nullptr;
    if (std::strcmp(name, "ffn_inp-0") == 0) {
        capture->ffn_inp = std::move(bytes);
        destination = capture->ffn_inp_ne;
        type = &capture->ffn_inp_type;
    } else if (std::strcmp(name, "ffn_swiglu-0") == 0) {
        capture->ffn_swiglu = std::move(bytes);
        destination = capture->ffn_swiglu_ne;
        type = &capture->ffn_swiglu_type;
    } else {
        capture->ffn_out = std::move(bytes);
        destination = capture->ffn_out_ne;
        type = &capture->ffn_out_type;
    }
    *type = tensor->type;
    for (int i = 0; i < 4; ++i) destination[i] = tensor->ne[i];
}

static bool eval_callback(ggml_tensor * tensor, bool ask, void * userdata) {
    auto * capture = static_cast<Capture *>(userdata);
    const char * name = ggml_get_name(tensor);
    if (std::strstr(name, "ffn") != nullptr && ask) {
        std::fprintf(stderr, "reference_node=%s type=%d ne0=%lld ne1=%lld bytes=%zu\n",
            name, static_cast<int>(tensor->type), static_cast<long long>(tensor->ne[0]),
            static_cast<long long>(tensor->ne[1]), ggml_nbytes(tensor));
    }
    if (!target_name(name)) return false;
    if (ask) {
        capture->pending = name;
        return true;
    }
    if (capture->pending != name) return false;
    capture_tensor(tensor, capture);
    capture->pending.clear();
    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab) {
    const char * prompt = "Hello world";
    std::vector<llama_token> tokens(64);
    int32_t count = llama_tokenize(vocab, prompt, std::strlen(prompt),
        tokens.data(), static_cast<int32_t>(tokens.size()), false, false);
    if (count < 0) {
        tokens.resize(static_cast<size_t>(-count));
        count = llama_tokenize(vocab, prompt, std::strlen(prompt),
            tokens.data(), static_cast<int32_t>(tokens.size()), false, false);
    }
    if (count < 0) return {};
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

static bool write_capture(const std::filesystem::path & directory,
    const char * name, const std::vector<uint8_t> & bytes) {
    std::ofstream output(directory / name, std::ios::binary);
    output.write(reinterpret_cast<const char *>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    return output.good();
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    const std::filesystem::path output_directory = argv[2];
    std::filesystem::create_directories(output_directory);
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.check_tensors = false;
    llama_model * model = llama_model_load_vbuf_direct(argv[1], model_params);
    if (model == nullptr) return 3;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = tokenize(vocab);
    if (tokens.empty()) return 4;
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_threads = 2;
    context_params.n_threads_batch = 2;
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) return 5;
    std::vector<int8_t> logits(tokens.size(), 1);
    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    batch.logits = logits.data();
    if (llama_decode(context, batch) != 0) return 6;
    Capture capture;
    std::vector<int8_t> second_logits(tokens.size(), 1);
    llama_batch second = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    second.logits = second_logits.data();
    ggml_backend_sched_set_eval_callback(context->get_sched(), eval_callback, &capture);
    const int decode_status = llama_decode(context, second);
    ggml_backend_sched_set_eval_callback(context->get_sched(), nullptr, nullptr);
    if (decode_status != 0 || capture.ffn_inp.empty() || capture.ffn_swiglu.empty() || capture.ffn_out.empty()) {
        std::fprintf(stderr, "reference_capture_failed decode=%d inp=%zu swiglu=%zu out=%zu\n",
            decode_status, capture.ffn_inp.size(), capture.ffn_swiglu.size(), capture.ffn_out.size());
        return 6;
    }
    if (!write_capture(output_directory, "ffn_inp.f32", capture.ffn_inp) ||
        !write_capture(output_directory, "ffn_swiglu.f32", capture.ffn_swiglu) ||
        !write_capture(output_directory, "ffn_out.f32", capture.ffn_out)) return 7;
    std::ofstream metadata(output_directory / "metadata.txt");
    metadata << "artifact=" << argv[1] << "\n"
             << "reference_prompt=Hello world\n"
             << "reference_revision=4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c\n"
             << "ffn_inp_type=" << static_cast<int>(capture.ffn_inp_type) << "\n"
             << "ffn_out_type=" << static_cast<int>(capture.ffn_out_type) << "\n"
             << "ffn_swiglu_type=" << static_cast<int>(capture.ffn_swiglu_type) << "\n"
             << "ffn_inp_ne=" << capture.ffn_inp_ne[0] << "," << capture.ffn_inp_ne[1] << "\n"
             << "ffn_out_ne=" << capture.ffn_out_ne[0] << "," << capture.ffn_out_ne[1] << "\n"
             << "ffn_swiglu_ne=" << capture.ffn_swiglu_ne[0] << "," << capture.ffn_swiglu_ne[1] << "\n"
             << "ffn_inp_bytes=" << capture.ffn_inp.size() << "\n"
             << "ffn_out_bytes=" << capture.ffn_out.size() << "\n"
             << "ffn_swiglu_bytes=" << capture.ffn_swiglu.size() << "\n";
    llama_free(context);
    llama_model_free_vbuf_direct(model);
    llama_backend_free();
    return metadata.good() ? 0 : 8;
}
