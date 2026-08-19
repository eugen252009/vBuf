#include <jni.h>
#include <android/log.h>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"
#include "llama_vbuf_loader.h"
#include "vbuf_remote_source.h"

namespace {
constexpr const char * TAG = "vbuf-android-chat";
std::mutex mutex;
llama_model * model = nullptr;
llama_context * context = nullptr;

void llama_log_to_android(enum ggml_log_level level, const char * text, void *) {
    const android_LogPriority priority = level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO;
    __android_log_print(priority, TAG, "%s", text);
}

jstring result(JNIEnv * env, const std::string & value) {
    return env->NewStringUTF(value.c_str());
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> tokens(256);
    int32_t count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(), static_cast<int32_t>(tokens.size()), true, false);
    if (count < 0) {
        tokens.resize(static_cast<size_t>(-count));
        count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(), static_cast<int32_t>(tokens.size()), true, false);
    }
    if (count < 0) return {};
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

llama_token greedy(const float * logits, int32_t count) {
    llama_token best = 0;
    for (int32_t i = 1; i < count; ++i) if (logits[i] > logits[best]) best = i;
    return best;
}

std::string piece(const llama_vocab * vocab, llama_token token) {
    std::string value(256, '\0');
    int32_t count = llama_token_to_piece(vocab, token, value.data(), static_cast<int32_t>(value.size()), 0, false);
    if (count < 0) { value.resize(static_cast<size_t>(-count)); count = llama_token_to_piece(vocab, token, value.data(), static_cast<int32_t>(value.size()), 0, false); }
    return count < 0 ? std::string() : value.substr(0, static_cast<size_t>(count));
}
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_eugen_vbufchat_NativeInference_open__Ljava_lang_String_2Ljava_lang_String_2(JNIEnv * env, jclass, jstring path, jstring endpoint) {
    std::lock_guard lock(mutex);
    if (model) return result(env, "OPEN_OK already_open");
    const char * raw_path = env->GetStringUTFChars(path, nullptr);
    const char * raw_endpoint = env->GetStringUTFChars(endpoint, nullptr);
    const bool remote = raw_endpoint[0] != '\0';
    const auto start = std::chrono::steady_clock::now();
    llama_log_set(llama_log_to_android, nullptr);
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    model = !remote
        ? llama_model_load_vbuf_direct(raw_path, params)
        : llama_model_load_vbuf_remote(raw_path, raw_endpoint, params);
    env->ReleaseStringUTFChars(path, raw_path);
    env->ReleaseStringUTFChars(endpoint, raw_endpoint);
    if (!model) return result(env, std::string("OPEN_FAIL model_load ") + (remote ? llama_model_last_remote_error() : "local"));
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 512;
    context_params.n_batch = 512;
    context_params.n_threads = 4;
    context_params.n_threads_batch = 4;
    context = llama_init_from_model(model, context_params);
    if (!context) { llama_model_free_vbuf_direct(model); model = nullptr; return result(env, "OPEN_FAIL context"); }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    __android_log_print(ANDROID_LOG_INFO, TAG, "opened vBuf model in %lld ms remote=%s", static_cast<long long>(elapsed), remote ? "yes" : "no");
    if (remote) {
        vbuf_llama::VbufRemoteMetrics metrics{};
        if (llama_model_vbuf_remote_metrics(model, &metrics)) {
            __android_log_print(ANDROID_LOG_INFO, TAG, "remote metrics requests=%llu bytes=%llu unique_bytes=%llu", static_cast<unsigned long long>(metrics.requests), static_cast<unsigned long long>(metrics.bytes), static_cast<unsigned long long>(metrics.unique_bytes));
        }
    }
    return result(env, remote ? "OPEN_OK qwen3 remote" : "OPEN_OK qwen3");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_eugen_vbufchat_NativeInference_probeRemote(JNIEnv * env, jclass, jstring path, jstring endpoint) {
    const char * raw_path = env->GetStringUTFChars(path, nullptr);
    const char * raw_endpoint = env->GetStringUTFChars(endpoint, nullptr);
    vbuf_llama::VbufRemoteMetrics metrics{};
    const bool ok = llama_model_probe_vbuf_remote(raw_path, raw_endpoint, &metrics);
    env->ReleaseStringUTFChars(path, raw_path);
    env->ReleaseStringUTFChars(endpoint, raw_endpoint);
    if (!ok) return result(env, "PROBE_FAIL remote_range");
    return result(env, "PROBE_OK requests=" + std::to_string(metrics.requests) + " bytes=" + std::to_string(metrics.bytes));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_eugen_vbufchat_NativeInference_generate(JNIEnv * env, jclass, jstring prompt, jint max_tokens) {
    std::lock_guard lock(mutex);
    if (!model || !context) return result(env, "GEN_FAIL model_not_open");
    const char * raw_prompt = env->GetStringUTFChars(prompt, nullptr);
    std::vector<llama_token> tokens = tokenize(llama_model_get_vocab(model), raw_prompt);
    env->ReleaseStringUTFChars(prompt, raw_prompt);
    if (tokens.empty()) return result(env, "GEN_FAIL tokenize");
    std::vector<int8_t> logits(tokens.size(), 1);
    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    batch.logits = logits.data();
    if (llama_decode(context, batch) != 0) return result(env, "GEN_FAIL prompt_decode");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string generated;
    const int32_t vocabulary = llama_vocab_n_tokens(vocab);
    for (int32_t step = 0; step < max_tokens; ++step) {
        const llama_token next = greedy(llama_get_logits_ith(context, static_cast<int32_t>(tokens.size()) - 1), vocabulary);
        generated += piece(vocab, next);
        if (llama_vocab_is_eog(vocab, next)) break;
        tokens.assign(1, next);
        int8_t logit = 1;
        batch = llama_batch_get_one(tokens.data(), 1);
        batch.logits = &logit;
        if (llama_decode(context, batch) != 0) return result(env, "GEN_FAIL token_decode");
    }
    return result(env, generated.empty() ? "GEN_OK <empty>" : generated);
}

extern "C" JNIEXPORT void JNICALL
Java_com_eugen_vbufchat_NativeInference_closeModel(JNIEnv *, jclass) {
    std::lock_guard lock(mutex);
    if (context) { llama_free(context); context = nullptr; }
    if (model) { llama_model_free_vbuf_direct(model); model = nullptr; }
    llama_backend_free();
}
