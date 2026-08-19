#include "llama.h"
#include "llama_vbuf_loader.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace {

struct Config {
    std::string model;
    std::string model_id = "vbuf-model";
    std::string format = "vbuf";
    std::string host = "127.0.0.1";
    int port = 8080;
    int threads = 2;
};

struct HttpRequest { std::string method, path, body; };

static void fail(const std::string & message) { throw std::runtime_error(message); }

static std::string json_escape(const std::string & value) {
    std::string out;
    for (const unsigned char c : value) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: if (c < 0x20) { char buffer[7]; std::snprintf(buffer, sizeof(buffer), "\\u%04x", c); out += buffer; } else out += static_cast<char>(c);
        }
    }
    return out;
}

static std::string json_string(const std::string & json, const std::string & key, const std::string & fallback = "") {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) return fallback;
    size_t pos = json.find(':', key_pos + needle.size());
    if (pos == std::string::npos) return fallback;
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return fallback;
    std::string value;
    for (++pos; pos < json.size(); ++pos) {
        if (json[pos] == '"') break;
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            const char escaped = json[++pos];
            value += escaped == 'n' ? '\n' : escaped == 'r' ? '\r' : escaped == 't' ? '\t' : escaped;
        } else value += json[pos];
    }
    return value;
}

static int json_int(const std::string & json, const std::string & key, int fallback) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) return fallback;
    const size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) return fallback;
    char * end = nullptr;
    const long value = std::strtol(json.c_str() + colon + 1, &end, 10);
    return end == json.c_str() + colon + 1 ? fallback : static_cast<int>(value);
}

static std::string request_prompt(const std::string & body) {
    const size_t messages = body.find("\"messages\"");
    if (messages != std::string::npos) {
        const size_t content = body.find("\"content\"", messages);
        if (content != std::string::npos) return json_string(body.substr(content), "content");
    }
    return json_string(body, "prompt");
}

static bool read_request(int fd, HttpRequest & request) {
    std::string data;
    char buffer[4096];
    size_t content_length = 0;
    while (data.find("\r\n\r\n") == std::string::npos) {
        const ssize_t count = ::recv(fd, buffer, sizeof(buffer), 0);
        if (count <= 0) return false;
        data.append(buffer, static_cast<size_t>(count));
        if (data.size() > 64 * 1024) fail("request headers too large");
    }
    const size_t header_end = data.find("\r\n\r\n");
    std::istringstream headers(data.substr(0, header_end));
    std::string line;
    if (!std::getline(headers, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream first(line);
    first >> request.method >> request.path;
    while (std::getline(headers, line)) {
        if (line.find("Content-Length:") == 0 || line.find("content-length:") == 0) content_length = std::strtoull(line.c_str() + line.find(':') + 1, nullptr, 10);
    }
    if (content_length > 16 * 1024 * 1024) fail("request body too large");
    request.body = data.substr(header_end + 4);
    while (request.body.size() < content_length) {
        const ssize_t count = ::recv(fd, buffer, sizeof(buffer), 0);
        if (count <= 0) return false;
        request.body.append(buffer, static_cast<size_t>(count));
    }
    request.body.resize(content_length);
    return true;
}

static void reply(int fd, int status, const std::string & body) {
    const char * reason = status == 200 ? "OK" : status == 404 ? "Not Found" : "Bad Request";
    std::ostringstream response;
    response << "HTTP/1.1 " << status << ' ' << reason << "\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Authorization, Content-Type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nContent-Length: " << body.size() << "\r\nConnection: close\r\n\r\n" << body;
    const std::string wire = response.str();
    (void)::send(fd, wire.data(), wire.size(), 0);
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> tokens(256);
    int32_t count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(), static_cast<int32_t>(tokens.size()), true, false);
    if (count < 0) { tokens.resize(static_cast<size_t>(-count)); count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(), static_cast<int32_t>(tokens.size()), true, false); }
    if (count < 0) return {};
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

static std::string generate(llama_model * model, const std::string & prompt, int max_tokens, int threads) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> prompt_tokens = tokenize(vocab, prompt);
    if (prompt_tokens.empty()) fail("tokenization failed");
    llama_context_params params = llama_context_default_params();
    params.n_ctx = std::max<uint32_t>(512, static_cast<uint32_t>(prompt_tokens.size() + max_tokens + 8));
    params.n_batch = params.n_ctx;
    params.n_threads = threads;
    params.n_threads_batch = threads;
    llama_context * context = llama_init_from_model(model, params);
    if (!context) fail("context initialization failed");
    std::string output;
    std::vector<llama_token> input = prompt_tokens;
    for (int step = 0; step < max_tokens; ++step) {
        std::vector<int8_t> flags(input.size(), 1);
        llama_batch batch = llama_batch_get_one(input.data(), static_cast<int32_t>(input.size()));
        batch.logits = flags.data();
        if (llama_decode(context, batch) != 0) { llama_free(context); fail("decode failed"); }
        const float * logits = llama_get_logits_ith(context, static_cast<int32_t>(input.size()) - 1);
        if (!logits) { llama_free(context); fail("logits unavailable"); }
        llama_token next = 0;
        for (int i = 1; i < llama_vocab_n_tokens(vocab); ++i) if (logits[i] > logits[next]) next = i;
        if (next == llama_vocab_eos(vocab)) break;
        char text[4096]{};
        const int32_t size = llama_token_to_piece(vocab, next, text, sizeof(text), 0, false);
        if (size > 0) output.append(text, static_cast<size_t>(size));
        input.assign(1, next);
    }
    llama_free(context);
    return output;
}

static void handle(int fd, llama_model * model, const Config & config) {
    try {
        HttpRequest request;
        if (!read_request(fd, request)) return;
        if (request.method == "OPTIONS") { reply(fd, 200, "{}"); return; }
        if (request.method != "POST" && request.path != "/v1/models" && request.path != "/health") { reply(fd, 400, R"({"error":{"message":"unsupported method"}})"); return; }
        if (request.path == "/health") { reply(fd, 200, R"({"status":"ok"})"); return; }
        if (request.path == "/v1/models") { reply(fd, 200, "{\"object\":\"list\",\"data\":[{\"id\":\"" + json_escape(config.model_id) + "\",\"object\":\"model\",\"owned_by\":\"vbuf\"}]}"); return; }
        if (request.path != "/v1/chat/completions" && request.path != "/v1/completions") { reply(fd, 404, R"({"error":{"message":"unknown endpoint"}})"); return; }
        const std::string prompt = request_prompt(request.body);
        if (prompt.empty()) { reply(fd, 400, R"({"error":{"message":"prompt or messages is required"}})"); return; }
        const int max_tokens = std::clamp(json_int(request.body, "max_tokens", 128), 1, 4096);
        const std::string text = generate(model, prompt, max_tokens, config.threads);
        const std::string id = "vbuf-" + std::to_string(static_cast<unsigned long long>(::getpid()));
        std::string body;
        if (request.path == "/v1/chat/completions") body = "{\"id\":\"" + id + "\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" + json_escape(text) + "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}";
        else body = "{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"" + json_escape(text) + "\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}";
        reply(fd, 200, body);
    } catch (const std::exception & error) { reply(fd, 400, "{\"error\":{\"message\":\"" + json_escape(error.what()) + "\"}}"); }
}

static Config parse_args(int argc, char ** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char * name) { if (i + 1 >= argc) fail(std::string("missing value for ") + name); return std::string(argv[++i]); };
        if (arg == "--model") config.model = value("--model"); else if (arg == "--format") config.format = value("--format"); else if (arg == "--model-id") config.model_id = value("--model-id"); else if (arg == "--host") config.host = value("--host"); else if (arg == "--port") config.port = std::stoi(value("--port")); else if (arg == "--threads") config.threads = std::stoi(value("--threads")); else fail("unknown argument: " + arg);
    }
    if (config.model.empty() || (config.format != "vbuf" && config.format != "gguf")) fail("usage: --model PATH [--format vbuf|gguf]");
    return config;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Config config = parse_args(argc, argv);
        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        llama_model * model = config.format == "vbuf" ? llama_model_load_vbuf_direct(config.model.c_str(), model_params) : llama_model_load_from_file(config.model.c_str(), model_params);
        if (!model) fail("model load failed");
        const int server = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server < 0) fail("socket failed");
        int reuse = 1; ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_ANY); address.sin_port = htons(static_cast<uint16_t>(config.port));
        if (::bind(server, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0 || ::listen(server, 16) < 0) fail(std::string("bind/listen failed: ") + std::strerror(errno));
        std::fprintf(stderr, "vbuf-openai-server listening on %s:%d model=%s format=%s\n", config.host.c_str(), config.port, config.model.c_str(), config.format.c_str());
        while (true) { const int client = ::accept(server, nullptr, nullptr); if (client < 0) continue; handle(client, model, config); ::close(client); }
    } catch (const std::exception & error) { std::fprintf(stderr, "vbuf-openai-server: %s\n", error.what()); return 1; }
}
