#define VBUF_POC22_LIBRARY_ONLY
#include "autoregressive_poc22.cpp"
#undef VBUF_POC22_LIBRARY_ONLY

#include "vbuf_generation.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <arpa/inet.h>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <netinet/in.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <strings.h>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {
struct VbufMlConsumerHandle;
VbufMlConsumerHandle * vbuf_ml_consumer_open_metadata(const char * path);
void vbuf_ml_consumer_close(VbufMlConsumerHandle * handle);
uint32_t vbuf_ml_consumer_token_count(const VbufMlConsumerHandle *, uint64_t * count);
uint32_t vbuf_ml_consumer_token_text(const VbufMlConsumerHandle *, uint64_t index,
    char * buffer, size_t capacity);
uint32_t vbuf_ml_consumer_merge_count(const VbufMlConsumerHandle *, uint64_t * count);
uint32_t vbuf_ml_consumer_merge_pair(const VbufMlConsumerHandle *, uint64_t index,
    uint64_t * left, uint64_t * right);
uint32_t vbuf_ml_consumer_merge_rank(const VbufMlConsumerHandle *, uint64_t left,
    uint64_t right, uint32_t * rank);
uint32_t vbuf_ml_consumer_special_token(const VbufMlConsumerHandle *, uint8_t kind,
    uint64_t * value);
uint32_t vbuf_ml_consumer_add_bos(const VbufMlConsumerHandle *, bool * value);
uint32_t vbuf_ml_consumer_chat_template(const VbufMlConsumerHandle *, char * buffer,
    size_t capacity);
uint32_t vbuf_ml_consumer_runtime_indexes(const VbufMlConsumerHandle *);
uint32_t vbuf_ml_consumer_token_id(const VbufMlConsumerHandle *, const uint8_t *, size_t,
    uint32_t * token_id);
}

namespace {

constexpr uint32_t OK = 0;
constexpr uint32_t BUFFER_TOO_SMALL = 3;
constexpr size_t MAX_REQUEST_BYTES = 4 * 1024 * 1024;

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

struct Message {
    std::string role;
    std::string content;
};

struct ServerConfig {
    std::string semantic_model;
    std::string source_url;
    std::string model_alias = "vbuf-model";
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    uint32_t blocks = 2;
    uint64_t capacity = 268435456;
    uint32_t max_new_tokens = 4;
    vbuf_ggml::RuntimeMode mode = vbuf_ggml::RuntimeMode::NormalInference;
};

struct JsonError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

static void fail(const std::string & message) { throw JsonError(message); }

static void append_utf8(std::string * output, uint32_t codepoint) {
    if (codepoint <= 0x7f) output->push_back(static_cast<char>(codepoint));
    else if (codepoint <= 0x7ff) {
        output->push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        output->push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        output->push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output->push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

static bool next_utf8(const std::string & value, size_t * offset, uint32_t * codepoint) {
    if (*offset >= value.size()) return false;
    const auto byte = static_cast<unsigned char>(value[*offset]);
    if (byte < 0x80) { *codepoint = byte; ++*offset; return true; }
    size_t count = byte >= 0xf0 ? 4 : byte >= 0xe0 ? 3 : 2;
    if (*offset + count > value.size()) return false;
    uint32_t result = byte & (count == 4 ? 0x07 : count == 3 ? 0x0f : 0x1f);
    for (size_t index = 1; index < count; ++index) {
        const auto continuation = static_cast<unsigned char>(value[*offset + index]);
        if ((continuation & 0xc0) != 0x80) return false;
        result = (result << 6) | (continuation & 0x3f);
    }
    *offset += count;
    *codepoint = result;
    return true;
}

static std::string json_escape(const std::string & value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const unsigned char byte : value) {
        switch (byte) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (byte < 0x20) output << "\\u" << std::setw(4) << static_cast<unsigned>(byte);
            else output << static_cast<char>(byte);
        }
    }
    return output.str();
}

static std::string json_string(const std::string & body, size_t * cursor) {
    if (*cursor >= body.size() || body[*cursor] != '"') fail("expected JSON string");
    ++*cursor;
    std::string output;
    while (*cursor < body.size()) {
        const unsigned char byte = static_cast<unsigned char>(body[*cursor]);
        ++*cursor;
        if (byte == '"') return output;
        if (byte != '\\') { output.push_back(static_cast<char>(byte)); continue; }
        if (*cursor >= body.size()) fail("unterminated JSON escape");
        const char escaped = body[(*cursor)++];
        switch (escaped) {
        case '"': output.push_back('"'); break;
        case '\\': output.push_back('\\'); break;
        case '/': output.push_back('/'); break;
        case 'b': output.push_back('\b'); break;
        case 'f': output.push_back('\f'); break;
        case 'n': output.push_back('\n'); break;
        case 'r': output.push_back('\r'); break;
        case 't': output.push_back('\t'); break;
        case 'u': {
            if (*cursor + 4 > body.size()) fail("short JSON unicode escape");
            uint32_t codepoint = 0;
            for (size_t index = 0; index < 4; ++index) {
                const char digit = body[(*cursor)++];
                const int value = digit >= '0' && digit <= '9' ? digit - '0' :
                    digit >= 'a' && digit <= 'f' ? digit - 'a' + 10 :
                    digit >= 'A' && digit <= 'F' ? digit - 'A' + 10 : -1;
                if (value < 0) fail("invalid JSON unicode escape");
                codepoint = (codepoint << 4) | static_cast<uint32_t>(value);
            }
            append_utf8(&output, codepoint);
            break;
        }
        default: fail("unsupported JSON escape");
        }
    }
    fail("unterminated JSON string");
    return {};
}

static size_t skip_space(const std::string & body, size_t cursor) {
    while (cursor < body.size() && std::isspace(static_cast<unsigned char>(body[cursor]))) ++cursor;
    return cursor;
}

static std::optional<size_t> top_level_key(const std::string & body, const std::string & wanted) {
    size_t cursor = skip_space(body, 0);
    if (cursor >= body.size() || body[cursor] != '{') return std::nullopt;
    ++cursor;
    while (true) {
        cursor = skip_space(body, cursor);
        if (cursor >= body.size()) return std::nullopt;
        if (body[cursor] == '}') return std::nullopt;
        if (body[cursor] != '"') return std::nullopt;
        const std::string key = json_string(body, &cursor);
        cursor = skip_space(body, cursor);
        if (cursor >= body.size() || body[cursor++] != ':') return std::nullopt;
        cursor = skip_space(body, cursor);
        if (key == wanted) return cursor;
        if (cursor >= body.size()) return std::nullopt;
        if (body[cursor] == '"') { (void)json_string(body, &cursor); }
        else if (body[cursor] == '[' || body[cursor] == '{') {
            const char open = body[cursor++];
            const char close = open == '[' ? ']' : '}';
            int depth = 1;
            bool string = false, escaped = false;
            while (cursor < body.size() && depth != 0) {
                const char value = body[cursor++];
                if (string) { if (escaped) escaped = false; else if (value == '\\') escaped = true; else if (value == '"') string = false; continue; }
                if (value == '"') string = true;
                else if (value == open) ++depth;
                else if (value == close) --depth;
            }
        } else {
            while (cursor < body.size() && body[cursor] != ',' && body[cursor] != '}') ++cursor;
        }
        cursor = skip_space(body, cursor);
        if (cursor < body.size() && body[cursor] == ',') ++cursor;
    }
}

static std::string top_level_string(const std::string & body, const std::string & key,
    const std::string & fallback = {}) {
    const auto position = top_level_key(body, key);
    if (!position) return fallback;
    size_t cursor = *position;
    return json_string(body, &cursor);
}

static bool top_level_bool(const std::string & body, const std::string & key, bool fallback) {
    const auto position = top_level_key(body, key);
    if (!position) return fallback;
    return body.compare(*position, 4, "true") == 0;
}

static uint32_t top_level_uint(const std::string & body, const std::string & key,
    uint32_t fallback) {
    const auto position = top_level_key(body, key);
    if (!position) return fallback;
    char * end = nullptr;
    const unsigned long value = std::strtoul(body.c_str() + *position, &end, 10);
    if (end == body.c_str() + *position || value > UINT32_MAX) fail("invalid numeric option: " + key);
    return static_cast<uint32_t>(value);
}

static bool has_top_level_key(const std::string & body, const std::string & key) {
    return top_level_key(body, key).has_value();
}

static size_t matching_array_end(const std::string & body, size_t start) {
    if (start >= body.size() || body[start] != '[') fail("messages must be an array");
    int depth = 0;
    bool string = false, escaped = false;
    for (size_t cursor = start; cursor < body.size(); ++cursor) {
        const char value = body[cursor];
        if (string) { if (escaped) escaped = false; else if (value == '\\') escaped = true; else if (value == '"') string = false; continue; }
        if (value == '"') string = true;
        else if (value == '[') ++depth;
        else if (value == ']' && --depth == 0) return cursor;
    }
    fail("unterminated messages array");
    return start;
}

static std::vector<Message> parse_messages(const std::string & body) {
    const auto position = top_level_key(body, "messages");
    if (!position) fail("messages is required");
    const size_t start = skip_space(body, *position);
    const size_t end = matching_array_end(body, start);
    std::vector<Message> messages;
    size_t cursor = start + 1;
    while (cursor < end) {
        const size_t role_key = body.find("\"role\"", cursor);
        if (role_key == std::string::npos || role_key >= end) break;
        cursor = body.find(':', role_key + 6);
        if (cursor == std::string::npos || cursor >= end) fail("message role is malformed");
        cursor = skip_space(body, cursor + 1);
        Message message;
        message.role = json_string(body, &cursor);
        const size_t content_key = body.find("\"content\"", cursor);
        if (content_key == std::string::npos || content_key >= end) fail("message content is required");
        cursor = body.find(':', content_key + 9);
        if (cursor == std::string::npos || cursor >= end) fail("message content is malformed");
        cursor = skip_space(body, cursor + 1);
        if (cursor >= body.size() || body[cursor] != '"')
            fail("only string message content is supported");
        message.content = json_string(body, &cursor);
        if (message.role != "system" && message.role != "user" && message.role != "assistant")
            fail("unsupported message role");
        messages.push_back(std::move(message));
    }
    if (messages.empty()) fail("messages must contain at least one message");
    return messages;
}

static std::string literal_value(const std::string & expression, size_t begin, size_t end) {
    while (begin < end && std::isspace(static_cast<unsigned char>(expression[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(expression[end - 1]))) --end;
    if (end - begin < 2 || expression[begin] != '\'' || expression[end - 1] != '\'')
        fail("unsupported chat-template expression");
    std::string value;
    for (size_t cursor = begin + 1; cursor + 1 < end; ++cursor) {
        if (expression[cursor] != '\\') { value.push_back(expression[cursor]); continue; }
        if (++cursor + 1 > end) fail("invalid chat-template literal");
        switch (expression[cursor]) {
        case 'n': value.push_back('\n'); break;
        case 'r': value.push_back('\r'); break;
        case 't': value.push_back('\t'); break;
        case '\\': value.push_back('\\'); break;
        case '\'': value.push_back('\''); break;
        default: value.push_back(expression[cursor]); break;
        }
    }
    return value;
}

static std::vector<std::string> split_plus(const std::string & expression) {
    std::vector<std::string> parts;
    size_t begin = 0;
    bool quoted = false;
    for (size_t cursor = 0; cursor < expression.size(); ++cursor) {
        if (expression[cursor] == '\'' && (cursor == 0 || expression[cursor - 1] != '\\')) quoted = !quoted;
        if (!quoted && expression[cursor] == '+') {
            parts.push_back(expression.substr(begin, cursor - begin));
            begin = cursor + 1;
        }
    }
    parts.push_back(expression.substr(begin));
    return parts;
}

struct PromptPart {
    std::string text;
    std::optional<uint32_t> special;
};

static void evaluate_template_expression(const std::string & expression,
    const Message * message, const std::string & bos_text, const std::string & eos_text,
    uint32_t bos_id, uint32_t eos_id, std::vector<PromptPart> * output) {
    for (const std::string & raw : split_plus(expression)) {
        size_t begin = 0, end = raw.size();
        while (begin < end && std::isspace(static_cast<unsigned char>(raw[begin]))) ++begin;
        while (end > begin && std::isspace(static_cast<unsigned char>(raw[end - 1]))) --end;
        const std::string term = raw.substr(begin, end - begin);
        if (term == "message['content']") output->push_back({ message ? message->content : "", std::nullopt });
        else if (term == "bos_token") output->push_back({ bos_text, bos_id });
        else if (term == "eos_token") output->push_back({ eos_text, eos_id });
        else output->push_back({ literal_value(raw, 0, raw.size()), std::nullopt });
    }
}

static std::vector<PromptPart> render_chat_template(const std::string & templ,
    const std::vector<Message> & messages, const std::string & bos_text, const std::string & eos_text,
    uint32_t bos_id, uint32_t eos_id) {
    const std::string loop_begin = "{% for message in messages %}";
    const std::string loop_end = "{% endfor %}";
    const size_t loop_start = templ.find(loop_begin);
    const size_t loop_stop = templ.find(loop_end, loop_start == std::string::npos ? 0 : loop_start);
    if (loop_start == std::string::npos || loop_stop == std::string::npos)
        fail("model chat template is outside the supported runtime subset");
    std::vector<PromptPart> output;
    auto render_expressions = [&](const std::string & source, const Message * message) {
        size_t cursor = 0;
        while ((cursor = source.find("{{", cursor)) != std::string::npos) {
            const size_t end = source.find("}}", cursor + 2);
            if (end == std::string::npos) fail("unterminated chat-template expression");
            evaluate_template_expression(source.substr(cursor + 2, end - cursor - 2), message,
                bos_text, eos_text, bos_id, eos_id, &output);
            cursor = end + 2;
        }
    };
    render_expressions(templ.substr(0, loop_start), nullptr);
    const std::string loop = templ.substr(loop_start + loop_begin.size(), loop_stop - loop_start - loop_begin.size());
    for (const Message & message : messages) {
        const std::string marker = "message['role'] == '" + message.role + "'";
        const size_t branch = loop.find(marker);
        if (branch == std::string::npos) continue;
        const size_t body_start = loop.find("%}", branch);
        if (body_start == std::string::npos) fail("malformed chat-template branch");
        size_t body_end = loop.find("{% elif", body_start + 2);
        body_end = std::min(body_end == std::string::npos ? loop.size() : body_end,
            loop.find("{% else", body_start + 2) == std::string::npos ? loop.size() : loop.find("{% else", body_start + 2));
        render_expressions(loop.substr(body_start + 2, body_end - body_start - 2), &message);
    }
    const size_t after_loop = loop_stop + loop_end.size();
    const size_t generation = templ.find("{% if add_generation_prompt %}", after_loop);
    if (generation != std::string::npos) {
        const size_t body_start = templ.find("%}", generation);
        const size_t body_end = templ.find("{% endif %}", body_start);
        if (body_start == std::string::npos || body_end == std::string::npos)
            fail("malformed generation-prompt branch");
        render_expressions(templ.substr(body_start + 2, body_end - body_start - 2), nullptr);
    }
    return output;
}

static std::string byte_encoded(const std::string & input) {
    std::array<uint32_t, 256> mapping{};
    std::vector<uint32_t> bytes;
    for (uint32_t value = '!' ; value <= '~'; ++value) bytes.push_back(value);
    for (uint32_t value = 0xa1; value <= 0xac; ++value) bytes.push_back(value);
    for (uint32_t value = 0xae; value <= 0xff; ++value) bytes.push_back(value);
    uint32_t next = 0;
    for (uint32_t value = 0; value < 256; ++value) {
        if (std::find(bytes.begin(), bytes.end(), value) == bytes.end()) bytes.push_back(value), mapping[value] = 256 + next++;
    }
    for (size_t index = 0; index < bytes.size(); ++index)
        if (mapping[bytes[index]] == 0) mapping[bytes[index]] = bytes[index];
    std::string output;
    for (const unsigned char byte : input) append_utf8(&output, mapping[byte]);
    return output;
}

static void init_byte_reverse(std::unordered_map<uint32_t, uint8_t> * reverse);

class VbufTokenizer {
public:
    explicit VbufTokenizer(const std::string & path) {
        init_byte_reverse(&reverse_byte_);
        handle_ = vbuf_ml_consumer_open_metadata(path.c_str());
        if (handle_ == nullptr) fail("vBuf tokenizer/model metadata open failed");
        uint64_t count = 0;
        if (vbuf_ml_consumer_token_count(handle_, &count) != OK) fail("vBuf tokenizer count failed");
        token_texts_.resize(count);
        for (uint64_t index = 0; index < count; ++index) {
            char buffer[8192]{};
            const uint32_t status = vbuf_ml_consumer_token_text(handle_, index, buffer, sizeof(buffer));
            if (status != OK) fail("vBuf token text lookup failed");
            token_texts_[index] = buffer;
            token_ids_[token_texts_[index]] = static_cast<uint32_t>(index);
        }
        uint64_t merge_count = 0;
        if (vbuf_ml_consumer_merge_count(handle_, &merge_count) != OK) fail("vBuf merge count failed");
        if (vbuf_ml_consumer_runtime_indexes(handle_) != OK) fail("vBuf tokenizer index build failed");
        for (uint64_t index = 0; index < merge_count; ++index) {
            uint64_t left = 0, right = 0;
            uint32_t rank = 0;
            if (vbuf_ml_consumer_merge_pair(handle_, index, &left, &right) != OK ||
                vbuf_ml_consumer_merge_rank(handle_, left, right, &rank) != OK)
                fail("vBuf merge lookup failed");
            merge_ranks_[(left << 32) | right] = rank;
        }
        for (uint8_t kind = 0; kind < 4; ++kind) {
            uint64_t value = 0;
            if (vbuf_ml_consumer_special_token(handle_, kind, &value) == OK) special_[kind] = value;
        }
        bool add_bos = false;
        if (vbuf_ml_consumer_add_bos(handle_, &add_bos) == OK) add_bos_ = add_bos;
        char template_buffer[1 << 20]{};
        if (vbuf_ml_consumer_chat_template(handle_, template_buffer, sizeof(template_buffer)) == OK)
            chat_template_ = template_buffer;
    }

    ~VbufTokenizer() { if (handle_ != nullptr) vbuf_ml_consumer_close(handle_); }
    VbufTokenizer(const VbufTokenizer &) = delete;

    std::vector<uint32_t> encode_text(const std::string & text, bool include_bos) const {
        std::vector<PromptPart> parts;
        if (include_bos && special_[0]) parts.push_back({ {}, static_cast<uint32_t>(*special_[0]) });
        parts.push_back({ text, std::nullopt });
        return encode_parts(parts);
    }

    std::vector<uint32_t> encode_chat(const std::vector<Message> & messages) const {
        if (chat_template_.empty()) fail("vBuf model has no chat template");
        const uint32_t bos = static_cast<uint32_t>(special_[0].value_or(0));
        const uint32_t eos = static_cast<uint32_t>(special_[1].value_or(0));
        std::vector<PromptPart> parts = render_chat_template(chat_template_, messages,
            token_texts_[bos], token_texts_[eos], bos, eos);
        return encode_parts(parts);
    }

    std::string decode(const std::vector<uint32_t> & tokens) const {
        std::string bytes;
        for (uint32_t token : tokens) {
            if (token >= token_texts_.size()) continue;
            size_t cursor = 0;
            uint32_t codepoint = 0;
            while (next_utf8(token_texts_[token], &cursor, &codepoint)) {
                const auto found = reverse_byte_.find(codepoint);
                if (found != reverse_byte_.end()) bytes.push_back(static_cast<char>(found->second));
                else append_utf8(&bytes, codepoint);
            }
        }
        return bytes;
    }

    std::string decode_token(uint32_t token) const { return decode({ token }); }
    std::optional<uint32_t> eos() const {
        return special_[1] ? std::optional<uint32_t>(static_cast<uint32_t>(*special_[1])) : std::nullopt;
    }
    bool add_bos() const { return add_bos_; }

private:
    std::vector<uint32_t> encode_parts(const std::vector<PromptPart> & parts) const {
        std::vector<uint32_t> output;
        for (const PromptPart & part : parts) {
            if (part.special) { output.push_back(*part.special); continue; }
            const std::string encoded = byte_encoded(part.text);
            std::vector<uint32_t> ids;
            size_t cursor = 0;
            while (cursor < encoded.size()) {
                uint32_t codepoint = 0;
                const size_t begin = cursor;
                if (!next_utf8(encoded, &cursor, &codepoint)) fail("vBuf tokenizer produced invalid UTF-8");
                const std::string symbol = encoded.substr(begin, cursor - begin);
                const auto found = token_ids_.find(symbol);
                if (found == token_ids_.end()) {
                    if (!special_[2]) fail("vBuf tokenizer cannot represent input byte");
                    ids.push_back(static_cast<uint32_t>(*special_[2]));
                } else ids.push_back(found->second);
            }
            while (ids.size() > 1) {
                size_t best = ids.size();
                uint32_t best_rank = UINT32_MAX;
                for (size_t index = 0; index + 1 < ids.size(); ++index) {
                    const auto found = merge_ranks_.find((static_cast<uint64_t>(ids[index]) << 32) | ids[index + 1]);
                    if (found != merge_ranks_.end() && found->second < best_rank) {
                        best = index; best_rank = found->second;
                    }
                }
                if (best == ids.size()) break;
                const std::string merged = token_texts_[ids[best]] + token_texts_[ids[best + 1]];
                const auto found = token_ids_.find(merged);
                if (found == token_ids_.end()) fail("vBuf tokenizer merge result is absent");
                ids[best] = found->second;
                ids.erase(ids.begin() + static_cast<std::ptrdiff_t>(best + 1));
            }
            output.insert(output.end(), ids.begin(), ids.end());
        }
        return output;
    }

    VbufMlConsumerHandle * handle_ = nullptr;
    std::vector<std::string> token_texts_;
    std::unordered_map<std::string, uint32_t> token_ids_;
    std::unordered_map<uint64_t, uint32_t> merge_ranks_;
    std::array<std::optional<uint64_t>, 4> special_{};
    std::unordered_map<uint32_t, uint8_t> reverse_byte_;
    std::string chat_template_;
    bool add_bos_ = false;
};

static void init_byte_reverse(std::unordered_map<uint32_t, uint8_t> * reverse) {
    std::vector<uint32_t> bytes;
    for (uint32_t value = '!' ; value <= '~'; ++value) bytes.push_back(value);
    for (uint32_t value = 0xa1; value <= 0xac; ++value) bytes.push_back(value);
    for (uint32_t value = 0xae; value <= 0xff; ++value) bytes.push_back(value);
    uint32_t next = 0;
    std::array<uint32_t, 256> mapping{};
    for (uint32_t value = 0; value < 256; ++value) {
        if (std::find(bytes.begin(), bytes.end(), value) == bytes.end()) bytes.push_back(value), mapping[value] = 256 + next++;
    }
    for (size_t index = 0; index < bytes.size(); ++index)
        if (mapping[bytes[index]] == 0) mapping[bytes[index]] = bytes[index];
    for (uint32_t value = 0; value < 256; ++value) (*reverse)[mapping[value]] = static_cast<uint8_t>(value);
}

struct TokenizerInitializer {
    TokenizerInitializer(std::unordered_map<uint32_t, uint8_t> * reverse) { init_byte_reverse(reverse); }
};

struct ServerRuntime {
    explicit ServerRuntime(const ServerConfig & config) : config(config), tokenizer(config.semantic_model),
        session(std::make_unique<vbuf_ggml::VbufGenerationSession>(config.semantic_model, config.blocks)) {}
    ServerConfig config;
    VbufTokenizer tokenizer;
    std::unique_ptr<vbuf_ggml::VbufGenerationSession> session;
};

static bool send_all(int fd, const std::string & data) {
    size_t sent = 0;
    while (sent < data.size()) {
        const ssize_t count = ::send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
        if (count <= 0) return false;
        sent += static_cast<size_t>(count);
    }
    return true;
}

static bool read_request(int fd, HttpRequest * request) {
    std::string data;
    char buffer[8192];
    size_t content_length = 0;
    while (data.find("\r\n\r\n") == std::string::npos) {
        const ssize_t count = ::recv(fd, buffer, sizeof(buffer), 0);
        if (count <= 0) return false;
        data.append(buffer, static_cast<size_t>(count));
        if (data.size() > 128 * 1024) fail("HTTP headers too large");
    }
    const size_t header_end = data.find("\r\n\r\n");
    std::istringstream headers(data.substr(0, header_end));
    std::string line;
    if (!std::getline(headers, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream first(line);
    first >> request->method >> request->path;
    while (std::getline(headers, line)) {
        if (line.size() >= 15 && strncasecmp(line.c_str(), "Content-Length:", 15) == 0)
            content_length = std::strtoull(line.c_str() + line.find(':') + 1, nullptr, 10);
    }
    if (content_length > MAX_REQUEST_BYTES) fail("HTTP request body too large");
    request->body = data.substr(header_end + 4);
    while (request->body.size() < content_length) {
        const ssize_t count = ::recv(fd, buffer, sizeof(buffer), 0);
        if (count <= 0) return false;
        request->body.append(buffer, static_cast<size_t>(count));
    }
    request->body.resize(content_length);
    return true;
}

static bool send_response(int fd, int status, const char * reason, const std::string & content_type,
    const std::string & body, bool chunked = false) {
    std::ostringstream headers;
    headers << "HTTP/1.1 " << status << ' ' << reason << "\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
        << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        << (chunked ? "X-Accel-Buffering: no\r\nTransfer-Encoding: chunked\r\n" :
            "Content-Length: " + std::to_string(body.size()) + "\r\n")
        << "Connection: close\r\n\r\n";
    return send_all(fd, headers.str()) && (chunked ? send_all(fd, std::to_string(body.size()) + "\r\n" + body + "\r\n0\r\n\r\n") : send_all(fd, body));
}

static bool send_sse_headers(int fd) {
    return send_all(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\nTransfer-Encoding: chunked\r\n"
        "Connection: close\r\n\r\n");
}

static bool send_chunk(int fd, const std::string & data) {
    std::ostringstream size;
    size << std::hex << data.size();
    return send_all(fd, size.str() + "\r\n" + data + "\r\n");
}

class StdoutSilencer {
public:
    StdoutSilencer() : saved_(::dup(STDOUT_FILENO)) {
        const int null_fd = ::open("/dev/null", O_WRONLY);
        if (saved_ >= 0 && null_fd >= 0) {
            ::dup2(null_fd, STDOUT_FILENO);
            ::close(null_fd);
            active_ = true;
        } else if (null_fd >= 0) {
            ::close(null_fd);
        }
    }
    ~StdoutSilencer() {
        if (active_) {
            std::fflush(stdout);
            ::dup2(saved_, STDOUT_FILENO);
        }
        if (saved_ >= 0) ::close(saved_);
    }
    StdoutSilencer(const StdoutSilencer &) = delete;

private:
    int saved_ = -1;
    bool active_ = false;
};

static std::string error_body(const std::string & message, const std::string & type = "invalid_request_error") {
    return "{\"error\":{\"message\":\"" + json_escape(message) + "\",\"type\":\"" + type + "\"}}";
}

static uint64_t now_seconds() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

static std::string request_id() {
    static std::atomic<uint64_t> next{1};
    return "chatcmpl-vbuf-" + std::to_string(next.fetch_add(1));
}

static void reject_unsupported(const std::string & body) {
    for (const char * key : {"temperature", "top_p", "stop", "seed", "tools", "tool_choice", "response_format", "stream_options"})
        if (has_top_level_key(body, key)) fail(std::string("unsupported generation option: ") + key);
}

static std::string make_chat_response(const std::string & id, const ServerConfig & config,
    const std::string & text, uint32_t prompt_tokens, uint32_t completion_tokens, const char * finish) {
    std::ostringstream output;
    output << "{\"id\":\"" << id << "\",\"object\":\"chat.completion\",\"created\":" << now_seconds()
        << ",\"model\":\"" << json_escape(config.model_alias) << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
        << json_escape(text) << "\"},\"finish_reason\":\"" << finish << "\"}],\"usage\":{\"prompt_tokens\":"
        << prompt_tokens << ",\"completion_tokens\":" << completion_tokens << ",\"total_tokens\":" << prompt_tokens + completion_tokens << "}}";
    return output.str();
}

static std::string make_completion_response(const std::string & id, const ServerConfig & config,
    const std::string & text, uint32_t prompt_tokens, uint32_t completion_tokens, const char * finish) {
    std::ostringstream output;
    output << "{\"id\":\"" << id << "\",\"object\":\"text_completion\",\"created\":" << now_seconds()
        << ",\"model\":\"" << json_escape(config.model_alias) << "\",\"choices\":[{\"index\":0,\"text\":\""
        << json_escape(text) << "\",\"logprobs\":null,\"finish_reason\":\"" << finish << "\"}],\"usage\":{\"prompt_tokens\":"
        << prompt_tokens << ",\"completion_tokens\":" << completion_tokens << ",\"total_tokens\":" << prompt_tokens + completion_tokens << "}}";
    return output.str();
}

static void handle_request(int fd, ServerRuntime * runtime) {
    try {
        HttpRequest request;
        if (!read_request(fd, &request)) return;
        if (request.method == "OPTIONS") { (void)send_response(fd, 200, "OK", "application/json", "{}"); return; }
        if (request.method == "GET" && request.path == "/health") {
            (void)send_response(fd, 200, "OK", "application/json", "{\"status\":\"ok\",\"runtime\":\"ready\"}");
            return;
        }
        if (request.method == "GET" && request.path == "/v1/models") {
            const std::string body = "{\"object\":\"list\",\"data\":[{\"id\":\"" +
                json_escape(runtime->config.model_alias) + "\",\"object\":\"model\",\"owned_by\":\"vbuf\"}]}";
            (void)send_response(fd, 200, "OK", "application/json", body);
            return;
        }
        if (request.method != "POST" ||
            (request.path != "/v1/chat/completions" && request.path != "/v1/completions")) {
            (void)send_response(fd, 404, "Not Found", "application/json", error_body("unknown endpoint"));
            return;
        }
        if (request.body.empty() || request.body.front() != '{') fail("request body must be a JSON object");
        reject_unsupported(request.body);
        const std::string model = top_level_string(request.body, "model");
        if (model != runtime->config.model_alias) fail("model '" + model + "' not found");
        const uint32_t max_tokens = top_level_uint(request.body, "max_tokens",
            top_level_uint(request.body, "max_completion_tokens", runtime->config.max_new_tokens));
        if (max_tokens == 0 || max_tokens > runtime->config.max_new_tokens)
            fail("max_tokens exceeds the configured bounded generation limit");
        const bool stream = top_level_bool(request.body, "stream", false);
        const std::string id = request_id();
        std::vector<uint32_t> prompt_tokens;
        if (request.path == "/v1/chat/completions")
            prompt_tokens = runtime->tokenizer.encode_chat(parse_messages(request.body));
        else {
            const std::string prompt = top_level_string(request.body, "prompt");
            if (prompt.empty()) fail("prompt is required");
            prompt_tokens = runtime->tokenizer.encode_text(prompt, runtime->tokenizer.add_bos());
        }
        if (prompt_tokens.empty()) fail("prompt tokenization produced no tokens");
        if (prompt_tokens.size() + max_tokens > 4096) fail("prompt exceeds the bounded context limit");

        std::string output;
        bool disconnected = false;
        bool first_stream_chunk = true;
        const bool headers_sent = !stream || send_sse_headers(fd);
        if (!headers_sent) return;
        vbuf_ggml::VbufGenerationConfig generation;
        generation.semantic_model = runtime->config.semantic_model;
        generation.source_endpoint = runtime->config.source_url;
        generation.block_count = runtime->config.blocks;
        generation.residency_capacity = runtime->config.capacity;
        generation.max_new_tokens = max_tokens;
        generation.mode = runtime->config.mode;
        generation.prompt_tokens = prompt_tokens;
        generation.stop_token = runtime->tokenizer.eos();
        generation.should_cancel = [&] { return disconnected; };
        generation.on_token = [&](uint32_t token, uint32_t) {
            const std::string piece = runtime->tokenizer.decode_token(token);
            output += piece;
            if (!stream) return true;
            std::ostringstream chunk;
            if (first_stream_chunk) {
                chunk << "data: {\"id\":\"" << id << "\",\"object\":\"chat.completion.chunk\",\"created\":" << now_seconds()
                    << ",\"model\":\"" << json_escape(runtime->config.model_alias) << "\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n";
                first_stream_chunk = false;
            }
            chunk << "data: {\"id\":\"" << id << "\",\"object\":\"chat.completion.chunk\",\"created\":" << now_seconds()
                << ",\"model\":\"" << json_escape(runtime->config.model_alias) << "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\""
                << json_escape(piece) << "\"},\"finish_reason\":null}]}\n\n";
            if (!send_chunk(fd, chunk.str())) {
                disconnected = true;
                return false;
            }
            return true;
        };
        vbuf_ggml::VbufGenerationResult result;
        {
            StdoutSilencer silence;
            result = runtime->session->run(generation);
        }
        const char * finish = result.tokens.size() >= max_tokens ? "length" : "stop";
        const char * logged_finish = result.cancelled ? "cancelled" : finish;
        std::cerr << "vbuf_request id=" << id << " model=" << runtime->config.model_alias
            << " prompt_tokens=" << prompt_tokens.size() << " generated_tokens=" << result.tokens.size()
            << " source_bytes=" << result.source_bytes << " materialized_bytes=" << result.materialized_bytes
            << " peak_residency_bytes=" << result.peak_resident_bytes << " elapsed_ns=" << result.elapsed_ns
            << " finish_reason=" << logged_finish << " cancelled=" << (result.cancelled ? "yes" : "no") << "\n";
        if (!result.error.empty() && !result.cancelled) fail(result.error);
        if (stream) {
            if (!result.cancelled) {
                std::ostringstream terminal;
                terminal << "data: {\"id\":\"" << id << "\",\"object\":\"chat.completion.chunk\",\"created\":" << now_seconds()
                    << ",\"model\":\"" << json_escape(runtime->config.model_alias) << "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"" << finish << "\"}]}\n\n"
                    << "data: [DONE]\n\n";
                (void)send_chunk(fd, terminal.str());
                (void)send_all(fd, "0\r\n\r\n");
            }
            return;
        }
        const std::string body = request.path == "/v1/chat/completions"
            ? make_chat_response(id, runtime->config, output, prompt_tokens.size(), result.tokens.size(), finish)
            : make_completion_response(id, runtime->config, output, prompt_tokens.size(), result.tokens.size(), finish);
        (void)send_response(fd, 200, "OK", "application/json", body);
    } catch (const JsonError & error) {
        (void)send_response(fd, 400, "Bad Request", "application/json", error_body(error.what()));
    } catch (const std::exception & error) {
        (void)send_response(fd, 500, "Internal Server Error", "application/json", error_body(error.what(), "server_error"));
    }
}

static ServerConfig parse_args(int argc, char ** argv) {
    ServerConfig config;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto value = [&] {
            if (index + 1 >= argc) fail("missing value for " + arg);
            return std::string(argv[++index]);
        };
        if (arg == "--semantic-model") config.semantic_model = value();
        else if (arg == "--source-url") config.source_url = value();
        else if (arg == "--model-alias") config.model_alias = value();
        else if (arg == "--host") config.host = value();
        else if (arg == "--port") config.port = static_cast<uint16_t>(std::stoul(value()));
        else if (arg == "--blocks") config.blocks = static_cast<uint32_t>(std::stoul(value()));
        else if (arg == "--capacity") config.capacity = std::stoull(value());
        else if (arg == "--max-new-tokens") config.max_new_tokens = static_cast<uint32_t>(std::stoul(value()));
        else if (arg == "--runtime-mode") {
            const std::string mode = value();
            if (mode == "qualification") config.mode = vbuf_ggml::RuntimeMode::Qualification;
            else if (mode != "normal") fail("runtime mode must be normal or qualification");
        } else fail("unknown argument: " + arg);
    }
    if (config.semantic_model.empty() || config.source_url.empty())
        fail("usage: --semantic-model PATH --source-url URL [--model-alias ID --host HOST --port PORT --blocks N --capacity BYTES --max-new-tokens N]");
    if (config.max_new_tokens == 0 || config.blocks == 0) fail("generation bounds must be positive");
    return config;
}

} // namespace

int main(int argc, char ** argv) {
    std::signal(SIGPIPE, SIG_IGN);
    try {
        const ServerConfig config = parse_args(argc, argv);
        if (config.host != "127.0.0.1" && config.host != "localhost")
            std::cerr << "warning: vBuf server is listening without authentication on " << config.host << "\n";
        ServerRuntime runtime(config);
        const int server = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server < 0) fail("socket failed");
        int reuse = 1;
        ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        if (config.host == "localhost") address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        else if (::inet_pton(AF_INET, config.host.c_str(), &address.sin_addr) != 1) fail("invalid listen host");
        address.sin_port = htons(config.port);
        if (::bind(server, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0 ||
            ::listen(server, 8) < 0) fail(std::string("bind/listen failed: ") + std::strerror(errno));
        std::cerr << "vbuf-compat-server listening on " << config.host << ':' << config.port
            << " model=" << config.model_alias << " blocks=" << config.blocks
            << " runtime_mode=" << (config.mode == vbuf_ggml::RuntimeMode::NormalInference ? "normal" : "qualification") << "\n";
        while (true) {
            const int client = ::accept(server, nullptr, nullptr);
            if (client < 0) continue;
            handle_request(client, &runtime);
            ::close(client);
        }
    } catch (const std::exception & error) {
        std::cerr << "vbuf-compat-server: " << error.what() << '\n';
        return 1;
    }
}
