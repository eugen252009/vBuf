#include "vbuf_agent_protocol.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <cstdlib>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace vbuf_agent {
namespace {

[[noreturn]] void bad(const std::string & message) {
    throw ProtocolError(ProtocolError::Category::MalformedRequest, message);
}
[[noreturn]] void state_error(const std::string & message) {
    throw ProtocolError(ProtocolError::Category::InvalidConversation, message);
}
[[noreturn]] void unsupported(const std::string & message) {
    throw ProtocolError(ProtocolError::Category::UnsupportedFeature, message);
}
bool valid_utf8(const std::string & value) {
    size_t i = 0;
    while (i < value.size()) {
        const auto c = static_cast<unsigned char>(value[i]);
        if (c <= 0x7f) { ++i; continue; }
        size_t n = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        if (n == 0 || i + n > value.size()) return false;
        for (size_t j = 1; j < n; ++j) if ((static_cast<unsigned char>(value[i + j]) & 0xc0) != 0x80) return false;
        const auto c1 = static_cast<unsigned char>(value[i + 1]);
        if ((c == 0xe0 && c1 < 0xa0) || (c == 0xed && c1 >= 0xa0) ||
            (c == 0xf0 && c1 < 0x90) || (c == 0xf4 && c1 >= 0x90)) return false;
        i += n;
    }
    return true;
}

bool valid_tool_name(const std::string & name) {
    return !name.empty() && name.size() <= 64 && std::all_of(name.begin(), name.end(), [](unsigned char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
    });
}

void append_utf8(std::string * out, uint32_t cp) {
    if (cp <= 0x7f) out->push_back(static_cast<char>(cp));
    else if (cp <= 0x7ff) { out->push_back(static_cast<char>(0xc0 | cp >> 6)); out->push_back(static_cast<char>(0x80 | (cp & 63))); }
    else if (cp <= 0xffff) { out->push_back(static_cast<char>(0xe0 | cp >> 12)); out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 63))); out->push_back(static_cast<char>(0x80 | (cp & 63))); }
    else { out->push_back(static_cast<char>(0xf0 | cp >> 18)); out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 63))); out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 63))); out->push_back(static_cast<char>(0x80 | (cp & 63))); }
}

class Parser {
public:
    Parser(const std::string & text, size_t depth) : text_(text), max_depth_(depth) {}
    JsonValue parse() {
        JsonValue value = parse_value(0);
        whitespace();
        if (cursor_ != text_.size()) bad("trailing bytes after JSON value");
        return value;
    }
private:
    const std::string & text_;
    size_t max_depth_;
    size_t cursor_ = 0;
    void whitespace() { while (cursor_ < text_.size() && (text_[cursor_] == ' ' || text_[cursor_] == '\t' || text_[cursor_] == '\r' || text_[cursor_] == '\n')) ++cursor_; }
    bool consume(char ch) { whitespace(); if (cursor_ < text_.size() && text_[cursor_] == ch) { ++cursor_; return true; } return false; }
    uint32_t hex4() {
        if (cursor_ + 4 > text_.size()) bad("truncated JSON unicode escape");
        uint32_t result = 0;
        for (int i = 0; i < 4; ++i) {
            char c = text_[cursor_++];
            int n = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
            if (n < 0) bad("invalid JSON unicode escape");
            result = (result << 4) | static_cast<uint32_t>(n);
        }
        return result;
    }
    std::string string() {
        if (cursor_ >= text_.size() || text_[cursor_++] != '"') bad("expected JSON string");
        std::string out;
        while (cursor_ < text_.size()) {
            unsigned char c = static_cast<unsigned char>(text_[cursor_++]);
            if (c == '"') { if (!valid_utf8(out)) bad("invalid UTF-8 in JSON string"); return out; }
            if (c < 0x20) bad("unescaped control byte in JSON string");
            if (c != '\\') { out.push_back(static_cast<char>(c)); continue; }
            if (cursor_ >= text_.size()) bad("truncated JSON escape");
            switch (text_[cursor_++]) {
            case '"': out.push_back('"'); break; case '\\': out.push_back('\\'); break; case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break; case 'f': out.push_back('\f'); break; case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break; case 't': out.push_back('\t'); break;
            case 'u': {
                uint32_t cp = hex4();
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    if (cursor_ + 2 > text_.size() || text_[cursor_] != '\\' || text_[cursor_ + 1] != 'u') bad("unpaired high surrogate");
                    cursor_ += 2; const uint32_t low = hex4();
                    if (low < 0xdc00 || low > 0xdfff) bad("invalid low surrogate");
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                } else if (cp >= 0xdc00 && cp <= 0xdfff) bad("unpaired low surrogate");
                append_utf8(&out, cp); break;
            }
            default: bad("invalid JSON escape");
            }
        }
        bad("unterminated JSON string");
    }
    JsonValue parse_value(size_t depth) {
        whitespace();
        if (depth > max_depth_) bad("JSON nesting exceeds configured limit");
        if (cursor_ >= text_.size()) bad("expected JSON value");
        const char c = text_[cursor_];
        if (c == '"') { JsonValue v; v.kind = JsonValue::Kind::String; v.scalar = string(); return v; }
        if (c == '{') {
            ++cursor_; JsonValue v; v.kind = JsonValue::Kind::Object; std::set<std::string> keys;
            whitespace(); if (consume('}')) return v;
            for (;;) {
                whitespace(); if (cursor_ >= text_.size() || text_[cursor_] != '"') bad("object key must be a string");
                std::string key = string(); if (!keys.insert(key).second) bad("duplicate JSON object key: " + key);
                if (!consume(':')) bad("expected colon after object key");
                v.object.emplace_back(std::move(key), parse_value(depth + 1));
                if (consume('}')) return v;
                if (!consume(',')) bad("expected comma in object");
            }
        }
        if (c == '[') {
            ++cursor_; JsonValue v; v.kind = JsonValue::Kind::Array;
            whitespace(); if (consume(']')) return v;
            for (;;) { v.array.push_back(parse_value(depth + 1)); if (consume(']')) return v; if (!consume(',')) bad("expected comma in array"); }
        }
        if (text_.compare(cursor_, 4, "true") == 0 || text_.compare(cursor_, 5, "false") == 0) {
            JsonValue v; v.kind = JsonValue::Kind::Boolean; v.boolean = text_[cursor_] == 't'; cursor_ += v.boolean ? 4 : 5; return v;
        }
        if (text_.compare(cursor_, 4, "null") == 0) { cursor_ += 4; return {}; }
        const size_t begin = cursor_;
        if (text_[cursor_] == '-') ++cursor_;
        if (cursor_ >= text_.size()) bad("invalid JSON number");
        if (text_[cursor_] == '0') ++cursor_;
        else { if (!std::isdigit(static_cast<unsigned char>(text_[cursor_]))) bad("invalid JSON value"); while (cursor_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[cursor_]))) ++cursor_; }
        if (cursor_ < text_.size() && text_[cursor_] == '.') { ++cursor_; const size_t d = cursor_; while (cursor_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[cursor_]))) ++cursor_; if (d == cursor_) bad("invalid JSON fraction"); }
        if (cursor_ < text_.size() && (text_[cursor_] == 'e' || text_[cursor_] == 'E')) { ++cursor_; if (cursor_ < text_.size() && (text_[cursor_] == '+' || text_[cursor_] == '-')) ++cursor_; const size_t d = cursor_; while (cursor_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[cursor_]))) ++cursor_; if (d == cursor_) bad("invalid JSON exponent"); }
        JsonValue v; v.kind = JsonValue::Kind::Number; v.scalar = text_.substr(begin, cursor_ - begin); return v;
    }
};

std::string quote(const std::string & s) {
    std::ostringstream out; out << '"';
    for (unsigned char c : s) {
        switch (c) { case '"': out << "\\\""; break; case '\\': out << "\\\\"; break; case '\b': out << "\\b"; break; case '\f': out << "\\f"; break; case '\n': out << "\\n"; break; case '\r': out << "\\r"; break; case '\t': out << "\\t"; break; default: if (c < 0x20) { char b[7]; std::snprintf(b, sizeof(b), "\\u%04x", c); out << b; } else out << static_cast<char>(c); }
    }
    out << '"'; return out.str();
}
std::string serialize(const JsonValue & v) {
    switch (v.kind) {
    case JsonValue::Kind::Null: return "null";
    case JsonValue::Kind::Boolean: return v.boolean ? "true" : "false";
    case JsonValue::Kind::Number: return v.scalar;
    case JsonValue::Kind::String: return quote(v.scalar);
    case JsonValue::Kind::Array: { std::string out = "["; for (size_t i=0;i<v.array.size();++i) { if(i) out += ','; out += serialize(v.array[i]); } return out + ']'; }
    case JsonValue::Kind::Object: { std::string out = "{"; for (size_t i=0;i<v.object.size();++i) { if(i) out += ','; out += quote(v.object[i].first) + ':' + serialize(v.object[i].second); } return out + '}'; }
    }
    return "null";
}
const JsonValue * required(const JsonValue & object, const std::string & key) {
    const auto * value = object.get(key); if (!value) bad("missing required field: " + key); return value;
}
std::string string_field(const JsonValue & object, const std::string & key, bool required_field = true) {
    const auto * v = object.get(key); if (!v) { if (required_field) bad("missing required string field: " + key); return {}; }
    if (v->kind != JsonValue::Kind::String) bad("field must be a string: " + key);
    return v->scalar;
}
void require_object(const JsonValue & v, const std::string & name) { if (!v.is_object()) bad(name + " must be an object"); }
void require_array(const JsonValue & v, const std::string & name) { if (!v.is_array()) bad(name + " must be an array"); }
JsonValue str(const std::string & s) { JsonValue v; v.kind = JsonValue::Kind::String; v.scalar = s; return v; }
JsonValue number(size_t n) { JsonValue v; v.kind = JsonValue::Kind::Number; v.scalar = std::to_string(n); return v; }
JsonValue object(std::initializer_list<std::pair<std::string, JsonValue>> fields) { JsonValue v; v.kind = JsonValue::Kind::Object; v.object.assign(fields); return v; }

ToolCall parse_call(const JsonValue & value, const Limits & limits) {
    require_object(value, "tool call");
    for (const auto & field : value.object)
        if (field.first != "id" && field.first != "type" && field.first != "function")
            unsupported("unsupported tool-call field: " + field.first);
    ToolCall call; call.id = string_field(value, "id");
    if (call.id.empty() || call.id.size() > limits.max_call_id_bytes) bad("tool call id is empty or exceeds limit");
    if (string_field(value, "type") != "function") bad("only function tool calls are supported");
    const auto * fn = required(value, "function"); require_object(*fn, "tool call function");
    for (const auto & field : fn->object)
        if (field.first != "name" && field.first != "arguments")
            unsupported("unsupported function-call field: " + field.first);
    call.name = string_field(*fn, "name"); if (!valid_tool_name(call.name) || call.name.size() > limits.max_tool_name_bytes) bad("tool function name is invalid or exceeds limit");
    call.arguments = string_field(*fn, "arguments");
    if (call.arguments.size() > limits.max_arguments_bytes) bad("tool arguments exceed configured limit");
    const JsonValue parsed = parse_json(call.arguments, limits.max_json_depth);
    if (parsed.kind != JsonValue::Kind::Object) bad("tool arguments must encode a JSON object");
    return call;
}

ToolDefinition parse_definition(const JsonValue & value, const Limits & limits) {
    require_object(value, "tool definition");
    for (const auto & field : value.object)
        if (field.first != "type" && field.first != "function")
            unsupported("unsupported tool-definition field: " + field.first);
    if (string_field(value, "type") != "function") bad("only function tool definitions are supported");
    const auto * fn = required(value, "function"); require_object(*fn, "tool function definition");
    for (const auto & field : fn->object)
        if (field.first != "name" && field.first != "description" && field.first != "parameters" && field.first != "strict")
            unsupported("unsupported function-definition field: " + field.first);
    ToolDefinition result; result.name = string_field(*fn, "name");
    if (!valid_tool_name(result.name) || result.name.size() > limits.max_tool_name_bytes) bad("tool name is invalid or exceeds limit");
    result.description = string_field(*fn, "description", false);
    if (result.description.size() > limits.max_tool_description_bytes) bad("tool description exceeds configured limit");
    result.parameters = *required(*fn, "parameters"); require_object(result.parameters, "function parameters");
    if (const auto * strict = fn->get("strict")) {
        if (strict->kind != JsonValue::Kind::Boolean) bad("function strict must be boolean");
        result.strict = strict->boolean;
    }
    if (serialize_json(result.parameters).size() > limits.max_schema_bytes) bad("tool schema exceeds configured limit");
    const auto * type = result.parameters.get("type");
    if (type && (!type->is_string() || type->scalar != "object")) bad("function parameters.type must be 'object'");
    const auto * props = result.parameters.get("properties"); if (props && !props->is_object()) bad("function parameters.properties must be an object");
    const auto * required_props = result.parameters.get("required");
    if (required_props) { require_array(*required_props, "function parameters.required"); for (const auto & name : required_props->array) if (!name.is_string()) bad("required property names must be strings"); }
    return result;
}

ChatMessage parse_message(const JsonValue & value, const Limits & limits) {
    require_object(value, "message"); ChatMessage m;
    const std::string role = string_field(value, "role");
    if (role == "system") m.role = ChatMessage::Role::System;
    else if (role == "user") m.role = ChatMessage::Role::User;
    else if (role == "assistant") m.role = ChatMessage::Role::Assistant;
    else if (role == "tool") m.role = ChatMessage::Role::Tool;
    else bad("unsupported message role: " + role);
    if (m.role == ChatMessage::Role::Tool) {
        for (const auto & field : value.object)
            if (field.first != "role" && field.first != "tool_call_id" && field.first != "content")
                unsupported("unsupported tool message field: " + field.first);
        m.tool_call_id = string_field(value, "tool_call_id");
        const auto * content = required(value, "content"); if (!content->is_string()) bad("tool message content must be a string"); m.content = content->scalar;
        if (m.content.size() > limits.max_tool_result_bytes) bad("tool result exceeds configured limit");
        return m;
    }
    for (const auto & field : value.object) {
        if (field.first != "role" && field.first != "content" && !(m.role == ChatMessage::Role::Assistant && field.first == "tool_calls"))
            unsupported("unsupported message field: " + field.first);
    }
    const auto * content = value.get("content");
    if (!content) { if (m.role == ChatMessage::Role::Assistant && value.get("tool_calls")) m.has_content = false; else bad("message content is required"); }
    else if (content->kind == JsonValue::Kind::Null && m.role == ChatMessage::Role::Assistant) m.has_content = false;
    else if (content->is_string()) { m.has_content = true; m.content = content->scalar; }
    else bad("only string content (or null assistant content with tool calls) is supported");
    if (m.role != ChatMessage::Role::Assistant && value.get("tool_calls")) bad("tool_calls are valid only on assistant messages");
    if (const auto * calls = value.get("tool_calls")) {
        require_array(*calls, "assistant tool_calls"); if (calls->array.empty()) bad("assistant tool_calls must not be empty when present");
        if (calls->array.size() > limits.max_calls_per_turn) bad("too many tool calls in one assistant turn");
        for (const auto & call : calls->array) m.tool_calls.push_back(parse_call(call, limits));
        if (!m.has_content && !value.get("tool_calls")) bad("null assistant content requires tool_calls");
    }
    if (m.role == ChatMessage::Role::Assistant && !m.has_content && m.tool_calls.empty()) bad("assistant message must contain content or tool_calls");
    return m;
}

JsonValue call_json(const ToolCall & call) {
    return object({{"id", str(call.id)}, {"type", str("function")}, {"function", object({{"name", str(call.name)}, {"arguments", str(call.arguments)}})}});
}
std::string event_json(const std::string & id, const std::string & model, const JsonValue & delta, const JsonValue & finish) {
    JsonValue choice = object({{"index", number(0)}, {"delta", delta}, {"finish_reason", finish}});
    JsonValue choices; choices.kind = JsonValue::Kind::Array; choices.array.push_back(std::move(choice));
    JsonValue event = object({{"id", str(id)}, {"object", str("chat.completion.chunk")}, {"model", str(model)}, {"choices", choices}});
    return serialize_json(event);
}

} // namespace

const JsonValue * JsonValue::get(const std::string & key) const {
    for (const auto & entry : object) if (entry.first == key) return &entry.second;
    return nullptr;
}
ProtocolError::ProtocolError(Category c, const std::string & message) : std::runtime_error(message), category(c) {}
JsonValue parse_json(const std::string & input, size_t max_depth) { return Parser(input, max_depth).parse(); }
std::string serialize_json(const JsonValue & value) { return serialize(value); }
const char * role_name(ChatMessage::Role role) { switch(role) { case ChatMessage::Role::System:return "system"; case ChatMessage::Role::User:return "user"; case ChatMessage::Role::Assistant:return "assistant"; case ChatMessage::Role::Tool:return "tool"; } return "user"; }
const char * tool_choice_name(ToolChoice::Kind kind) { switch(kind) { case ToolChoice::Kind::Auto:return "auto"; case ToolChoice::Kind::None:return "none"; case ToolChoice::Kind::ForcedFunction:return "function"; } return "auto"; }

ChatRequest parse_chat_request(const std::string & body, const Limits & limits) {
    if (body.size() > limits.request_bytes) bad("request body exceeds configured limit");
    const JsonValue root = parse_json(body, limits.max_json_depth); require_object(root, "request");
    static const std::set<std::string> supported_fields = {"model", "messages", "tools", "tool_choice", "max_tokens", "max_completion_tokens", "stream"};
    for (const auto & field : root.object)
        if (!supported_fields.count(field.first)) unsupported("unsupported request field: " + field.first);
    ChatRequest request; request.model = string_field(root, "model");
    if (request.model.empty()) bad("model must not be empty");
    bool token_limit_seen = false;
    for (const char * key : {"max_tokens", "max_completion_tokens"}) {
        if (const auto * value = root.get(key)) {
            if (value->kind != JsonValue::Kind::Number || value->scalar.empty() || value->scalar.find_first_not_of("0123456789") != std::string::npos)
                bad(std::string(key) + " must be a non-negative integer");
            if (token_limit_seen) bad("specify only one of max_tokens and max_completion_tokens");
            token_limit_seen = true;
            try { request.max_tokens = static_cast<size_t>(std::stoull(value->scalar)); }
            catch (...) { bad(std::string(key) + " exceeds supported integer range"); }
        }
    }
    if (const auto * stream = root.get("stream")) if (stream->kind != JsonValue::Kind::Boolean) bad("stream must be boolean");
    const auto * messages = required(root, "messages"); require_array(*messages, "messages");
    if (messages->array.empty()) bad("messages must not be empty");
    for (const auto & message : messages->array) request.messages.push_back(parse_message(message, limits));
    validate_conversation(request.messages, limits);
    if (const auto * tools = root.get("tools")) {
        request.tools_were_supplied = true; require_array(*tools, "tools");
        if (tools->array.size() > limits.max_tools) bad("too many tool definitions");
        std::set<std::string> names;
        for (const auto & item : tools->array) {
            auto tool = parse_definition(item, limits);
            if (!names.insert(tool.name).second) bad("duplicate tool name: " + tool.name);
            request.tools.push_back(std::move(tool));
        }
    }
    if (const auto * choice = root.get("tool_choice")) {
        request.tool_choice_was_supplied = true;
        if (choice->is_string()) {
            if (choice->scalar == "auto") request.tool_choice.kind = ToolChoice::Kind::Auto;
            else if (choice->scalar == "none") request.tool_choice.kind = ToolChoice::Kind::None;
            else bad("tool_choice string must be 'auto' or 'none'");
        } else {
            require_object(*choice, "tool_choice");
            for (const auto & field : choice->object)
                if (field.first != "type" && field.first != "function")
                    unsupported("unsupported tool_choice field: " + field.first);
            if (string_field(*choice, "type") != "function") bad("forced tool_choice.type must be 'function'");
            const auto * fn = required(*choice, "function"); require_object(*fn, "tool_choice.function");
            for (const auto & field : fn->object)
                if (field.first != "name") unsupported("unsupported forced tool_choice function field: " + field.first);
            request.tool_choice.kind = ToolChoice::Kind::ForcedFunction;
            request.tool_choice.function_name = string_field(*fn, "name");
            if (!valid_tool_name(request.tool_choice.function_name)) bad("forced tool_choice function name is invalid");
            if (!request.tools_were_supplied || std::none_of(request.tools.begin(), request.tools.end(), [&](const ToolDefinition & t){ return t.name == request.tool_choice.function_name; })) bad("tool_choice references unavailable function: " + request.tool_choice.function_name);
        }
    }
    return request;
}

void validate_conversation(const std::vector<ChatMessage> & messages, const Limits & limits) {
    std::unordered_set<std::string> seen;
    std::set<std::string> outstanding;
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & m = messages[i];
        if (!outstanding.empty() && m.role != ChatMessage::Role::Tool)
            state_error("assistant tool calls must receive all tool results before another non-tool message");
        if (m.role == ChatMessage::Role::Tool) {
            if (outstanding.empty()) state_error("tool message has no preceding outstanding tool call");
            if (m.tool_call_id.empty()) state_error("tool message is missing tool_call_id");
            if (m.tool_call_id.size() > limits.max_call_id_bytes) state_error("tool_call_id exceeds configured limit");
            if (!outstanding.erase(m.tool_call_id)) state_error("unknown or duplicate tool_call_id result: " + m.tool_call_id);
            if (m.content.size() > limits.max_tool_result_bytes) state_error("tool result exceeds configured limit");
            continue;
        }
        if (m.role == ChatMessage::Role::Assistant && !m.tool_calls.empty()) {
            for (const auto & call : m.tool_calls) {
                if (call.id.empty()) state_error("assistant tool call is missing id");
                if (!seen.insert(call.id).second) state_error("duplicate tool_call_id: " + call.id);
                if (!valid_tool_name(call.name)) state_error("assistant tool call has invalid or missing function name");
                if (call.arguments.size() > limits.max_arguments_bytes) state_error("tool arguments exceed configured limit");
                const JsonValue args = parse_json(call.arguments, limits.max_json_depth);
                if (!args.is_object()) state_error("tool arguments must be a JSON object");
                outstanding.insert(call.id);
            }
        }
    }
    if (!outstanding.empty()) state_error("conversation ends with unresolved tool calls");
}

void validate_assistant_output(const AssistantOutput & output, const Limits & limits) {
    if (output.tool_calls.size() > limits.max_calls_per_turn) throw ProtocolError(ProtocolError::Category::Internal, "adapter emitted too many tool calls");
    if (output.content.size() > limits.max_assistant_content_bytes) throw ProtocolError(ProtocolError::Category::Internal, "adapter emitted oversized assistant content");
    if (output.has_content && !valid_utf8(output.content)) throw ProtocolError(ProtocolError::Category::Internal, "adapter emitted invalid UTF-8 assistant content");
    std::set<std::string> ids;
    for (const auto & call : output.tool_calls) {
        if (call.id.empty() || call.id.size() > limits.max_call_id_bytes || !ids.insert(call.id).second) throw ProtocolError(ProtocolError::Category::Internal, "adapter emitted missing, oversized, or duplicate tool-call ID");
        if (!valid_tool_name(call.name) || call.name.size() > limits.max_tool_name_bytes) throw ProtocolError(ProtocolError::Category::Internal, "adapter emitted invalid function name");
        if (call.arguments.size() > limits.max_arguments_bytes) throw ProtocolError(ProtocolError::Category::Internal, "adapter emitted oversized tool arguments");
        try {
            if (!parse_json(call.arguments, limits.max_json_depth).is_object())
                throw ProtocolError(ProtocolError::Category::Internal, "adapter emitted malformed tool arguments");
        } catch (const ProtocolError & error) {
            if (error.category == ProtocolError::Category::Internal) throw;
            throw ProtocolError(ProtocolError::Category::Internal, "adapter emitted malformed tool arguments");
        }
    }
    if (!output.has_content && output.tool_calls.empty()) throw ProtocolError(ProtocolError::Category::Internal, "adapter emitted neither content nor tool calls");
    if (output.tool_calls.empty() && output.finish_reason == AssistantOutput::FinishReason::ToolCalls)
        throw ProtocolError(ProtocolError::Category::Internal, "adapter marked text output as tool_calls");
    if (!output.tool_calls.empty() && output.finish_reason != AssistantOutput::FinishReason::ToolCalls)
        throw ProtocolError(ProtocolError::Category::Internal, "adapter tool calls require tool_calls finish reason");
}

void validate_assistant_output_for_request(const ChatRequest & request,
    const AssistantOutput & output, const Limits & limits) {
    validate_assistant_output(output, limits);
    if (output.tool_calls.empty()) {
        if (request.tool_choice.kind == ToolChoice::Kind::ForcedFunction)
            throw ProtocolError(ProtocolError::Category::Internal, "adapter did not honor forced tool_choice");
        return;
    }
    if (!request.tools_were_supplied || request.tools.empty())
        throw ProtocolError(ProtocolError::Category::Internal, "adapter emitted tool calls without declared tools");
    for (const auto & call : output.tool_calls) {
        const bool available = std::any_of(request.tools.begin(), request.tools.end(),
            [&](const ToolDefinition & tool) { return tool.name == call.name; });
        if (!available) throw ProtocolError(ProtocolError::Category::Internal, "adapter emitted undeclared tool: " + call.name);
        if (request.tool_choice.kind == ToolChoice::Kind::None)
            throw ProtocolError(ProtocolError::Category::Internal, "adapter emitted a tool call despite tool_choice=none");
        if (request.tool_choice.kind == ToolChoice::Kind::ForcedFunction && call.name != request.tool_choice.function_name)
            throw ProtocolError(ProtocolError::Category::Internal, "adapter emitted a function different from forced tool_choice");
    }
    if (request.tool_choice.kind == ToolChoice::Kind::ForcedFunction && output.tool_calls.size() != 1)
        throw ProtocolError(ProtocolError::Category::Internal, "forced tool_choice must produce exactly one matching call");
}

std::string serialize_chat_response(const std::string & id, const std::string & model, const AssistantOutput & output, size_t prompt_tokens, size_t completion_tokens, const Limits & limits) {
    validate_assistant_output(output, limits);
    JsonValue message = object({{"role", str("assistant")}, {"content", output.has_content ? str(output.content) : JsonValue{}}});
    if (!output.tool_calls.empty()) { JsonValue calls; calls.kind = JsonValue::Kind::Array; for (const auto & c : output.tool_calls) calls.array.push_back(call_json(c)); message.object.emplace_back("tool_calls", std::move(calls)); }
    const char * finish = output.finish_reason == AssistantOutput::FinishReason::Length ? "length" :
        output.finish_reason == AssistantOutput::FinishReason::ToolCalls ? "tool_calls" : "stop";
    JsonValue choice = object({{"index", number(0)}, {"message", std::move(message)}, {"finish_reason", str(finish)}});
    JsonValue choices; choices.kind = JsonValue::Kind::Array; choices.array.push_back(std::move(choice));
    JsonValue usage = object({{"prompt_tokens", number(prompt_tokens)}, {"completion_tokens", number(completion_tokens)}, {"total_tokens", number(prompt_tokens + completion_tokens)}});
    return serialize_json(object({{"id", str(id)}, {"object", str("chat.completion")}, {"model", str(model)}, {"choices", std::move(choices)}, {"usage", std::move(usage)}}));
}

std::vector<std::string> stream_chat_response(const std::string & id, const std::string & model, const AssistantOutput & output, const Limits & limits) {
    validate_assistant_output(output, limits);
    std::vector<std::string> events;
    events.push_back(event_json(id, model, object({{"role", str("assistant")}}), JsonValue{}));
    if (output.has_content) {
        // Split only at UTF-8 leading-byte boundaries; every SSE delta remains valid text.
        if (output.content.empty()) events.push_back(event_json(id, model, object({{"content", str("")}}), JsonValue{}));
        for (size_t i = 0; i < output.content.size();) {
            size_t end = std::min(i + 1024, output.content.size());
            while (end < output.content.size() &&
                (static_cast<unsigned char>(output.content[end]) & 0xc0) == 0x80) --end;
            if (end == i) { end = std::min(i + 4, output.content.size()); }
            events.push_back(event_json(id, model, object({{"content", str(output.content.substr(i, end - i))}}), JsonValue{}));
            i = end;
        }
    }
    if (output.tool_calls.empty()) {
        const char * finish = output.finish_reason == AssistantOutput::FinishReason::Length ? "length" : "stop";
        events.push_back(event_json(id, model, object({}), str(finish)));
    } else {
        for (size_t i=0;i<output.tool_calls.size();++i) {
            const auto & c=output.tool_calls[i];
            // OpenAI deltas may fragment all string fields. The index is stable for reconstruction.
            events.push_back(event_json(id,model,object({{"tool_calls", [&](){ JsonValue a; a.kind=JsonValue::Kind::Array; a.array.push_back(object({{"index",number(i)},{"id",str(c.id.substr(0, c.id.size()/2))},{"type",str("function")}})); return a;}()}}),JsonValue{}));
            if(c.id.size()>c.id.size()/2) events.push_back(event_json(id,model,object({{"tool_calls", [&](){ JsonValue a; a.kind=JsonValue::Kind::Array; a.array.push_back(object({{"index",number(i)},{"id",str(c.id.substr(c.id.size()/2))}})); return a;}()}}),JsonValue{}));
            for(size_t n=0;n<c.name.size();n+=2) events.push_back(event_json(id,model,object({{"tool_calls", [&](){ JsonValue a; a.kind=JsonValue::Kind::Array; a.array.push_back(object({{"index",number(i)},{"function",object({{"name",str(c.name.substr(n,2))}})}})); return a;}()}}),JsonValue{}));
            for(size_t n=0;n<c.arguments.size();n+=256) events.push_back(event_json(id,model,object({{"tool_calls", [&](){ JsonValue a; a.kind=JsonValue::Kind::Array; a.array.push_back(object({{"index",number(i)},{"function",object({{"arguments",str(c.arguments.substr(n,256))}})}})); return a;}()}}),JsonValue{}));
        }
        events.push_back(event_json(id, model, object({}), str("tool_calls")));
    }
    if (events.size() + 1 > limits.max_stream_events) throw ProtocolError(ProtocolError::Category::Internal, "serialized stream exceeds event limit");
    events.push_back("[DONE]"); return events;
}

AssistantOutput reconstruct_stream(const std::vector<std::string> & records, const Limits & limits) {
    AssistantOutput result;
    result.has_content = false;
    bool terminal = false, done = false;
    std::map<size_t, ToolCall> calls;
    std::set<size_t> function_types;
    if (records.size() > limits.max_stream_events) bad("stream exceeds configured event limit");
    for (const std::string & record : records) {
        if (record == "[DONE]") { if (done) bad("duplicate [DONE]"); done = true; continue; }
        if (done) bad("SSE data received after [DONE]");
        if (terminal) bad("SSE data received after finish reason");
        const JsonValue event = parse_json(record, limits.max_json_depth);
        const auto * choices = event.get("choices");
        if (!choices || !choices->is_array() || choices->array.size() != 1) bad("invalid chat completion stream choice");
        const JsonValue & choice = choices->array[0];
        const auto * finish = choice.get("finish_reason");
        if (finish && finish->kind != JsonValue::Kind::Null && finish->kind != JsonValue::Kind::String)
            bad("SSE finish_reason must be string or null");
        const auto * delta = choice.get("delta");
        if (!delta || !delta->is_object()) bad("SSE choice delta is missing");
        if (const auto * text = delta->get("content")) {
            if (!text->is_string()) bad("invalid text SSE delta");
            result.has_content = true;
            result.content += text->scalar;
            if (result.content.size() > limits.max_assistant_content_bytes) bad("streamed assistant content exceeds limit");
        }
        if (const auto * values = delta->get("tool_calls")) {
            if (!values->is_array()) bad("tool_calls SSE delta must be an array");
            for (const auto & part : values->array) {
                if (!part.is_object()) bad("tool-call SSE fragment must be an object");
                const auto * ix = part.get("index");
                if (!ix || ix->kind != JsonValue::Kind::Number || ix->scalar.empty() ||
                    ix->scalar.find_first_not_of("0123456789") != std::string::npos) bad("tool-call SSE index must be a non-negative integer");
                size_t index = 0;
                try { index = static_cast<size_t>(std::stoull(ix->scalar)); }
                catch (...) { bad("tool-call SSE index is out of range"); }
                if (index >= limits.max_calls_per_turn) bad("tool-call SSE index exceeds call limit");
                auto & call = calls[index];
                if (const auto * id = part.get("id")) {
                    if (!id->is_string()) bad("tool-call SSE id must be string");
                    call.id += id->scalar;
                    if (call.id.size() > limits.max_call_id_bytes) bad("streamed tool-call id exceeds limit");
                }
                if (const auto * type = part.get("type")) {
                    if (!type->is_string() || type->scalar != "function") bad("unsupported streamed tool-call type");
                    function_types.insert(index);
                }
                if (const auto * fn = part.get("function")) {
                    if (!fn->is_object()) bad("streamed function must be object");
                    if (const auto * name = fn->get("name")) {
                        if (!name->is_string()) bad("streamed function name must be string");
                        call.name += name->scalar;
                        if (call.name.size() > limits.max_tool_name_bytes) bad("streamed function name exceeds limit");
                    }
                    if (const auto * args = fn->get("arguments")) {
                        if (!args->is_string()) bad("streamed function arguments must be string");
                        call.arguments += args->scalar;
                        if (call.arguments.size() > limits.max_arguments_bytes) bad("streamed function arguments exceed limit");
                    }
                }
            }
        }
        if (finish && finish->kind == JsonValue::Kind::String) {
            if (terminal) bad("duplicate SSE finish reason");
            terminal = true;
            if (finish->scalar == "tool_calls") {
                if (calls.empty()) bad("tool_calls finish reason without calls");
                result.finish_reason = AssistantOutput::FinishReason::ToolCalls;
            } else if (finish->scalar == "stop") {
                if (!calls.empty()) bad("stop finish reason with tool calls");
                result.finish_reason = AssistantOutput::FinishReason::Stop;
            } else if (finish->scalar == "length") {
                if (!calls.empty()) bad("length finish reason with tool calls");
                result.finish_reason = AssistantOutput::FinishReason::Length;
            } else bad("unsupported SSE finish reason");
        }
    }
    if (!terminal || !done) bad("SSE stream lacks finish reason or [DONE]");
    if (calls.size() > limits.max_calls_per_turn) bad("stream contains too many tool calls");
    size_t expected_index = 0;
    for (auto & entry : calls) {
        if (entry.first != expected_index++ || !function_types.count(entry.first)) bad("stream has missing index or function type");
        result.tool_calls.push_back(std::move(entry.second));
    }
    validate_assistant_output(result, limits);
    return result;
}

DeterministicFakeAdapter::DeterministicFakeAdapter(std::vector<AssistantOutput> outputs):outputs_(std::move(outputs)){}
std::string DeterministicFakeAdapter::render(const ChatRequest & request) const {
    // Deterministic diagnostic representation for protocol tests only; never used as production model syntax.
    std::ostringstream out; out << "messages=" << request.messages.size() << ";tools=" << request.tools.size() << ";choice=" << tool_choice_name(request.tool_choice.kind);
    for (const auto & message : request.messages) {
        out << "\\n" << role_name(message.role) << ':';
        if (message.has_content) out << message.content;
        if (message.role == ChatMessage::Role::Tool) out << "[call=" << message.tool_call_id << ']';
        for (const auto & call : message.tool_calls) out << "[call=" << call.id << ",name=" << call.name << ",args=" << call.arguments << ']';
    }
    return out.str();
}
AssistantOutput DeterministicFakeAdapter::parse_generation(const std::string &, const ChatRequest & request) const {
    if(next_>=outputs_.size()) throw ProtocolError(ProtocolError::Category::Internal,"fake adapter output queue exhausted");
    auto result=outputs_[next_++]; validate_assistant_output_for_request(request, result); return result;
}
size_t DeterministicFakeAdapter::remaining() const {return outputs_.size()-next_;}

} // namespace vbuf_agent
