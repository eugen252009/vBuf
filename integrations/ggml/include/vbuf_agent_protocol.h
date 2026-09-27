#pragma once

// Model-independent OpenAI Chat Completions agent/tool protocol types.
// This layer validates and transports tool calls; it never executes them.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vbuf_agent {

struct JsonValue {
    enum class Kind { Null, Boolean, Number, String, Array, Object };
    Kind kind = Kind::Null;
    bool boolean = false;
    std::string scalar; // String contents or validated raw JSON number.
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object; // insertion order, unique keys

    const JsonValue * get(const std::string & key) const;
    bool is_object() const { return kind == Kind::Object; }
    bool is_array() const { return kind == Kind::Array; }
    bool is_string() const { return kind == Kind::String; }
};

struct Limits {
    size_t request_bytes = 4 * 1024 * 1024;
    size_t max_json_depth = 64;
    size_t max_tools = 64;
    size_t max_tool_name_bytes = 64;
    size_t max_tool_description_bytes = 64 * 1024;
    size_t max_schema_bytes = 256 * 1024;
    size_t max_calls_per_turn = 32;
    size_t max_call_id_bytes = 128;
    size_t max_arguments_bytes = 64 * 1024;
    size_t max_tool_result_bytes = 1024 * 1024;
    size_t max_assistant_content_bytes = 1024 * 1024;
    size_t max_stream_events = 20000;
};

struct ToolDefinition {
    std::string name;
    std::string description;
    JsonValue parameters;
    bool strict = false;
};

struct ToolChoice {
    enum class Kind { Auto, None, ForcedFunction };
    Kind kind = Kind::Auto;
    std::string function_name;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments; // OpenAI's JSON-encoded function.arguments string
};

struct ToolResult {
    std::string tool_call_id;
    std::string content;
};

struct ChatMessage {
    enum class Role { System, User, Assistant, Tool };
    Role role = Role::User;
    bool has_content = true;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id; // role=tool
};

struct ChatRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    std::vector<ToolDefinition> tools;
    bool tools_were_supplied = false;
    bool tool_choice_was_supplied = false;
    ToolChoice tool_choice;
    size_t max_tokens = 0;
};

struct AssistantOutput {
    enum class FinishReason { Stop, Length, ToolCalls };
    bool has_content = true;
    std::string content;
    std::vector<ToolCall> tool_calls;
    FinishReason finish_reason = FinishReason::Stop;
};

struct ProtocolError : std::runtime_error {
    enum class Category { MalformedRequest, UnsupportedFeature, InvalidConversation, Internal };
    ProtocolError(Category category, const std::string & message);
    Category category;
};

JsonValue parse_json(const std::string & input, size_t max_depth = 64);
std::string serialize_json(const JsonValue & value);
ChatRequest parse_chat_request(const std::string & body, const Limits & limits = {});
void validate_conversation(const std::vector<ChatMessage> & messages, const Limits & limits = {});
void validate_assistant_output(const AssistantOutput & output, const Limits & limits = {});
void validate_assistant_output_for_request(const ChatRequest & request,
    const AssistantOutput & output, const Limits & limits = {});
std::string serialize_chat_response(const std::string & id, const std::string & model,
    const AssistantOutput & output, size_t prompt_tokens, size_t completion_tokens,
    const Limits & limits = {});

// Each emitted item is one SSE data record. Includes terminal finish chunk and [DONE].
std::vector<std::string> stream_chat_response(const std::string & id, const std::string & model,
    const AssistantOutput & output, const Limits & limits = {});
AssistantOutput reconstruct_stream(const std::vector<std::string> & data_records,
    const Limits & limits = {});

// Model adapter boundary. Production model-specific adapters are intentionally absent.
class ModelConversationAdapter {
public:
    virtual ~ModelConversationAdapter() = default;
    virtual std::string render(const ChatRequest & request) const = 0;
    virtual AssistantOutput parse_generation(const std::string & native_output,
        const ChatRequest & request) const = 0;
};

// Test-only deterministic adapter: consumes queued semantic outputs, not model syntax.
class DeterministicFakeAdapter final : public ModelConversationAdapter {
public:
    explicit DeterministicFakeAdapter(std::vector<AssistantOutput> outputs);
    std::string render(const ChatRequest & request) const override;
    AssistantOutput parse_generation(const std::string & native_output,
        const ChatRequest & request) const override;
    size_t remaining() const;
private:
    std::vector<AssistantOutput> outputs_;
    mutable size_t next_ = 0;
};

const char * role_name(ChatMessage::Role role);
const char * tool_choice_name(ToolChoice::Kind kind);

} // namespace vbuf_agent
