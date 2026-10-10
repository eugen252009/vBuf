#include "vbuf_agent_protocol.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace vbuf_agent;

static bool rejects(const std::string & body, ProtocolError::Category expected) {
    try { (void)parse_chat_request(body); }
    catch (const ProtocolError & e) { return e.category == expected; }
    return false;
}
int main() {
    const std::string tool = R"({"type":"function","function":{"name":"add","description":"Add numbers","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}})";
    const std::string echo_tool = R"({"type":"function","function":{"name":"echo","description":"Echo text","parameters":{"type":"object","properties":{"text":{"type":"string"}}}}})";
    const std::string prefix = R"({"model":"test","messages":[{"role":"user","content":"calculate"}],"tools":[)";
    ChatRequest one = parse_chat_request(prefix + tool + "]}");
    assert(one.tools.size() == 1 && one.tools[0].name == "add");
    const ChatRequest string_content = parse_chat_request(
        R"({"model":"test","messages":[{"role":"user","content":"Reply with exactly: PI_VBUF_OK"}]})");
    const ChatRequest text_part_content = parse_chat_request(
        R"({"model":"test","messages":[{"role":"user","content":[{"type":"text","text":"Reply with exactly: PI_VBUF_OK"}]}]})");
    assert(string_content.messages[0].content == "Reply with exactly: PI_VBUF_OK");
    assert(text_part_content.messages[0].content == string_content.messages[0].content);
    const ChatRequest multiple_text_parts = parse_chat_request(
        R"({"model":"test","messages":[{"role":"user","content":[{"type":"text","text":"foo"},{"type":"text","text":"bar"}]}]})");
    const ChatRequest joined_text = parse_chat_request(
        R"({"model":"test","messages":[{"role":"user","content":"foobar"}]})");
    assert(multiple_text_parts.messages[0].content == "foobar");
    assert(multiple_text_parts.messages[0].content == joined_text.messages[0].content);
    const ChatRequest text_roles = parse_chat_request(
        R"({"model":"test","messages":[{"role":"system","content":[{"type":"text","text":"rules"}]},{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"text","text":"hello"}]}]})");
    assert(text_roles.messages[0].content == "rules" && text_roles.messages[2].content == "hello");
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]})", ProtocolError::Category::UnsupportedFeature));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":[{"type":"unknown_future_type"}]}]})", ProtocolError::Category::UnsupportedFeature));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AA=="}}]}]})", ProtocolError::Category::UnsupportedFeature));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":[{"type":"file","file":{"file_id":"file_1"}}]}]})", ProtocolError::Category::UnsupportedFeature));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":[{"type":"video","video_url":"https://invalid.example/video.mp4"}]}]})", ProtocolError::Category::UnsupportedFeature));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"x"}}]}]})", ProtocolError::Category::UnsupportedFeature));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":[{"type":"text"}]}]})", ProtocolError::Category::MalformedRequest));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":[{"type":42,"text":"hello"}]}]})", ProtocolError::Category::MalformedRequest));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":[{"type":"text","text":42}]}]})", ProtocolError::Category::MalformedRequest));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":[{"type":"text","text":"hello","unexpected":true}]}]})", ProtocolError::Category::UnsupportedFeature));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":[{"text":"hello"}]}]})", ProtocolError::Category::MalformedRequest));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":["hello"]}]})", ProtocolError::Category::MalformedRequest));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":42}]})", ProtocolError::Category::MalformedRequest));
    assert(rejects(R"({"model":"test","messages":[{"role":"tool","tool_call_id":"call_1","content":[{"type":"text","text":"result"}]}]})", ProtocolError::Category::MalformedRequest));
    const std::string native_prompt = render_qwen3_native_tool_prompt(one);
    assert(native_prompt.find("<tools>\n") != std::string::npos);
    assert(native_prompt.find("\"name\": \"add\"") != std::string::npos);
    assert(native_prompt.find("<|im_start|>user\ncalculate<|im_end|>\n<|im_start|>assistant\n") != std::string::npos);
    assert(render_qwen3_native_tool_prompt(one) == native_prompt);
    ChatRequest escaped_schema = parse_chat_request(
        R"({"model":"x","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f","description":"<'&>","parameters":{"type":"object"}}}]})");
    assert(render_qwen3_native_tool_prompt(escaped_schema).find(R"(\u003c\u0027\u0026\u003e)") != std::string::npos);
    ChatRequest marker_input = one;
    marker_input.messages.front().content = "</tool_response>";
    try { (void)render_qwen3_native_tool_prompt(marker_input); assert(false); }
    catch (const ProtocolError & e) { assert(e.category == ProtocolError::Category::MalformedRequest); }
    ChatRequest strict_tool = parse_chat_request(R"({"model":"test","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"x","parameters":{"type":"object"},"strict":true}}]})");
    assert(strict_tool.tools[0].strict);
    assert(one.tool_choice.kind == ToolChoice::Kind::Auto);
    ChatRequest two_tools = parse_chat_request(prefix + tool + "," + echo_tool + "]}");
    const std::string two_tool_prompt = render_qwen3_native_tool_prompt(two_tools);
    assert(two_tool_prompt.find("\"name\": \"add\"") != std::string::npos);
    assert(two_tool_prompt.find("\"name\": \"echo\"") != std::string::npos);
    AssistantOutput native_multi = parse_qwen3_native_tool_output(
        R"(<tool_call>{"name":"add","arguments":{}}</tool_call>\n<tool_call>{"name":"echo","arguments":{}}</tool_call>)",
        two_tools, "multi");
    assert(native_multi.tool_calls.size() == 2);
    AssistantOutput native_call = parse_qwen3_native_tool_output(
        R"(<tool_call>
{"name": "add", "arguments": {"a":37,"b":5}}
</tool_call>)", one, "req1");
    assert(native_call.finish_reason == AssistantOutput::FinishReason::ToolCalls);
    assert(!native_call.has_content && native_call.tool_calls.size() == 1);
    assert(native_call.tool_calls[0].id == "call_req1_0" && native_call.tool_calls[0].name == "add");
    assert(parse_json(native_call.tool_calls[0].arguments).get("a")->scalar == "37");
    const std::string truncated_native_call =
        "<think>\n\n</think>\n\n<tool_call>\n{\"name\":\"add\",\"arguments\":{\"a\":37";
    try {
        (void)parse_qwen3_native_tool_output(truncated_native_call, one, "truncated_without_limit");
        assert(false);
    } catch (const ProtocolError & e) { assert(e.category == ProtocolError::Category::Internal); }
    const auto truncated_native = parse_qwen3_native_tool_output(
        truncated_native_call, one, "truncated_at_limit", {}, true);
    assert(truncated_native.finish_reason == AssistantOutput::FinishReason::Length &&
        truncated_native.has_content && truncated_native.content.empty() && truncated_native.tool_calls.empty());
    const auto truncated_response = parse_json(serialize_chat_response(
        "truncated_at_limit", "model", truncated_native, 916, 6144));
    assert(truncated_response.get("choices")->array[0].get("finish_reason")->scalar == "length");
    assert(truncated_response.get("choices")->array[0].get("message")->get("content")->scalar.empty());
    assert(truncated_response.get("choices")->array[0].get("message")->get("tool_calls") == nullptr);
    const auto truncated_sse = stream_chat_response("truncated_at_limit", "model", truncated_native);
    const auto truncated_sse_result = reconstruct_stream(truncated_sse);
    assert(truncated_sse_result.finish_reason == AssistantOutput::FinishReason::Length &&
        truncated_sse_result.has_content && truncated_sse_result.content.empty() &&
        truncated_sse_result.tool_calls.empty());
    const auto partial_marker_at_limit = parse_qwen3_native_tool_output(
        "<tool_call", one, "partial_marker_at_limit", {}, true);
    assert(partial_marker_at_limit.finish_reason == AssistantOutput::FinishReason::Length &&
        partial_marker_at_limit.tool_calls.empty() && partial_marker_at_limit.content.empty());
    const auto partial_close_at_limit = parse_qwen3_native_tool_output(
        R"(<tool_call>{"name":"add","arguments":{}}</tool_call)",
        one, "partial_close_at_limit", {}, true);
    assert(partial_close_at_limit.finish_reason == AssistantOutput::FinishReason::Length &&
        partial_close_at_limit.tool_calls.empty());
    const auto completed_call_at_limit = parse_qwen3_native_tool_output(
        R"(<tool_call>{"name":"add","arguments":{}}</tool_call>)",
        one, "completed_at_limit", {}, true);
    assert(completed_call_at_limit.finish_reason == AssistantOutput::FinishReason::ToolCalls &&
        completed_call_at_limit.tool_calls.size() == 1);
    const auto complete_then_truncated = parse_qwen3_native_tool_output(
        R"(<tool_call>{"name":"add","arguments":{}}</tool_call><tool_call>{"name":"add","arguments":{"a":)",
        one, "complete_then_truncated", {}, true);
    assert(complete_then_truncated.finish_reason == AssistantOutput::FinishReason::Length &&
        complete_then_truncated.tool_calls.empty());
    try {
        (void)parse_qwen3_native_tool_output(
            R"(<tool_call>{bad}</tool_call>)", one, "malformed_at_limit", {}, true);
        assert(false);
    } catch (const ProtocolError & e) { assert(e.category == ProtocolError::Category::Internal); }
    const ChatRequest forced_length_request = parse_chat_request(
        prefix + tool + R"(],"tool_choice":{"type":"function","function":{"name":"add"}}})");
    const auto forced_truncated = parse_qwen3_native_tool_output(
        truncated_native_call, forced_length_request, "forced_truncated", {}, true);
    assert(forced_truncated.finish_reason == AssistantOutput::FinishReason::Length &&
        forced_truncated.tool_calls.empty());
    assert(parse_qwen3_native_tool_output("plain answer", one, "req2").content == "plain answer");
    AssistantOutput reasoned_call = parse_qwen3_native_tool_output(
        R"(<think>private reasoning</think>
<tool_call>{"name":"add","arguments":{}}</tool_call>)", one, "reasoned");
    assert(!reasoned_call.has_content && reasoned_call.tool_calls.size() == 1);
    try {
        (void)parse_qwen3_native_tool_output("<think>unfinished<tool_call>{}", one, "bad_reasoning");
        assert(false);
    } catch (const ProtocolError & e) { assert(e.category == ProtocolError::Category::Internal); }
    assert(parse_qwen3_native_tool_output(R"(preface <tool_call>{"name":"add","arguments":{}}</tool_call>)", one, "req3").content == "preface");
    const std::vector<std::string> bad_native = {
        R"(<tool_call>{"name":"add","arguments":{}})", "<tool_call></tool_call>", "<tool_call>{}</tool_call>",
        R"(<tool_call>{"name":"missing","arguments":{}}</tool_call>)",
        R"(<tool_call>{"name":"add","arguments":[]}</tool_call>)", "</tool_call>", "<tool_call>{bad}</tool_call>",
        "<tool_call"
        R"(<tool_call>{"name":"add","arguments":{}}</tool_call></tool_call>)"
    };
    for (const auto & invalid : bad_native) {
        try { (void)parse_qwen3_native_tool_output(invalid, one, "bad"); assert(false); }
        catch (const ProtocolError & e) { assert(e.category == ProtocolError::Category::Internal); }
        bool emitted = false;
        try {
            const auto parsed = parse_qwen3_native_tool_output(invalid, one, "bad_stream");
            (void)stream_chat_response("bad_stream", "model", parsed);
            emitted = true;
        } catch (const ProtocolError &) {}
        assert(!emitted);
    }
    ChatRequest empty = parse_chat_request(R"({"model":"test","messages":[{"role":"user","content":"hi"}],"tools":[]})");
    assert(empty.tools_were_supplied && empty.tools.empty());
    ChatRequest none = parse_chat_request(R"({"model":"test","messages":[{"role":"user","content":"hi"}],"tools":[],"tool_choice":"none"})");
    assert(none.tool_choice.kind == ToolChoice::Kind::None);
    ChatRequest none_without_tools = parse_chat_request(R"({"model":"test","messages":[{"role":"user","content":"hi"}],"tool_choice":"none"})");
    assert(none_without_tools.tool_choice.kind == ToolChoice::Kind::None && !none_without_tools.tools_were_supplied);
    ChatRequest forced = parse_chat_request(prefix + tool + R"(],"tool_choice":{"type":"function","function":{"name":"add"}}})");
    assert(forced.tool_choice.kind == ToolChoice::Kind::ForcedFunction && forced.tool_choice.function_name == "add");
    assert(rejects(prefix + tool + "," + tool + "]}", ProtocolError::Category::MalformedRequest));
    assert(rejects(prefix + R"({"type":"function","function":{"description":"x","parameters":{"type":"object"}}} ]})", ProtocolError::Category::MalformedRequest));
    assert(rejects(prefix + R"({"type":"function","function":{"name":"x","parameters":[]}}]})", ProtocolError::Category::MalformedRequest));
    assert(rejects(prefix + R"({"type":"not-function","function":{"name":"x","parameters":{"type":"object"}}}]})", ProtocolError::Category::MalformedRequest));
    assert(rejects(prefix + R"({"type":"function","function":{"name":"x","parameters":{"type":"object"},"strict":"yes"}}]})", ProtocolError::Category::MalformedRequest));
    assert(rejects(prefix + R"({"type":"function","function":{"name":"x","parameters":{"type":"object"},"extra":true}}]})", ProtocolError::Category::UnsupportedFeature));
    assert(rejects(prefix + tool + R"(],"tool_choice":{"type":"function","function":{"name":"missing"}}})", ProtocolError::Category::MalformedRequest));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":"x"}],"tool_choice":"required"})", ProtocolError::Category::MalformedRequest));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":"x"}],"max_tokens":0,"max_completion_tokens":1})", ProtocolError::Category::MalformedRequest));
    assert(rejects(R"({"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"x"}}]})", ProtocolError::Category::MalformedRequest));

    const std::string call_a = R"({"id":"call_a","type":"function","function":{"name":"add","arguments":"{\"a\":37,\"b\":5}"}})";
    const std::string call_b = R"({"id":"call_b","type":"function","function":{"name":"echo","arguments":"{\"text\":\"ok\"}"}})";
    const std::string prefix_messages = R"({"model":"test","messages":[{"role":"user","content":"do"},{"role":"assistant","content":null,"tool_calls":[)";
    ChatRequest multi = parse_chat_request(prefix_messages + call_a + "," + call_b + R"(]},{"role":"tool","tool_call_id":"call_b","content":"ok"},{"role":"tool","tool_call_id":"call_a","content":"42"},{"role":"assistant","content":"done"}]})");
    assert(multi.messages.size() == 5 && multi.messages[1].tool_calls.size() == 2);
    assert(multi.messages[2].tool_call_id == "call_b");
    ChatRequest multi_turn = parse_chat_request(R"({"model":"test","messages":[{"role":"user","content":"do"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"echo","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":"result"},{"role":"assistant","content":"used result"},{"role":"user","content":"follow up"}]})");
    assert(multi_turn.messages.size() == 5 && multi_turn.messages.back().content == "follow up");
    const std::string continuation_native = render_qwen3_native_tool_prompt(multi_turn);
    assert(continuation_native.find("<tool_response>\nresult\n</tool_response>") != std::string::npos);
    assert(continuation_native.find("<tool_call>") != std::string::npos);

    assert(rejects(prefix_messages + call_a + R"(]},{"role":"tool","tool_call_id":"unknown","content":"x"}]})", ProtocolError::Category::InvalidConversation));
    assert(rejects(prefix_messages + call_a + R"(]},{"role":"tool","tool_call_id":"call_a","content":"x"},{"role":"tool","tool_call_id":"call_a","content":"again"}]})", ProtocolError::Category::InvalidConversation));
    assert(rejects(prefix_messages + call_a + R"(]},{"role":"assistant","content":"skip tool result"}]})", ProtocolError::Category::InvalidConversation));
    assert(rejects(prefix_messages + R"({"id":"call_a","type":"function","function":{"name":"add","arguments":"[]"}}]})", ProtocolError::Category::MalformedRequest));
    assert(rejects(prefix_messages + R"({"id":"call_a","type":"function","function":{"name":"add","arguments":"{}"}},{"id":"call_a","type":"function","function":{"name":"add","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_a","content":"x"}]})", ProtocolError::Category::InvalidConversation));
    assert(rejects(prefix_messages + R"({"type":"function","function":{"name":"add","arguments":"{}"}}]})", ProtocolError::Category::MalformedRequest));
    assert(rejects(prefix_messages + R"({"id":"missing_name","type":"function","function":{"arguments":"{}"}}]})", ProtocolError::Category::MalformedRequest));
    assert(rejects(R"({"messages":[{"role":"user","content":"x"}],"messages":[{"role":"user","content":"y"}]})", ProtocolError::Category::MalformedRequest));
    assert(rejects(R"({"model":"test","messages":[{"role":"user","content":"x","name":"ignored"}]})", ProtocolError::Category::UnsupportedFeature));
    assert(rejects(R"({"model":"test","messages":[{"role":"tool","tool_call_id":"orphan","content":"x"}]})", ProtocolError::Category::InvalidConversation));

    AssistantOutput text; text.content = "hello π";
    AssistantOutput tools; tools.has_content = false; tools.finish_reason = AssistantOutput::FinishReason::ToolCalls; tools.tool_calls = {{"call_1", "add", R"({"a":37,"b":5})"}, {"call_2", "echo", R"({"text":"ok"})"}};
    const std::string response = serialize_chat_response("id", "model", tools, 7, 12);
    JsonValue decoded = parse_json(response);
    assert(decoded.get("choices")->array[0].get("finish_reason")->scalar == "tool_calls");
    assert(decoded.get("choices")->array[0].get("message")->get("tool_calls")->array.size() == 2);
    const auto events = stream_chat_response("id", "model", tools);
    const JsonValue first_tool_delta = parse_json(events[1]);
    const JsonValue & first_call_delta = first_tool_delta.get("choices")->array[0].get("delta")->get("tool_calls")->array[0];
    assert(first_call_delta.get("index")->scalar == "0");
    assert(first_call_delta.get("id")->scalar == tools.tool_calls[0].id);
    assert(first_call_delta.get("type")->scalar == "function");
    assert(first_call_delta.get("function")->get("name")->scalar == tools.tool_calls[0].name);
    AssistantOutput reconstructed = reconstruct_stream(events);
    assert(!reconstructed.has_content && reconstructed.tool_calls.size() == 2);
    assert(reconstructed.tool_calls[0].id == tools.tool_calls[0].id);
    assert(reconstructed.tool_calls[0].name == tools.tool_calls[0].name);
    assert(reconstructed.tool_calls[0].arguments == tools.tool_calls[0].arguments);
    assert(reconstructed.tool_calls[1].arguments == tools.tool_calls[1].arguments);
    std::vector<std::string> wire_records;
    assert(deliver_stream_records(events, [&](const std::string & frame) {
        wire_records.push_back(frame); return true;
    }));
    std::vector<std::string> delivered_records;
    for (const auto & frame : wire_records) {
        assert(frame.rfind("data: ", 0) == 0 && frame.size() >= 8 && frame.substr(frame.size() - 2) == "\n\n");
        delivered_records.push_back(frame.substr(6, frame.size() - 8));
    }
    assert(reconstruct_stream(delivered_records).tool_calls[0].arguments == tools.tool_calls[0].arguments);
    size_t write_attempts = 0;
    assert(!deliver_stream_records(events, [&](const std::string &) { return ++write_attempts < 3; }));
    assert(write_attempts == 3);
    AssistantOutput unicode_call; unicode_call.has_content = false;
    unicode_call.finish_reason = AssistantOutput::FinishReason::ToolCalls;
    unicode_call.tool_calls = {{"call_unicode", "echo", R"({"x":")" + std::string(249, 'a') + "π\"}"}};
    const auto unicode_events = stream_chat_response("id", "model", unicode_call);
    std::vector<std::string> argument_fragments;
    for (const auto & event : unicode_events) {
        if (event == "[DONE]") continue;
        const auto parsed = parse_json(event);
        const auto * calls = parsed.get("choices")->array[0].get("delta")->get("tool_calls");
        if (calls && calls->array[0].get("function") && calls->array[0].get("function")->get("arguments"))
            argument_fragments.push_back(calls->array[0].get("function")->get("arguments")->scalar);
    }
    assert(argument_fragments.size() == 2 && argument_fragments[1].rfind("π", 0) == 0);
    assert(reconstruct_stream(unicode_events).tool_calls[0].arguments == unicode_call.tool_calls[0].arguments);
    auto terminal_delta = events;
    const std::string last_call_delta = terminal_delta[terminal_delta.size() - 3];
    terminal_delta.erase(terminal_delta.end() - 3);
    JsonValue last_fragment = parse_json(last_call_delta);
    JsonValue terminal_event = parse_json(terminal_delta[terminal_delta.size() - 2]);
    JsonValue delta = *terminal_event.get("choices")->array[0].get("delta");
    const auto * fragmented_calls = last_fragment.get("choices")->array[0].get("delta")->get("tool_calls");
    delta.object.emplace_back("tool_calls", *fragmented_calls);
    terminal_event.object[3].second.array[0].object[1].second = delta;
    terminal_delta[terminal_delta.size() - 2] = serialize_json(terminal_event);
    AssistantOutput terminal_reconstructed = reconstruct_stream(terminal_delta);
    assert(terminal_reconstructed.tool_calls.size() == tools.tool_calls.size());
    assert(terminal_reconstructed.tool_calls.back().arguments == tools.tool_calls.back().arguments);
    AssistantOutput mixed; mixed.content = "preface"; mixed.finish_reason = AssistantOutput::FinishReason::ToolCalls; mixed.tool_calls = {{"call_mixed", "echo", "{}"}};
    const auto mixed_wire = stream_chat_response("id", "model", mixed);
    AssistantOutput mixed_reconstructed = reconstruct_stream(mixed_wire);
    JsonValue mixed_json = parse_json(serialize_chat_response("id", "model", mixed, 1, 2));
    assert(mixed_json.get("choices")->array[0].get("message")->get("content")->scalar == "preface");
    assert(mixed_json.get("choices")->array[0].get("message")->get("tool_calls")->array.size() == 1);
    assert(mixed_reconstructed.has_content && mixed_reconstructed.content == "preface");
    assert(mixed_reconstructed.tool_calls.size() == 1 && mixed_reconstructed.tool_calls[0].id == "call_mixed");
    try { auto invalid = events; invalid.pop_back(); (void)reconstruct_stream(invalid); assert(false); }
    catch (const ProtocolError &) {}
    try { auto invalid = events; invalid.push_back("[DONE]"); (void)reconstruct_stream(invalid); assert(false); }
    catch (const ProtocolError &) {}
    const auto text_events = stream_chat_response("id", "model", text);
    AssistantOutput text_reconstructed = reconstruct_stream(text_events);
    assert(text_reconstructed.has_content && text_reconstructed.content == text.content);
    assert(parse_json(serialize_chat_response("id", "model", text, 2, 3)).get("choices")->array[0].get("finish_reason")->scalar == "stop");
    AssistantOutput length_text = text; length_text.finish_reason = AssistantOutput::FinishReason::Length;
    assert(parse_json(serialize_chat_response("id", "model", length_text, 2, 3)).get("choices")->array[0].get("finish_reason")->scalar == "length");
    assert(reconstruct_stream(stream_chat_response("id", "model", length_text)).finish_reason == AssistantOutput::FinishReason::Length);
    DeterministicFakeAdapter fake({tools, text});
    ChatRequest plain = parse_chat_request(prefix + tool + "," + echo_tool + "]}");
    assert(fake.render(plain).find("messages=1") != std::string::npos);
    AssistantOutput fake_call = fake.parse_generation("ignored native bytes", plain);
    assert(fake_call.tool_calls.size() == 2);
    assert(fake.parse_generation("ignored", plain).content == "hello π");
    assert(fake.remaining() == 0);
    assert(rejects(R"({"messages":[{"role":"user","content":"x"}],"temperature":0})", ProtocolError::Category::UnsupportedFeature));

    // The test harness, not vBuf, executes these harmless deterministic fixture tools.
    DeterministicFakeAdapter no_tool_fake({text});
    ChatRequest no_tool_request = parse_chat_request(R"({"model":"test","messages":[{"role":"user","content":"Reply hello"}]})");
    assert(no_tool_fake.parse_generation("fixture text", no_tool_request).content == "hello π");
    AssistantOutput first_call; first_call.has_content = false; first_call.finish_reason = AssistantOutput::FinishReason::ToolCalls; first_call.tool_calls = {{"call_a", "add", R"({"a":37,"b":5})"}};
    AssistantOutput second_call_fixture; second_call_fixture.has_content = false; second_call_fixture.finish_reason = AssistantOutput::FinishReason::ToolCalls; second_call_fixture.tool_calls = {{"call_b", "echo", R"({"text":"42"})"}};
    DeterministicFakeAdapter loop({first_call, second_call_fixture, text});
    ChatRequest initial = parse_chat_request(prefix + tool + "]}");
    AssistantOutput requested = loop.parse_generation("fixture generation", initial);
    assert(requested.tool_calls.size() == 1 && requested.tool_calls[0].name == "add" && requested.tool_calls[0].id == "call_a");
    std::string continued_json = prefix_messages + call_a + R"(]},{"role":"tool","tool_call_id":"call_a","content":"42"}],"tools":[)" + tool + "," + echo_tool + "]}";
    ChatRequest continuation = parse_chat_request(continued_json);
    assert(loop.render(continuation).find("tool:42[call=call_a]") != std::string::npos);
    AssistantOutput second_call = loop.parse_generation("fixture second tool call", continuation);
    assert(second_call.tool_calls.size() == 1 && second_call.tool_calls[0].name == "echo");
    ChatRequest followup = parse_chat_request(prefix_messages + call_a + R"(]},{"role":"tool","tool_call_id":"call_a","content":"42"},{"role":"assistant","content":null,"tool_calls":[)" + call_b + R"(]},{"role":"tool","tool_call_id":"call_b","content":"42"},{"role":"assistant","content":"hello π"},{"role":"user","content":"follow up"}],"tools":[)" + tool + "," + echo_tool + "]}");
    assert(loop.render(followup).find("user:follow up") != std::string::npos);
    assert(loop.parse_generation("fixture next turn", followup).content == "hello π");
    assert(loop.remaining() == 0);
    ChatRequest forced_request = parse_chat_request(prefix + tool + R"(],"tool_choice":{"type":"function","function":{"name":"add"}}})");
    try { validate_assistant_output_for_request(forced_request, text); assert(false); }
    catch (const ProtocolError & e) { assert(e.category == ProtocolError::Category::Internal); }
    ChatRequest none_with_tools = parse_chat_request(prefix + tool + R"(],"tool_choice":"none"})");
    try { validate_assistant_output_for_request(none_with_tools, tools); assert(false); }
    catch (const ProtocolError & e) { assert(e.category == ProtocolError::Category::Internal); }
    AssistantOutput undeclared; undeclared.has_content = false; undeclared.finish_reason = AssistantOutput::FinishReason::ToolCalls; undeclared.tool_calls = {{"unknown_call", "missing", "{}"}};
    try { validate_assistant_output_for_request(initial, undeclared); assert(false); }
    catch (const ProtocolError & e) { assert(e.category == ProtocolError::Category::Internal); }
    AssistantOutput malformed; malformed.has_content = false; malformed.finish_reason = AssistantOutput::FinishReason::ToolCalls; malformed.tool_calls = {{"x", "f", "not-json"}};
    try { DeterministicFakeAdapter bad_fake({malformed}); (void)bad_fake.parse_generation("", plain); assert(false); }
    catch (const ProtocolError & e) { assert(e.category == ProtocolError::Category::Internal); }

    const std::string invalid_utf8 = R"({"messages":[{"role":"user","content":")" + std::string(1, static_cast<char>(0xff)) + R"("}]})";
    assert(rejects(invalid_utf8, ProtocolError::Category::MalformedRequest));
    Limits tiny; tiny.request_bytes = 20;
    try { (void)parse_chat_request(R"({"messages":[]})", tiny); assert(false); }
    catch (const ProtocolError & e) { assert(e.category == ProtocolError::Category::MalformedRequest); }
    Limits shallow; shallow.max_json_depth = 2;
    try { (void)parse_chat_request(R"({"messages":[{"role":"user","content":"x"}]})", shallow); assert(false); }
    catch (const ProtocolError & e) { assert(e.category == ProtocolError::Category::MalformedRequest); }
    Limits small_schema; small_schema.max_schema_bytes = 12;
    try { (void)parse_chat_request(prefix + tool + "]}", small_schema); assert(false); }
    catch (const ProtocolError & e) { assert(e.category == ProtocolError::Category::MalformedRequest); }
    Limits small_result; small_result.max_tool_result_bytes = 2;
    try { (void)parse_chat_request(prefix_messages + call_a + R"(]},{"role":"tool","tool_call_id":"call_a","content":"long"}]})", small_result); assert(false); }
    catch (const ProtocolError & e) { assert(e.category == ProtocolError::Category::MalformedRequest); }
    std::cout << "AGENT_PROTOCOL_CONTRACT=PASS\n";
}
