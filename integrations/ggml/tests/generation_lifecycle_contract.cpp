#include "vbuf_generation.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char ** argv) {
    require(argc == 2 || argc == 3, "expected semantic model and optional source endpoint");
    const std::string semantic_model = argv[1];

    auto runtime = std::make_shared<vbuf_ggml::VbufModelRuntime>(semantic_model, 1);
    std::weak_ptr<vbuf_ggml::VbufModelRuntime> runtime_lifetime = runtime;
    auto session_a = runtime->create_session();
    auto session_b = runtime->create_session();
    require(session_a && session_b, "runtime failed to create both sessions");
    require(session_a->current_context_length() == 0, "session A did not start empty");
    require(session_b->current_context_length() == 0, "session B did not start empty");

    session_a->reset();
    require(session_a->current_context_length() == 0, "reset did not clear session A");
    require(session_b->current_context_length() == 0, "reset affected session B");

    vbuf_ggml::VbufGenerationConfig invalid_request;
    invalid_request.block_count = 1;
    const auto failed_a = session_a->run(invalid_request);
    require(!failed_a.error.empty(), "empty request was not rejected");
    require(session_a->current_context_length() == 0, "failed request left session A state");
    require(session_b->current_context_length() == 0, "failed request changed session B");
    require(runtime->snapshot().request_count == 1, "runtime request count was not shared");

    runtime.reset();
    require(!runtime_lifetime.expired(), "runtime did not outlive its external owner");
    session_a.reset();
    require(!runtime_lifetime.expired(), "destroying session A invalidated session B runtime");

    const auto failed_b = session_b->run(invalid_request);
    require(!failed_b.error.empty(), "second empty request was not rejected");
    require(session_b->current_context_length() == 0, "failed request left session B state");
    require(session_b->snapshot().request_count == 2, "session snapshot did not reflect runtime requests");
    session_b.reset();
    require(runtime_lifetime.expired(), "runtime remained alive after its last session was destroyed");

    vbuf_ggml::VbufGenerationConfig compatibility_request;
    compatibility_request.semantic_model = semantic_model;
    compatibility_request.block_count = 1;
    const auto compatibility_result = vbuf_ggml::run_vbuf_generation(compatibility_request);
    require(!compatibility_result.error.empty(), "whole-request run wrapper changed empty-prompt behavior");
    auto reusable_runtime = std::make_shared<vbuf_ggml::VbufModelRuntime>(semantic_model, 1);
    const auto runtime_result = reusable_runtime->run(compatibility_request);
    require(!runtime_result.error.empty(), "model-runtime run wrapper changed empty-prompt behavior");
    require(reusable_runtime->snapshot().request_count == 1,
        "model-runtime run wrapper did not execute through a temporary session");

    bool rejected_stack_owned_runtime = false;
    try {
        vbuf_ggml::VbufModelRuntime stack_runtime(semantic_model, 1);
        (void) stack_runtime.create_session();
    } catch (const std::runtime_error &) {
        rejected_stack_owned_runtime = true;
    }
    require(rejected_stack_owned_runtime, "stack-owned runtime unexpectedly created a session");

    bool rejected_null_runtime = false;
    try {
        vbuf_ggml::VbufGenerationSession invalid_session(nullptr);
        (void) invalid_session;
    } catch (const std::runtime_error &) {
        rejected_null_runtime = true;
    }
    require(rejected_null_runtime, "null runtime unexpectedly created a session");

    bool rejected_invalid_model = false;
    try {
        vbuf_ggml::VbufModelRuntime invalid_runtime(semantic_model + ".missing", 1);
        (void) invalid_runtime;
    } catch (const std::exception &) {
        rejected_invalid_model = true;
    }
    require(rejected_invalid_model, "invalid model runtime unexpectedly initialized");

    if (argc == 3) {
        auto shared_runtime = std::make_shared<vbuf_ggml::VbufModelRuntime>(semantic_model, 1);
        auto first = shared_runtime->create_session();
        auto second = shared_runtime->create_session();
        vbuf_ggml::VbufGenerationConfig request;
        request.semantic_model = semantic_model;
        request.source_endpoint = argv[2];
        request.block_count = 1;
        request.max_new_tokens = 2;
        request.prompt_tokens = {1};
        request.source_failure_requests = 1;
        const auto injected_failure = first->run(request);
        require(!injected_failure.error.empty() && injected_failure.source_failure_injected,
            "injected source failure was not reported");
        require(first->current_context_length() == 0, "failed source request left session state");
        require(second->current_context_length() == 0, "failed source request changed the other session");

        request.source_failure_requests = 0;
        const auto first_result = first->run(request);
        require(first_result.error.empty() && first_result.completed, "session A generation failed");
        require(first->current_context_length() == 2, "session A logical length is incorrect");
        require(second->current_context_length() == 0, "session A changed session B state");
        const auto after_first = shared_runtime->snapshot();
        require(after_first.source_requests > 0, "first generation did not exercise the shared source");

        request.prompt_tokens = {2};
        const auto second_result = second->run(request);
        require(second_result.error.empty() && second_result.completed, "session B generation failed");
        require(second->current_context_length() == 2, "session B logical length is incorrect");
        require(first->current_context_length() == 2, "session B changed session A state");
        const auto after_second = shared_runtime->snapshot();
        require(after_second.source_requests >= after_first.source_requests &&
                after_second.source_bytes >= after_first.source_bytes,
            "model source diagnostics were recreated instead of reused");
        require(after_second.materializations >= after_first.materializations,
            "model residency was recreated instead of reused");
        second->reset();
        require(second->current_context_length() == 0, "reset did not clear session B");
        require(first->current_context_length() == 2, "reset of session B changed session A");
        first->reset();
        require(first->current_context_length() == 0, "reset did not clear session A");
        require(shared_runtime->snapshot().request_count == 3,
            "shared model runtime did not retain all request counts");
    }

    return 0;
}
